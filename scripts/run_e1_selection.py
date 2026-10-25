"""E1: воронка отбора признаков. Каждый этап сохраняет результат в artifacts/E1/ и при перезапуске
пропускается, если результат уже есть (контрольные точки на случай падения WSL).

  python scripts/run_e1_selection.py filter   # 1. фильтры
  python scripts/run_e1_selection.py rank     # 2. ранжирование: SHAP внутри групп, SHAP общий, L1 (+ повтор на другой выборке для стабильности)
  python scripts/run_e1_selection.py greedy   # 3. жадный отбор по GAUC на val_a (долго, ~1–1.5 ч)
  python scripts/run_e1_selection.py greedy_nosess   # 3б. то же без сессионных признаков (JOURNAL I7)
  python scripts/run_e1_selection.py curves   # 4. кривые «качество vs число признаков» для всех способов отбора
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src import memguard
from src.candidates import load_candidates
from src.features import E0_FEATURES
from src.metrics import evaluate
from src.selection import (catalog, filter_features, fit_light, gauc, greedy_forward, l1_rank, load_matrix,
                           shap_within_rank)

ART = Path(__file__).resolve().parents[1] / "artifacts" / "E1"
ART.mkdir(parents=True, exist_ok=True)
POOL = 60
SEED = 0


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(ART / "selection_log.txt", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def samples(seed=SEED):
    c = load_candidates()
    #случайно выбираем из них 600 тыс и упорядочиванием по номеру строки
    # метрики label берутся по позиции потом
    tr = c.filter(pl.col("part") == "train").sample(600_000, seed=seed).sort("row_id")
    va = c.filter(pl.col("part") == "val_a")
    vb = c.filter(pl.col("part") == "val_b")
    return c, tr, va, vb


#первый этап воронки отбора, где убираются явно бесполезные признаки
def stage_filter():
    out = ART / "filter.json"
    #если рухнет wsl, то этап уже сохранен и идем дальше
    if out.exists():
        return log("filter: уже есть")
    
    _, tr, _, _ = samples()
    cat = catalog()
    s = tr.sample(200_000, seed=1)

    #меньше 3 ГБ памяти, стоп
    memguard.check(3.0, "фильтры")
    t0 = time.perf_counter()
    # из 43 файлов читаем нужные колонки и присоединяем к строкам
    X = load_matrix(s, cat["feature"].to_list(), cat)

    kept, rep = filter_features(X, cat["feature"].to_list(), s["label"].to_numpy())
    
    
    rep.write_parquet(ART / "filter_removed.parquet")
    out.write_text(json.dumps({"all": cat.height, "kept": kept, "seconds": time.perf_counter() - t0}, ensure_ascii=False))
    log(f"filter: {cat.height} -> {len(kept)} признаков за {time.perf_counter() - t0:.0f} с; "
        f"пик {memguard.peak_rss_gb():.1f} ГБ")


def _rank(seed: int, suffix: str):
    out = ART / f"ranking{suffix}.parquet"
    if out.exists():
        return log(f"rank{suffix}: уже есть")

    
    kept = json.loads((ART / "filter.json").read_text())["kept"]

    _, tr, va, _ = samples(seed)
    cat = catalog()

    rng = np.random.default_rng(seed)
    t_tr = tr.sample(300_000, seed=seed)

    #берем именно треть пользователей, потому что SHAP внутр групп и GAUC считаеются по группа пользователь*день
    #группы должны быть целыми
    users = va["uid"].unique().to_numpy()
    #replace=False- чтобы каждый пользователь не был выбран несколько раз
    va_s = va.filter(pl.col("uid").is_in(rng.choice(users, len(users) // 3, replace=False).tolist()))
    memguard.check(3.0, "ранжирование")
    t0 = time.perf_counter()

    Xtr = load_matrix(t_tr, kept, cat).select(kept).to_numpy()
    memguard.check(2.0, 'ранжирование: после train')
    Xva = load_matrix(va_s, kept, cat).select(kept).to_numpy()
    #ЛЕГКАЯ МОДЕЛЬ на 666 признаках
    m = fit_light(Xtr, t_tr["label"].to_numpy(), Xva, va_s["label"].to_numpy())
    t_fit = time.perf_counter() - t0

    #SHAP внутри групп и общий
    groups = va_s.select("uid", "day").unique().sort("uid", "day")
    #сулчайные 600 групп
    pick = groups[rng.choice(groups.height, 600, replace=False)]

    g = va_s.with_row_index("_i").join(pick, on=["uid", "day"]).sort("uid", "day")
    shap_tab = shap_within_rank(m, Xva[g["_i"].to_numpy()], g, kept)
    t_shap = time.perf_counter() - t0 - t_fit
        #ПРИЗНАК трека разный у разных треков, его SHAP внутри группы гуляет -> РАНЖИРУЕТ! 

    gauc_all = gauc(m, Xva, va_s.select("uid", "day", "label"))
    del Xva #свободим память перед L1

    l1 = l1_rank(Xtr[:100_000], t_tr["label"].to_numpy()[:100_000], kept)
    t_l1 = time.perf_counter() - t0 - t_fit - t_shap

    #исползьуем встроенную важность lgnm
    gain = dict(zip(kept, m.feature_importance("gain")))
    tab = (shap_tab.join(l1, on="feature")
           #repalce_strict - заменяет имя признака на его значение из словаря и так добавляет колонку gain
           .with_columns(pl.col("feature").replace_strict(gain, return_dtype=pl.Float64).alias("gain"))
           .sort("shap_within", descending=True))      # join в polars 2 не сохраняет порядок — сортируем явно
    tab.write_parquet(out)

    (ART / f"ranking{suffix}_cost.json").write_text(json.dumps(
        {"fit_s": t_fit, "shap_s": t_shap, "l1_s": t_l1, "trees": m.best_iteration,
         "gauc_all_kept": gauc_all}))
    log(f"rank{suffix}: модель {t_fit:.0f} с ({m.best_iteration} деревьев), SHAP {t_shap:.0f} с, L1 {t_l1:.0f} с; "
        f"пик {memguard.peak_rss_gb():.1f} ГБ")


def stage_rank():
    _rank(SEED, "")


def stage_rank_seed1():
    _rank(SEED + 1, "_seed1")       # повтор на другой выборке — для оценки стабильности ранжирования


def stage_greedy(session: bool = True, seed: int = SEED):
    """session=False — без сессионного семейства: они описывают момент, а не трек
    seed=1 — вся воронка заново на другой подвыборке (ранжирование ranking_seed1 + другой train) — проверка стабильности"""

    suffix = "" if seed == SEED else f"_seed{seed}"

    rank = pl.read_parquet(ART / f"ranking{suffix}.parquet").sort("shap_within", descending=True)
    
    if not session:
        #так как мы тестировать потом стали без сессионных признаков, то вот опция
        rank = rank.filter(~pl.col("feature").str.starts_with("s__"))

    #жадые отбор выбирает отлько из POOL фильтров
    #по умолчанию 60, или наче wsl было бы плохо
    pool = rank["feature"][:POOL].to_list()
    name = ("greedy" if session else "greedy_nosess") + suffix
    _, tr, va, _ = samples(seed)
    cat = catalog()
    Xtr = load_matrix(tr, pool, cat)
    Xva = load_matrix(va, pool, cat)

    log(f"{name}: пул {len(pool)} признаков (топ по SHAP внутри групп), train {tr.height}, val_a {va.height}")
    # без сессии — без порога остановки, до 15 признаков: число признаков потом выбираем по пику GAUC на val_a
    max_k, min_gain = (25, 0.0005) if session else (15, -1.0)
    greedy_forward(Xtr, tr["label"].to_numpy(), Xva, va["label"].to_numpy(), va.select("uid", "day", "label"),
                   pool, max_k=max_k, min_gain=min_gain, checkpoint=ART / f"{name}.json", log=log)
    log(f"{name}: готово; пик {memguard.peak_rss_gb():.1f} ГБ")


def stage_curves():
    out = ART / "curves.parquet"
    if out.exists():
        return log("curves: уже есть")
    
    rank = pl.read_parquet(ART / "ranking.parquet")
    greedy = [s["added"] for s in json.loads((ART / "greedy.json").read_text())]

    #пять упорядоченных списков приизнаков, по одному на способ
    #в каждом первый элемент - самый важный признак по мнению этого способа
    orders = {
        "SHAP внутри групп": rank.sort("shap_within", descending=True)["feature"].to_list(),
        "SHAP общий": rank.sort("shap_total", descending=True)["feature"].to_list(),
        "L1": rank.filter(pl.col("l1_abs") > 0).sort("l1_abs", descending=True)["feature"].to_list(),
        "gain": rank.sort("gain", descending=True)["feature"].to_list(),
        "жадный (GAUC)": greedy,
    }
    #точки кривой
    ks =  [3, 5, 10, 15, 20, 25, 30, 40, 60] 
    # все признаки, котоыре понадобятся хоть в одной точке: обьединение первыз 60 из каждого списка
    need = sorted({f for o in orders.values() for f in o[:max(ks)]})

    _, tr, va, vb = samples()
    cat = catalog()
    memguard.check(3.0, "кривые")

    X = {n: load_matrix(d, need, cat) for n, d in [("tr", tr), ("va", va), ("vb", vb)]}
    
    from src.features import build_e0_features
    feats0 = pl.read_parquet(Path(__file__).resolve().parents[1] / "data" / "processed" / "features_E0.parquet")
    rows = []

    #одна точка кривой
    def run(name, k, cols, Xs):
        t0 = time.perf_counter()
        #легкая модель
        m = fit_light(Xs["tr"].select(cols).to_numpy(), tr["label"].to_numpy(),
                      Xs["va"].select(cols).to_numpy(), va["label"].to_numpy())
        fit_s = time.perf_counter() - t0
        for part, d in [("val_a", va), ("val_b", vb)]:
            key = "va" if part == "val_a" else "vb"
            s = d.select("uid", "day", "label").with_columns(
                    p=pl.Series( m.predict(Xs[key].select(cols).to_numpy(), 
                                           num_iteration=m.best_iteration)))
            #всеми метриками оцениваем
            rows.append({"selector": name, "k": k, "part": part, "fit_s": fit_s, "trees": m.best_iteration,
                         **evaluate(s, "p")})

    for name, order in orders.items():
        for k in ks:
            if k <= len(order):
                run(name, k, order[:k], X)
                log(f"curves: {name} k={k} готово")

    X0 = {n: d.select("row_id").join(feats0, on="row_id", how="left", maintain_order="left") for n, d in [("tr", tr), ("va", va), ("vb", vb)]}
    run("E0 (ручные)", len(E0_FEATURES), E0_FEATURES, X0)
    pl.DataFrame(rows).write_parquet(out)
    log(f"curves: готово; пик {memguard.peak_rss_gb():.1f} ГБ")


if __name__ == "__main__":
    {"filter": stage_filter, "rank": stage_rank, "rank_seed1": stage_rank_seed1, "greedy": stage_greedy,
     "greedy_nosess": lambda: stage_greedy(session=False),
     "greedy_nosess_seed1": lambda: stage_greedy(session=False, seed=SEED + 1),
     "curves": stage_curves}[sys.argv[1]]()
