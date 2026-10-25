"""E1: сборка матрицы признаков из файлов грамматики и воронка отбора

Воронка:
  1. filter_features   — константы, почти пустые, почти-дубли (|корреляция| > 0.98)
  2. shap_within_rank  — ранжирование по внутригрупповому вкладу SHAP 
     l1_rank           — ранжирование по L1-логрегрессии (второй, независимый способ)
  3. greedy_forward    — жадный отбор по GAUC на val_a из пула лучших по SHAP

финальная модель — LGB_PARAMS как в E0
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src import memguard
from src.grammar import E1_DIR, FAMILIES, WINDOWS, SESSION_FEATURES, len_bucket, wname
from src.metrics import ranking_metrics

LIGHT_PARAMS = {
    "objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 1.0, "bagging_fraction": 1.0, "lambda_l2": 1.0,
    "num_threads": 10, "seed": 0, "deterministic": True, "force_row_wise": True, "verbose": -1,
}
LIGHT_ROUNDS, LIGHT_STOP = 400, 30


# ------------------------------------------------------------------ каталог признаков
def catalog() -> pl.DataFrame:
    """Все признаки E1 с разбором имени:
      семейство, фильтр, агрегат, окно, файл."""
    
    rows = []
    for fam in FAMILIES:
        for w in WINDOWS:
            path = E1_DIR / f"{fam}__{wname(w)}.parquet"

            #нам нужны чисто имена и типы колонок, а памяти то мало
            for name in pl.read_parquet_schema(path):
                #колонки ключи типо uid, utem_id пропускаем
                if not name.startswith(f"{fam}__"):
                    continue

                parts = name.split("__")
                fa = parts[1].split("_", 1)
                rows.append({"feature": name, "family": fam, "window": parts[2] if len(parts) > 2 else "-",
                             
                             "filter": fa[0] if len(fa) == 2 else "-", "agg": fa[1] if len(fa) == 2 else fa[0],
                             "file": path.name})
    
    for name, _ in SESSION_FEATURES:
        rows.append({"feature": name, "family": "s", "window": "сессия", "filter": "-",
                     "agg": name.split("__")[1], "file": "s.parquet"})
    
    # для жадного отбора:
    #- по каталогу находим, в каких файлах что лежит и группируем по файлам
    # из каждого файла читаем нужные колонки
    # для обычных семейств признеаки хранятся на уровне (ключ, день) -> поэтому к кандидатам прикрепляем их по ключу и дню
    # для s.parque -> соединяем по row_id

    #без каталога в мою память бы не влезло
    return pl.DataFrame(rows)


def load_matrix(cands: pl.DataFrame, features: list[str], cat: pl.DataFrame | None = None) -> pl.DataFrame:
    """row_id + запрошенные признаки для строк cands. Из каждого файла читаем только нужные колонки"""
    
    cat = catalog() if cat is None else cat
    need = cat.filter(pl.col("feature").is_in(features))
    base = cands.select("row_id", "uid", "item_id", "artist_id", "day", "track_length_seconds").with_columns(len_bucket())
    out = base.select("row_id")
    
    for file, grp in need.group_by("file", maintain_order=True):
        cols = grp["feature"].to_list()
        
        if file[0] == "s.parquet":
            t = pl.scan_parquet(E1_DIR / file[0]).select("row_id", *cols).join(base.lazy().select("row_id"), on="row_id")
            out = out.join(t.collect(), on="row_id", how="left", maintain_order="left")
        
        else:
            keys = [*FAMILIES[grp["family"][0]], "day"]
            t = (pl.scan_parquet(E1_DIR / file[0]).select(*keys, *cols)
                 .join(base.lazy().select(*keys).unique(), on=keys, how="semi"))
            out = out.join(base.select("row_id", *keys).join(t.collect(), on=keys, how="left").select("row_id", *cols),
                           on="row_id", how="left", maintain_order="left")
    
    
    # порядок строк = порядку cands: по нему сопоставляются метки (в polars 2 join без maintain_order его не гарантирует)
    assert out["row_id"].equals(cands["row_id"]), "load_matrix: порядок строк нарушен"
    return out.select("row_id", *features)


# ------------------------------------------------------------------ лёгкая модель и оценка
def fit_light(X_tr, y_tr, X_va, y_va):
    dtr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    return lgb.train(LIGHT_PARAMS, dtr, LIGHT_ROUNDS, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(LIGHT_STOP, verbose=False)])


def gauc(model, X, meta: pl.DataFrame) -> float:
    s = meta.with_columns(p=pl.Series(model.predict(X, num_iteration=model.best_iteration)))
    return ranking_metrics(s, "p")["gauc"]


# ------------------------------------------------------------------ 1. фильтры
def filter_features(sample: pl.DataFrame, feats: list[str], y: np.ndarray,
                    max_null: float = 0.99, max_corr: float = 0.98) -> tuple[list[str], pl.DataFrame]:
    """Возвращает (оставшиеся признаки, таблицу причин удаления)"""
    from sklearn.metrics import roc_auc_score
    
    reasons = {}
    X = sample.select(feats)
    #число пропусков в каждой колонке
    null = X.null_count().row(0)
    #стандратное отклд каждой колонки
    std = X.select(pl.all().std()).row(0)
    keep = []
    
    
    for f, n, s in zip(feats, null, std):
        if n / X.height > max_null:
            reasons[f] = "почти пустой"
        elif s is None or s == 0:
            reasons[f] = "константа"
        else:
            keep.append(f)
    
    
    # одномерная сила признака: |AUC − 0.5| (пропуски заполняем медианой колонки) — при корреляции оставляем сильнейший
    A = X.select(keep).with_columns(pl.all().fill_null(pl.all().median())).to_numpy().astype(np.float64)
    #A[:,j] - вся колонка номер j - каждый признак
    #если бы какждллый признак был моделью - его значение = скор
    power = np.array([abs(roc_auc_score(y, A[:, j]) - 0.5) for j in range(A.shape[1])])
    #от самого сильного признака к самому слабому
    order = np.argsort(-power)
    
    #стандартизуем
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    #смотрим корреляцию признаков
    C = np.abs(A.T @ A / len(A))

    # так как идем от сильного к слабому, то убираются слабые, а сильные остаются
    kept_idx = []
    for j in order:
        dup = [k for k in kept_idx if C[j, k] > max_corr]
        if dup:
            reasons[keep[j]] = f"дубль {keep[dup[0]]}"
        else:
            kept_idx.append(j)
    
    kept = [keep[j] for j in sorted(kept_idx)]
    rep = pl.DataFrame({"feature": list(reasons), "причина": list(reasons.values())})
    return kept, rep


# ------------------------------------------------------------------ 2. ранжирование
def shap_within_rank(model, X_groups: np.ndarray, meta_groups: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """Внутригрупповой вклад: std SHAP признака внутри группы user×day, усреднённое по группам"""
    
    import shap
    sv = shap.TreeExplainer(model).shap_values(X_groups)
    sv = sv[1] if isinstance(sv, list) else sv
    
    S = pl.DataFrame(sv, schema=feats).with_columns(meta_groups.select("uid", "day"))
    
    within = S.group_by("uid", "day").agg(pl.col(feats).std()).select(pl.col(feats).mean()).row(0)
    total = S.select(pl.col(feats).std()).row(0)
    
    return (pl.DataFrame({"feature": feats, "shap_within": within, "shap_total": total})
            .fill_null(0).sort("shap_within", descending=True))


def l1_rank(X: np.ndarray, y: np.ndarray, feats: list[str], C: float = 0.01) -> pl.DataFrame:
    """L1-логрегрессия на стандартизированных признаках (log1p для счётчиков, пропуски = медиана).
    Сила признака = |вес|; признаки с нулевым весом L1 «выключила»"""
    
    from sklearn.linear_model import LogisticRegression
    from src.models import is_count_feature
    
    Z = X.astype(np.float64).copy()
    
    for j, f in enumerate(feats):
        col = Z[:, j]
        col[np.isnan(col)] = np.nanmedian(col) if np.isfinite(np.nanmedian(col)) else 0
        if is_count_feature(f.split("__")[1] if "__" in f else f):
            col[:] = np.log1p(np.maximum(col, 0))
        Z[:, j] = (col - col.mean()) / (col.std() + 1e-12)
    
    m = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=200).fit(Z, y)
    return (pl.DataFrame({"feature": feats, "l1_weight": m.coef_[0]})
            .with_columns(pl.col("l1_weight").abs().alias("l1_abs")).sort("l1_abs", descending=True))


# ------------------------------------------------------------------ 3. жадный отбор
def greedy_forward(Xtr: pl.DataFrame, ytr, Xva: pl.DataFrame, yva, meta_va: pl.DataFrame, pool: list[str],
                   max_k: int = 25, min_gain: float = 0.0005, checkpoint: Path | None = None, log=print) -> list[dict]:
    """На каждом шаге пробует добавить каждый признак пула и берёт тот, что сильнее всего поднял GAUC на val_a.
    Останавливается, когда прирост < min_gain или набрано max_k признаков.
    Контрольная точка (json) после каждого шага: при перезапуске продолжает с того же места"""
    
    steps = json.loads(checkpoint.read_text()) if checkpoint and checkpoint.exists() else []
    chosen = [s["added"] for s in steps]
    best = steps[-1]["gauc"] if steps else 0.5
    
    while len(chosen) < max_k:
        memguard.check(1.5, f"жадный шаг {len(chosen) + 1}")
        t0 = time.perf_counter()
        trials = []
        
        for f in pool:
            if f in chosen:
                continue
            #берем выбранные признаки  + этот кандидат
            cols = [*chosen, f]
            #обучаем легкую модель
            m = fit_light(Xtr.select(cols).to_numpy(), ytr, Xva.select(cols).to_numpy(), yva)
            #смотрми как ищемнилась gauc
            trials.append((gauc(m, Xva.select(cols).to_numpy(), meta_va), f, m.best_iteration))
        
        trials.sort(reverse=True)
        g, f, it = trials[0]
        #выбираем самый лучший
        step = {"step": len(chosen) + 1, "added": f, "gauc": g, "gain": g - best, "trees": it,
                "seconds": time.perf_counter() - t0, "fits": len(trials),
                "runner_up": [{"feature": t[1], "gauc": t[0]} for t in trials[1:4]]}
        log(f"шаг {step['step']:2d}: +{f:40s} GAUC {g:.4f} (прирост {g - best:+.4f}), "
            f"{step['seconds']:.0f} с, {len(trials)} обучений")
        
        if g - best < min_gain:
            log("прирост меньше порога — остановка")
            break
        steps.append(step)
        chosen.append(f)
        best = g
        
        if checkpoint:
            #если wsl рухнет, есть бэкап
            checkpoint.write_text(json.dumps(steps, ensure_ascii=False, indent=1))
    return steps
