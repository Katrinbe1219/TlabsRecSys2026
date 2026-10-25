"""Проверка I7: ранжируют ли сессионные признаки треки или только «моменты» внутри дня.

Проверка 1 — «зафиксировать сессию».
 Модель обучена с сессионными признаками, но при оценке всем
строкам группы подставляем одно и то же состояние сессии (как в реальном выборе: в момент выбора
состояние одно, кандидатов много). Тогда сессия может влиять на порядок только через
взаимодействия с признаками трека. Сравниваем GAUC с моделью без сессии.

Проверка 2 — «сузить группы». 
GAUC в группах пользователь × день, × час и × сессия
(сессия = прослушивания без перерыва > 30 мин). 
Если выигрыш сессионных признаков — от
ранжирования моментов, в узких группах он должен заметно уменьшиться.

Результат: artifacts/E1/session_check.parquet (+ строки в selection_log.txt).
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
from src.data import HOUR, PROCESSED
from src.features import E0_FEATURES
from src.metrics import ranking_metrics
from src.selection import catalog, fit_light, load_matrix

ART = Path(__file__).resolve().parents[1] / "artifacts" / "E1"
SESSION_GAP = 30 * 60
# «чистые» признаки момента: одинаковы у всех треков в один момент (s__today_artist_* зависят от трека — их нет)
MOMENT = ["s__prev_dur_min", "s__today_cnt", "s__today_rate", "s__skip_streak", "s__hour_rate"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(ART / "selection_log.txt", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + "[I7] " + msg + "\n")


def main():
    memguard.check(3.0, "проверка I7")
    t0 = time.perf_counter()
    c = load_candidates()
    tr = c.filter(pl.col("part") == "train").sample(600_000, seed=0).sort("row_id")
    va = c.filter(pl.col("part") == "val_a")
    vb = (c.filter(pl.col("part") == "val_b").sort("uid", "timestamp")
          .with_columns(hour=(pl.col("timestamp") // HOUR).cast(pl.Int32),
                        session=((pl.col("timestamp").cast(pl.Int64).diff().over("uid") > SESSION_GAP)
                                 .fill_null(True).cast(pl.Int32).cum_sum().over("uid"))))
    del c

    greedy = [s["added"] for s in json.loads((ART / "greedy.json").read_text())]
    no_sess = [f for f in greedy if not f.startswith("s__")]
    feats0 = pl.read_parquet(PROCESSED / "features_E0.parquet")
    e1_need = sorted(set(greedy) | set(MOMENT))
    cat = catalog()
    X = {}
    for n, d in [("tr", tr), ("va", va), ("vb", vb)]:
        X[n] = load_matrix(d, e1_need, cat).join(feats0, on="row_id", how="left", maintain_order="left")
    X["vb"] = vb.select("row_id", "uid", "day", "hour", "session", "timestamp").join(X["vb"], on="row_id", maintain_order="left")
    log(f"данные загружены за {time.perf_counter() - t0:.0f} с, пик {memguard.peak_rss_gb():.1f} ГБ")

    sess_in = [f for f in greedy if f.startswith("s__")]
    sets = {
        "E1 жадный (с сессией)": greedy,
        "E1 жадный без сессии": no_sess,
        "E0": E0_FEATURES,
        "E0 + сессия": [*E0_FEATURES, *sess_in],
    }
    models = {}
    for name, cols in sets.items():
        models[name] = fit_light(X["tr"].select(cols).to_numpy(), tr["label"].to_numpy(),
                                 X["va"].select(cols).to_numpy(), va["label"].to_numpy())
        log(f"обучена «{name}» ({len(cols)} признаков, {models[name].best_iteration} деревьев)")

    groupings = {"пользователь × день": ["uid", "day"], "пользователь × час": ["uid", "hour"],
                 "пользователь × сессия": ["uid", "session"]}
    rng = np.random.default_rng(0)
    base = X["vb"].with_columns(_r=pl.Series(rng.random(X["vb"].height)), label=vb["label"])
    rows = []

    def score(name, cols, data, grp, mode):
        p = models[name].predict(data.select(cols).to_numpy(), num_iteration=models[name].best_iteration)
        m = ranking_metrics(data.select(*grp, "label").with_columns(p=pl.Series(p)), "p", group=grp)
        rows.append({"model": name, "grouping": gname, "session_mode": mode, **m})

    for gname, grp in groupings.items():
        for name, cols in sets.items():
            score(name, cols, base, grp, "как есть")
            s_cols = [f for f in cols if f.startswith("s__")]
            if not s_cols:
                continue
            # одно состояние сессии на всю группу — три способа его выбрать
            fixed = {
                "первое в группе": base.sort("timestamp").with_columns(pl.col(s_cols).first().over(grp)),
                "медиана группы": base.with_columns(pl.col(s_cols).median().over(grp)),
                "случайная строка группы": base.sort("_r").with_columns(pl.col(s_cols).first().over(grp)),
            }
            for mode, data in fixed.items():
                score(name, cols, data, grp, f"фиксирована: {mode}")
        # одиночные признаки как скор (без модели)
        for f in ["s__today_rate", "s__prev_dur_min", "s__today_cnt", "ui__all_srate__all", "i__all_srate__all"]:
            m = ranking_metrics(base.select(*grp, "label", f).with_columns(pl.col(f).fill_null(-1)), f, group=grp)
            rows.append({"model": f"признак {f}", "grouping": gname, "session_mode": "как есть", **m})
        log(f"группировка «{gname}» готова")

    out = pl.DataFrame(rows)
    out.write_parquet(ART / "session_check.parquet")
    log(f"готово за {time.perf_counter() - t0:.0f} с, пик {memguard.peak_rss_gb():.1f} ГБ")
    with pl.Config(tbl_rows=100, tbl_width_chars=200, fmt_str_lengths=60):
        print(out.select("grouping", "model", "session_mode", pl.col("gauc").round(4),
                         pl.col("ndcg@10").round(4), "groups"))


if __name__ == "__main__":
    main()
