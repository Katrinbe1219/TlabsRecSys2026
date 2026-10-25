"""Проверка шума подбора h в E2: насколько GAUC на val_a меняется от подвыборки train

Подбор (run_e2_tuning.py) обучал каждую конфигурацию один раз на подвыборке train (seed 0) и принимал
смену h при приросте > 0.0005
На тепловой карте база (h = 120) лучше всех соседей у ui и ua на +-0.002 —
похоже, что шум больше порога
Здесь те же конфигурации обучаются на подвыборках с seed 0…3;
val_a и параметры модели те же

Главное — разброс РАЗНОСТЕЙ конфигураций по seed (с ним и сравнивается порог)
seed 0 должен совпасть с tuning.json (проверка, что условия те же)

Результат: artifacts/E2/noise_check.json (+ noise_check_log.txt)
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
from src.decay import load_decay
from src.grammar import FAMILIES
from src.selection import fit_light, gauc

ART = Path(__file__).resolve().parents[1] / "artifacts" / "E2"
OUT = ART / "noise_check.json"
SEEDS = [0, 1, 2, 3]
REF = "h по семействам"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(ART / "noise_check_log.txt", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def same_h(h):
    return {f: [h] for f in FAMILIES}


CONFIGS = {
    "одно h = 30": same_h(30),
    "одно h = 60": same_h(60),
    "одно h = 120": same_h(120),
    "одно h = 240": same_h(240),
    REF: {**same_h(120), "a": [480]},
    "ui = 60": {**same_h(120), "a": [480], "ui": [60]},
    "ui = 240": {**same_h(120), "a": [480], "ui": [240]},
    "ua = 60": {**same_h(120), "a": [480], "ua": [60]},
    "ua = 240": {**same_h(120), "a": [480], "ua": [240]},
}


def main():
    trials = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(t["name"], t["seed"]) for t in trials}
    # только нужные h
    need = {f: sorted({h for cfg in CONFIGS.values() for h in cfg[f]}) for f in FAMILIES}

    c = load_candidates()
    train = c.filter(pl.col("part") == "train")
    va = c.filter(pl.col("part") == "val_a")
    del c

    memguard.check(2.0, "E2 шум: val_a")
    Xva = load_decay(va, need)
    meta = va.select("uid", "day", "label")

    for seed in SEEDS:
        if all((n, seed) in done for n in CONFIGS):
            continue

        memguard.check(1.5, f"E2 шум: train seed {seed}")
        tr = train.sample(600_000, seed=seed).sort("row_id")
        Xtr = load_decay(tr, need)

        for name, cfg in CONFIGS.items():
            if (name, seed) in done:
                continue

            cols = [f for f in Xtr.columns if f != "row_id"
                    and int(f.rsplit("__h", 1)[1]) in cfg.get(f.split("__")[0], [])]
            
            t = time.perf_counter()
            m = fit_light(Xtr.select(cols).to_numpy(), tr["label"].to_numpy(),
                          Xva.select(cols).to_numpy(), va["label"].to_numpy())
            g = gauc(m, Xva.select(cols).to_numpy(), meta)

            trials.append({"name": name, "seed": seed, "config": cfg, "gauc": g,
                           "logloss": float(m.best_score["valid_0"]["binary_logloss"]),
                           "seconds": time.perf_counter() - t})
            OUT.write_text(json.dumps(trials, ensure_ascii=False, indent=1))
            log(f"seed {seed}  {name:16s} → GAUC {g:.4f}")

        del Xtr, tr

    t = pl.DataFrame([{k: r[k] for k in ("name", "seed", "gauc")} for r in trials])
    ref = t.filter(pl.col("name") == REF).select("seed", ref_gauc="gauc")
    t = t.join(ref, on="seed").with_columns(diff=pl.col("gauc") - pl.col("ref_gauc"))

    summary = (t.group_by("name", maintain_order=True)
               .agg(pl.col("gauc").mean().alias("gauc_mean"), pl.col("gauc").std().alias("gauc_std"),
                    pl.col("diff").mean().alias("diff_mean"), pl.col("diff").std().alias("diff_std")))
    log("итог (diff = GAUC − GAUC «h по семействам» на том же seed):")
    
    for r in summary.iter_rows(named=True):
        log(f"  {r['name']:16s} GAUC {r['gauc_mean']:.4f} ± {r['gauc_std']:.4f}   "
            f"diff {r['diff_mean']:+.4f} ± {r['diff_std']:.4f}")
    stds = summary.filter(pl.col("name") != REF)["diff_std"].to_numpy()
    
    log(f"типичный разброс разности двух конфигураций (медиана std): {np.median(stds):.4f}; порог подбора 0.0005")
    log(f"готово; пик {memguard.peak_rss_gb():.1f} ГБ")


if __name__ == "__main__":
    main()
