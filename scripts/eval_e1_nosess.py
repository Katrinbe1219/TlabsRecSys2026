"""E1 без сессии: кривая по шагам жадного отбора на val_a/val_b и сравнение с E0 (лёгкая модель).
Разница GAUC с E0 — парный бутстреп по группам «пользователь × день» (оценка шума).
Результат: artifacts/E1/nosess_eval.parquet, artifacts/E1/nosess_bootstrap.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src.candidates import load_candidates
from src.data import PROCESSED
from src.features import E0_FEATURES
from src.metrics import evaluate
from src.selection import catalog, fit_light, load_matrix

ART = Path(__file__).resolve().parents[1] / "artifacts" / "E1"


def group_gauc(meta: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """GAUC каждой группы (только группы с обеими метками) — для бутстрепа."""
    return (meta.with_columns(p=pl.Series(p))
            .with_columns(r=pl.col("p").rank("average").over("uid", "day"),
                          npos=pl.col("label").sum().over("uid", "day"), n=pl.len().over("uid", "day"))
            .filter((pl.col("npos") > 0) & (pl.col("npos") < pl.col("n")))
            .group_by("uid", "day")
            .agg(g=((pl.col("r") * pl.col("label")).sum() - pl.col("npos").first() * (pl.col("npos").first() + 1) / 2)
                 / (pl.col("npos").first() * (pl.col("n").first() - pl.col("npos").first())))
            .sort("uid", "day"))


def main():
    c = load_candidates()
    tr = c.filter(pl.col("part") == "train").sample(600_000, seed=0).sort("row_id")
    va = c.filter(pl.col("part") == "val_a")
    vb = c.filter(pl.col("part") == "val_b")
    del c

    steps = json.loads((ART / "greedy_nosess.json").read_text())
    chosen = [s["added"] for s in steps]
    feats0 = pl.read_parquet(PROCESSED / "features_E0.parquet")
    cat = catalog()

    X = {n: load_matrix(d, chosen, cat).join(feats0, on="row_id", how="left", maintain_order="left") for n, d in [("tr", tr), ("va", va), ("vb", vb)]}
    
    sets = {f"E1 без сессии, k={k}": chosen[:k] for k in range(1, len(chosen) + 1)}
    sets["E0"] = E0_FEATURES
    sets["E0 + E1 без сессии"] = [*E0_FEATURES, *[f for f in chosen if f not in E0_FEATURES]]
    
    rows, per_group = [], {}
    for name, cols in sets.items():
        m = fit_light(X["tr"].select(cols).to_numpy(), tr["label"].to_numpy(),
                      X["va"].select(cols).to_numpy(), va["label"].to_numpy())
        
        for part, d, key in [("val_a", va, "va"), ("val_b", vb, "vb")]:
            p = m.predict(X[key].select(cols).to_numpy(), num_iteration=m.best_iteration)
            
            rows.append({"set": name, "k": len(cols), "part": part, "trees": m.best_iteration,
                         **evaluate(d.select("uid", "day", "label").with_columns(p=pl.Series(p)), "p")})
            
            if part == "val_b":
                per_group[name] = group_gauc(d.select("uid", "day", "label"), p)["g"].to_numpy()
        
        print(name, "готово", flush=True)
    
    out = pl.DataFrame(rows)
    out.write_parquet(ART / "nosess_eval.parquet")
    
    rng = np.random.default_rng(0)
    base = per_group["E0"]
    idx = rng.integers(0, len(base), (1000, len(base)))
    
    boot = {}
    for name in [f"E1 без сессии, k={len(chosen)}", "E0 + E1 без сессии"]:
        diff = per_group[name] - base
        bs = diff[idx].mean(1)
        boot[name] = {"diff": float(diff.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    (ART / "nosess_bootstrap.json").write_text(json.dumps(boot, ensure_ascii=False, indent=1))
    
    with pl.Config(tbl_rows=50, tbl_width_chars=200, fmt_str_lengths=40):
        print(out.select("set", "part", "trees", *[pl.col(c).round(4) for c in ["logloss", "roc_auc", "gauc", "ndcg@10", "recall@10"]]))
    
    print(json.dumps(boot, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
