"""E2: подбор периода полураспада h на val_a (лёгкая модель, та же подвыборка train, что в отборе E1).

  1. global   — одно h для всех семейств;
  2. per_fam  — своё h для каждого семейства (покоординатно: от лучшего общего h меняем h одного семейства,
                остальные фиксированы; принимаем смену при приросте ≥ MIN_GAIN; до 2 проходов) — проверка H5;
  3. all_h    — все h сразу (несколько временных масштабов, как несколько окон в E1).
Результат: artifacts/E2/tuning.json (все испытания) — при перезапуске уже сделанные испытания не повторяются.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src import memguard
from src.candidates import load_candidates
from src.decay import HALF_LIVES, load_decay
from src.grammar import FAMILIES
from src.selection import fit_light, gauc

#папка для результатов
ART = Path(__file__).resolve().parents[1] / "artifacts" / "E2"
ART.mkdir(parents=True, exist_ok=True)
#к файлу, где после каждого обучения дописывается запись: конфиг h, GAUC, logloss, число деревьев, время
OUT = ART / "tuning.json"
MIN_GAIN = 0.0005     # как порог жадного отбора E1: меньшие приросты — шум 


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(ART / "tuning_log.txt", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def main():
    # возобновляемость
    trials = json.loads(OUT.read_text()) if OUT.exists() else []
    #сортировка для key in function run, потому что u:" ", i:" " могут стоять в разном порядке, что разный ключ
    done = {json.dumps(t["config"], sort_keys=True): t for t in trials}

    #загружаем кандидатов, поделенных на части
    c = load_candidates()
    tr = c.filter(pl.col("part") == "train").sample(600_000, seed=0).sort("row_id")
    va = c.filter(pl.col("part") == "val_a")
    del c
    
    allc = {f: HALF_LIVES for f in FAMILIES}
    memguard.check(2.5, "E2: загрузка")
    t0 = time.perf_counter()

    Xtr, Xva = load_decay(tr, allc), load_decay(va, allc)
    log(f"матрицы: {Xtr.width - 1} признаков, {time.perf_counter() - t0:.0f} с, пик {memguard.peak_rss_gb():.1f} ГБ")
    #gauc считается внутри групп пользователь*день
    #минимальный набор из va
    meta = va.select("uid", "day", "label")

    def run(stage, config):
        key = json.dumps(config, sort_keys=True)
        if key in done:
            return done[key]["gauc"]
        
        cols = [f for f in Xtr.columns if f != "row_id"
                #[0] - семейство
                #[1] - h
                #rsplit - режет с конца
                #проверка, что такая конфигурация для семейства есть в требовании нашем
                and int(f.rsplit("__h", 1)[1]) in config.get(f.split("__")[0], [])]
        
        t = time.perf_counter()
        m = fit_light(Xtr.select(cols).to_numpy(), tr["label"].to_numpy(), Xva.select(cols).to_numpy(), va["label"].to_numpy())
        
        g = gauc(m, Xva.select(cols).to_numpy(), meta)
        #запониманием результат и пишем в файл
        rec = {"stage": stage, "config": config, "n_features": len(cols), "gauc": g,
               "logloss": float(m.best_score["valid_0"]["binary_logloss"]), "trees": m.best_iteration,
               "seconds": time.perf_counter() - t}
        trials.append(rec)
        done[key] = rec
        OUT.write_text(json.dumps(trials, ensure_ascii=False, indent=1))
        log(f"{stage:8s} {config} → GAUC {g:.4f}, logloss {rec['logloss']:.4f} ({len(cols)} призн., {rec['seconds']:.0f} с)")
        return g

    # 1. одно h для всех
    #120: {f:120, f2:120...}
    glob = {h: run("global", {f: [h] for f in FAMILIES}) for h in HALF_LIVES}
    best_h = max(glob, key=glob.get)
    log(f"лучшее общее h = {best_h}")

    # 2. своё h для каждого семейства (покоординатно)
    #сначала запишем для них всех best_h как best
    cur = {f: [best_h] for f in FAMILIES}
    best = glob[best_h]
    for p in range(2):
        # на всякий случай 2 прохода, вель перый был проход, когда у всез был 120, 
        # теперь стали возможно новые
        changed = False
        for fam in FAMILIES:
            for h in HALF_LIVES:
                g = run(f"per_fam{p + 1}", {**cur, fam: [h]})
                if g > best + MIN_GAIN:
                    best, cur, changed = g, {**cur, fam: [h]}, True
        log(f"проход {p + 1}: {cur}, GAUC {best:.4f}")
        if not changed:
            break

    # 3. все h сразу
    #нужно ли несколкьо масштабов одновременно?
    #деревья умеют нахожить комбинации
    run("all_h", allc)
    (ART / "best_config.json").write_text(json.dumps({"global_h": best_h, "per_family": cur}, indent=1))
    log(f"готово; пик {memguard.peak_rss_gb():.1f} ГБ")


if __name__ == "__main__":
    main()
