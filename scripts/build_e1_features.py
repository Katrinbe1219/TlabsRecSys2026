"""Генерация всех признаков E1 на диск: data/processed/e1/{семейство}__{окно}.parquet и s.parquet.

Можно перезапускать: уже посчитанные файлы пропускаются (контрольные точки на случай падения WSL).
Запуск:  python scripts/build_e1_features.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import memguard
from src.candidates import load_candidates
from src.grammar import E1_DIR, FAMILIES, build_family, build_session_features, priors

E1_DIR.mkdir(parents=True, exist_ok=True)
log = open(E1_DIR / "build_log.jsonl", "a")
cands = load_candidates()
p0_path = E1_DIR / "priors.json"

if not p0_path.exists():
    p0_path.write_text(json.dumps(priors()))
p0 = json.loads(p0_path.read_text())
print("приоры (до начала train):", p0, flush=True)

for fam in FAMILIES:
    t0 = time.perf_counter()
    build_family(cands, fam, p0=p0, out_dir=E1_DIR, verbose=True)
    log.write(json.dumps({"family": fam, "seconds": time.perf_counter() - t0,
                          "peak_rss_gb": memguard.peak_rss_gb()}) + "\n")
    log.flush()

s_path = E1_DIR / "s.parquet"

if not s_path.exists():
    memguard.check(2.5, "сессионные признаки")
    t0 = time.perf_counter()

    build_session_features(cands).write_parquet(s_path)

    sec = time.perf_counter() - t0

    print(f"  s: {sec:.1f} с, пик {memguard.peak_rss_gb():.1f} ГБ", flush=True)
    log.write(json.dumps({"family": "s", "seconds": sec, "peak_rss_gb": memguard.peak_rss_gb()}) + "\n")

print("ГОТОВО", flush=True)
