"""Генерация признаков E2 (затухающие счётчики) на диск: data/processed/e2/{семейство}__h{h}.parquet.

Можно перезапускать: уже посчитанные файлы пропускаются. Приоры для сглаживания — те же, что в E1.
Запуск:  python scripts/build_e2_features.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidates import load_candidates
from src.decay import E2_DIR, build_all
from src.grammar import E1_DIR

E2_DIR.mkdir(parents=True, exist_ok=True)
p0 = json.loads((E1_DIR / "priors.json").read_text())
cands = load_candidates()
with open(E2_DIR / "build_log.jsonl", "a") as log:
    for c in build_all(cands, p0, log=lambda m: print(m, flush=True)):
        log.write(json.dumps(c) + "\n")
print("ГОТОВО", flush=True)
