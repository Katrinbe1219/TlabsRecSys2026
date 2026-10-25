"""Разбиение по времени: части идут по порядку, не пересекаются, между ними зазор"""
import polars as pl

from src.data import LAST_TIMESTAMP
from src.split import GAP, PARTS, SPLITS, WEEK


def test_parts_are_ordered_with_gaps():
    bounds = [SPLITS[p] for p in PARTS]
    for (a1, b1), (a2, b2) in zip(bounds, bounds[1:]):
        assert a1 < b1 <= a2 < b2
        assert a2 - b1 == GAP


def test_test_part_is_last_week():
    a, b = SPLITS["test"]
    assert b == LAST_TIMESTAMP + 1 and b - 1 - a == WEEK


def test_each_candidate_in_its_part_bounds(candidates):
    for name, (a, b) in SPLITS.items():
        ts = candidates.filter(pl.col("part") == name)["timestamp"]
        assert ts.min() >= a and ts.max() < b, name


def test_no_candidates_in_gaps_or_history(candidates):
    assert candidates["part"].null_count() == 0
    assert candidates["timestamp"].min() >= SPLITS["train"][0]
