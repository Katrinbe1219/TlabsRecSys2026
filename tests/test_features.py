"""Главный тест на утечку для признаков E0

Идея: признаки для кандидатов дня D не должны зависеть ни от чего, что случилось в день D
и позже. Проверяем буквально: считаем признаки (1) по всем событиям и (2) по событиям,
обрезанным до начала дня D. Результаты должны совпасть до последнего значения

Обратная проверка (чтобы тест не был пустым): если обрезать ещё на день раньше (до D-1),
признаки ДОЛЖНЫ измениться — значит, вчерашние события действительно используются

Для скорости берём 30 пользователей и только их события (свойство «не смотрит в будущее»
от этого не зависит)
"""
import polars as pl
import pytest

from src.data import DAY, scan_events
from src.features import E0_FEATURES, build_e0_features
from src.split import SPLITS


@pytest.fixture(scope="module")
def setup(candidates):
    uids = candidates["uid"].unique().sort()[:30].implode()
    events = scan_events().filter(pl.col("uid").is_in(uids))
    # первый полный день каждой части: на границах частей утечка вероятнее всего
    days = [SPLITS[p][0] // DAY + 1 for p in SPLITS]
    cands = candidates.filter(pl.col("uid").is_in(uids) & pl.col("day").is_in(days))
    full = build_e0_features(cands, events).sort("row_id")
    return events, cands, full, days


def _same(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    return a.select(E0_FEATURES).equals(b.select(E0_FEATURES), null_equal=True)


def test_features_ignore_current_and_future_days(setup):
    events, cands, full, days = setup

    for d in days:
        c = cands.filter(pl.col("day") == d)
        assert c.height > 0

        cut = build_e0_features(c, events.filter(pl.col("timestamp") < d * DAY)).sort("row_id")
        assert _same(full.filter(pl.col("row_id").is_in(c["row_id"].implode())), cut), f"утечка в день {d}"


def test_features_do_use_yesterday(setup):
    """Убираем вчершаний день -> результат меняется"""
    events, cands, full, days = setup
    d = days[0]
    c = cands.filter(pl.col("day") == d)
    cut = build_e0_features(c, events.filter(pl.col("timestamp") < (d - 1) * DAY)).sort("row_id")
    assert not _same(full.filter(pl.col("row_id").is_in(c["row_id"].implode())), cut)
