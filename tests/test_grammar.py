"""Тесты на утечку для E1: грамматика (на начало дня) и сессионное семейство (на момент трека)

Логика та же, что в test_features.py: признаки не должны меняться, если стереть из событий
всё, что случилось начиная с момента, на который признак считается
"""
import polars as pl
import pytest

from src.data import DAY, scan_events
from src.grammar import FAMILIES, SESSION_FEATURES, build_family, build_session_features, priors
from src.split import SPLITS

SESSION = [n for n, _ in SESSION_FEATURES]


@pytest.fixture(scope="module")
def small(candidates):
    uids = candidates["uid"].unique().sort()[:20].implode()
    events = scan_events().filter(pl.col("uid").is_in(uids))
    return candidates.filter(pl.col("uid").is_in(uids)), events


def _eq(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    return a.equals(b, null_equal=True)


@pytest.mark.parametrize("fam", list(FAMILIES))
def test_family_ignores_current_and_future_days(small, fam):
    cands, events = small
    # приоры по истории до train — общие для обоих расчётов
    p0 = priors()                    
    d = SPLITS["val_a"][0] // DAY + 1
    c = cands.filter(pl.col("day") == d)
    keys = [*FAMILIES[fam], "day"]

    for w in [7, None]:
        full = build_family(c, fam, events, p0, windows=[w])[w].sort(keys)
        cut = build_family(c, fam, events.filter(pl.col("timestamp") < d * DAY), p0, windows=[w])[w].sort(keys)
        assert _eq(full, cut), f"утечка: {fam}, окно {w}"


def test_family_uses_yesterday(small):
    cands, events = small
    p0 = priors()
    d = SPLITS["val_a"][0] // DAY + 1
    c = cands.filter(pl.col("day") == d)
    full = build_family(c, "u", events, p0, windows=[1])[1].sort("uid")
    cut = build_family(c, "u", events.filter(pl.col("timestamp") < (d - 1) * DAY), p0, windows=[1])[1].sort("uid")
    assert not _eq(full, cut)


def test_session_ignores_current_moment_and_future(small):
    """Для каждой строки отдельно: стираем все события с timestamp >= timestamp строки
    (включая события в тот же 5-секундный интервал) — сессионные признаки не меняются"""
    cands, events = small
    rows = cands.sample(40, seed=0)
    full = build_session_features(rows, events)

    #named - получаем словарь, а не кортеж
    for r in rows.iter_rows(named=True): 
        #передаем одного кандидата
        one = rows.filter(pl.col("row_id") == r["row_id"])
        cut = build_session_features(one, events.filter(pl.col("timestamp") < r["timestamp"]))
        assert _eq(full.filter(pl.col("row_id") == r["row_id"]), cut), f"утечка в строке {r['row_id']}"


def test_session_uses_recent_past(small):
    cands, events = small
    rows = cands.filter(pl.col("day") == cands["day"].max()).sample(40, seed=1)
    full = build_session_features(rows, events)
    cut = build_session_features(rows, events.filter(pl.col("timestamp") < rows["timestamp"].min() - 3 * 3600))
    assert not _eq(full, cut)


def test_session_is_deterministic_and_batch_independent(small):
    """Значение признака строки не зависит ни от запуска, ни от того, какие ещё строки считаются вместе"""
    cands, events = small
    rows = cands.sample(200, seed=2)

    a = build_session_features(rows, events)
    b = build_session_features(rows, events)
    assert _eq(a, b)

    half = rows.sample(50, seed=3)
    c = build_session_features(half, events)
    # мы получаем кандидатов - следовательно расчет по ним везде должен быть одинаков и не зависит от того,
    #  какие еще пользователи быди отобраны
    assert _eq(a.filter(pl.col("row_id").is_in(half["row_id"].implode())), c)


def test_session_does_not_depend_on_current_duration(small):
    """timestamp = ОКОНЧАНИЕ прослушивания, поэтому t − (окончание предыдущего) +- длительность
    текущего прослушивания +- label
      Проверка: если «сдвинуть» окончание текущего трека раньше
    (как будто его пропустили сразу), сессионные признаки не должны измениться

    Новое окончание: сразу после окончания предыдущего прослушивания, в тот же день"""
    cands, events = small
    rows = cands.sample(300, seed=4)

    listens = (events.filter(pl.col("event_type") == "listen").select("uid", pl.col("timestamp").alias("ts"))
               .collect().unique().sort("ts"))
    
    prev = (rows.select("row_id", "uid", "day", pl.col("timestamp").cast(pl.Int64).alias("t"))
            .with_columns(_k=pl.col("t") - 1).sort("_k")
            .join_asof(listens.with_columns(pl.col("ts").cast(pl.Int64)), left_on="_k", right_on="ts",
                       by="uid", strategy="backward", check_sortedness=False))

    #округление до 5 секунд, поэтому + 5, самый ранний следующий момент
    #day*DAY - полность - начало дня кандидата - не уезжать во вчера
    new_t = prev.select("row_id", pl.max_horizontal(pl.col("ts").fill_null(0) + 5, pl.col("day") * DAY)
                        .alias("new_t"))
    #новое время, но не позже старого
    moved = (rows.join(new_t, on="row_id")
             .with_columns(pl.min_horizontal("new_t", pl.col("timestamp").cast(pl.Int64)).cast(pl.UInt32).alias("timestamp"))
             .drop("new_t"))

    
    a = build_session_features(rows, events).sort("row_id")
    b = build_session_features(moved, events).sort("row_id")
    assert _eq(a, b)
