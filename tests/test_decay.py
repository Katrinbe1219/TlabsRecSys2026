"""Тесты E2 (затухающие счётчики): нет утечки, вчерашний день учитывается, значения совпадают
с прямым подсчётом по формуле Σ 2^(−(d − e)/h)."""
import numpy as np
import polars as pl
import pytest

from src.data import DAY, LISTEN_THRESHOLD, scan_events
from src.decay import build_decay_family
from src.grammar import FAMILIES, priors
from src.split import SPLITS

#выопняем один раз на всесь файл
@pytest.fixture(scope="module")
def small(candidates):
    #implode = сворачивает Series из 20 значений в одно значение типа list
    #is_in может подумать: проверка вхождения или сравнить поэлементно
    uids = candidates["uid"].unique().sort()[:20].implode()
    events = scan_events().filter(pl.col("uid").is_in(uids))
    return candidates.filter(pl.col("uid").is_in(uids)), events, priors()


#запускаем 6 раз - для каждой из 6 семей
@pytest.mark.parametrize("fam", list(FAMILIES))
def test_decay_ignores_current_and_future_days(small, fam):
    """Проверка утечки из будущего
    - считаем full - по всем событиям
    - считаем cut - по обрезанным строго до начала дня d
    
    Если признаки честные, события дня d и дальшне ни на что не влияют, и обе таблицы совпдадут до последнего знака"""
    cands, events, p0 = small
    #из секунд! получаем день, плюс один , потому что будет день 279.9 - начинаем с точного целого дня
    d = SPLITS["val_a"][0] // DAY + 1
    c = cands.filter(pl.col("day") == d)
    keys = [*FAMILIES[fam], "day"]
    full = build_decay_family(c, fam, 7, events, p0).sort(keys)
    cut = build_decay_family(c, fam, 7, events.filter(pl.col("timestamp") < d * DAY), p0).sort(keys)
    assert full.equals(cut, null_equal=True), f"утечка: {fam}"


def test_decay_uses_yesterday(small):
    """Проверка похожая на прошлый тест, только теперь удаляет вчерашйни день,
      который должен был быть в подсчете """
    cands, events, p0 = small

    d = SPLITS["val_a"][0] // DAY + 1
    c = cands.filter(pl.col("day") == d)

    full = build_decay_family(c, "u", 7, events, p0).sort("uid")
    #удаляем вчерщаний день
    cut = build_decay_family(c, "u", 7, events.filter(pl.col("timestamp") < (d - 1) * DAY), p0).sort("uid")
    
    assert not full.equals(cut, null_equal=True)


@pytest.mark.parametrize("h", [3, 30])
def test_decay_matches_direct_formula(small, h):
    """Сверка с прямым подсчётом: для каждого (пользователь, день) суммируем веса 2^(−(d − e)/h)
    по прошлым прослушиваниям и дослушиваниям"""
    #в тесте исползуется быстрая реализация с растущим множителем, кумулятивной суммйо, json_asof и обратное домножение
    #тест считает наинвым способом
    cands, events, p0 = small

    got = build_decay_family(cands, "u", h, events, p0)
    #из grammar.quiries мы получим уникальные пары!
    q = got.select("uid", "day").sample(60, seed=0)

    ev = (events.filter(pl.col("event_type") == "listen")
          .select("uid", (pl.col("timestamp") // DAY).cast(pl.Int32).alias("e"),
                  (pl.col("played_ratio_pct") >= LISTEN_THRESHOLD).cast(pl.Float64).alias("full"))
          .collect())
    
    direct = (q.join(ev, on="uid").filter(pl.col("e") < pl.col("day"))
              .with_columns(w=2.0 ** (-(pl.col("day") - pl.col("e")).cast(pl.Float64) / h))
              .group_by("uid", "day").agg(cnt=pl.col("w").sum(), 
                                          full=(pl.col("w") * pl.col("full")).sum()))
    
    #заполняем full_null я пар в первый день например, filter null убирает
    #долавяем наше и ответ функции
    j = (q.join(direct, on=["uid", "day"], how="left").with_columns(pl.col("cnt", "full").fill_null(0.0))
         .join(got, on=["uid", "day"]))

    #assert_allclose считает числа равными, если |ответ − эталон| <= atol + rtol · |эталон|
    np.testing.assert_allclose(j[f"u__all_cnt__h{h}"].to_numpy(), j["cnt"].to_numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(j[f"u__all_full__h{h}"].to_numpy(), j["full"].to_numpy(), rtol=1e-5, atol=1e-6)
