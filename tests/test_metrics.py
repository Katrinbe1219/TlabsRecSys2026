"""Метрики ранжирования на маленьких примерах, посчитанных вручную"""
import math

import polars as pl
import pytest

from src.metrics import ranking_metrics


def _df(labels, scores, uid=1, day=1):
    return pl.DataFrame({"uid": [uid] * len(labels), "day": [day] * len(labels),
                         "label": labels, "s": scores})


def test_perfect_ranking():
    m = ranking_metrics(_df([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]), "s", k=10)
    assert m["ndcg@10"] == pytest.approx(1.0) and m["recall@10"] == pytest.approx(1.0)


def test_known_ndcg_value():
    # порядок по скору: [0, 1]; единица на 2-й позиции: DCG = 1/log2(3), IDCG = 1
    m = ranking_metrics(_df([0, 1], [0.9, 0.1]), "s", k=10)
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(3))


def test_recall_at_k_cut():
    # k = 1: наверху единица, всего единиц 2 -> recall = 1/2
    m = ranking_metrics(_df([1, 1, 0], [0.9, 0.5, 0.1]), "s", k=1)
    assert m["recall@1"] == pytest.approx(0.5) and m["ndcg@1"] == pytest.approx(1.0)


def test_pure_groups_are_ignored():
    df = pl.concat([_df([1, 1], [0.1, 0.2], uid=1), _df([0, 0], [0.1, 0.2], uid=2),
                    _df([1, 0], [0.9, 0.1], uid=3)])
    m = ranking_metrics(df, "s", k=10)
    assert m["groups"] == 1 and m["ndcg@10"] == pytest.approx(1.0)


def test_gauc():
    # единицы со скорами 0.9 и 0.3, нули — 0.5 и 0.1: из 4 пар (1, 0) правильно упорядочены 3
    m = ranking_metrics(_df([1, 0, 1, 0], [0.9, 0.5, 0.3, 0.1]), "s", k=10)
    assert m["gauc"] == pytest.approx(0.75)


def test_constant_score_is_random_within_group():
    # постоянный скор внутри группы: GAUC ровно 0.5, NDCG не зависит от порядка строк
    m = ranking_metrics(_df([1, 0], [0.5, 0.5]), "s", k=10)
    assert m["gauc"] == pytest.approx(0.5)
