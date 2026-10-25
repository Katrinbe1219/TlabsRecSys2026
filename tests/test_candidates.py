"""Таблица кандидатов: только рекомендации, корректный label, исход не протекает в колонки."""
import polars as pl

from src.data import LISTEN_THRESHOLD, scan_listens


def test_label_is_binary(candidates):
    assert set(candidates["label"].unique().to_list()) <= {0, 1}


def test_outcome_column_not_present(candidates):
    # played_ratio_pct — это исход, из него сделан label; в таблице кандидатов его быть не должно
    assert "played_ratio_pct" not in candidates.columns


def test_row_ids_unique(candidates):
    assert candidates["row_id"].n_unique() == candidates.height


def test_only_recommendation_listens_and_label_matches_raw(candidates):
    """Сверяем с сырыми данными на выборке пользователей: кандидаты — это ровно рекомендательные
    прослушивания из частей, и дослушанные всегда получают label = 1

    Сравниваем агрегатами по ключу (uid, item_id, timestamp), а не построчно: из-за округления
    времени до 5 с один трек может встречаться несколько раз с одним и тем же timestamp

    Проверка, что кандидаты - ровно рекомендательные прослушивания, без лишних и без потерянных,
    что каждое дослушанное прослушивание получило label=1
    """
    keys = ["uid", "item_id", "timestamp"]
    uids = candidates["uid"].unique().sort()[:50]
    sub = candidates.filter(pl.col("uid").is_in(uids.implode()))

    raw = (scan_listens(["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct"])
           .filter(pl.col("uid").is_in(uids.implode()) & (pl.col("is_organic") == 0)
                   # чтобы соответствовать временным рамкам кандидатов
                   & pl.col("timestamp").is_in(sub["timestamp"].unique().implode()))
           .collect())
    
    c = sub.group_by(keys).agg(n=pl.len(), pos=pl.col("label").sum())
    r = raw.group_by(keys).agg(n=pl.len(), full=(pl.col("played_ratio_pct") >= LISTEN_THRESHOLD).sum())
    j = c.join(r, on=keys, how="full", coalesce=True, suffix="_raw")

    # множества ключей совпадают
    assert j["n"].null_count() == 0 and j["n_raw"].null_count() == 0   
    # и число строк на ключ
    assert (j["n"] == j["n_raw"]).all()    
    # каждое дослушивание дало label = 1 (лайки могут добавить)                              
    assert (j["pos"] >= j["full"]).all()        


def test_row_id_is_deterministic(candidates):
    """Пересборка даёт те же row_id для тех же строк (признаки связаны с кандидатами по row_id)"""
    from src.candidates import build_candidates
    again = build_candidates()
    cols = ["row_id", "uid", "item_id", "timestamp", "label"]
    assert again.select(cols).equals(candidates.select(cols))
