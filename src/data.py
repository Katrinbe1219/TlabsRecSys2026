"""Загрузка данных Yambda-50M.

Все функции возвращают polars.LazyFrame: данные не читаются в память,
пока не вызван .collect()
У нас лимит +- 7ГБ!!!!
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

EVENTS_PATH = RAW / "flat" / "50m" / "multi_event.parquet"
ARTIST_PATH = RAW / "artist_item_mapping.parquet"
ALBUM_PATH = RAW / "album_item_mapping.parquet"

# timestamp в Yambda — секунды от начала датасета, округлённые до 5 с
# (см. benchmarks/yambda/constants.py в репозитории датасета).
HOUR = 3600
DAY = 24 * HOUR
LAST_TIMESTAMP = 26_000_000

# Официальный порог «трек прослушан» из бенчмарка Yambda.
LISTEN_THRESHOLD = 50


def scan_events(columns: list[str] | None = None) -> pl.LazyFrame:
    """Все события (listen/like/dislike/unlike/undislike)"""
    lf = pl.scan_parquet(EVENTS_PATH)
    return lf.select(columns) if columns else lf


def scan_listens(columns: list[str] | None = None) -> pl.LazyFrame:
    """Только прослушивания"""
    lf = pl.scan_parquet(EVENTS_PATH).filter(pl.col("event_type") == "listen")
    return lf.select(columns) if columns else lf


def scan_artist_map() -> pl.LazyFrame:
    """item_id -> artist_id. У небольшой части треков несколько артистов:
    берём минимальный artist_id, чтобы связь была один-к-одному"""
    return pl.scan_parquet(ARTIST_PATH).group_by("item_id").agg(pl.col("artist_id").min())


def scan_album_map() -> pl.LazyFrame:
    """item_id -> album_id. Треки часто входят в несколько альбомов
    (сборники), берём минимальный album_id"""
    return pl.scan_parquet(ALBUM_PATH).group_by("item_id").agg(pl.col("album_id").min())
