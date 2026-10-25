"""Таблица кандидатов — общая для всех экспериментов (E0, E1, E2)

Кандидат = рекомендательное прослушивание (is_organic = 0), попавшее в одну из частей
train / val_a / val_b / test
 Label = дослушал (played_ratio_pct >= 50) или лайкнул этот трек
с момента начала прослушивания до часа после его окончания.
 Начало прослушивания = timestamp − played_ratio * длина трека
"""
import polars as pl

from src.data import DAY, HOUR, LISTEN_THRESHOLD, PROCESSED, scan_artist_map, scan_events, scan_listens
from src.split import SPLITS, part_expr

CANDIDATES_PATH = PROCESSED / "candidates.parquet"
LIKE_WINDOW = HOUR


def build_candidates() -> pl.DataFrame:
    start = SPLITS["train"][0]
    reco = (
        scan_listens(["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "track_length_seconds"])
        .filter((pl.col("is_organic") == 0) & (pl.col("timestamp") >= start))
        .drop("is_organic")
        .collect()
        .sort("timestamp")
    )
    likes = (
        scan_events(["uid", "item_id", "timestamp", "event_type"])
        .filter((pl.col("event_type") == "like") & (pl.col("timestamp") >= start - LIKE_WINDOW))
        .select("uid", "item_id", pl.col("timestamp").cast(pl.Int64).alias("like_ts"))
        .collect()
        .sort("like_ts")
    )
    # начало прослушивания (Int64: разности времени со знаком; UInt32 «переворачивает» отрицательные)
    reco = reco.with_columns(
        _start=(pl.col("timestamp").cast(pl.Int64)
                # при 150% "начало" уехало бы раньше реального начала, в зону предыдущего лйака
                # -5 : время округлено до 5 секунд
                - (pl.col("played_ratio_pct").clip(0, 100) / 100 * pl.col("track_length_seconds")).cast(pl.Int64) - 5)
    ).sort("_start")

    cands = (
        # первый лайк этого трека после начала прослушивания\, засчитываем, если он не позже часа после окончания
        #json-asof - соединение по ближайшииму,а время оайка и начала прослушивания никогда +- равны
        #лайкнул ли челоек в свзяис  жтим прослушиванием
        reco.join_asof(likes, left_on="_start", right_on="like_ts", by=["uid", "item_id"],
                       strategy="forward", check_sortedness=False)
                       # если поставил через неделю, то такой рещультат нам не нужен
        .with_columns(pl.when(pl.col("like_ts") <= pl.col("timestamp").cast(pl.Int64) + LIKE_WINDOW)
                      .then(pl.col("like_ts")).alias("like_ts"))
        .drop("_start")
        .with_columns(
            part=part_expr(),
            day=(pl.col("timestamp") // DAY).cast(pl.Int32),
            label=((pl.col("played_ratio_pct") >= LISTEN_THRESHOLD) | pl.col("like_ts").is_not_null()).cast(pl.Int8),
        )
        .filter(pl.col("part").is_not_null())          # выкидываем строки из зазоров
        #добавляем artist-id
        .join(scan_artist_map().collect(), on="item_id", how="left")
        # played_ratio_pct — это исход (из него сделан label), в признаки он НЕ идёт.
        .drop("like_ts", "played_ratio_pct")
        # Детерминированный row_id: сортировка по всем полям. При одинаковых (uid, timestamp) — а такие
        # бывают — сортировка только по ним расставляла строки по-разному от сборки к сборке
        .sort("uid", "timestamp", "item_id", "label", "track_length_seconds")
        .with_row_index("row_id")
    )
    return cands


def load_candidates(rebuild: bool = False) -> pl.DataFrame:
    if rebuild or not CANDIDATES_PATH.exists():
        PROCESSED.mkdir(parents=True, exist_ok=True)
        build_candidates().write_parquet(CANDIDATES_PATH)
    return pl.read_parquet(CANDIDATES_PATH)
