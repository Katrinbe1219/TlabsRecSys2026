"""Ручные признаки E0, посчитанные на начало дня

Правило против утечки: для кандидата в день d используются только события
с timestamp < d * DAY, то есть дни 0 … d-1. Текущий день не используется совсем.

Как считаем быстро и в небольшой памяти:
1. Для каждого ключа (user, item, artist, user×item, user×artist) считаем дневные суммы
   статистик STATS и накопленные итоги по дням (cum_sum)
2. "Накоплено к концу дня d-1" находим через join_asof (последний день <= d-1)
3. Сумма за окно w дней (дни d-w … d-1) = накоплено(d-1) − накоплено(d-1-w)
Так любое окно стоит два join_asof, а не проход по миллионам событий
"""
import time

import polars as pl

from src.data import DAY, LISTEN_THRESHOLD, scan_artist_map, scan_events

# Дневные статистики, из которых собираются признаки.
STATS = ["n_listen", "n_full", "n_reco", "n_reco_full", "n_like", "n_dislike"]

# Ключи сущностей: имя -> колонки.
KEYS = {
    "u": ["uid"],
    "i": ["item_id"],
    "a": ["artist_id"],
    "ui": ["uid", "item_id"],
    "ua": ["uid", "artist_id"],
}


def event_base(events: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """Одна строка = одно событие с флагами-статистиками и днём.
    events — LazyFrame в формате multi_event (можно передать усечённый, для тестов на утечку)"""
    events = scan_events() if events is None else events

    listen = pl.col("event_type") == "listen"
    full = listen & (pl.col("played_ratio_pct") >= LISTEN_THRESHOLD)
    reco = listen & (pl.col("is_organic") == 0)

    return (
        events.filter(pl.col("event_type").is_in(["listen", "like", "dislike"]))
        .join(scan_artist_map(), on="item_id", how="left")
        .select(

            "uid", "item_id", "artist_id",
            (pl.col("timestamp") // DAY).cast(pl.Int32).alias("day"),
            
            listen.cast(pl.UInt32).alias("n_listen"),
            full.cast(pl.UInt32).alias("n_full"),
            reco.cast(pl.UInt32).alias("n_reco"),
            (reco & full).cast(pl.UInt32).alias("n_reco_full"),

            (pl.col("event_type") == "like").cast(pl.UInt32).alias("n_like"),
            (pl.col("event_type") == "dislike").cast(pl.UInt32).alias("n_dislike"),
        )
    )


def key_aggregates(base: pl.LazyFrame, cands: pl.DataFrame, prefix: str,
                   windows: list[int]) -> pl.DataFrame:
    """Для каждой пары (ключ, день) из кандидатов: суммы STATS за окна и за всё время,
    плюс день последней активности. Возвращает колонки {prefix}_{stat}_{w}d, {prefix}_{stat}_all,
    {prefix}_last_day"""

    keys = KEYS[prefix]
    q = cands.select(*keys, "day").drop_nulls().unique()

    cum = (
        base.join(q.select(keys).unique().lazy(), on=keys, how="semi")   # только нужные ключи
        .group_by(*keys, "day").agg(pl.col(s).sum() for s in STATS)
        .sort(*keys, "day")
        .with_columns(pl.col(s).cum_sum().over(keys) for s in STATS)
        .collect(engine="streaming")
        .sort("day")
    )

    def asof(lag: int) -> pl.DataFrame:
        left = q.with_columns((pl.col("day") - 1 - lag).alias("_q")).sort("_q")
        return left.join_asof(cum.rename({"day": "_last_day"}), left_on="_q", right_on="_last_day",
                              by=keys, strategy="backward", check_sortedness=False).drop("_q")

    now = asof(0)
    out = now.select(
        *keys, "day",
        *[pl.col(s).fill_null(0).alias(f"{prefix}_{s}_all") for s in STATS],
        pl.col("_last_day").alias(f"{prefix}_last_day"),
    )
    for w in windows:
        past = asof(w).select(*keys, "day", *[pl.col(s).fill_null(0).alias(f"_p_{s}") for s in STATS])
        out = out.join(past, on=[*keys, "day"], how="left").with_columns(
            (pl.col(f"{prefix}_{s}_all") - pl.col(f"_p_{s}")).alias(f"{prefix}_{s}_{w}d") for s in STATS
        ).drop(f"_p_{s}" for s in STATS)
    return out


def _rate(num: str, den: str) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None)


# Ручной набор E0: (имя, выражение, группа, описание). Описания попадают в ноутбуки.
E0_SPEC = [
    # --- пользователь
    ("u_listen_7d", pl.col("u_n_listen_7d"), "user", "прослушиваний за 7 дней"),
    ("u_listen_30d", pl.col("u_n_listen_30d"), "user", "прослушиваний за 30 дней"),
    ("u_listen_all", pl.col("u_n_listen_all"), "user", "прослушиваний за всю историю"),
    ("u_full_rate_7d", _rate("u_n_full_7d", "u_n_listen_7d"), "user", "доля дослушанных за 7 дней"),
    ("u_full_rate_30d", _rate("u_n_full_30d", "u_n_listen_30d"), "user", "доля дослушанных за 30 дней"),
    ("u_reco_30d", pl.col("u_n_reco_30d"), "user", "рекомендательных прослушиваний за 30 дней"),
    ("u_reco_full_rate_30d", _rate("u_n_reco_full_30d", "u_n_reco_30d"), "user",
     "доля дослушанных рекомендаций за 30 дней"),
    ("u_organic_share_30d", 1 - _rate("u_n_reco_30d", "u_n_listen_30d"), "user", "доля органики за 30 дней"),
    ("u_like_30d", pl.col("u_n_like_30d"), "user", "лайков за 30 дней"),
    ("u_dislike_30d", pl.col("u_n_dislike_30d"), "user", "дизлайков за 30 дней"),
    ("u_days_since_last", pl.col("day") - pl.col("u_last_day"), "user", "дней с последней активности"),
    # --- трек
    ("i_listen_7d", pl.col("i_n_listen_7d"), "item", "прослушиваний трека за 7 дней (все пользователи)"),
    ("i_listen_30d", pl.col("i_n_listen_30d"), "item", "прослушиваний трека за 30 дней"),
    ("i_listen_all", pl.col("i_n_listen_all"), "item", "прослушиваний трека за всю историю"),
    ("i_full_rate_30d", _rate("i_n_full_30d", "i_n_listen_30d"), "item", "доля дослушиваний трека за 30 дней"),
    ("i_full_rate_all", _rate("i_n_full_all", "i_n_listen_all"), "item", "доля дослушиваний трека за всё время"),
    ("i_like_30d", pl.col("i_n_like_30d"), "item", "лайков трека за 30 дней"),
    ("track_length_seconds", pl.col("track_length_seconds"), "item", "длина трека, с"),
    # --- артист
    ("a_listen_30d", pl.col("a_n_listen_30d"), "artist", "прослушиваний артиста за 30 дней"),
    ("a_full_rate_30d", _rate("a_n_full_30d", "a_n_listen_30d"), "artist", "доля дослушиваний артиста за 30 дней"),
    ("a_listen_all", pl.col("a_n_listen_all"), "artist", "прослушиваний артиста за всё время"),
    # --- пользователь × трек (прямые связи)
    ("ui_listen_all", pl.col("ui_n_listen_all"), "user×item", "сколько раз пользователь слушал этот трек"),
    ("ui_full_all", pl.col("ui_n_full_all"), "user×item", "сколько раз дослушал этот трек"),
    ("ui_skip_all", pl.col("ui_n_listen_all") - pl.col("ui_n_full_all"), "user×item", "сколько раз пропустил этот трек"),
    ("ui_listen_30d", pl.col("ui_n_listen_30d"), "user×item", "слушал этот трек за 30 дней"),
    ("ui_like_all", pl.col("ui_n_like_all"), "user×item", "лайкал этот трек"),
    ("ui_dislike_all", pl.col("ui_n_dislike_all"), "user×item", "дизлайкал этот трек"),
    ("ui_days_since_last", pl.col("day") - pl.col("ui_last_day"), "user×item", "дней с последнего события с треком"),
    # --- пользователь × артист (косвенные связи)
    ("ua_listen_30d", pl.col("ua_n_listen_30d"), "user×artist", "прослушиваний артиста пользователем за 30 дней"),
    ("ua_full_rate_30d", _rate("ua_n_full_30d", "ua_n_listen_30d"), "user×artist",
     "доля дослушиваний артиста пользователем за 30 дней"),
    ("ua_listen_all", pl.col("ua_n_listen_all"), "user×artist", "прослушиваний артиста пользователем за всё время"),
    ("ua_full_rate_all", _rate("ua_n_full_all", "ua_n_listen_all"), "user×artist",
     "доля дослушиваний артиста пользователем за всё время"),
    ("ua_like_all", pl.col("ua_n_like_all"), "user×artist", "лайков артиста пользователем"),
    ("ua_dislike_all", pl.col("ua_n_dislike_all"), "user×artist", "дизлайков артиста пользователем"),
    ("ua_days_since_last", pl.col("day") - pl.col("ua_last_day"), "user×artist", "дней с последнего прослушивания артиста"),
    ("ua_share_30d", _rate("ua_n_listen_30d", "u_n_listen_30d"), "user×artist",
     "доля артиста в прослушиваниях пользователя за 30 дней"),
]
E0_FEATURES = [name for name, *_ in E0_SPEC]
E0_WINDOWS = {"u": [7, 30], "i": [7, 30], "a": [30], "ui": [30], "ua": [30]}


def build_e0_features(cands: pl.DataFrame, events: pl.LazyFrame | None = None,
                      verbose: bool = False) -> pl.DataFrame:
    """Возвращает row_id + признаки E0 (Float32) для каждой строки кандидатов"""

    base = event_base(events)
    out = cands.select("row_id", "uid", "item_id", "artist_id", "day", "track_length_seconds")

    for prefix, windows in E0_WINDOWS.items():
        t0 = time.perf_counter()
        agg = key_aggregates(base, cands, prefix, windows)
        out = out.join(agg, on=[*KEYS[prefix], "day"], how="left")
        
        if verbose:
            print(f"  {prefix:3s}: {agg.height:>10,} пар (ключ, день), {time.perf_counter() - t0:5.1f} с")

    return out.select("row_id", *[expr.cast(pl.Float32).alias(name) for name, expr, *_ in E0_SPEC])
