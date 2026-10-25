"""E1: грамматика признаков.

Признак = КЛЮЧ × ФИЛЬТР × АГРЕГАТ × ОКНО

  ключи     u (пользователь), i (трек), a (артист), ui (пользователь×трек),
            ua (пользователь×артист), ul (пользователь×корзина длины трека)

  фильтры   all (все прослушивания), reco (рекомендации), org (органика)

  агрегаты  cnt (прослушиваний), full (дослушиваний), skip (пропусков), rate (доля дослушиваний),
            srate (сглаженная доля), played (средняя доля прослушанного)
            + like, dislike (без фильтра) + days_since (без окна)

  окна      1, 3, 7, 14, 30, 90 дней, all (вся история)

Имя признака: "{ключ}__{фильтр}_{агрегат}__{окно}", например "ua__reco_rate__30d"

признаки «дневных» семейств считаются на начало дня: для кандидата в день d
используются только события дней 0 … d-1

Хранятся на уровне (ключ, день)  по файлу на (семейство, окно)

Сессионное семейство (s) — отдельная функция ниже: оно считается на момент НАЧАЛА каждого
прослушивания. timestamp в Yambda — момент окончания прослушивания, поэтому
точка отсчёта — окончание предыдущего прослушивания пользователя, а не timestamp кандидата
"""
import time

import polars as pl

from src import memguard
from src.data import DAY, HOUR, LISTEN_THRESHOLD, PROCESSED, scan_artist_map, scan_events
from src.split import SPLITS

E1_DIR = PROCESSED / "e1"

FAMILIES = {
    "u": ["uid"],
    "i": ["item_id"],
    "a": ["artist_id"],
    "ui": ["uid", "item_id"],
    "ua": ["uid", "artist_id"],
    "ul": ["uid", "len_bucket"],
}
WINDOWS = [1, 3, 7, 14, 30, 90, None]          # None = вся история
LEN_EDGES = (150, 240)                          # корзины длины трека, с: короткий / средний / длинный
ALPHA = 10                                      # сила сглаживания доли

STATS = ["n_listen", "n_full", "n_reco", "n_reco_full", "n_like", "n_dislike", "sum_played", "sum_played_reco"]
FILTERS = {  # фильтр -> (прослушиваний, дослушиваний, сумма played)
    "all": ("n_listen", "n_full", "sum_played"),
    "reco": ("n_reco", "n_reco_full", "sum_played_reco"),
    #ради экономии памяти, они буду создаваться
    "org": None,
}
AGGS = ["cnt", "full", "skip", "rate", "srate", "played"]


def wname(w: int | None) -> str:
    """Добавление значения окна в название признака"""
    return "all" if w is None else f"{w}d"


def feature_names(fam: str, w: int | None) -> list[str]:
    """Для каждого признака создаем один набор признаков по грамматике"""
    s = wname(w)
    names = [f"{fam}__{f}_{a}__{s}" for f in FILTERS for a in AGGS]
    names += [f"{fam}__like__{s}", f"{fam}__dislike__{s}"]
    if w is None:
        names.append(f"{fam}__days_since")
    return names


def len_bucket(col: str = "track_length_seconds") -> pl.Expr:
    """Делим треки по длительности на три группы, ради семейства пользователь* корзина длины"""
    return (pl.when(pl.col(col) < LEN_EDGES[0]).then(0)
            .when(pl.col(col) < LEN_EDGES[1]).then(1)
            .otherwise(2).cast(pl.Int8).alias("len_bucket"))


def event_base(events: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """Для подготовки событий - каждое осбытие: дослушано? рекомендовано? и тп"""
    events = scan_events() if events is None else events
    #логические маски
    listen = pl.col("event_type") == "listen"
    full = listen & (pl.col("played_ratio_pct") >= LISTEN_THRESHOLD)
    reco = listen & (pl.col("is_organic") == 0)
    #доля прослушанного трека
    #нам хочется знать от 0 до 100, а если если несколько  раз подряд слушали, нам важно именно 0-100
    played = pl.when(listen).then(pl.col("played_ratio_pct").clip(0, 100) / 100).otherwise(0.0)
    
    return (
        events.filter(pl.col("event_type").is_in(["listen", "like", "dislike"]))
        .join(scan_artist_map(), on="item_id", how="left")
        .select(
            "uid", "item_id", "artist_id",
            pl.when(listen).then(len_bucket()).alias("len_bucket"),
            # чтобы бы потом кандидант длня d видел дни <=d-1
            (pl.col("timestamp") // DAY).cast(pl.Int32).alias("day"),
            #unit32  - счетсчики целые и точны, а также логические начения можно будет потом суммировать
            #пока это просто флаги, но потом они станят числами после суммирования в build-family
            listen.cast(pl.UInt32).alias("n_listen"),
            full.cast(pl.UInt32).alias("n_full"),
            reco.cast(pl.UInt32).alias("n_reco"),
            (reco & full).cast(pl.UInt32).alias("n_reco_full"),
            (pl.col("event_type") == "like").cast(pl.UInt32).alias("n_like"),
            (pl.col("event_type") == "dislike").cast(pl.UInt32).alias("n_dislike"),
            # Float64: накопленные суммы по популярным трекам большие, Float32 потерял бы точность в разностях окон
            played.cast(pl.Float64).alias("sum_played"),
            pl.when(reco).then(played).otherwise(0.0).cast(pl.Float64).alias("sum_played_reco"),
        )
    )


def priors(events: pl.LazyFrame | None = None) -> dict[str, float]:
    """Средние доли дослушиваний для сглаживания — только по истории ДО начала train (без утечки)"""
    
    cut = SPLITS["train"][0] // DAY
    s = (event_base(events).filter(pl.col("day") < cut)
         .select(pl.col(c).sum() for c in ["n_listen", "n_full", "n_reco", "n_reco_full"]).collect().row(0, named=True))
    
    return {"all": s["n_full"] / s["n_listen"], "reco": s["n_reco_full"] / s["n_reco"],
            "org": (s["n_full"] - s["n_reco_full"]) / (s["n_listen"] - s["n_reco"])}


def _derive(fam: str, w: int | None, p0: dict) -> list[pl.Expr]:
    """прервзаем суммы в признаки, из 8 сырых сумм за окно получаем осмысленные признаки"""
    s = wname(w)
    #col = lambda c: (pl.col(f"{c}_w") if isinstance(c, str) else c)
    out = []
    win = lambda names: [pl.col(f"{c}_w") for c in names]

    for f, names in FILTERS.items():
        if names is None:   # org = all − reco
            n, k, pl_sum = (a - b for a, b in zip(win(FILTERS["all"]), win(FILTERS["reco"])))
        else:
            n, k, pl_sum = win(names)
        has = n > 0
        out += [
            #число прослушгиваний в окне
            n.alias(f"{fam}__{f}_cnt__{s}"),
            #число дослушиваний
            k.alias(f"{fam}__{f}_full__{s}"),
            # число пропусков
            (n - k).alias(f"{fam}__{f}_skip__{s}"),
            pl.when(has).then(k / n).alias(f"{fam}__{f}_rate__{s}"),
            #сглаживание
            # k -дослушал
            # n - всего слушал
            #p0- средняя доля дослушиваний по всем
            #alpha - кф
            ((k + ALPHA * p0[f]) / (n + ALPHA)).alias(f"{fam}__{f}_srate__{s}"),
            pl.when(has).then(pl_sum / n).alias(f"{fam}__{f}_played__{s}"),
        ]
    out += [pl.col("n_like_w").alias(f"{fam}__like__{s}"), pl.col("n_dislike_w").alias(f"{fam}__dislike__{s}")]
    #для экоеномии памяти
    return [e.cast(pl.Float32) for e in out]


def queries(cands: pl.DataFrame, fam: str) -> pl.DataFrame:
    """Уникальные пары (ключ, день), для которых нужны признаки."""
    c = cands.with_columns(len_bucket()) if fam == "ul" else cands
    return c.select(*FAMILIES[fam], "day").drop_nulls().unique()


def build_family(cands: pl.DataFrame, fam: str, events: pl.LazyFrame | None = None,
                 p0: dict | None = None, windows=WINDOWS, out_dir=None, verbose=False) -> dict:
    """Считает признаки семейства для всех окон. 
    Если out_dir задан — пишет по файлу на окно
    (и пропускает уже посчитанные файлы)
    
    иначе возвращает {окно: DataFrame}"""

    keys = FAMILIES[fam]
    p0 = p0 or priors(events)
    q = queries(cands, fam)
    todo = [w for w in windows if out_dir is None or not (out_dir / f"{fam}__{wname(w)}.parquet").exists()]
    if not todo:
        return {}
    
    memguard.check(2.5, f"{fam}: накопленные суммы")
    t0 = time.perf_counter()
    #ЕГО АНАЛОГ будет написан в build_sesion_features !
    cum = (
        event_base(events).drop_nulls(keys)
        #semi - оставляет только события тех ключей, что встречаются среди кандиадтов, и не добавляет колонок
        # что всетрчаются среди кандидатов, и не добавляет колонок
        #как фильтр!
        .join(q.select(keys).unique().lazy(), on=keys, how="semi")
        .group_by(*keys, "day").agg(pl.col(s).sum() for s in STATS)
        .sort(*keys, "day")
        # значения у каждого ключа становятся cumsum
        .with_columns(pl.col(s).cum_sum().over(keys) for s in STATS)
        #запускаем реальные вычисления
        .collect(engine="streaming")
        #сортируем по дня для дальнейшего join_asof
        .sort("day")
    )

    def asof(lag: int, keep_day: bool = False) -> pl.DataFrame:
        #day - 1 для сегеодняшенго дня и так далее
        left = q.with_columns((pl.col("day") - 1 - lag).alias("_q")).sort("_q")
        #backward - ближайшее прошлое, может не быть прошлого дня
        #то есть та стастика, которую мы можем брать без утечки данных
        r = left.join_asof(cum.rename({"day": "_last_day"}), left_on="_q", right_on="_last_day",
                           by=keys, strategy="backward", check_sortedness=False).drop("_q")
        
        r = r.with_columns(pl.col(s).fill_null(0) for s in STATS)
        return r if keep_day else r.drop("_last_day")

    now = asof(0, keep_day=True)
    results = {}

    for w in todo:
        memguard.check(2.0, f"{fam}: окно {wname(w)}")
        if w is None:
            #все нормально, просто добавляем суффикс, чтобы _derive работал для всех одинаково
            tab = now.with_columns(pl.col(s).alias(f"{s}_w") for s in STATS)
        else:
            past = asof(w).rename({s: f"_p_{s}" for s in STATS})
            tab = now.join(past, on=[*keys, "day"], how="left").with_columns(
                (pl.col(s) - pl.col(f"_p_{s}")).alias(f"{s}_w") for s in STATS)
            
        
        # строится если нет окна, потому что в окне и так будет присутствовать
        # у нас самая послеждняя опция None, которая означае все и на которой будет считаться extra
        #"как давно слушали - артиста/трек" и все как давно смотря на вопрос признака
        extra = [(pl.col("day") - pl.col("_last_day")).cast(pl.Float32).alias(f"{fam}__days_since")] if w is None else []
        tab = tab.select(*keys, "day", *_derive(fam, w, p0), *extra)

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            tab.write_parquet(out_dir / f"{fam}__{wname(w)}.parquet")
        else:
            results[w] = tab

        if verbose:
            print(f"  {fam:3s} {wname(w):>4s}: {tab.height:>9,} × {tab.width - len(keys) - 1} признаков, "
                  f"{time.perf_counter() - t0:6.1f} с, пик {memguard.peak_rss_gb():.1f} ГБ, "
                  f"свободно {memguard.available_gb():.1f} ГБ", flush=True)
        del tab

    return results


# ------------------------------------------------------------------ сессионное семейство
SESSION_START = SPLITS["train"][0] - DAY     # сессионные признаки смотрят на события не раньше этого момента

SESSION_FEATURES = [
    ("s__today_cnt", "прослушиваний сегодня до этого трека"),
    ("s__today_skip", "пропусков сегодня до этого трека"),
    ("s__today_rate", "доля дослушиваний сегодня до этого трека"),
    ("s__today_reco_cnt", "рекомендаций сегодня до этого трека"),
    ("s__today_reco_rate", "доля дослушанных рекомендаций сегодня"),
    ("s__hour_cnt", "прослушиваний за последний час"),
    ("s__hour_rate", "доля дослушиваний за последний час"),
    ("s__skip_streak", "пропусков подряд прямо перед этим треком (по моментам времени)"),
    ("s__prev_full", "в предыдущий момент времени было дослушивание (1/0)"),
    ("s__prev_dur_min", "сколько минут длилось предыдущее прослушивание"),
    ("s__today_artist_cnt", "прослушиваний этого артиста сегодня"),
    ("s__today_artist_rate", "доля дослушиваний этого артиста сегодня"),
]


def build_session_features(cands: pl.DataFrame, events: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Признаки на момент НАЧАЛА прослушивания-кандидата.

    timestamp = окончание прослушивания.

    Прослушивания, закончившиеся раньше timestamp
    кандидата (ts <= t-1; тот же 5-секундный момент не берём), — это те, что были ДО начала
    текущего. Порядок внутри этого 5-секундного моментна не известен, так что на всякий случай не берем его
    
    Брать текущий признак t не будем, потому что так невольно можно получить значение длительности прослушивания трека пользоватлеем
    """
    events = scan_events() if events is None else events
    # Фиксированное начало сессионной истории (не зависит от того, какие кандидаты переданы):
    # иначе значение признака строки зависело бы от "соседей по расчёту"
    start = SESSION_START
    uids = cands["uid"].unique().implode()
    L = (
        events.filter((pl.col("event_type") == "listen") & (pl.col("timestamp") >= start) & pl.col("uid").is_in(uids))
        .join(scan_artist_map(), on="item_id", how="left")
        .select("uid", "artist_id", pl.col("timestamp").alias("ts"),
                (pl.col("played_ratio_pct") >= LISTEN_THRESHOLD).cast(pl.Int32).alias("full"),
                (pl.col("is_organic") == 0).cast(pl.Int32).alias("reco"))
        .collect().sort("uid", "ts")
    )
    # Одна строка на (пользователь, момент времени): состояние ПОСЛЕ всех событий этого момента
    # Порядок событий внутри одного 5-секундного интервала не определён , поэтому
    # момент описываем только тем, что от порядка не зависит: числом событий и дослушиваний

    G = (
        L.group_by("uid", "ts")
        .agg(n=pl.len(), full=pl.col("full").sum(), reco=pl.col("reco").sum(),
             reco_full=(pl.col("reco") * pl.col("full")).sum())
        .sort("uid", "ts")
        .with_columns(last_full=(pl.col("full") > 0).cast(pl.Int32))
        #каждый момент ДОСЛУШИВАНИЯ начинает новый отрезок
        .with_columns(_seg=pl.col("last_full").cum_sum().over("uid"))
        # streak - сколкьо треков подряд пользователь пропустил с сомента последнего дослушивания
        .with_columns(streak=pl.when(pl.col("last_full") == 1).then(0)
                      .otherwise(pl.col("n")).cum_sum().over("uid", "_seg"))
        .with_columns(day=(pl.col("ts") // DAY).cast(pl.Int32),
                      prev_dur=(pl.col("ts").cast(pl.Int64) - pl.col("ts").cast(pl.Int64).shift(1).over("uid")))
        .with_columns(
            #накопленные поменты что и в buil;d_family, но по моментам
            #аналог cum
            c_n=pl.col("n").cum_sum().over("uid"), 
            c_full=pl.col("full").cum_sum().over("uid"),
            d_n=pl.col("n").cum_sum().over("uid", "day"), 
            d_full=pl.col("full").cum_sum().over("uid", "day"),
            d_reco=pl.col("reco").cum_sum().over("uid", "day"),
            d_reco_full=pl.col("reco_full").cum_sum().over("uid", "day"))
        .sort("ts")
    )
    #накопленные сумму по артисту
    GA = (
        L.drop_nulls("artist_id").group_by("uid", "artist_id", "ts")
        .agg(n=pl.len(), full=pl.col("full").sum())
        .sort("uid", "artist_id", "ts")
        .with_columns(day=(pl.col("ts") // DAY).cast(pl.Int32))
        .with_columns(a_n=pl.col("n").cum_sum().over("uid", "artist_id", "day"),
                      a_full=pl.col("full").cum_sum().over("uid", "artist_id", "day"))
        .select("uid", "artist_id", "ts", pl.col("day").alias("a_day"), "a_n", "a_full").sort("ts")
    )
    del L


    q = cands.select("row_id", "uid", "artist_id", "day", pl.col("timestamp").cast(pl.Int64).alias("t"))
    #момент пользователя 
    prev = (q.with_columns(_k=pl.col("t") - 1).sort("_k")
            .join_asof(G.with_columns(pl.col("ts").cast(pl.Int64)), left_on="_k", right_on="ts", by="uid",
                       strategy="backward", check_sortedness=False, suffix="_g"))
    # «последний час» = (prev_ts − 1 ч, prev_ts]
    #  prev_ts — окончание предыдущего прослушивания
    hour = (prev.select("row_id", "uid", (pl.col("ts") - HOUR).alias("_k")).drop_nulls("_k").sort("_k")
            .join_asof(G.select("uid", pl.col("ts").cast(pl.Int64), pl.col("c_n").alias("h_n"),
                                pl.col("c_full").alias("h_full")),
                       left_on="_k", right_on="ts", by="uid", strategy="backward", check_sortedness=False)
            .select("row_id", "h_n", "h_full"))
    #момент с артистом
    art = (q.drop_nulls("artist_id").with_columns(_k=pl.col("t") - 1).sort("_k")
           .join_asof(GA.with_columns(pl.col("ts").cast(pl.Int64)), left_on="_k", right_on="ts",
                      by=["uid", "artist_id"], strategy="backward", check_sortedness=False)
           .select("row_id", "a_day", "a_n", "a_full"))

    same_day = pl.col("day_g") == pl.col("day")
    rate = lambda k, n: pl.when(n > 0).then(k / n)
    out = (
        prev.join(hour, on="row_id", how="left").join(art, on="row_id", how="left")
        .with_columns(
            #d* - накоплены с начала дня предыбущего момента
            #если предыдубщий - вчера - то до сегодня ничего не было
            t_n=pl.when(same_day).then(pl.col("d_n")).otherwise(0),
            t_full=pl.when(same_day).then(pl.col("d_full")).otherwise(0),
            t_reco=pl.when(same_day).then(pl.col("d_reco")).otherwise(0),
            t_reco_full=pl.when(same_day).then(pl.col("d_reco_full")).otherwise(0),
            #за последний час «последний час» = (prev_ts − 1 ч, prev_ts]
            #если нет предыдущего момента или до начала часа событий не было = fill_null(0)
            hn=pl.col("c_n").fill_null(0) - pl.col("h_n").fill_null(0),
            hf=pl.col("c_full").fill_null(0) - pl.col("h_full").fill_null(0),
            ta_n=pl.when(pl.col("a_day") == pl.col("day")).then(pl.col("a_n")).otherwise(0),
            ta_full=pl.when(pl.col("a_day") == pl.col("day")).then(pl.col("a_full")).otherwise(0),
        )
        .select(
            "row_id",
            pl.col("t_n").alias("s__today_cnt"),
            (pl.col("t_n") - pl.col("t_full")).alias("s__today_skip"),
            rate(pl.col("t_full"), pl.col("t_n")).alias("s__today_rate"),
            pl.col("t_reco").alias("s__today_reco_cnt"),
            rate(pl.col("t_reco_full"), pl.col("t_reco")).alias("s__today_reco_rate"),
            pl.col("hn").alias("s__hour_cnt"),
            rate(pl.col("hf"), pl.col("hn")).alias("s__hour_rate"),
            pl.col("streak").alias("s__skip_streak"),
            pl.col("last_full").alias("s__prev_full"),
            (pl.col("prev_dur") / 60).alias("s__prev_dur_min"),
            pl.col("ta_n").alias("s__today_artist_cnt"),
            rate(pl.col("ta_full"), pl.col("ta_n")).alias("s__today_artist_rate"),
        )
        .with_columns(pl.exclude("row_id").cast(pl.Float32))
        .sort("row_id")
    )
    return out
