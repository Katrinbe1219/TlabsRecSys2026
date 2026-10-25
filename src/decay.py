"""E2: затухающие счётчики вместо фиксированных окон.

Каждое событие дня e входит в признак на начало
дня d (e < d) с весом 2^(−(d − e)/h), где h — период
полураспада в днях: вчерашнее событие весит 2^(−1/h),
событие h дней назад — вдвое меньше и т. д.

Вместо выбора из окон 1…90 дней и всей истории подбирается одно число h.

Имя признака: "{ключ}__{фильтр}_{агрегат}__h{h}", например "ui__reco_srate__h30".
"""
import time

import polars as pl

from src import memguard
from src.data import PROCESSED
from src.grammar import ALPHA, FAMILIES, event_base, priors, queries

E2_DIR = PROCESSED / "e2"
HALF_LIVES = [3, 7, 14, 30, 60, 120, 240, 480]
#строятся в event_base in grammar.py
STATS = ["n_listen", "n_full", "n_reco", "n_reco_full", "sum_played", "sum_played_reco"]
FILTERS = {"all": ("n_listen", "n_full", "sum_played"), "reco": ("n_reco", "n_reco_full", "sum_played_reco")}
AGGS = ["cnt", "full", "srate", "played"]
EPS = 1e-9          # «истории нет»: затухающий счётчик меньше этого — доля не определена


def feature_names(fam: str, h: int) -> list[str]:
    return [f"{fam}__{f}_{a}__h{h}" for f in FILTERS for a in AGGS]


def build_decay_family(cands: pl.DataFrame, fam: str, h: int, events: pl.LazyFrame | None = None,
                       p0: dict | None = None) -> pl.DataFrame:
    """Признаки семейства fam с периодом полураспада h на уровне (ключ, день)"""
    keys = FAMILIES[fam]
    p0 = p0 or priors(events)
    q = queries(cands, fam)

    memguard.check(2.5, f"E2 {fam} h={h}")
    #растущий множитель, работает как функция
    grow = (2.0 ** (pl.col("day").cast(pl.Float64) / h))
    #кумуллятивные суммы как в grammar.py
    cum = (
        event_base(events).drop_nulls(keys)
        #semi! - фильтр, а не соединение, ключи, которые есть у кандидатов
        .join(q.select(keys).unique().lazy(), on=keys, how="semi")
        #суточные суммы
        .group_by(*keys, "day").agg(pl.col(s).sum().cast(pl.Float64) for s in STATS)
        .with_columns(pl.col(s) * grow for s in STATS)
        .sort(*keys, "day")
        .with_columns(pl.col(s).cum_sum().over(keys) for s in STATS)
        .collect(engine="streaming")
        .sort("day")
    )


    # накопленная сумма на конец дня d − 1 (последний день с событиями ≤ d − 1), затем × 2^(−d/h)
    #последний разрешенный день это вчера!
    #проверка в test_decay_ignores_current_and_future_days
    left = q.with_columns((pl.col("day") - 1).alias("_q")).sort("_q")
    tab = (left.join_asof(cum.rename({"day": "_last"}), left_on="_q", right_on="_last", by=keys,
                          strategy="backward", check_sortedness=False)
           .with_columns((pl.col(s).fill_null(0.0) * 2.0 ** (-pl.col("day").cast(pl.Float64) / h)) for s in STATS))
    
    out = []
    
    for f, (n, k, played) in FILTERS.items():
        n, k, played = pl.col(n), pl.col(k), pl.col(played)
        out += [n.alias(f"{fam}__{f}_cnt__h{h}"),
                k.alias(f"{fam}__{f}_full__h{h}"),
                # k -дослушал взвешенное количетво - отличия от кода смысла в grammar.py
                            # n -взвешенное количетсво всего
                            #p0- средняя доля дослушиваний по всем
                            #alpha - кф
                ((k + ALPHA * p0[f]) / (n + ALPHA)).alias(f"{fam}__{f}_srate__h{h}"),
                pl.when(n > EPS).then(played / n).alias(f"{fam}__{f}_played__h{h}")]
    
    return tab.select(*keys, "day", *[e.cast(pl.Float32) for e in out])


def build_all(cands: pl.DataFrame, p0: dict, half_lives=HALF_LIVES, out_dir=E2_DIR, log=print) -> list[dict]:
    """Все семейства × все h на диск: {семейство}__h{h}.parquet. Уже посчитанные файлы пропускаются"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = []
    for fam in FAMILIES:
        for h in half_lives:

            path = out_dir / f"{fam}__h{h}.parquet"
            if path.exists():
                continue

            t0 = time.perf_counter()
            tab = build_decay_family(cands, fam, h, p0=p0)
            tab.write_parquet(path)
            c = {"family": fam, "h": h, "seconds": time.perf_counter() - t0, "rows": tab.height,
                 "peak_rss_gb": memguard.peak_rss_gb()}
            cost.append(c)
            
            log(f"  {fam:3s} h={h:3d}: {tab.height:>9,} строк, {c['seconds']:5.1f} с, пик {c['peak_rss_gb']:.1f} ГБ")
            del tab
    
    return cost


def load_decay(cands: pl.DataFrame, config: dict[str, list[int]], aggs: list[str] | None = None,
               in_dir=E2_DIR) -> pl.DataFrame:
    """row_id + признаки E2 для строк cands. 
    config: {семейство: [h, ...]}; 
    порядок строк = порядку cands"""
    from src.grammar import len_bucket
    
    #работают как доп ключи, как для представителя семейства
    base = cands.select("row_id", "uid", "item_id", "artist_id", "day", "track_length_seconds").with_columns(len_bucket())
    #копилка для цикла
    out = base.select("row_id")
    
    for fam, hs in config.items():
        keys = [*FAMILIES[fam], "day"]
        for h in hs:

            # получаем нужные колонки и оставляем с нужными агрегатами, либо со вснеми
            cols = [f for f in feature_names(fam, h) if aggs is None or f.split("__")[1].split("_", 1)[1] in aggs]
            
            # из файлов получаем определенные столбцы и значения, которые есть в base - semi попогм
            t = (pl.scan_parquet(in_dir / f"{fam}__h{h}.parquet").select(*keys, *cols)
                 .join(base.lazy().select(*keys).unique(), on=keys, how="semi").collect())
            
            out = out.join(base.select("row_id", *keys).join(t, on=keys, how="left", maintain_order="left")
                           .select("row_id", *cols), on="row_id", how="left", maintain_order="left")
    
    assert out["row_id"].equals(cands["row_id"]), "load_decay: порядок строк нарушен"
    return out
