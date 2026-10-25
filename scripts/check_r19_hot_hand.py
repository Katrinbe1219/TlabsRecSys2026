"""Проверка : почему s__today_rate внутри групп работает «наоборот» (эффект «горячей руки», Miller–Sanjurjo)
Сравниваем GAUC признака «доля единиц у предыдущих строк группы»
 на настоящих и перемешанных внутри группы метках"""

#чтобы првоерить нужен контролььный эксперимент: мир, где связи между прошлым и текузим точно нет
#если GAUC <0/5 появится и там, значит, дело в метрике, а не в признаке

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, polars as pl
from src.candidates import load_candidates
from src.data import HOUR
from src.grammar import E1_DIR
from src.metrics import ranking_metrics

#обязательно сортируем по времени и пользователю, ведь потом будут считаться предыдущие строки группы
vb = (load_candidates().filter(pl.col("part") == "val_b").sort("uid", "timestamp", "row_id")
      .with_columns(hour=(pl.col("timestamp") // HOUR).cast(pl.Int32),
                    #номер часа от начала датасета
                    #если пауза больше 1800 сек = 30 мин, началась новая сессия
                
                    session=((pl.col("timestamp").cast(pl.Int64).diff().over("uid") > 1800)
                             #у первой строки польлзоватлея предыдущей нет -> full_null(True)
                             .fill_null(True).cast(pl.Int32)
                             #накопленная сумма  - флаг превращается в номер сессии
                             .cum_sum().over("uid")))
      .select("row_id", "uid", "day", "hour", "session", "label"))


s = pl.read_parquet(E1_DIR / "s.parquet", columns=["row_id", "s__today_rate", "s__prev_full", "s__today_cnt"])
d = vb.join(s, on="row_id", how="left", maintain_order="left")
print("доля пустых today_rate:", round(d["s__today_rate"].null_count() / d.height, 3))


rng = np.random.default_rng(1)

def g(df, col, grp):
    return round(ranking_metrics(df.select(*grp, "label", col).drop_nulls(col), col, group=grp)["gauc"], 4)

for gname, grp in {"день": ["uid", "day"], "сессия": ["uid", "session"], "час": ["uid", "hour"]}.items():
    # «доля единиц среди предыдущих строк ЭТОЙ ЖЕ группы» — на настоящих и на перемешанных метках
    def prior_rate(df, lab): 
        return df.with_columns(
            #единиц накопилось включаая текущую строку - текущая строка
            _c=pl.col(lab).cum_sum().over(grp) - pl.col(lab),
            #номер сроки в группе с нуля, так же число предыдущих строк
              _n=pl.int_range(pl.len()).over(grp)
              #доля единиц среди предыдущих
        ).with_columns(prior=pl.when(pl.col("_n") > 0).then(pl.col("_c") / pl.col("_n")))
    
    
    shuf = d.with_columns(_r=pl.Series(rng.random(d.height))).with_columns(
        lab_sh=pl.col("label").sort_by("_r").over(grp))      # перемешали метки внутри группы
    
    real = prior_rate(d, "label")
    fake = prior_rate(shuf, "lab_sh").with_columns(label=pl.col("lab_sh"))

    print(f"{gname:7s} today_rate: {g(d, 's__today_rate', grp)} | prev_full: {g(d, 's__prev_full', grp)} | "
          f"доля 1 у предыдущих в группе — настоящие метки: {g(real, 'prior', grp)}, "
          f"ПЕРЕМЕШАННЫЕ метки: {g(fake, 'prior', grp)}")
