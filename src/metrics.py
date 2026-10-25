"""Метрики: общие для всех экспериментов.

- LogLoss и ROC-AUC — по всем строкам части.
- NDCG@K, Recall@K и GAUC — внутри групп «пользователь × день», только в группах,
  где есть и 0, и 1 

 «общие», и «групповые» метрики:
-  ROC-AUC по всем строкам в основном
измеряет, умеет ли модель отличать «дослушивающих» пользователей от «пропускающих».
Признак пользователя одинаков для всех его треков за день и внутри группы ничего не
ранжирует
- Групповые метрики (NDCG, GAUC) измеряют именно ранжирование треков для
одного пользователя — то, что нужно рекомендательной системе

Ничьи в скоре разбиваются случайно (с фиксированным seed), а не по порядку строк:
иначе постоянный внутри группы скор ранжировал бы треки по времени
"""
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

GROUP = ["uid", "day"]

#k - глубина топа для групповых метрик
def ranking_metrics(df: pl.DataFrame, score: str, k: int = 10, group: list[str] = GROUP) -> dict:
    """group — ключи групп (по умолчанию «пользователь × день»; другие — только для проверок"""
    GROUP = group
    #единица на позиици дает  r дает влкда по формуле ниже
    disc = 1.0 / np.log2(np.arange(2, k + 2))            # 1/log2(r+1) для r = 1..k
    #idcg - DCG иедальной выдачи, где все единицы стоят наверху
    #NDCG = DCG / IDCG
    ideal = np.concatenate([[0.0], np.cumsum(disc)])  
    #чтобы разбивать ничьи по скору, то есть расставлять строки с одинаковым скором в случайном порядке   
    tiebreak = np.random.default_rng(0).random(df.height)    # своё случайное число у каждой строки
    
    
    g = (
        df.select(*GROUP, "label", score)
        #привязка идет но номеру строки
        .with_columns(_tb=pl.Series(tiebreak))
        .with_columns(
            # прибавление к целому рангу различия при ничьих
            #.rank("ordinal").over(GROUP - ранжирует получившиеся дробные числа и превращает
            # в уникальные позициию Ordinal-rank никода не наедт одинаковых значений
            # OVER (PARTITION BY uid, day)
            # dense -  жто как min, но бещ пропуском: следующий ранг идет подряд (при ничье min пропускает один порядок)
            rank=(pl.col(score).rank("dense", descending=True) + pl.col("_tb")).rank("ordinal").over(GROUP),
            #ранг во возрастанию внутри группы, average - равные скоры получают средний ранг
            #ранг нужен для AUC< средний ранг автоматически засчитывает ничью как 0.5 пары
            arank=pl.col(score).rank("average").over(GROUP),        # для GAUC: ничья = 1/2
            #число единиц в группе, занчение щаписывает в каждую строку группы
            npos=pl.col("label").sum().over(GROUP),
            n=pl.len().over(GROUP),
        )
        #работает по сторчно,  но у группы знчения одинаковые, поэтому оставляется так точно целая гурппа
        # где есть хотя бы один ноль и хотя бы одна единица для NCGD
        .filter((pl.col("npos") > 0) & (pl.col("npos") < pl.col("n")))
        .with_columns(top=pl.col("rank") <= k)
        #считаем DCG , кто в топе
        .with_columns(gain=pl.when(pl.col("top")).then(pl.col("label") / (pl.col("rank") + 1).log(2)).otherwise(0.0))
        .group_by(GROUP)
        .agg(dcg=pl.col("gain").sum(),
             #равно 1 умножение, когда строка одновременно положительна и в топе
             # сколкьо единиц попало в топ-k
             hits=(pl.col("label") * pl.col("top")).sum(),
             
             npos=pl.col("npos").first(),
             n=pl.col("n").first(),
             #у нулей дает 0 - сумма возрватсающих рангов
             pos_rank_sum=(pl.col("arank") * pl.col("label")).sum())
        
        # AUC через ранги : доля пар (1, 0), где у единицы скор выше
        #формула Манна-Уитни
        #npos - количество единиц, n-npos - количество нулей
        #pos_rank_sum - сумма рангов единиц
        .with_columns(auc=(pl.col("pos_rank_sum") - pl.col("npos") * (pl.col("npos") + 1) / 2)
                      / (pl.col("npos") * (pl.col("n") - pl.col("npos"))))
    )
    idcg = ideal[np.minimum(g["npos"].to_numpy(), k)]
    return {
        f"ndcg@{k}": float(np.mean(g["dcg"].to_numpy() / idcg)),
        f"recall@{k}": float(np.mean(g["hits"].to_numpy() / g["npos"].to_numpy())),
        "gauc": float(g["auc"].mean()),
        "groups": g.height,
    }


def evaluate(df: pl.DataFrame, score: str, k: int = 10) -> dict:
    """df должен содержать uid, day, label и колонку со скором (вероятностью для LogLoss)."""
    y = df["label"].to_numpy()
    p = np.clip(df[score].to_numpy().astype(np.float64), 1e-7, 1 - 1e-7)
    return {"logloss": float(log_loss(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            **ranking_metrics(df, score, k)}
