"""Модели — общие для всех экспериментов, с одинаковыми параметрами

- LightGBM: основная модель. Гиперпараметры фиксированы и одинаковы для E0–E2,
  число деревьев выбирается ранней остановкой по val_a
- Логистическая регрессия (GLM): вторая модель. Счётчики сильно скошены, поэтому
  log1p; пропуски — медиана + флаг пропуска; затем стандартизация
"""
import time

import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 10,          
    "seed": SEED,
    "verbose": -1,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100


def train_lgb(X_tr, y_tr, X_va, y_va, params: dict | None = None):
    """Обучает LightGBM с ранней остановкой по logloss на (X_va, y_va).
    Возвращает (модель, история logloss по итерациям, время в секундах)"""
    params = {**LGB_PARAMS, **(params or {})}
    dtr = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    history: dict = {}
    t0 = time.perf_counter()
    model = lgb.train(
        params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dtr, dva], valid_names=["train", "val_a"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.record_evaluation(history)],
    )
    return model, history, time.perf_counter() - t0


def is_count_feature(name: str) -> bool:
    """Счётчики (неотрицательные, с длинным хвостом) логарифмируем; доли и длину трека — нет"""
    return "rate" not in name and "share" not in name and name != "track_length_seconds"


def _log1p_pos(x):
    """Перед логарифмом отрицательные заменяем на 0 -> страховка"""
    return np.log1p(np.maximum(x, 0))


def make_logreg(features: list[str]):
    counts = [i for i, f in enumerate(features) if is_count_feature(f)]
    others = [i for i, f in enumerate(features) if not is_count_feature(f)]
    
    log1p = FunctionTransformer(_log1p_pos, feature_names_out="one-to-one")
    pre = ColumnTransformer([
        ("counts", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), log1p), counts),
        ("others", SimpleImputer(strategy="median", add_indicator=True), others),
    ])
    return make_pipeline(pre, StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
