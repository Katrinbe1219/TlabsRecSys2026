"""Разбиение по времени

Границы — полуинтервалы [начало, конец) в секундах.
 Между частями зазор 30 минут,
чтобы одна сессия не попала в две части. Всё до начала train — история только для признаков
"""
import polars as pl

from src.data import DAY, HOUR, LAST_TIMESTAMP

GAP = HOUR // 2
WEEK = 7 * DAY
TRAIN_DAYS = 28

_test = (LAST_TIMESTAMP - WEEK, LAST_TIMESTAMP + 1)
_val_b = (_test[0] - GAP - WEEK, _test[0] - GAP)
_val_a = (_val_b[0] - GAP - WEEK, _val_b[0] - GAP)
_train = (_val_a[0] - GAP - TRAIN_DAYS * DAY, _val_a[0] - GAP)

SPLITS: dict[str, tuple[int, int]] = {
    "train": _train,
    "val_a": _val_a,
    "val_b": _val_b,
    "test": _test,
}
PARTS = list(SPLITS)


def part_expr(ts: str = "timestamp") -> pl.Expr:
    """Выражение polars: имя части для каждой строки, null — если строка в истории или в зазоре"""
    expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
    for name, (a, b) in SPLITS.items():
        expr = expr.when((pl.col(ts) >= a) & (pl.col(ts) < b)).then(pl.lit(name))
    return expr.cast(pl.Enum(PARTS))
