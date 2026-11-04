from __future__ import annotations

import numpy as np
import polars as pl

FLOAT_PRECISIONS = ("float64", "float32")


def check_precision(precision: str) -> str:
    precision = precision.lower()
    if precision not in FLOAT_PRECISIONS:
        raise ValueError(f"precision must be one of: {list(FLOAT_PRECISIONS)}")
    return precision


def float_dtype(precision: str) -> pl.DataType:
    return pl.Float32 if check_precision(precision) == "float32" else pl.Float64


def np_float_dtype(precision: str) -> type[np.floating]:
    return np.float32 if check_precision(precision) == "float32" else np.float64


def float_lit(value: float | None, precision: str) -> pl.Expr:
    return pl.lit(value, dtype=float_dtype(precision))
