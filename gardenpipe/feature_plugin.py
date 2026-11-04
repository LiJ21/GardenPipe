from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

from gardenpipe.precision import check_precision


def ewma_unnormalized(
    input_expr: pl.Expr,
    time_expr: pl.Expr,
    half_life_ns: float,
    precision: str = "float64",
) -> pl.Expr | None:
    lib = _plugin_lib()
    if lib is None:
        return None
    function_name = (
        "ewma_unnormalized_f32"
        if check_precision(precision) == "float32"
        else "ewma_unnormalized"
    )
    return register_plugin_function(
        plugin_path=lib,
        function_name=function_name,
        args=[input_expr, time_expr],
        kwargs={"half_life_ns": float(half_life_ns)},
        is_elementwise=False,
        use_abs_path=True,
    )


@cache
def _plugin_lib() -> Path | None:
    if os.environ.get("GARDENPIPE_DISABLE_FEATURE_RS"):
        return None
    if sys.platform == "darwin":
        filename = "libfeature_rs.dylib"
    elif sys.platform == "win32":
        filename = "feature_rs.dll"
    else:
        filename = "libfeature_rs.so"
    path = (
        Path(__file__).resolve().parent
        / "feature_rs"
        / "target"
        / "release"
        / filename
    )
    return path if path.exists() else None
