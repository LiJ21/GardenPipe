from __future__ import annotations

import re
from collections.abc import Sequence

import polars as pl

from gardenpipe.precision import check_precision, float_dtype, float_lit

Frame = pl.DataFrame | pl.LazyFrame

__all__ = ["add_forward_change", "add_forward_value"]

CHANGE_KINDS = ("diff", "pct", "log")
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h|d|w)")
_NS = {"ns": 1, "us": 1_000, "µs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
_NS |= {
    "m": 60 * _NS["s"],
    "h": 3_600 * _NS["s"],
    "d": 86_400 * _NS["s"],
    "w": 604_800 * _NS["s"],
}


def add_forward_value(
    df: Frame,
    expr: pl.Expr,
    horizons: Sequence[str] | str,
    weights: Sequence[float] | float = 1.0,
    time: str = "time",
    name: str = "forward_value",
    by: Sequence[str] = (),
    precision: str = "float64",
    tolerance: str | None = None,
    max_time: int | pl.DataFrame | None = None,
) -> Frame:
    """Append the horizon-weighted value of ``expr`` at ``time + horizon``.

    Each horizon looks up the last row at or before ``time + horizon`` within
    the ``by`` group (as-of join). Rows whose lookup time passes the group's
    last timestamp get null. ``tolerance`` bounds how far before
    ``time + horizon`` that row may be; ``"0s"`` accepts exact matches only.

    ``max_time`` gives that last timestamp when ``df`` holds only part of the
    data, such as one window and its look-ahead: epoch nanoseconds, or a
    frame of the ``by`` columns and ``max_time_ns``. By default it is the
    last timestamp in ``df``.
    """

    precision = check_precision(precision)
    horizons, weights = _weighted_horizons(horizons, weights)
    if len(horizons) == 0:
        return df
    lazy = isinstance(df, pl.LazyFrame)
    lf = df if lazy else df.lazy()
    join_by = [c for c in by if c in lf.collect_schema().names()]

    base = lf.with_row_index("__fv_row").with_columns(
        pl.col(time).cast(pl.Datetime("ns")).alias("__fv_t"),
        expr.cast(float_dtype(precision)).alias("__fv_src"),
    )
    if isinstance(max_time, pl.DataFrame) and not join_by:
        max_time = max_time.get_column("max_time_ns").max()
    if max_time is None:
        max_t = pl.col("__fv_t").max().over(join_by) if join_by else pl.col("__fv_t").max()
        base = base.with_columns(max_t.alias("__fv_max_t"))
    elif isinstance(max_time, pl.DataFrame):
        last = max_time.lazy().select(
            *join_by, pl.col("max_time_ns").cast(pl.Datetime("ns")).alias("__fv_max_t")
        )
        base = base.join(last, on=join_by, how="left", maintain_order="left")
    else:
        base = base.with_columns(pl.lit(max_time).cast(pl.Datetime("ns")).alias("__fv_max_t"))

    out = base
    terms, future_cols, total = [], [], 0.0
    for i, (horizon, weight) in enumerate(zip(horizons, weights)):
        if weight == 0:
            continue
        col = f"__fv_{i}"
        future_cols.append(col)
        total += weight
        left = base.select(
            "__fv_row",
            *join_by,
            (pl.col("__fv_t") + pl.duration(nanoseconds=_duration_ns(horizon))).alias("__fv_target"),
            "__fv_max_t",
        ).sort([*join_by, "__fv_target"])
        right = base.select(*join_by, "__fv_t", "__fv_row", pl.col("__fv_src").alias(col)).sort(
            [*join_by, "__fv_t", "__fv_row"]
        )
        joined = left.join_asof(
            right,
            left_on="__fv_target",
            right_on="__fv_t",
            by=join_by or None,
            strategy="backward",
            tolerance=tolerance,
            check_sortedness=False,
        ).select(
            "__fv_row",
            pl.when(pl.col("__fv_target") <= pl.col("__fv_max_t")).then(pl.col(col)).alias(col),
        )
        out = out.join(joined, on="__fv_row", how="left")
        terms.append(pl.col(col) * float_lit(weight, precision))

    if not terms or total == 0:
        raise ValueError("sum of non-zero weights must not be zero")
    out = out.with_columns(
        (sum(terms, float_lit(0.0, precision)) / float_lit(total, precision))
        .cast(float_dtype(precision))
        .alias(name)
    )
    out = out.sort("__fv_row").drop(["__fv_row", "__fv_t", "__fv_src", "__fv_max_t", *future_cols])
    return out if lazy else out.collect(engine="streaming")


def add_forward_change(
    df: Frame,
    expr: pl.Expr,
    horizons: Sequence[str] | str,
    weights: Sequence[float] | float = 1.0,
    time: str = "time",
    name: str = "forward_change",
    by: Sequence[str] = (),
    precision: str = "float64",
    kind: str = "diff",
    scale: float = 1.0,
    tolerance: str | None = None,
    max_time: int | pl.DataFrame | None = None,
) -> Frame:
    """Append ``scale * change(current, forward)`` for ``expr``.

    ``kind`` is ``"diff"`` (forward - current), ``"pct"`` (forward / current - 1)
    or ``"log"`` (log(forward / current)).
    """

    precision = check_precision(precision)
    if kind not in CHANGE_KINDS:
        raise ValueError(f"kind must be one of: {list(CHANGE_KINDS)}")
    lazy = isinstance(df, pl.LazyFrame)
    out = add_forward_value(
        df,
        expr,
        horizons,
        weights,
        time,
        "__forward_value",
        by,
        precision,
        tolerance,
        max_time,
    ).lazy()
    if "__forward_value" in out.collect_schema():
        out = out.with_columns(expr.cast(float_dtype(precision)).alias("__current"))
        forward, current = pl.col("__forward_value"), pl.col("__current")
        if kind == "diff":
            change = forward - current
        elif kind == "pct":
            change = forward / current - float_lit(1.0, precision)
        else:
            change = (forward / current).log()
        out = out.with_columns(
            (change * float_lit(scale, precision))
            .cast(float_dtype(precision))
            .alias(name)
        )
        out = out.drop("__forward_value", "__current")
    return out if lazy else out.collect(engine="streaming")


def _duration_ns(duration: str) -> int:
    pos = 0
    ns = 0.0
    for match in _DURATION_RE.finditer(duration):
        if match.start() != pos:
            raise ValueError(f"invalid duration {duration!r}")
        ns += float(match[1]) * _NS[match[2]]
        pos = match.end()
    if pos != len(duration) or ns <= 0:
        raise ValueError(f"invalid duration {duration!r}")
    return round(ns)


def _weighted_horizons(
    horizons: Sequence[str] | str,
    weights: Sequence[float] | float,
) -> tuple[list[str], list[float]]:
    horizons = [horizons] if isinstance(horizons, str) else list(horizons)
    weights = [weights] * len(horizons) if isinstance(weights, (int, float)) else list(weights)
    if len(horizons) != len(weights):
        raise ValueError("horizons and weights must be equal length")
    return horizons, weights
