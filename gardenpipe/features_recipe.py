from __future__ import annotations

import copy
import hashlib
import inspect
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite, log
from numbers import Real
from typing import Any

import polars as pl

from gardenpipe.feature_plugin import ewma_unnormalized
from gardenpipe.precision import check_precision, float_dtype, float_lit

HalfLife = str | timedelta | int | float
SELF_CARRYOVER = "__self__"
CHILD_CARRYOVERS = "children"
_DURATION_RE = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|µs|ms|s|m|h|d|w|i)")
_DURATION_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "µs": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
    "m": 60_000_000_000.0,
    "h": 3_600_000_000_000.0,
    "d": 86_400_000_000_000.0,
    "w": 604_800_000_000_000.0,
    "i": 1.0,
}

__all__ = [
    "CHILD_CARRYOVERS",
    "SELF_CARRYOVER",
    "CarryForwardCarryover",
    "CarryForwardFeature",
    "EwmaCarryover",
    "EwmaFeature",
    "ExprFeature",
    "LookbackFeature",
]


@dataclass(frozen=True)
class CarryForwardCarryover:
    value: float


@dataclass(frozen=True)
class EwmaCarryover:
    time_ns: int
    value: float


@dataclass(frozen=True)
class ExprFeature:
    name: str
    expr: pl.Expr
    sub_features: Sequence[Any] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sub_features", _copy_sub_features(self.sub_features))
        self._validate_tree()

    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        lf = self._apply_subfeatures(lf, carryover, front_pad=front_pad)
        return lf.with_columns(self.expr.alias(self.name))

    def get_carryover(
        self, df: pl.DataFrame, previous: Any | None = None
    ) -> dict[str, Any] | None:
        children = self._get_child_carryovers(df, previous)
        return {CHILD_CARRYOVERS: children} if children else None

    def internal_cols(self) -> list[str]:
        cols: list[str] = []
        for feature in self.sub_features:
            cols.append(feature.name)
            cols.extend(feature.internal_cols())
        return _ordered_unique(cols)

    def to_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "stateful_feature",
            "type": type(self).__name__,
            "expr": _expr_config(self.expr),
            "sub_features": [_feature_config(feature) for feature in self.sub_features],
        }

    def _apply_subfeatures(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        child_carryovers = _child_carryovers(carryover)
        for feature in self.sub_features:
            lf = _apply_feature(
                feature,
                lf,
                child_carryovers.get(feature.name),
                front_pad=front_pad,
            )
        return lf

    def _get_child_carryovers(
        self, df: pl.DataFrame, previous: Any | None = None
    ) -> dict[str, Any]:
        prior = _child_carryovers(previous)
        out: dict[str, Any] = {}
        for feature in self.sub_features:
            carryover = _feature_carryover(feature, df, prior.get(feature.name))
            if carryover is not None:
                out[feature.name] = carryover
        return out

    def _validate_tree(self) -> None:
        names = [self.name]
        for feature in self.sub_features:
            names.extend(_feature_tree_names(feature))
        duplicates = _duplicates(names)
        if duplicates:
            raise ValueError(
                f"feature tree for {self.name!r} has duplicate names: {duplicates}"
            )


@dataclass(frozen=True)
class CarryForwardFeature(ExprFeature):
    precision: str = "float64"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "precision", check_precision(self.precision))

    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        lf = self._apply_subfeatures(lf, carryover, front_pad=front_pad)
        own_carryover = _own_carryover(carryover)

        seed_col = self._col("seed")
        input_col = self._col("input")
        current_col = self._col("current")

        lf = lf.with_columns(
            pl.lit(False).alias(seed_col),
            self.expr.cast(float_dtype(self.precision)).alias(input_col),
        )
        if own_carryover is not None and front_pad > 0:
            return self._apply_front_padded_carry(
                lf,
                own_carryover,
                seed_col,
                input_col,
                current_col,
                front_pad,
            )
        if own_carryover is not None:
            seed = pl.DataFrame(
                {
                    seed_col: [True],
                    input_col: pl.Series(
                        [_carry_forward_value(own_carryover)],
                        dtype=float_dtype(self.precision),
                    ),
                }
            ).lazy()
            lf = pl.concat([seed, lf], how="diagonal_relaxed")

        lf = self._with_carry_columns(lf, input_col, current_col)
        return lf.filter(~pl.col(seed_col)) if own_carryover is not None else lf

    def _apply_front_padded_carry(
        self,
        lf: pl.LazyFrame,
        own_carryover: Any,
        seed_col: str,
        input_col: str,
        current_col: str,
        front_pad: int,
    ) -> pl.LazyFrame:
        order_col = self._col("front_order")
        seed_idx = front_pad - 1
        lf = lf.with_row_index(order_col)
        active = (
            lf.filter(pl.col(order_col) >= seed_idx)
            .with_columns(
                pl.when(pl.col(order_col) == seed_idx)
                .then(True)
                .otherwise(pl.col(seed_col))
                .alias(seed_col),
                pl.when(pl.col(order_col) == seed_idx)
                .then(float_lit(_carry_forward_value(own_carryover), self.precision))
                .otherwise(pl.col(input_col))
                .alias(input_col),
            )
            .pipe(self._with_carry_columns, input_col, current_col)
        )
        if front_pad <= 1:
            return active.drop(order_col)

        inactive = lf.filter(pl.col(order_col) < seed_idx).with_columns(
            float_lit(None, self.precision).alias(current_col),
            float_lit(None, self.precision).alias(self.name),
        )
        return (
            pl.concat([inactive, active], how="diagonal_relaxed")
            .sort(order_col)
            .drop(order_col)
        )

    def get_carryover(
        self, df: pl.DataFrame, previous: Any | None = None
    ) -> dict[str, Any] | None:
        out: dict[str, Any] = {}
        own = self._get_own_carryover(df)
        if own is not None:
            out[SELF_CARRYOVER] = own
        children = self._get_child_carryovers(df, previous)
        if children:
            out[CHILD_CARRYOVERS] = children
        return out or None

    def internal_cols(self) -> list[str]:
        return _ordered_unique(
            [
                *super().internal_cols(),
                self._col("seed"),
                self._col("input"),
                self._col("current"),
            ]
        )

    def _get_own_carryover(self, df: pl.DataFrame) -> CarryForwardCarryover | None:
        current_col = self._col("current")
        if df.height == 0 or current_col not in df.columns:
            return None
        value = df.select(current_col).tail(1).item()
        if value is None:
            return None
        return CarryForwardCarryover(float(value))

    def _with_carry_columns(
        self, lf: pl.LazyFrame, input_col: str, current_col: str
    ) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col(input_col).forward_fill().alias(current_col)
        ).with_columns(pl.col(current_col).shift(1).alias(self.name))

    def to_config(self) -> dict[str, Any]:
        return {**super().to_config(), "precision": self.precision}

    def _col(self, suffix: str) -> str:
        return f"__{self.name}_{suffix}"


@dataclass(frozen=True)
class EwmaFeature(ExprFeature):
    half_life: HalfLife = 1.0
    time: str = "time"
    normalized: bool = True
    precision: str = "float64"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "precision", check_precision(self.precision))

    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        lf = self._apply_subfeatures(lf, carryover, front_pad=front_pad)
        own_carryover = _own_carryover(carryover)

        seed_col = self._col("seed")
        time_col = self._col("time_ns")
        state_time_col = self._col("state_time_ns")
        input_col = self._col("input")

        lf = lf.with_columns(
            pl.lit(False).alias(seed_col),
            pl.col(self.time).cast(pl.Datetime("ns")).dt.epoch("ns").alias(time_col),
            self._input_expr().alias(input_col),
        )
        if own_carryover is not None and front_pad > 0:
            return self._apply_front_padded_ewm(
                lf,
                own_carryover,
                seed_col,
                time_col,
                state_time_col,
                input_col,
                front_pad,
            )
        if own_carryover is not None:
            seed = pl.DataFrame(
                {
                    seed_col: [True],
                    time_col: [own_carryover.time_ns],
                    input_col: pl.Series(
                        [own_carryover.value],
                        dtype=float_dtype(self.precision),
                    ),
                }
            ).lazy()
            lf = pl.concat([seed, lf], how="diagonal_relaxed")

        lf = lf.with_columns(
            self._state_time_expr(input_col, time_col).alias(state_time_col),
            self._output_expr(input_col, time_col).alias(self.name),
        )
        return lf.filter(~pl.col(seed_col)) if own_carryover is not None else lf

    def _apply_front_padded_ewm(
        self,
        lf: pl.LazyFrame,
        own_carryover: EwmaCarryover,
        seed_col: str,
        time_col: str,
        state_time_col: str,
        input_col: str,
        front_pad: int,
    ) -> pl.LazyFrame:
        order_col = self._col("front_order")
        seed_idx = front_pad - 1
        lf = lf.with_row_index(order_col)
        active = (
            lf.filter(pl.col(order_col) >= seed_idx)
            .with_columns(
                pl.when(pl.col(order_col) == seed_idx)
                .then(True)
                .otherwise(pl.col(seed_col))
                .alias(seed_col),
                pl.when(pl.col(order_col) == seed_idx)
                .then(pl.lit(own_carryover.time_ns))
                .otherwise(pl.col(time_col))
                .alias(time_col),
                pl.when(pl.col(order_col) == seed_idx)
                .then(float_lit(own_carryover.value, self.precision))
                .otherwise(pl.col(input_col))
                .alias(input_col),
            )
            .with_columns(
                self._state_time_expr(input_col, time_col).alias(state_time_col),
                self._output_expr(input_col, time_col).alias(self.name),
            )
        )
        if front_pad <= 1:
            return active.drop(order_col)

        inactive = lf.filter(pl.col(order_col) < seed_idx).with_columns(
            float_lit(None, self.precision).alias(self.name)
        )
        return (
            pl.concat([inactive, active], how="diagonal_relaxed")
            .sort(order_col)
            .drop(order_col)
        )

    def get_carryover(
        self, df: pl.DataFrame, previous: Any | None = None
    ) -> dict[str, Any] | None:
        out: dict[str, Any] = {}
        own = self._get_own_carryover(df)
        if own is not None:
            out[SELF_CARRYOVER] = own
        children = self._get_child_carryovers(df, previous)
        if children:
            out[CHILD_CARRYOVERS] = children
        return out or None

    def internal_cols(self) -> list[str]:
        return _ordered_unique(
            [
                *super().internal_cols(),
                self._col("seed"),
                self._col("time_ns"),
                self._col("state_time_ns"),
                self._col("input"),
            ]
        )

    def to_config(self) -> dict[str, Any]:
        return {
            **super().to_config(),
            "half_life": _json_ready_half_life(self.half_life),
            "time": self.time,
            "normalized": self.normalized,
            "precision": self.precision,
        }

    def _get_own_carryover(self, df: pl.DataFrame) -> EwmaCarryover | None:
        time_col = (
            self._col("state_time_ns") if self.normalized else self._col("time_ns")
        )
        if time_col not in df.columns:
            time_col = self._col("time_ns")
        cols = [time_col, self.name]
        if df.height == 0 or any(col not in df.columns for col in cols):
            return None
        time_ns, value = df.select(cols).tail(1).row(0)
        if time_ns is None or value is None:
            return None
        return EwmaCarryover(int(time_ns), float(value))

    def _input_expr(self) -> pl.Expr:
        expr = self.expr.cast(float_dtype(self.precision))
        return expr if self.normalized else expr.fill_null(float_lit(0.0, self.precision))

    def _state_time_expr(self, input_col: str, time_col: str) -> pl.Expr:
        if not self.normalized:
            return pl.col(time_col)
        return (
            pl.when(pl.col(input_col).is_not_null())
            .then(pl.col(time_col))
            .otherwise(None)
            .forward_fill()
        )

    def _output_expr(self, input_col: str, time_col: str) -> pl.Expr:
        expr = self._ewm_expr(input_col, time_col)
        expr = expr.forward_fill() if self.normalized else expr
        return expr.cast(float_dtype(self.precision))

    def _ewm_expr(self, input_col: str, time_col: str) -> pl.Expr:
        if self.normalized:
            return self._normalized_ewm_expr(input_col, time_col)
        return self._unnormalized_ewm_expr(input_col, time_col)

    def _normalized_ewm_expr(self, input_col: str, time_col: str) -> pl.Expr:
        return pl.col(input_col).ewm_mean_by(
            pl.col(time_col),
            half_life=_half_life(self.half_life),
        ).cast(float_dtype(self.precision))

    def _unnormalized_ewm_expr(self, input_col: str, time_col: str) -> pl.Expr:
        rust_expr = ewma_unnormalized(
            pl.col(input_col),
            pl.col(time_col),
            float(_half_life_ns(self.half_life)),
            precision=self.precision,
        )
        if rust_expr is not None:
            return rust_expr.cast(float_dtype(self.precision))

        input_expr = pl.col(input_col)
        time_expr = pl.col(time_col)
        previous_time = time_expr.shift(1)
        first_at_time = previous_time.is_null() | (time_expr != previous_time)

        group_total = input_expr.sum().over(time_col)
        group_cumulative = input_expr.cum_sum().over(time_col)
        adjusted_input = (
            pl.when(first_at_time)
            .then(group_total / self._alpha_expr(time_col).fill_null(float_lit(1.0, self.precision)))
            .otherwise(float_lit(0.0, self.precision))
        )
        group_end = adjusted_input.ewm_mean_by(
            time_expr,
            half_life=_half_life(self.half_life),
        )
        return (group_end - (group_total - group_cumulative)).cast(
            float_dtype(self.precision)
        )

    def _alpha_expr(self, time_col: str) -> pl.Expr:
        dt = (pl.col(time_col) - pl.col(time_col).shift(1)).cast(
            float_dtype(self.precision)
        )
        exponent = -log(2.0) * dt / float(_half_life_ns(self.half_life))
        return (float_lit(1.0, self.precision) - exponent.exp()).cast(
            float_dtype(self.precision)
        )

    def _col(self, suffix: str) -> str:
        return f"__{self.name}_{suffix}"


@dataclass(frozen=True)
class LookbackFeature(ExprFeature):
    """``expr`` over the current rows and earlier rows carried across batches.

    ``expr`` may look back at most ``rows`` rows, or ``period`` of time on
    ``time`` (a duration string such as ``"5m"``). That much history of the
    columns it reads is kept as state and put in front of the next batch, so
    a shift or a rolling statistic gives the same values batched as in a
    full pass: ``pl.col("x").shift(3)`` with ``rows=3``,
    ``pl.col("x").rolling_mean(20)`` with ``rows=19``, or
    ``pl.col("x").rolling_mean_by("time", "5m")`` with ``period="5m"``.
    Pad rows the loader already gives a batch are not repeated.
    """

    rows: int | None = None
    period: str | None = None
    time: str = "time"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.rows is None) == (self.period is None):
            raise ValueError("LookbackFeature takes exactly one of rows and period")
        if self.rows is not None and self.rows < 1:
            raise ValueError("rows must be positive")
        if self.period is not None:
            _duration_str_ns(self.period)

    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        lf = self._apply_subfeatures(lf, carryover, front_pad=front_pad)
        seed_col = self._col("seed")
        lf = lf.with_columns(
            pl.lit(False).alias(seed_col),
            *(pl.col(col).alias(self._input_col(col)) for col in self._inputs()),
        )
        history = _own_carryover(carryover)
        if history is not None:
            # The loader's pad rows are the newest rows of the history. Their
            # inputs were recomputed here without the rows before them (a
            # sub-feature's diff, a nested shift), so take the stored values.
            pad = min(front_pad, history.height)
            if pad:
                lf = self._restore_pad_inputs(lf, history.tail(pad), front_pad)
            older = history.head(history.height - pad)
            if older.height:
                seed = older.with_columns(pl.lit(True).alias(seed_col)).lazy()
                lf = pl.concat([seed, lf], how="diagonal_relaxed")
        lf = lf.with_columns(self.expr.alias(self.name))
        return lf.filter(~pl.col(seed_col))

    def get_carryover(
        self, df: pl.DataFrame, previous: Any | None = None
    ) -> dict[str, Any] | None:
        out: dict[str, Any] = {}
        own = self._get_own_carryover(df, _own_carryover(previous))
        if own is not None:
            out[SELF_CARRYOVER] = own
        children = self._get_child_carryovers(df, previous)
        if children:
            out[CHILD_CARRYOVERS] = children
        return out or None

    def internal_cols(self) -> list[str]:
        return _ordered_unique(
            [
                *super().internal_cols(),
                self._col("seed"),
                *(self._input_col(col) for col in self._inputs()),
            ]
        )

    def to_config(self) -> dict[str, Any]:
        return {
            **super().to_config(),
            "rows": self.rows,
            "period": self.period,
            "time": self.time,
        }

    def _get_own_carryover(
        self, df: pl.DataFrame, previous: pl.DataFrame | None
    ) -> pl.DataFrame | None:
        inputs = self._inputs()
        if df.height == 0 or any(self._input_col(c) not in df.columns for c in inputs):
            return previous
        rows = df.select(pl.col(self._input_col(c)).alias(c) for c in inputs)
        if previous is not None:
            rows = pl.concat([previous, rows], how="vertical_relaxed")
        if self.rows is not None:
            return rows.tail(self.rows)
        t_ns = pl.col(self.time).cast(pl.Datetime("ns")).dt.epoch("ns")
        last = rows.select(t_ns.max()).item()
        if last is None:
            return rows
        return rows.filter(t_ns >= last - _duration_str_ns(self.period))

    def _restore_pad_inputs(
        self, lf: pl.LazyFrame, stored: pl.DataFrame, front_pad: int
    ) -> pl.LazyFrame:
        """Put ``stored``, the newest history rows, on the last pad rows of ``lf``."""
        order = self._col("order")
        first = front_pad - stored.height
        stored = stored.rename({c: self._col(f"stored_{c}") for c in stored.columns})
        on_pad = pl.col(order).is_between(first, front_pad - 1)
        return (
            lf.with_row_index(order)
            .join(
                stored.with_row_index(order, offset=first).lazy(),
                on=order,
                how="left",
                maintain_order="left",
            )
            .with_columns(
                pl.when(on_pad)
                .then(pl.col(self._col(f"stored_{c}")))
                .otherwise(pl.col(c))
                .alias(c)
                for c in self._inputs()
            )
            .drop(order, *stored.columns)
        )

    def _inputs(self) -> list[str]:
        cols = self.expr.meta.root_names()
        if self.period is not None:
            cols = [*cols, self.time]
        return _ordered_unique(cols)

    def _input_col(self, col: str) -> str:
        return self._col(f"in_{col}")

    def _col(self, suffix: str) -> str:
        return f"__{self.name}_{suffix}"


def _feature_carryover(feature: Any, df: pl.DataFrame, previous: Any | None) -> Any | None:
    """``feature``'s state after ``df``, given its state before when it takes it."""
    if previous is not None:
        params = inspect.signature(feature.get_carryover).parameters
        if "previous" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return feature.get_carryover(df, previous=previous)
    return feature.get_carryover(df)


def _child_carryovers(carryover: Any | None) -> Mapping[str, Any]:
    if not isinstance(carryover, Mapping):
        return {}
    children = carryover.get(CHILD_CARRYOVERS)
    if isinstance(children, Mapping):
        return children
    return {
        key: value
        for key, value in carryover.items()
        if key not in {SELF_CARRYOVER, CHILD_CARRYOVERS}
    }


def _own_carryover(carryover: Any | None) -> Any | None:
    if isinstance(carryover, Mapping):
        return carryover.get(SELF_CARRYOVER)
    return carryover


def _carry_forward_value(carryover: Any) -> float:
    if isinstance(carryover, CarryForwardCarryover):
        return carryover.value
    if isinstance(carryover, Real) and not isinstance(carryover, bool):
        return float(carryover)
    raise TypeError("carry-forward carryover must be CarryForwardCarryover or a number")


def _copy_sub_features(sub_features: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(copy.copy(feature) for feature in sub_features)


def _apply_feature(
    feature: Any,
    lf: pl.LazyFrame,
    carryover: Any | None,
    front_pad: int,
) -> pl.LazyFrame:
    sig = inspect.signature(feature.apply)
    if "front_pad" in sig.parameters or any(
        p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
    ):
        return feature.apply(lf, carryover, front_pad=front_pad)
    return feature.apply(lf, carryover)


def _feature_tree_names(feature: Any) -> list[str]:
    names = [feature.name]
    for child in getattr(feature, "sub_features", ()):
        names.extend(_feature_tree_names(child))
    return names


def _duplicates(values: Sequence[str]) -> list[str]:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _feature_config(feature: Any) -> dict[str, Any]:
    if hasattr(feature, "to_config"):
        return feature.to_config()
    return {"name": feature.name, "type": type(feature).__name__}


def _expr_config(expr: pl.Expr) -> dict[str, Any]:
    out: dict[str, Any] = {"repr": str(expr)}
    try:
        out["roots"] = expr.meta.root_names()
    except Exception:
        pass
    try:
        blob = expr.meta.serialize(format="json")
    except Exception:
        return out
    out.update(
        {
            "format": "json",
            "polars_version": pl.__version__,
            "expr": blob,
            "fingerprint": hashlib.sha256(blob.encode()).hexdigest(),
        }
    )
    return out


def _json_ready_half_life(half_life: HalfLife) -> str | int | float:
    if isinstance(half_life, timedelta):
        return half_life.total_seconds()
    return half_life


def _half_life_ns(half_life: HalfLife) -> int:
    if isinstance(half_life, bool):
        raise TypeError(
            "half_life must be a duration string, timedelta, or positive seconds"
        )
    if isinstance(half_life, Real):
        return _seconds_ns(float(half_life))
    if isinstance(half_life, timedelta):
        return _seconds_ns(half_life.total_seconds())
    if isinstance(half_life, str):
        return _duration_str_ns(half_life)
    raise TypeError(
        "half_life must be a duration string, timedelta, or positive seconds"
    )


def _half_life(half_life: HalfLife) -> str | timedelta:
    if isinstance(half_life, bool):
        raise TypeError(
            "half_life must be a duration string, timedelta, or positive seconds"
        )
    if isinstance(half_life, Real):
        return f"{_seconds_ns(float(half_life))}ns"
    if isinstance(half_life, str):
        # polars duration strings reject fractional values (e.g. "0.1s");
        # ours allow them, so re-emit as integer nanoseconds.
        return f"{_duration_str_ns(half_life)}ns"
    if isinstance(half_life, timedelta):
        return half_life
    raise TypeError(
        "half_life must be a duration string, timedelta, or positive seconds"
    )


def _seconds_ns(seconds: float) -> int:
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError("numeric half_life is in seconds and must be positive")
    ns = round(seconds * 1_000_000_000)
    if ns <= 0:
        raise ValueError("numeric half_life is in seconds and must be at least 0.5ns")
    return ns


def _duration_str_ns(duration: str) -> int:
    text = duration.strip()
    if not text:
        raise ValueError("duration string half_life must not be empty")
    total = 0.0
    pos = 0
    while pos < len(text):
        match = _DURATION_RE.match(text, pos)
        if match is None:
            raise ValueError(f"unsupported duration string half_life: {duration!r}")
        value, unit = match.groups()
        total += float(value) * _DURATION_NS[unit]
        pos = match.end()
    ns = round(total)
    if ns <= 0:
        raise ValueError("duration string half_life must be at least 0.5ns")
    return ns
