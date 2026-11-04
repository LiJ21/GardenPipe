from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from gardenpipe.data import _mask
from gardenpipe.features_recipe import (
    CarryForwardFeature,
    EwmaCarryover,
    EwmaFeature,
    ExprFeature,
    LookbackFeature,
)

Frame = pl.DataFrame | pl.LazyFrame
Source = Frame | str | Path
__all__ = [
    "CarryForwardFeature",
    "EwmaCarryover",
    "EwmaFeature",
    "ExprFeature",
    "LookbackFeature",
    "StatefulFeature",
    "add_features",
    "compute_features",
    "to_features",
]


def compute_features(
    lf: pl.LazyFrame,
    feature_exprs: Mapping[str, pl.Expr],
    filters: Sequence[pl.Expr] = (),
    time: str = "time",
) -> pl.LazyFrame:
    return lf.filter(_mask(filters)).select(
        pl.col(time),
        *[expr.alias(name) for name, expr in feature_exprs.items()],
    )


def add_features(
    lf: pl.LazyFrame, feature_exprs: Mapping[str, pl.Expr]
) -> pl.LazyFrame:
    return lf.with_columns([expr.alias(name) for name, expr in feature_exprs.items()])


class StatefulFeature(ABC):
    name: str

    @abstractmethod
    def apply(
        self, lf: pl.LazyFrame, carryover: Any | None, front_pad: int = 0
    ) -> pl.LazyFrame:
        raise NotImplementedError

    @abstractmethod
    def get_carryover(self, df: pl.DataFrame) -> Any | None:
        """The state after ``df``, a batch with this feature's internal columns.

        A feature whose state spans more than one batch's rows may also take
        ``previous``, its state before ``df``; it is then passed by keyword.
        """
        raise NotImplementedError

    def internal_cols(self) -> list[str]:
        return []

    def to_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "stateful_feature",
            "type": type(self).__name__,
        }


def to_features(
    df: Source,
    feature_exprs: Mapping[str, pl.Expr] | None = None,
    filters: Sequence[pl.Expr] = (),
    meta_cols: Sequence[str] | None = None,
    batch_size: int = 65_536,
) -> Frame | Iterator[pl.DataFrame]:
    """Filter a frame and append feature columns.

    A parquet path yields collected batches of ``batch_size`` rows; a frame
    returns a frame of the same (lazy or eager) kind.
    """

    feature_names = tuple(feature_exprs or ())
    if isinstance(df, (str, Path)):
        lf = _apply(
            pl.scan_parquet(df), feature_exprs, feature_names, filters, meta_cols
        )
        return lf.collect_batches(chunk_size=batch_size, maintain_order=True)

    lazy = isinstance(df, pl.LazyFrame)
    lf = _apply(df if lazy else df.lazy(), feature_exprs, feature_names, filters, meta_cols)
    return lf if lazy else lf.collect(engine="streaming")


def _apply(
    lf: pl.LazyFrame,
    feature_exprs: Mapping[str, pl.Expr] | None,
    feature_names: Sequence[str],
    filters: Sequence[pl.Expr],
    meta_cols: Sequence[str] | None,
) -> pl.LazyFrame:
    if filters:
        lf = lf.filter(_mask(filters))
    if feature_exprs:
        lf = add_features(lf, feature_exprs)
    if meta_cols is not None:
        lf = lf.select(*dict.fromkeys([*meta_cols, *feature_names]))
    return lf
