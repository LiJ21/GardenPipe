from __future__ import annotations

import os
import re
import inspect
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from gardenpipe.precision import check_precision, float_dtype

CTX_COLS = ("date", "tag")
FRONT_PAD_COL = "__load_front_pad"
SAMPLED_COL = "__sampled"
RAW_STATE_PREFIX = "__raw_state_"


@dataclass(frozen=True)
class DateFrame:
    date: str
    tag: str
    lf: pl.LazyFrame
    stateful_features: tuple[Any, ...] = ()


@dataclass
class LoadState:
    carryovers: dict[str, Any] = field(default_factory=dict)

    def get(self, feature: Any) -> Any | None:
        return self.carryovers.get(feature.name)

    def update(self, features: Sequence[Any], df: pl.DataFrame) -> None:
        # No rows, nothing learned. A feature tree would return only the
        # parts that keep state without rows and lose the rest.
        if df.height == 0:
            return
        for feature in features:
            previous = self.carryovers.get(feature.name)
            if previous is not None and _accepts_kw(feature.get_carryover, "previous"):
                carryover = feature.get_carryover(df, previous=previous)
            else:
                carryover = feature.get_carryover(df)
            if carryover is not None:
                self.carryovers[feature.name] = carryover

    def reset(self, features: Sequence[Any]) -> None:
        for feature in features:
            self.carryovers.pop(feature.name, None)


@dataclass(frozen=True)
class LoadHint:
    batch_size: int | None = None
    polars_engine: str = "streaming"
    state: LoadState = field(default_factory=LoadState)
    front_pad: int = 0
    precision: str = "float64"

    def __post_init__(self) -> None:
        if self.front_pad < 0:
            raise ValueError("front_pad must be nonnegative")
        object.__setattr__(self, "precision", check_precision(self.precision))


Loader = Callable[[list[str]], list[DateFrame]]
# "in-memory" streams only the reads of files; each batch's features, carried
# state and targets are computed in memory, where order-dependent work (EWMAs,
# shifts, look-backs) runs several times faster than on the streaming engine.
POLARS_ENGINES = {"auto", "streaming", "in-memory", "gpu"}
Window = tuple[int, int, int, int, int, pl.DataFrame | None]


@dataclass(frozen=True)
class Batch:
    x: np.ndarray
    y: np.ndarray
    ctx: dict[str, Any]
    weight: np.ndarray | None = None

    def __iter__(self) -> Iterator[Any]:
        """Preserve the historical ``x, y, ctx = batch`` interface."""
        yield self.x
        yield self.y
        yield self.ctx

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int | slice) -> Any:
        return (self.x, self.y, self.ctx)[index]


def as_batch(value: Batch | tuple[np.ndarray, np.ndarray, dict[str, Any]]) -> Batch:
    if isinstance(value, Batch):
        return value
    x, y, ctx = value
    return Batch(x=x, y=y, ctx=ctx)


def expand_dates(
    dates: str,
    exclude: str | Callable[[date], bool] | None = None,
    str_result: bool = True,
    end_date: bool = True,
):
    """Expand ``YYYYMMDD-YYYYMMDD`` (or a single ``YYYYMMDD``) into days.

    ``exclude`` drops days: ``"weekends"`` drops Saturday and Sunday, and a
    callable drops every day for which it returns true.
    """

    parts = dates.split("-")
    start = datetime.strptime(parts[0], "%Y%m%d").date()
    end = datetime.strptime(parts[-1], "%Y%m%d").date()
    if end < start:
        raise ValueError(f"end date {parts[-1]} precedes start date {parts[0]}")

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if exclude == "weekends":
        days = [d for d in days if d.weekday() < 5]
    elif callable(exclude):
        days = [d for d in days if not exclude(d)]
    elif exclude is not None:
        raise ValueError(
            f"exclude must be 'weekends', a callable or None, got {exclude!r}"
        )

    if not end_date:
        days = [d for d in days if d != end]
    return [d.isoformat() for d in days] if str_result else days


def _as_dates(dates: str | Sequence[str]) -> list[str]:
    return list(dates) if not isinstance(dates, str) else expand_dates(dates)


def _mask(filters: Sequence[pl.Expr]) -> pl.Expr:
    if not filters:
        return pl.lit(True)
    mask = filters[0]
    for expr in filters[1:]:
        mask = mask | expr
    return mask


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _stateful_internal_cols(features: Sequence[Any]) -> list[str]:
    cols: list[str] = []
    for feature in features:
        cols.extend(feature.internal_cols())
    return _ordered_unique(cols)


def _raw_state_col(name: str) -> str:
    return f"{RAW_STATE_PREFIX}{name}"


def _restore_raw_state(df: pl.DataFrame, features: Sequence[Any]) -> pl.DataFrame:
    """Put the untransformed stateful outputs back under their own names."""
    names = [f.name for f in features if _raw_state_col(f.name) in df.columns]
    if not names:
        return df
    return df.with_columns(pl.col(_raw_state_col(name)).alias(name) for name in names)


def _time_step_ns(dtype: pl.DataType) -> int:
    """The smallest step, in ns, that a value of ``dtype`` can represent."""
    if isinstance(dtype, pl.Datetime):
        return {"ns": 1, "us": 1_000, "ms": 1_000_000}[dtype.time_unit]
    if dtype == pl.Date:
        return 86_400_000_000_000
    return 1


def _epoch_ns(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Datetime("ns")).dt.epoch("ns")


def _time_windows(
    chunks: Iterable[pl.Series], batch_size: int, name: str
) -> Iterator[tuple[int, int, int]]:
    """Cut sorted times (epoch ns, in chunks) into ``(start, stop, end_ns)``.

    A window ends after the last row of its last timestamp, so no timestamp
    is split. It holds at most ``batch_size`` rows, unless one timestamp
    alone has more; that timestamp is then a window of its own. Memory is a
    few chunks, not the whole column.
    """

    buf = np.empty(0, dtype=np.int64)
    base = 0
    last: int | None = None
    for series in chunks:
        if series.null_count():
            raise ValueError(f"{name} has null times; windowed loading needs them all")
        chunk = series.to_numpy().astype(np.int64, copy=False)
        if chunk.size == 0:
            continue
        if (last is not None and chunk[0] < last) or bool(np.any(chunk[1:] < chunk[:-1])):
            raise ValueError(
                f"{name} is not sorted by time; windowed loading needs rows in time order"
            )
        last = int(chunk[-1])
        buf = np.concatenate([buf, chunk]) if buf.size else chunk
        while (cut := _window_cut(buf, batch_size, final=False)) is not None:
            yield base, base + cut, int(buf[cut - 1])
            buf, base = buf[cut:], base + cut
    while buf.size:
        cut = _window_cut(buf, batch_size, final=True)
        assert cut is not None
        yield base, base + cut, int(buf[cut - 1])
        buf, base = buf[cut:], base + cut


def _window_cut(t: np.ndarray, batch_size: int, final: bool) -> int | None:
    """Rows in the first window of sorted ``t``; None until more rows decide it."""

    n = t.size
    if n <= batch_size:
        return n if final else None
    if t[batch_size - 1] != t[batch_size]:
        return batch_size
    first = int(np.searchsorted(t, t[batch_size - 1], side="left"))
    if first > 0:
        return first
    # The first timestamp alone has more than batch_size rows.
    end = int(np.searchsorted(t, t[0], side="right"))
    return end if end < n or final else None


_CELLS_PER_WINDOW = 64
_MAX_CELLS = 1 << 20


def _union_window_ends(
    times: Sequence[pl.LazyFrame], batch_size: int, step_ns: int, engine: str
) -> list[int]:
    """Window end times (epoch ns, inclusive) over several sorted time columns.

    Each frame has one integer column ``t``. Rows are counted on a time grid
    of about 64 cells per window; a cell with more than ``batch_size`` rows
    is counted again on a finer grid, down to one time unit. A window then
    holds at most ``batch_size`` rows unless one timestamp has more, and
    memory is the number of cells, not the number of rows.
    """

    stats = pl.concat(
        [
            lf.select(
                pl.col("t").min().alias("lo"),
                pl.col("t").max().alias("hi"),
                pl.len().cast(pl.Int64).alias("n"),
            )
            for lf in times
        ]
    ).collect(engine=engine)
    n = int(stats.get_column("n").sum())
    if n == 0:
        return []
    lo = int(stats.get_column("lo").min())
    hi = int(stats.get_column("hi").max()) + 1

    def histogram(a: int, b: int, cells: int) -> list[tuple[int, int, int]]:
        width = max(1, -(-(b - a) // cells))
        counts = (
            pl.concat(
                [
                    lf.filter((pl.col("t") >= a) & (pl.col("t") < b)).select(
                        ((pl.col("t") - a) // width).alias("cell")
                    )
                    for lf in times
                ]
            )
            .group_by("cell")
            .len()
            .sort("cell")
            .collect(engine=engine)
        )
        return [
            (a + cell * width, min(a + (cell + 1) * width, b), count)
            for cell, count in counts.iter_rows()
        ]

    def refine(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        for a, b, count in cells:
            if count > batch_size and b - a > step_ns:
                out.extend(refine(histogram(a, b, _CELLS_PER_WINDOW)))
            else:
                out.append((a, b, count))
        return out

    cells = min(_MAX_CELLS, _CELLS_PER_WINDOW * -(-n // batch_size))
    ends: list[int] = []
    rows, stop = 0, lo
    for _, b, count in refine(histogram(lo, hi, cells)):
        if rows and rows + count > batch_size:
            ends.append(stop - 1)
            rows = 0
        rows += count
        stop = b
    if rows:
        ends.append(stop - 1)
    return ends


def _drop_cols(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    existing = [col for col in cols if col in df.columns]
    return df.drop(existing) if existing else df


def _hstack_unique(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        cols = [col for col in frame.columns if col not in out.columns]
        if cols:
            out = out.hstack(frame.select(cols))
    return out


def _ctx_from_df(df: pl.DataFrame) -> dict[str, Any]:
    ctx = {"n": df.height}
    for col in CTX_COLS:
        if col in df.columns:
            values = df.get_column(col).to_numpy()
            ctx[col] = values
            ctx[f"{col}s"] = _ordered_unique([str(v) for v in values.tolist()])
    return ctx


def _to_batch(
    df: pl.DataFrame,
    features: Sequence[str],
    target: str,
    sample_weight_col: str | None = None,
    require_positive_weight: bool = False,
) -> Batch:
    x = df.select(features).to_numpy()
    y = df.get_column(target).to_numpy()
    weight = (
        df.get_column(sample_weight_col).to_numpy()
        if sample_weight_col is not None
        else None
    )
    if weight is not None:
        validate_sample_weight(
            weight,
            len(y),
            sample_weight_col,
            require_positive=require_positive_weight,
        )
    return Batch(x=x, y=y, ctx=_ctx_from_df(df), weight=weight)


def validate_sample_weight(
    weight: np.ndarray,
    n: int,
    name: str = "sample_weight",
    require_positive: bool = False,
) -> None:
    weight = np.asarray(weight)
    if weight.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {weight.shape}")
    if len(weight) != n:
        raise ValueError(f"batch length mismatch: y={n} {name}={len(weight)}")
    if not np.issubdtype(weight.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {weight.dtype}")
    if not np.all(np.isfinite(weight)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(weight < 0):
        raise ValueError(f"{name} must be nonnegative")
    if require_positive and n and not np.any(weight > 0):
        raise ValueError(f"{name} must contain at least one positive value per batch")


def _streamed_engine(engine: str) -> str:
    """The engine for reading files and collecting batches.

    With "in-memory", loaders compute their windows in memory themselves;
    whatever else is cut into batches (a date with no windows, another
    loader's frames) still streams, as with "streaming".
    """
    return "streaming" if engine == "in-memory" else engine


def _check_polars_engine(engine: str) -> str:
    engine = engine.lower()
    if engine not in POLARS_ENGINES:
        raise ValueError(f"polars_engine must be one of: {sorted(POLARS_ENGINES)}")
    return engine


def _accepts_kw(fn: Any, name: str) -> bool:
    # Feature methods are checked for every batch; cache by the function.
    func = getattr(fn, "__func__", None)
    if func is None:
        return _signature_accepts(fn, name)
    key = (func, name)
    accepts = _ACCEPTS_KW.get(key)
    if accepts is None:
        accepts = _ACCEPTS_KW[key] = _signature_accepts(func, name)
    return accepts


_ACCEPTS_KW: dict[tuple[Any, str], bool] = {}


def _signature_accepts(fn: Any, name: str) -> bool:
    sig = inspect.signature(fn)
    return name in sig.parameters or any(
        p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
    )


@dataclass
class DataSource:
    dates: list[str]
    loader: Loader
    target: str
    features: list[str]
    # The sampler: a row is used if any expression matches. Features and
    # carried state are always computed from every row; to remove rows from
    # the data itself, use the loader's own filters.
    filters: tuple[pl.Expr, ...] = ()
    transform: Any = None
    cache: dict[tuple[Any, ...], Batch] | None = None
    cache_key: tuple[Any, ...] | None = None
    polars_engine: str = "streaming"
    precision: str = "float64"
    sample_weight_col: str | None = None
    _frames: list[DateFrame] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.polars_engine = _check_polars_engine(self.polars_engine)
        self.precision = check_precision(self.precision)
        if self.sample_weight_col == "":
            raise ValueError("sample_weight_col must be non-empty or None")

    def date_frames(self, hint: LoadHint | None = None) -> list[DateFrame]:
        if hint is not None:
            return self._load_frames(hint)
        if self._frames is None:
            self._frames = self._load_frames(None)
        return self._frames

    def _load_frames(self, hint: LoadHint | None) -> list[DateFrame]:
        if hint is not None and _accepts_kw(self.loader, "hint"):
            return self.loader(list(self.dates), hint=hint)
        return self.loader(list(self.dates))

    def iter_date_frames(self, hint: LoadHint | None = None) -> Iterator[DateFrame]:
        if hint is not None and hasattr(self.loader, "iter_date_frames"):
            fn = getattr(self.loader, "iter_date_frames")
            if _accepts_kw(fn, "hint"):
                yield from fn(list(self.dates), hint=hint)
            else:
                yield from fn(list(self.dates))
            return
        yield from self.date_frames(hint)

    def frame(self, select: bool = True) -> pl.LazyFrame:
        frames = [self._prepare(item, select=select) for item in self.date_frames()]
        if not frames:
            raise ValueError("DataSource has no dates")
        return pl.concat(frames, how="vertical")

    def materialize(self) -> Batch:
        if (
            self.cache is not None
            and self.cache_key is not None
            and self.cache_key in self.cache
        ):
            return self.cache[self.cache_key]
        batch = _to_batch(
            self.frame().collect(engine=self.polars_engine),
            self.features,
            self.target,
            self.sample_weight_col,
            require_positive_weight=True,
        )
        if self.cache is not None and self.cache_key is not None:
            self.cache[self.cache_key] = batch
        return batch

    def batches(
        self, batch_size: int | None = None, multicollect: int = -1
    ) -> Iterator[Batch]:
        from tqdm import tqdm

        saw_weight = False
        saw_positive_weight = False
        with tqdm(desc="Loading data", unit="row", unit_scale=True) as bar:
            for df in self.dataframe_batches(batch_size, multicollect):
                batch = _to_batch(
                    df,
                    self.features,
                    self.target,
                    self.sample_weight_col,
                )
                if batch.weight is not None:
                    saw_weight = True
                    saw_positive_weight |= bool(np.any(batch.weight > 0))
                bar.update(batch.ctx["n"])
                yield batch
        if saw_weight and not saw_positive_weight:
            raise ValueError(
                f"{self.sample_weight_col} must contain at least one positive value"
            )

    def dataframe_batches(
        self,
        batch_size: int | None = None,
        multicollect: int = -1,
        cols: Sequence[str] | None = None,
        front_pad: int = 0,
    ) -> Iterator[pl.DataFrame]:
        if multicollect == 0:
            raise ValueError("multicollect must be positive, or negative to disable")
        if front_pad < 0:
            raise ValueError("front_pad must be nonnegative")

        hint = (
            LoadHint(
                batch_size,
                self.polars_engine,
                front_pad=front_pad,
                precision=self.precision,
            )
            if batch_size is not None or front_pad > 0
            else None
        )
        selected = _ordered_unique(cols or self._default_cols())

        for item in self.iter_date_frames(hint):
            internal_cols = _stateful_internal_cols(item.stateful_features)
            if multicollect < 0:
                yield from self._combined_dataframe_batches(
                    item,
                    selected,
                    internal_cols,
                    hint,
                    batch_size,
                )
                continue

            yield from self._multicollect_dataframe_batches(
                item,
                selected,
                hint,
                batch_size,
                multicollect,
            )

    def _combined_dataframe_batches(
        self,
        item: DateFrame,
        selected: Sequence[str],
        internal_cols: Sequence[str],
        hint: LoadHint | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        defer = self._defer_sampling(item, hint)
        raw_state = self._raw_state_names(item, hint)
        # Model-input selection must not discard a feature's output before
        # its carryover is read, even when only downstream features are used.
        # Transforms already retain these outputs under raw-state aliases.
        state_cols = (
            [feature.name for feature in item.stateful_features]
            if hint is not None and not raw_state
            else []
        )
        lf = self._prepare(
            item,
            cols=[*selected, *state_cols, *internal_cols, *map(_raw_state_col, raw_state)],
            mark=defer,
            raw_state=raw_state,
        )
        if batch_size is None:
            df = lf.collect(engine=self.polars_engine)
            if hint is not None:
                hint.state.update(
                    item.stateful_features, _restore_raw_state(df, item.stateful_features)
                )
            if defer:
                df = df.filter(pl.col(SAMPLED_COL))
            if df.height:
                yield _drop_cols(df, internal_cols).select(selected)
            return

        for df in lf.collect_batches(
            chunk_size=batch_size,
            maintain_order=True,
            engine=_streamed_engine(self.polars_engine),
        ):
            if hint is not None:
                hint.state.update(
                    item.stateful_features, _restore_raw_state(df, item.stateful_features)
                )
            if defer:
                df = df.filter(pl.col(SAMPLED_COL))
                if df.height == 0:
                    continue
            yield _drop_cols(df, internal_cols).select(selected)

    def _multicollect_dataframe_batches(
        self,
        item: DateFrame,
        selected: Sequence[str],
        hint: LoadHint | None,
        batch_size: int | None,
        multicollect: int,
    ) -> Iterator[pl.DataFrame]:
        feature_cols = [col for col in self.features if col in selected]
        feature_set = set(feature_cols)
        base_cols = [col for col in selected if col not in feature_set]
        if hint is not None:
            # Collect unselected stateful outputs in the same bounded groups;
            # they supply carryover but are not returned as model inputs.
            feature_cols = _ordered_unique(
                [*feature_cols, *(feature.name for feature in item.stateful_features)]
            )
        feature_groups = [
            feature_cols[i : i + multicollect]
            for i in range(0, len(feature_cols), multicollect)
        ]
        stateful_by_name = {feature.name: feature for feature in item.stateful_features}
        group_stateful_features = [
            tuple(stateful_by_name[name] for name in group if name in stateful_by_name)
            for group in feature_groups
        ]
        group_internal_cols = [
            _stateful_internal_cols(features) for features in group_stateful_features
        ]
        defer = self._defer_sampling(item, hint)
        raw_state = self._raw_state_names(item, hint)
        base_lf = self._prepare(item, select=False, mark=defer, raw_state=raw_state)
        target_cols = [*base_cols, SAMPLED_COL] if defer else base_cols
        group_raw_cols = [
            [_raw_state_col(f.name) for f in features if f.name in raw_state]
            for features in group_stateful_features
        ]
        target_lf = base_lf.select(self._select_exprs(target_cols))
        feature_lfs = [
            base_lf.select(
                self._select_exprs([
                    *(col for col in group if col in selected or col not in raw_state),
                    *cols,
                    *raw,
                ])
            )
            for group, cols, raw in zip(
                feature_groups, group_internal_cols, group_raw_cols
            )
        ]

        if batch_size is None:
            df = target_lf.collect(engine=self.polars_engine)
            for group, features, lf in zip(
                feature_groups,
                group_stateful_features,
                feature_lfs,
            ):
                feature_df = lf.collect(engine=self.polars_engine)
                if hint is not None:
                    hint.state.update(features, _restore_raw_state(feature_df, features))
                df = _hstack_unique(
                    [df, feature_df.select([col for col in group if col in selected])]
                )
            if defer:
                df = df.filter(pl.col(SAMPLED_COL))
            if df.height:
                yield df.select(selected)
            return

        feature_iters = [
            lf.collect_batches(
                chunk_size=batch_size,
                maintain_order=True,
                engine=_streamed_engine(self.polars_engine),
            )
            for lf in feature_lfs
        ]
        for df in target_lf.collect_batches(
            chunk_size=batch_size,
            maintain_order=True,
            engine=_streamed_engine(self.polars_engine),
        ):
            for group, features, it in zip(
                feature_groups,
                group_stateful_features,
                feature_iters,
            ):
                try:
                    feature_df = next(it)
                except StopIteration as exc:
                    raise RuntimeError(
                        "multicollect feature stream ended before target stream"
                    ) from exc
                if feature_df.height != df.height:
                    raise RuntimeError(
                        "multicollect feature and target batches have "
                        f"different heights: {feature_df.height} != {df.height}"
                    )
                if hint is not None:
                    hint.state.update(features, _restore_raw_state(feature_df, features))
                df = _hstack_unique(
                    [df, feature_df.select([col for col in group if col in selected])]
                )
            if defer:
                df = df.filter(pl.col(SAMPLED_COL))
                if df.height == 0:
                    continue
            yield df.select(selected)
        for it in feature_iters:
            try:
                next(it)
            except StopIteration:
                continue
            raise RuntimeError(
                "multicollect feature stream produced more batches than target stream"
            )

    def labels(self) -> tuple[np.ndarray, dict[str, Any]]:
        parts = []
        cols = [self.target, *CTX_COLS]
        for item in self.date_frames():
            parts.append(
                self._prepare(item, cols=cols).collect(engine=self.polars_engine)
            )
        df = pl.concat(parts, how="vertical") if parts else pl.DataFrame(schema=cols)
        return df.get_column(self.target).to_numpy(), _ctx_from_df(df)

    def count(self) -> int:
        return int(
            self.frame().select(pl.len()).collect(engine=self.polars_engine).item()
        )

    def with_transform(self, transform: Any) -> "DataSource":
        from gardenpipe.transform import compose_transform

        cache_key = (
            (*self.cache_key, "transform", id(transform))
            if self.cache_key is not None
            else None
        )
        out = DataSource(
            dates=list(self.dates),
            loader=self.loader,
            target=self.target,
            features=list(self.features),
            filters=self.filters,
            transform=compose_transform(self.transform, transform),
            sample_weight_col=self.sample_weight_col,
            cache=self.cache,
            cache_key=cache_key,
            polars_engine=self.polars_engine,
            precision=self.precision,
        )
        out._frames = self._frames
        return out

    def _prepare(
        self,
        item: DateFrame,
        cols: Sequence[str] | None = None,
        select: bool = True,
        mark: bool = False,
        raw_state: Sequence[str] = (),
    ) -> pl.LazyFrame:
        """Add ctx, sample, transform and select one date frame.

        With ``mark`` the sampler adds the boolean SAMPLED_COL instead of
        dropping rows, so a batch keeps every row until its state is read.
        ``raw_state`` names stateful outputs to copy before the transform.
        """
        lf = item.lf.with_columns(
            pl.lit(item.date).alias("date"),
            pl.lit(item.tag).alias("tag"),
        )
        mark = mark and bool(self.filters)
        if self.filters:
            sampled = _mask(self.filters)
            if mark:
                lf = lf.with_columns(sampled.alias(SAMPLED_COL))
            else:
                lf = lf.filter(sampled)
        if FRONT_PAD_COL in lf.collect_schema():
            lf = lf.filter(~pl.col(FRONT_PAD_COL).fill_null(False)).drop(
                FRONT_PAD_COL
            )
        if self.transform is not None:
            if raw_state:
                lf = lf.with_columns(
                    pl.col(name).alias(_raw_state_col(name)) for name in raw_state
                )
            lf = self.transform.transform(lf)
        if not select:
            return lf
        selected = list(cols or self._default_cols())
        if mark:
            selected.append(SAMPLED_COL)
        return lf.select(self._select_exprs(selected))

    def _raw_state_names(self, item: DateFrame, hint: LoadHint | None) -> list[str]:
        # The transform may rewrite a stateful output before its state is read
        # from the batch, so keep untransformed copies while state is recorded.
        if hint is None or self.transform is None:
            return []
        return [feature.name for feature in item.stateful_features]

    def _defer_sampling(self, item: DateFrame, hint: LoadHint | None) -> bool:
        # The carried state must be read from every row of a batch, so while
        # it is being recorded the sampler drops rows only after that.
        return hint is not None and bool(item.stateful_features) and bool(self.filters)

    def _select_exprs(self, cols: Sequence[str]) -> list[pl.Expr]:
        public_cols = {*self.features, self.target}
        if self.sample_weight_col is not None:
            public_cols.add(self.sample_weight_col)
        dtype = float_dtype(self.precision)
        exprs: list[pl.Expr] = []
        for col in _ordered_unique(cols):
            expr = pl.col(col)
            if col in public_cols:
                expr = expr.cast(dtype)
            exprs.append(expr.alias(col))
        return exprs

    def _default_cols(self) -> list[str]:
        weight_cols = (
            [self.sample_weight_col] if self.sample_weight_col is not None else []
        )
        return [*self.features, self.target, *weight_cols, *CTX_COLS]


class Raw:
    """Resolve and scan one parquet file per date.

    ``path`` is a template. Its file name may use ``{date}`` (``YYYY-MM-DD``),
    ``{date_compact}`` (``YYYYMMDD``), ``{source}``, ``{source_lower}`` and
    ``{tag}``; the directory part may use every placeholder except ``{tag}``,
    plus ``{date_slash}`` (``YYYY/MM/DD``). ``{tag}`` matches any text, and the
    matched text becomes the frame's tag. Exactly one file must match.
    """

    @classmethod
    def resolve_path(
        cls,
        d: str,
        path: str,
        source: str | None = None,
    ) -> tuple[str, str]:
        if source is None and "{source" in path:
            raise ValueError(f"path {path!r} needs a source")
        fields = {
            "date": d,
            "date_compact": d.replace("-", ""),
            "date_slash": d.replace("-", "/"),
            "source": source or "",
            "source_lower": (source or "").lower(),
        }
        d_dir, name = os.path.split(path)
        d_dir = d_dir.format(**fields) or "."
        name_pat = re.escape(name.format(**fields, tag="\0")).replace(
            re.escape("\0"), "(?P<tag>.+)"
        )
        regex = re.compile("^" + name_pat + "$")
        matched = []
        for fname in os.listdir(d_dir):
            match = regex.match(fname)
            if not match:
                continue
            matched.append((fname, match.groupdict().get("tag") or ""))
        if len(matched) != 1:
            raise ValueError(
                f"expected exactly one file matching {name!r} in {d_dir!r}, "
                f"matched {[fname for fname, _ in matched]}"
            )

        fname, tag = matched[0]
        return os.path.join(d_dir, fname), tag

    @classmethod
    def load_date(
        cls,
        d: str,
        path: str,
        source: str | None = None,
        filters: Sequence[pl.Expr] = (),
        cols: Sequence[str] | None = None,
    ) -> DateFrame:
        fpath, tag = cls.resolve_path(d, path, source)
        lf = pl.scan_parquet(fpath)
        if filters:
            lf = lf.filter(_mask(filters))
        if cols is not None:
            lf = lf.select(cols)
        return DateFrame(date=d, tag=tag, lf=lf)

    @classmethod
    def load_dates(
        cls, dates: str | Sequence[str], path: str, **kwargs: Any
    ) -> list[DateFrame]:
        return [cls.load_date(d, path, **kwargs) for d in _as_dates(dates)]


@dataclass
class FeatureLoader:
    """Load per-date parquet files, add features and forward-change targets.

    ``targets`` maps target names to expressions; each target is
    ``target_scale * change(expr now, expr at time + horizon)`` with
    ``target_kind`` in ``diff``/``pct``/``log`` (see ``add_forward_change``),
    averaged over ``horizons`` with ``weights``. Horizons are resolved as-of
    on ``target_time`` within each ``target_by`` group; ``target_tolerance``
    (``"0s"`` for exact timestamps) nulls a horizon with no row close enough.
    A target may read an earlier target, but streamed in batches it is null
    near the end of each window, which looks ahead only one horizon.

    Rows come from one parquet file per date (``path``, see ``Raw``) or,
    for data already in memory, from ``frames``, a mapping from each date to
    its DataFrame. ``filters`` remove rows before anything is computed; the
    pipeline's samplers only choose rows afterwards.

    Streamed in batches, a date is cut into windows of whole timestamps
    (every row of a timestamp is computed in the same window), so the rows
    must be sorted by ``target_time``. Each window is computed from its own
    rows, so ``feature_exprs``, target expressions, ``filters`` and samplers
    must be batch-local: row-local, or over rows sharing a timestamp. History
    that crosses windows belongs in ``stateful_features``, which carry their
    state from one window to the next. ``front_pad`` gives every batch that many
    earlier rows, for inputs that need a few previous rows (such as
    ``x.shift(1)`` inside a stateful feature). Pad rows are filtered again in
    the batch they pad, without the rest of their timestamp, so with
    ``front_pad`` the ``filters`` should be row-local.
    """

    path: str | None = None
    source: str | None = None
    feature_exprs: Mapping[str, pl.Expr] = field(default_factory=dict)
    stateful_features: Sequence[Any] = field(default_factory=tuple)
    targets: Mapping[str, pl.Expr] = field(default_factory=dict)
    target_kind: str = "diff"
    target_scale: float = 1.0
    horizons: Sequence[str] | str = "1s"
    weights: Sequence[float] | float = 1.0
    filters: tuple[pl.Expr, ...] = ()
    meta_cols: tuple[str, ...] | None = None
    target_time: str = "time"
    target_by: tuple[str, ...] = ()
    precision: str = "float64"
    front_pad: int = 0
    target_tolerance: str | None = None
    frames: Mapping[str, pl.DataFrame] | None = None

    def __post_init__(self) -> None:
        if not self.path and self.frames is None:
            raise ValueError("FeatureLoader needs a path template or frames")
        if self.path and self.frames is not None:
            raise ValueError("FeatureLoader takes a path template or frames, not both")
        if self.front_pad < 0:
            raise ValueError("front_pad must be nonnegative")
        self.precision = check_precision(self.precision)

    def __call__(
        self,
        dates: str | Sequence[str],
        source: str | None = None,
        hint: LoadHint | None = None,
    ) -> list[DateFrame]:
        return self.load_dates(dates, source=source, hint=hint)

    def load_dates(
        self,
        dates: str | Sequence[str],
        source: str | None = None,
        hint: LoadHint | None = None,
    ) -> list[DateFrame]:
        src = source or self.source
        hint = self._padded(hint)
        out = []
        for d in _as_dates(dates):
            scan, tag, name = self._resolve(d, src)
            if self._use_window_hint(hint):
                if self.stateful_features:
                    raise ValueError(
                        "stateful window loading requires iter_date_frames"
                    )
                out.extend(self._load_windows(d, scan, tag, name, hint))
            else:
                out.append(self._load_date_frame(d, scan, tag))
        return out

    def iter_date_frames(
        self,
        dates: str | Sequence[str],
        source: str | None = None,
        hint: LoadHint | None = None,
    ) -> Iterator[DateFrame]:
        src = source or self.source
        hint = self._padded(hint)
        for d in _as_dates(dates):
            scan, tag, name = self._resolve(d, src)
            if self._use_window_hint(hint):
                yield from self._iter_windows(d, scan, tag, name, hint)
            else:
                yield self._load_date_frame(d, scan, tag)

    def _padded(self, hint: LoadHint | None) -> LoadHint | None:
        # The loader's pad is a floor: a caller may add rows, never remove them.
        # replace() keeps hint.state, so carried state still reaches the caller.
        if hint is None or hint.front_pad >= self.front_pad:
            return hint
        return replace(hint, front_pad=self.front_pad)

    def load_date(self, d: str, source: str | None = None) -> DateFrame:
        scan, tag, _ = self._resolve(d, source or self.source)
        return self._load_date_frame(d, scan, tag)

    def _resolve(
        self, d: str, source: str | None
    ) -> tuple[Callable[[], pl.LazyFrame], str, str]:
        """A scan of one date's raw rows, its tag, and a name for messages.

        The scan is a factory: every query gets a fresh plan node, because
        Polars 1.41 can mix up the columns of queries sharing one scan.
        """
        if self.frames is not None:
            if d not in self.frames:
                raise ValueError(f"FeatureLoader has no frame for {d!r}")
            frame = self.frames[d]
            if not isinstance(frame, pl.DataFrame):
                raise TypeError("FeatureLoader frames must be DataFrames")
            return frame.lazy, "", f"frame {d!r}"
        fpath, tag = Raw.resolve_path(d, self.path, source)
        return partial(pl.scan_parquet, fpath), tag, fpath

    def _load_date_frame(
        self,
        d: str,
        scan: Callable[[], pl.LazyFrame],
        tag: str,
        window: Window | None = None,
        carryovers: Mapping[str, Any] | None = None,
        in_memory: bool = False,
    ) -> DateFrame:
        raw = scan()
        kept_front_pad = 0
        if window is not None:
            kept_front_pad = self._kept_front_pad(scan, window)
            # Cut before anything is computed: each batch sees only its own
            # rows plus the pad, whether or not there are stateful features.
            raw = self._filter_window(raw.with_row_index("__load_row"), window)
        if not in_memory:
            return self._frame_from_raw(d, tag, raw, window, carryovers, kept_front_pad)
        # Stream the read, then compute the whole batch in memory: callers
        # only select from it. (Polars 1.41 also fails some column subsets of
        # a lazy plan over in-memory rows, such as multicollect's.)
        raw = raw.collect(engine="streaming").lazy()
        frame = self._frame_from_raw(d, tag, raw, window, carryovers, kept_front_pad)
        return replace(frame, lf=frame.lf.collect(engine="in-memory").lazy())

    def _frame_from_raw(
        self,
        d: str,
        tag: str,
        raw: pl.LazyFrame,
        window: Window | None,
        carryovers: Mapping[str, Any] | None,
        kept_front_pad: int,
        filters: tuple[pl.Expr, ...] | None = None,
    ) -> DateFrame:
        """Features, carried state and targets of ``raw``.

        With a window, ``raw`` holds a ``__load_row`` index: rows below the
        window's start are pad rows, rows from its stop on are look-ahead
        rows for the targets, and only ``kept_front_pad`` pad rows passed
        ``filters`` (the loader's own unless given).
        """
        from gardenpipe.features import to_features
        from gardenpipe.targets import add_forward_change

        filters = self.filters if filters is None else filters
        lf = to_features(raw, self.feature_exprs, filters)
        lf = self._push_stateful_features(lf, carryovers, front_pad=kept_front_pad)

        # A window holds its rows and one horizon of look-ahead, not the whole
        # date. Its lookups stop at the date's last time, as in a full pass,
        # or where the loaded rows end, past which a lookup would take the last
        # loaded row: the window's own rows never look that far, a target that
        # reads another target's look-ahead values does, and gets null there.
        max_time: int | pl.DataFrame | None = None
        if window is not None:
            _, _, end_t_ns, pad_ns, _, group_max_times = window
            max_time = end_t_ns + pad_ns
            if group_max_times is not None:
                max_time = group_max_times.select(
                    pl.exclude("__load_group_max_t_ns"),
                    pl.min_horizontal("__load_group_max_t_ns", pl.lit(max_time)).alias(
                        "max_time_ns"
                    ),
                )

        for name, expr in self.targets.items():
            lf = add_forward_change(
                lf,
                expr,
                self.horizons,
                self.weights,
                self.target_time,
                name,
                self.target_by,
                precision=self.precision,
                kind=self.target_kind,
                scale=self.target_scale,
                tolerance=self.target_tolerance,
                max_time=max_time,
            )
        if window is not None:
            start, stop, _, _, actual_front_pad, _ = window
            front_start = start - actual_front_pad
            lf = lf.filter(
                (pl.col("__load_row") >= front_start)
                & (pl.col("__load_row") < stop)
            )
            if actual_front_pad > 0:
                lf = lf.with_columns(
                    (pl.col("__load_row") < start).alias(FRONT_PAD_COL)
                )
            lf = lf.drop("__load_row")
        if self.meta_cols is not None:
            cols = [
                *self.meta_cols,
                *self.feature_exprs,
                *self._stateful_feature_names(),
                *self.targets,
            ]
            if window is not None and self._actual_front_pad(window) > 0:
                cols.append(FRONT_PAD_COL)
            lf = lf.select(_ordered_unique(cols))
        return DateFrame(
            date=d,
            tag=tag,
            lf=lf,
            stateful_features=tuple(self.stateful_features),
        )

    def _load_windows(
        self,
        d: str,
        scan: Callable[[], pl.LazyFrame],
        tag: str,
        name: str,
        hint: LoadHint | None,
    ) -> list[DateFrame]:
        return list(self._iter_windows(d, scan, tag, name, hint))

    def _iter_windows(
        self,
        d: str,
        scan: Callable[[], pl.LazyFrame],
        tag: str,
        name: str,
        hint: LoadHint | None,
    ) -> Iterator[DateFrame]:
        assert hint is not None and hint.batch_size is not None
        hint.state.reset(self.stateful_features)
        pad_ns = self._max_horizon_ns()
        engine = _streamed_engine(hint.polars_engine)
        group_max_times = self._target_group_max_times(scan(), engine)
        times = (
            df.get_column("t")
            for df in scan().select(_epoch_ns(self.target_time).alias("t")).collect_batches(
                chunk_size=hint.batch_size,
                maintain_order=True,
                engine=engine,
            )
        )
        for start, stop, end_t_ns in _time_windows(
            times, hint.batch_size, f"{name} ({self.target_time})"
        ):
            actual_front_pad = min(hint.front_pad, start)
            carryovers = {
                feature.name: hint.state.get(feature)
                for feature in self.stateful_features
            }
            yield self._load_date_frame(
                d,
                scan,
                tag,
                (
                    start,
                    stop,
                    end_t_ns,
                    pad_ns,
                    actual_front_pad,
                    group_max_times,
                ),
                carryovers,
                in_memory=hint.polars_engine == "in-memory",
            )

    def _filter_window(
        self,
        lf: pl.LazyFrame,
        window: Window,
    ) -> pl.LazyFrame:
        start, _, end_t_ns, pad_ns, actual_front_pad, _ = window
        front_start = start - actual_front_pad
        return lf.filter(
            (pl.col("__load_row") >= front_start)
            & (_epoch_ns(self.target_time) <= end_t_ns + pad_ns)
        )

    def _target_group_max_times(
        self, raw: pl.LazyFrame, polars_engine: str
    ) -> pl.DataFrame | None:
        """Each target group's last time among the rows ``filters`` keep.

        A full pass looks targets up among those rows only, so a window's
        lookups stop at the last of them.
        """
        if not self.targets:
            return None
        schema = raw.collect_schema()
        if self.target_time not in schema:
            return None
        lf = raw.filter(_mask(self.filters)) if self.filters else raw
        by = [col for col in self.target_by if col in schema]
        max_t = _epoch_ns(self.target_time).max().alias("__load_group_max_t_ns")
        if by:
            return lf.group_by(by).agg(max_t).collect(engine=polars_engine)
        out = lf.select(max_t).collect(engine=polars_engine)
        return out if out.item() is not None else None

    def _push_stateful_features(
        self,
        lf: pl.LazyFrame,
        carryovers: Mapping[str, Any] | None,
        front_pad: int = 0,
    ) -> pl.LazyFrame:
        for feature in self.stateful_features:
            carryover = carryovers.get(feature.name) if carryovers is not None else None
            if _accepts_kw(feature.apply, "front_pad"):
                lf = feature.apply(lf, carryover, front_pad=front_pad)
            else:
                lf = feature.apply(lf, carryover)
        return lf

    def _stateful_feature_names(self) -> list[str]:
        return [feature.name for feature in self.stateful_features]

    def _use_window_hint(self, hint: LoadHint | None) -> bool:
        return (
            hint is not None
            and hint.batch_size is not None
            and hint.batch_size > 0
            and bool(hint.front_pad > 0 or self.targets)
        )

    def _actual_front_pad(self, window: Window) -> int:
        return window[4]

    def _kept_front_pad(self, scan: Callable[[], pl.LazyFrame], window: Window) -> int:
        """Pad rows left after ``filters``; stateful features seed on the last.

        ``filters`` run before any feature, on raw columns, so counting the
        pad rows that pass them in the file is exact.
        """
        start, _, _, _, pad, _ = window
        if pad == 0 or not self.filters:
            return pad
        return int(
            scan()
            .slice(start - pad, pad)
            .filter(_mask(self.filters))
            .select(pl.len())
            .collect()
            .item()
        )

    def _max_horizon_ns(self) -> int:
        from gardenpipe.targets import _duration_ns

        horizons = (
            [self.horizons] if isinstance(self.horizons, str) else list(self.horizons)
        )
        return max((_duration_ns(h) for h in horizons), default=0)


# One stream's window is a small frame already in memory; the streaming
# engine costs about 4 ms of setup per such query, the in-memory one 1.5 ms.
_SMALL = "in-memory"


@dataclass
class _Stream:
    """One label's rows on one date, and what they carry between windows."""

    label: Any
    loader: FeatureLoader
    tag: str
    schema: pl.Schema
    scan: Callable[[], pl.LazyFrame] | None = None
    front_pad: int = 0
    max_times: pl.DataFrame | None = None
    state: LoadState = field(default_factory=LoadState)
    pad: pl.DataFrame | None = None
    seen: bool = False


@dataclass
class _Date:
    """The streams of one date, and the shared file they are split from."""

    d: str
    tag: str
    name: str
    schema: pl.Schema
    streams: dict[Any, _Stream]
    scan: Callable[[], pl.LazyFrame] | None = None
    front_pad: int = 0
    max_times: pl.DataFrame | None = None
    max_time_cols: tuple[str, ...] = ()


@dataclass
class MultiSourceLoader:
    """Load several streams per date, each on its own, and join them on ``on``.

    Each entry of ``sources`` labels one stream. With ``label_col``, the
    streams are the rows of each value of that column in one shared file
    per date, read by ``loader``. Otherwise each source has its own file:
    ``loader``'s path uses ``{source}``, or ``loader`` maps each source to
    its own loader.

    When the streams come from ``FeatureLoader``s, each one computes its
    features, carried state and targets from its own rows only, exactly as
    a loader reading that stream alone would, and its loader's ``filters``
    apply within it. Given a batch size, a date is cut into time windows
    that every stream shares: no timestamp is split, and a window holds at
    most ``batch_size`` rows of all streams together unless one timestamp
    has more. Each window is read once per file; every stream carries its
    state and its last ``front_pad`` rows to the next window, and the
    streams are joined per window, so memory is one window, not one date.
    ``workers`` streams are computed at a time. The shared file, or each
    stream's file, must be sorted by time.

    ``layout="wide"`` gives one row per value of ``on``. Each stream is
    deduplicated on ``on`` (its last row wins), its columns are prefixed with
    its lowercased label and ``prefix_sep``, and missing values are
    forward-filled across streams and windows unless ``forward_fill`` is
    false. ``tick_col`` adds a boolean ``<label><sep><tick_col>`` per
    stream, true on the stream's own rows and never filled. A stream with
    no rows contributes typed null columns, so the columns stay the same on
    every date. ``feature_exprs`` are evaluated on these wide rows: differences,
    ratios or averages across streams.

    ``layout="long"`` keeps every stream's own rows, with its label in
    ``label_col`` (or a ``"source"`` column for per-file streams), sorted by
    ``on`` and label; ``feature_exprs`` are computed on the wide rows and
    joined back on ``on``. With ``label_col`` and neither ``sources`` nor
    ``feature_exprs``, every label found is loaded.

    Any other loader is called with ``source=`` per whole date and joined
    lazily, without windows.
    """

    sources: Sequence[Any]
    loader: Any
    on: tuple[str, ...] = ("time",)
    forward_fill: bool = True
    dedup_on: bool = True
    prefix_sep: str = "__"
    label_col: str | None = None
    feature_exprs: Mapping[str, pl.Expr] = field(default_factory=dict)
    layout: str = "wide"
    tick_col: str | None = None
    workers: int = 8

    def __post_init__(self) -> None:
        self.sources = list(self.sources)
        self.on = tuple(self.on)
        if self.layout not in ("wide", "long"):
            raise ValueError("layout must be 'wide' or 'long'")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if not self.sources and (
            self.label_col is None or self.layout == "wide" or self.feature_exprs
        ):
            raise ValueError(
                "sources must list the streams; only the long layout with "
                "label_col and no feature_exprs loads every label it finds"
            )
        prefixes = [self._prefix(source) for source in self.sources]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError(f"sources give duplicate column prefixes: {prefixes}")
        if self.label_col is not None and not isinstance(self.loader, FeatureLoader):
            raise ValueError("label_col needs a FeatureLoader that reads the shared file")
        self._warned: set[Any] = set()

    def __call__(
        self,
        dates: str | Sequence[str],
        source: str | None = None,
        hint: LoadHint | None = None,
    ) -> list[DateFrame]:
        return self.load_dates(dates, hint=hint)

    def load_dates(
        self,
        dates: str | Sequence[str],
        source: str | None = None,
        hint: LoadHint | None = None,
    ) -> list[DateFrame]:
        """Whole dates, joined lazily; ``iter_date_frames`` gives windows."""
        if hint is not None and hint.batch_size:
            raise ValueError("windowed loading requires iter_date_frames")
        ds = _as_dates(dates)
        if not self._streamed():
            return self._lazy_dates(ds)
        return [self._whole_date(d) for d in ds]

    def iter_date_frames(
        self,
        dates: str | Sequence[str],
        source: str | None = None,
        hint: LoadHint | None = None,
    ) -> Iterator[DateFrame]:
        ds = _as_dates(dates)
        if hint is None or not hint.batch_size or not self._streamed():
            yield from self.load_dates(ds)
            return
        self._check_windowed_on()
        for d in ds:
            yield from self._windows(d, hint)

    # -- streams ---------------------------------------------------------

    def _stream_loader(self, label: Any) -> Any:
        return self.loader[label] if isinstance(self.loader, Mapping) else self.loader

    def _streamed(self) -> bool:
        if self.label_col is not None:
            return True
        return all(
            isinstance(self._stream_loader(source), FeatureLoader)
            for source in self.sources
        )

    def _prefix(self, label: Any) -> str:
        return f"{str(label).lower()}{self.prefix_sep}"

    def _check_windowed_on(self) -> None:
        """Windows are cut by time and joined one at a time, so ``on`` leads with time."""
        loaders = [self.loader] if self.label_col is not None else [
            self._stream_loader(source) for source in self.sources
        ]
        times = _ordered_unique([ld.target_time for ld in loaders])
        if times != [self.on[0]]:
            raise ValueError(
                f"windowed loading joins each time window on {list(self.on)}; "
                f"`on` must start with the time column {times}"
            )

    def _label_name(self) -> str:
        return self.label_col or "source"

    def _open(self, d: str, hint: LoadHint | None = None) -> _Date:
        """Resolve one date's streams; with a hint, also what windows need."""
        engine = _streamed_engine(hint.polars_engine) if hint is not None else "streaming"
        hint_pad = hint.front_pad if hint is not None else 0
        if self.label_col is not None:
            loader = self.loader
            file_scan, tag, name = loader._resolve(d, loader.source)
            scan = file_scan
            if self.sources:
                universe = pl.col(self.label_col).is_in(self.sources)
                scan = lambda: file_scan().filter(universe)  # noqa: E731
            date = _Date(
                d,
                tag,
                name,
                _stream_schema(loader, d, tag, scan()),
                {},
                scan=scan,
                front_pad=max(loader.front_pad, hint_pad),
            )
            if hint is not None and loader.targets:
                raw = scan()
                schema = raw.collect_schema()
                by = [c for c in loader.target_by if c in schema and c != self.label_col]
                keep_label = self.label_col in loader.target_by
                kept = raw.filter(_mask(loader.filters)) if loader.filters else raw
                date.max_times = (
                    kept.group_by([self.label_col, *by])
                    .agg(_epoch_ns(loader.target_time).max().alias("__load_group_max_t_ns"))
                    .collect(engine=engine)
                )
                date.max_time_cols = (self.label_col, *by) if keep_label else tuple(by)
            for label in self.sources:
                date.streams[label] = self._new_stream(date, label)
            return date

        streams: dict[Any, _Stream] = {}
        for label in self.sources:
            loader = self._stream_loader(label)
            scan, tag, _ = loader._resolve(d, label)
            stream = _Stream(label, loader, tag, _stream_schema(loader, d, tag, scan()), scan)
            if hint is not None:
                stream.front_pad = max(loader.front_pad, hint_pad)
                stream.max_times = loader._target_group_max_times(scan(), engine)
            streams[label] = stream
        tag = "+".join(_ordered_unique([stream.tag for stream in streams.values()]))
        first = next(iter(streams.values()))
        return _Date(d, tag, tag, first.schema, streams)

    def _new_stream(self, date: _Date, label: Any) -> _Stream:
        max_times = None
        if date.max_times is not None:
            part = date.max_times.filter(pl.col(self.label_col).eq_missing(label))
            if part.height:
                max_times = part.select(*date.max_time_cols, "__load_group_max_t_ns")
        return _Stream(
            label,
            self.loader,
            date.tag,
            date.schema,
            front_pad=date.front_pad,
            max_times=max_times,
        )

    def _stream_window(
        self,
        d: str,
        stream: _Stream,
        rows: pl.DataFrame,
        end_ns: int | None,
        pad_ns: int,
    ) -> pl.DataFrame | None:
        """One stream's rows of one window, or of the whole date without ``end_ns``.

        ``rows`` are the stream's raw rows up to ``end_ns`` plus the
        look-ahead its targets need. The stream's carried pad rows go in
        front, its state seeds the stateful features, and both are updated
        from the window's own rows.
        """
        loader = stream.loader
        if end_ns is None:
            if rows.height == 0:
                return None
            stream.seen = True
            df = loader._frame_from_raw(d, stream.tag, rows.lazy(), None, None, 0).lf
            return _conform(df.collect(engine=_SMALL), stream.schema)

        t_ns = rows.select(_epoch_ns(loader.target_time)).to_series()
        if not t_ns.is_sorted():
            raise ValueError(
                f"{stream.label!r} on {d} is not sorted by {loader.target_time}; "
                "windowed loading needs rows in time order"
            )
        n_win = int(t_ns.search_sorted(end_ns, side="right"))
        if n_win == 0:
            return None
        stream.seen = True
        win = rows.slice(0, n_win)
        pad = stream.pad if stream.pad is not None else rows.clear()
        raw = pl.concat([pad, rows], how="vertical_relaxed").with_row_index("__load_row")
        start = pad.height
        window = (start, start + n_win, end_ns, pad_ns, start, stream.max_times)
        carryovers = {f.name: stream.state.get(f) for f in loader.stateful_features}
        filters = loader.filters
        if start and filters:
            # The pad rows passed the filters with all of their timestamp's
            # rows; here they would be judged with only part of it.
            filters = (*filters, pl.col("__load_row") < start)
        lf = loader._frame_from_raw(
            d, stream.tag, raw.lazy(), window, carryovers, start, filters
        ).lf
        if start:
            lf = lf.filter(~pl.col(FRONT_PAD_COL).fill_null(False)).drop(FRONT_PAD_COL)
        df = lf.collect(engine=_SMALL)
        stream.state.update(loader.stateful_features, df)
        if stream.front_pad:
            kept = win.filter(_mask(loader.filters)) if loader.filters else win
            stream.pad = pl.concat([pad, kept], how="vertical_relaxed").tail(
                stream.front_pad
            )
        return _conform(df, stream.schema)

    # -- dates and windows -----------------------------------------------

    def _whole_date(self, d: str) -> DateFrame:
        date = self._open(d)
        schema = self._join(self._frames(date, {}))[0].collect_schema()
        return DateFrame(
            date=d,
            tag=date.tag,
            lf=pl.defer(lambda: self._load_whole(d), schema=schema),
        )

    def _load_whole(self, d: str) -> pl.DataFrame:
        date = self._open(d)
        if date.scan is not None:
            tasks = self._split(date, date.scan().collect())
        else:
            tasks = [(stream, stream.scan().collect()) for stream in date.streams.values()]
        outs = self._run(
            lambda task: self._stream_window(d, task[0], task[1], None, 0), tasks
        )
        self._warn_unseen(date)
        return self._join(self._frames(date, outs))[0]

    def _windows(self, d: str, hint: LoadHint) -> Iterator[DateFrame]:
        assert hint.batch_size is not None
        date = self._open(d, hint)
        engine = _streamed_engine(hint.polars_engine)
        carry: pl.DataFrame | None = None
        prev: int | None = None
        pool = ThreadPoolExecutor(self.workers) if self.workers > 1 else None
        try:
            for end in self._window_ends(date, hint.batch_size, engine):
                outs = self._window_outs(date, prev, end, engine, pool)
                prev = end
                frame, carry = self._join(self._frames(date, outs), carry)
                if frame.height:
                    yield DateFrame(date=d, tag=date.tag, lf=frame.lazy())
        finally:
            if pool is not None:
                pool.shutdown()
        self._warn_unseen(date)

    def _window_ends(self, date: _Date, batch_size: int, engine: str) -> list[int]:
        if date.scan is not None:
            sources = [(date.scan, self.loader.target_time, date.name)]
        else:
            sources = [
                (stream.scan, stream.loader.target_time, f"{stream.label!r} on {date.d}")
                for stream in date.streams.values()
            ]
        if len(sources) == 1:
            scan, time, name = sources[0]
            chunks = (
                df.get_column("t")
                for df in scan().select(_epoch_ns(time).alias("t")).collect_batches(
                    chunk_size=batch_size, maintain_order=True, engine=engine
                )
            )
            return [end for _, _, end in _time_windows(chunks, batch_size, f"{name} ({time})")]
        step = min(_time_step_ns(scan().collect_schema()[time]) for scan, time, _ in sources)
        times = [scan().select(_epoch_ns(time).alias("t")) for scan, time, _ in sources]
        return _union_window_ends(times, batch_size, step, engine)

    def _window_outs(
        self,
        date: _Date,
        prev: int | None,
        end: int,
        engine: str,
        pool: ThreadPoolExecutor | None,
    ) -> dict[Any, pl.DataFrame | None]:
        def span(loader: FeatureLoader) -> pl.Expr:
            t = _epoch_ns(loader.target_time)
            upper = t <= end + loader._max_horizon_ns()
            return upper if prev is None else (t > prev) & upper

        if date.scan is not None:
            loader = self.loader
            rows = date.scan().filter(span(loader)).collect(engine=engine)
            pad_ns = loader._max_horizon_ns()
            return self._run(
                lambda task: self._stream_window(date.d, task[0], task[1], end, pad_ns),
                self._split(date, rows),
                pool,
            )

        def one(stream: _Stream) -> pl.DataFrame | None:
            rows = stream.scan().filter(span(stream.loader)).collect(engine=engine)
            pad_ns = stream.loader._max_horizon_ns()
            return self._stream_window(date.d, stream, rows, end, pad_ns)

        streams = list(date.streams.values())
        return self._run(one, streams, pool)

    def _split(self, date: _Date, rows: pl.DataFrame) -> list[tuple[_Stream, pl.DataFrame]]:
        """The shared rows of each label, opening streams for labels first seen."""
        if rows.height == 0:
            return []
        tasks = []
        for key, part in rows.partition_by(
            self.label_col, as_dict=True, maintain_order=True
        ).items():
            label = key[0]
            if label not in date.streams:
                date.streams[label] = self._new_stream(date, label)
            tasks.append((date.streams[label], part))
        return tasks

    def _run(
        self,
        fn: Callable[[Any], pl.DataFrame | None],
        tasks: Sequence[Any],
        pool: ThreadPoolExecutor | None = None,
    ) -> dict[Any, pl.DataFrame | None]:
        labels = [task.label if isinstance(task, _Stream) else task[0].label for task in tasks]
        if pool is None and self.workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(self.workers) as own:
                return dict(zip(labels, own.map(fn, tasks)))
        if pool is not None and len(tasks) > 1:
            return dict(zip(labels, pool.map(fn, tasks)))
        return dict(zip(labels, map(fn, tasks)))

    def _frames(
        self, date: _Date, outs: Mapping[Any, pl.DataFrame | None]
    ) -> dict[Any, pl.DataFrame]:
        if not date.streams:
            return {None: pl.DataFrame(schema=date.schema)}
        return {
            label: out if (out := outs.get(label)) is not None else pl.DataFrame(schema=stream.schema)
            for label, stream in date.streams.items()
        }

    def _warn_unseen(self, date: _Date) -> None:
        if not self.sources:
            return
        for label, stream in date.streams.items():
            if not stream.seen and label not in self._warned:
                self._warned.add(label)
                warnings.warn(
                    f"{label!r} has no rows on {date.d}; later dates are not reported",
                    RuntimeWarning,
                    stacklevel=3,
                )

    # -- joins -------------------------------------------------------------

    def _lazy_dates(self, ds: list[str]) -> list[DateFrame]:
        by_source = {
            source: {
                item.date: item
                for item in _load_source(self._stream_loader(source), ds, source)
            }
            for source in self.sources
        }
        out = []
        for d in ds:
            items = {source: by_source[source][d] for source in self.sources}
            frame, _ = self._join({source: item.lf for source, item in items.items()})
            tag = "+".join(_ordered_unique([item.tag for item in items.values()]))
            out.append(DateFrame(date=d, tag=tag, lf=frame))
        return out

    def _join(
        self, frames: Mapping[Any, Any], carry: pl.DataFrame | None = None
    ) -> tuple[Any, pl.DataFrame | None]:
        """Join one window (or date) of every stream; ``carry`` is the last wide row."""
        wide = None
        if self.layout == "wide" or self.feature_exprs:
            wide, carry = self._wide(frames, carry)
        if self.layout == "wide":
            return wide, carry
        return self._long(frames, wide), carry

    def _wide(
        self, frames: Mapping[Any, Any], carry: pl.DataFrame | None
    ) -> tuple[Any, pl.DataFrame | None]:
        parts, ticks = [], []
        for label, frame in frames.items():
            if self.label_col is not None and self.label_col in _names(frame):
                frame = frame.drop(self.label_col)
            if self.dedup_on:
                frame = frame.unique(subset=list(self.on), keep="last", maintain_order=True)
            if self.tick_col is not None:
                frame = frame.with_columns(pl.lit(True).alias(self.tick_col))
                ticks.append(f"{self._prefix(label)}{self.tick_col}")
            # The aligned concat sorts by the keys in column order: put them
            # first, in the order of ``on``, so the rows run in that order.
            frame = frame.select(*self.on, pl.exclude(self.on))
            parts.append(_prefix_non_keys(frame, self._prefix(label), self.on))
        counts = Counter(c for part in parts for c in _names(part) if c not in self.on)
        repeated = [c for c, n in counts.items() if n > 1]
        if repeated:
            raise ValueError(f"streams give the same prefixed columns: {repeated}")
        wide = pl.concat(parts, how="align")
        if self.forward_fill:
            fill = [c for c in _names(wide) if c not in self.on and c not in ticks]
            if carry is not None:
                wide = pl.concat([carry, wide], how="vertical_relaxed")
            if fill:
                wide = wide.with_columns(pl.col(fill).forward_fill())
            if carry is not None:
                wide = wide.slice(1)
            if isinstance(wide, pl.DataFrame) and wide.height:
                carry = wide.tail(1)
        if ticks:
            wide = wide.with_columns(pl.col(ticks).fill_null(False))
        if self.feature_exprs:
            wide = wide.with_columns(
                expr.alias(name) for name, expr in self.feature_exprs.items()
            )
        return wide, carry

    def _long(self, frames: Mapping[Any, Any], wide: Any) -> Any:
        label = self._label_name()
        parts = [
            frame
            if self.label_col is not None
            else frame.with_columns(pl.lit(source).alias(label))
            for source, frame in frames.items()
        ]
        long = pl.concat(parts, how="diagonal_relaxed").sort(
            _ordered_unique([*self.on, label]), maintain_order=True
        )
        if wide is not None:
            cross = wide.select(*self.on, *self.feature_exprs)
            long = long.join(cross, on=list(self.on), how="left", maintain_order="left")
        return long


def scan_labels(
    loader: FeatureLoader, dates: str | Sequence[str], label_col: str
) -> list[Any]:
    """Every value of ``label_col`` in ``loader``'s rows over ``dates``, sorted.

    A wide ``MultiSourceLoader`` needs its labels listed up front: scan the
    dates once and pass the result as ``sources``.
    """
    labels: set[Any] = set()
    for d in _as_dates(dates):
        scan, _, _ = loader._resolve(d, loader.source)
        values = scan().select(pl.col(label_col).unique()).collect().to_series()
        labels.update(values.drop_nulls().to_list())
    return sorted(labels)


def _stream_schema(
    loader: FeatureLoader, d: str, tag: str, raw: pl.LazyFrame
) -> pl.Schema:
    """The columns one stream gives, from a run on no rows."""
    empty = pl.DataFrame(schema=raw.collect_schema()).lazy()
    df = loader._frame_from_raw(d, tag, empty, None, None, 0).lf.collect(engine=_SMALL)
    return _drop_cols(
        df, [*_stateful_internal_cols(loader.stateful_features), FRONT_PAD_COL]
    ).schema


def _conform(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    out = df.select(schema.names())
    return out if out.schema == schema else out.cast(dict(schema))


def _names(frame: pl.DataFrame | pl.LazyFrame) -> list[str]:
    return frame.collect_schema().names()


def _load_source(loader: Any, dates: Sequence[str], source: str) -> list[DateFrame]:
    fn = loader.load_dates if hasattr(loader, "load_dates") else loader
    sig = inspect.signature(fn)
    accepts_source = "source" in sig.parameters or any(
        p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
    )
    return fn(dates, source=source) if accepts_source else fn(dates)


def _prefix_non_keys(
    frame: pl.DataFrame | pl.LazyFrame, prefix: str, keys: Sequence[str]
) -> pl.DataFrame | pl.LazyFrame:
    key_set = set(keys)
    return frame.rename({c: f"{prefix}{c}" for c in _names(frame) if c not in key_set})
