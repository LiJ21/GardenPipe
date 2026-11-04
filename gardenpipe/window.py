from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl

from gardenpipe.data import Batch, DataSource, validate_sample_weight

_TRAIN_ROLES = frozenset({"train", "final_train"})
_VAL_ROLES = frozenset({"val", "final_val"})
_WINDOW_ROLES = _TRAIN_ROLES | _VAL_ROLES | {"test"}


@dataclass(frozen=True)
class WindowSpec:
    """Serializable day-local window and endpoint sampling policy."""

    length: int = 100
    endpoint_col: str | None = "__window_endpoint"
    group_cols: tuple[str, ...] = ()
    context_cols: tuple[str, ...] = ()
    train_samples: int | None = None
    val_stride: int = 10
    test_stride: int = 1
    seed: int = 0
    max_resident_bytes: int = 4 << 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_cols", tuple(self.group_cols))
        object.__setattr__(self, "context_cols", tuple(self.context_cols))
        if self.length < 1:
            raise ValueError("length must be positive")
        if self.endpoint_col == "":
            raise ValueError("endpoint_col must be non-empty or None")
        if self.train_samples is not None and self.train_samples < 0:
            raise ValueError("train_samples must be nonnegative or None")
        if self.val_stride < 1 or self.test_stride < 1:
            raise ValueError("validation and test strides must be positive")
        if self.max_resident_bytes < 0:
            raise ValueError("max_resident_bytes must be nonnegative")
        for name, cols in (
            ("group_cols", self.group_cols),
            ("context_cols", self.context_cols),
        ):
            if any(not col for col in cols) or len(set(cols)) != len(cols):
                raise ValueError(f"{name} must contain unique non-empty names")


@dataclass
class _Day:
    x: np.ndarray
    ends: np.ndarray
    y: np.ndarray
    weight: np.ndarray | None
    context: dict[str, np.ndarray]
    date: str
    tag: str
    nbytes: int


class WindowDataSource:
    """Gather endpoint-aligned ``(batch, time, feature)`` windows by day."""

    def __init__(self, source: DataSource, spec: WindowSpec, role: str) -> None:
        if role not in _WINDOW_ROLES:
            raise ValueError(f"unsupported window source role: {role!r}")
        self._source = source
        self.spec = spec
        self.role = role
        self._cache: OrderedDict[str, _Day] = OrderedDict()
        self._resident_bytes = 0
        self._counts: dict[str, int] = {}
        self._run_index = 0

    def __getattr__(self, name: str) -> Any:
        try:
            source = self.__dict__["_source"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(source, name)

    def __repr__(self) -> str:
        return f"WindowDataSource({self._source!r}, {self.spec!r}, role={self.role!r})"

    @property
    def source(self) -> DataSource:
        return self._source

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def resident_dates(self) -> tuple[str, ...]:
        return tuple(self._cache)

    @property
    def is_shuffled(self) -> bool:
        return self.role in _TRAIN_ROLES

    @property
    def is_deterministic(self) -> bool:
        return self.role not in _TRAIN_ROLES

    def with_transform(self, transform: Any) -> "WindowDataSource":
        return WindowDataSource(
            self._source.with_transform(transform), self.spec, self.role
        )

    def materialize(self) -> Batch:
        raise NotImplementedError("window sources are streaming; use batches(batch_size)")

    def dataframe_batches(self, *args: Any, **kwargs: Any) -> Iterator[pl.DataFrame]:
        raise NotImplementedError("window sources expose tensor batches only")

    def batches(
        self, batch_size: int | None = None, multicollect: int = -1
    ) -> Iterator[Batch]:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive or None")
        run = self._run_index
        self._run_index += 1
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.spec.seed, run, 0 if self.role == "train" else 1]
            )
        )
        dates = list(self._source.dates)
        saw_weight = False
        saw_positive_weight = False

        selected: dict[str, np.ndarray] = {}
        if self.role in _TRAIN_ROLES:
            if self.spec.train_samples is None:
                dates = [dates[i] for i in rng.permutation(len(dates))]
            else:
                for date in dates:
                    if date not in self._counts:
                        self._counts[date] = len(self._day(date).ends)
                counts = np.array([self._counts[d] for d in dates], dtype=np.int64)
                bounds = np.r_[0, counts.cumsum()]
                n = min(self.spec.train_samples, int(bounds[-1]))
                chosen = (
                    rng.choice(int(bounds[-1]), n, replace=False)
                    if n
                    else np.empty(0, np.int64)
                )
                for i, date in enumerate(dates):
                    local = (
                        chosen[(chosen >= bounds[i]) & (chosen < bounds[i + 1])]
                        - bounds[i]
                    )
                    if len(local):
                        selected[date] = local
                dates = [
                    dates[i]
                    for i in rng.permutation(len(dates))
                    if dates[i] in selected
                ]

        stride = (
            self.spec.val_stride
            if self.role in _VAL_ROLES
            else self.spec.test_stride
        )
        for date in dates:
            day = self._day(date)
            if self.role in _TRAIN_ROLES:
                pick = selected.get(date)
                if pick is None:
                    pick = rng.permutation(len(day.ends))
                else:
                    rng.shuffle(pick)
            else:
                pick = np.arange(0, len(day.ends), stride, dtype=np.int64)
            size = len(pick) if batch_size is None else batch_size
            for start in range(0, len(pick), max(1, size)):
                at = pick[start : start + size]
                ends = day.ends[at]
                rows = ends[:, None] + np.arange(1 - self.spec.length, 1)
                ctx: dict[str, Any] = {
                    "n": len(at),
                    "date": np.full(len(at), day.date),
                    "dates": [day.date],
                    "tag": np.full(len(at), day.tag),
                    "tags": [day.tag],
                }
                ctx.update({name: values[at] for name, values in day.context.items()})
                weight = day.weight[at] if day.weight is not None else None
                if weight is not None:
                    saw_weight = True
                    saw_positive_weight |= bool(np.any(weight > 0))
                yield Batch(
                    x=np.ascontiguousarray(day.x[rows]),
                    y=day.y[at],
                    ctx=ctx,
                    weight=weight,
                )
        if saw_weight and not saw_positive_weight:
            raise ValueError(
                f"{self._source.sample_weight_col} must contain at least one "
                "positive value"
            )

    def _day(self, date: str) -> _Day:
        if date in self._cache:
            self._cache.move_to_end(date)
            return self._cache[date]

        cols = [
            *self._source.features,
            self._source.target,
            *(
                [self._source.sample_weight_col]
                if self._source.sample_weight_col
                else []
            ),
            *([self.spec.endpoint_col] if self.spec.endpoint_col else []),
            *self.spec.group_cols,
            *self.spec.context_cols,
            "date",
            "tag",
        ]
        validator: Callable[[pl.DataFrame], Any] | None = None
        owner: Any = self._source
        seen: set[int] = set()
        while owner is not None and id(owner) not in seen:
            seen.add(id(owner))
            validator = validator or getattr(owner, "assert_aligned", None)
            alignment = getattr(owner, "alignment_cols", ())
            if isinstance(alignment, str):
                alignment = (alignment,)
            cols.extend(alignment)
            marker = getattr(owner, "ALIGNMENT_COL", None)
            if marker:
                cols.append(marker)
            owner = getattr(owner, "loader", None)
        cols = list(dict.fromkeys(cols))
        parts = list(
            replace(self._source, dates=[date]).dataframe_batches(None, cols=cols)
        )
        if not parts:
            day = _Day(
                np.empty((0, len(self._source.features)), np.float32),
                np.empty(0, np.int64),
                np.empty(0, np.float32),
                None,
                {},
                date,
                "",
                0,
            )
        else:
            df = (
                parts[0]
                if len(parts) == 1
                else pl.concat(parts, how="vertical_relaxed")
            )
            if validator is not None and validator(df) is False:
                raise ValueError(f"source alignment failed for {date}")
            dates = df.get_column("date").unique().to_list()
            tags = df.get_column("tag").unique().to_list()
            if len(dates) > 1 or len(tags) > 1:
                raise ValueError("each window day must have one date and tag")

            x = np.ascontiguousarray(
                df.select(self._source.features).to_numpy(), dtype=np.float32
            )
            target = df.get_column(self._source.target).to_numpy()
            mask = np.ones(df.height, dtype=bool)
            if self.spec.endpoint_col is not None:
                mask &= (
                    df.get_column(self.spec.endpoint_col)
                    .fill_null(False)
                    .cast(pl.Boolean)
                    .to_numpy()
                )
            try:
                mask &= np.isfinite(target)
            except TypeError:
                mask &= np.fromiter(
                    (value is not None for value in target), bool, df.height
                )
            ends = np.flatnonzero(
                mask & (np.arange(df.height) >= self.spec.length - 1)
            )
            if len(ends):
                last_bad = np.where(
                    np.isfinite(x).all(axis=1),
                    -1,
                    np.arange(df.height, dtype=np.int64),
                )
                np.maximum.accumulate(last_bad, out=last_bad)
                ends = ends[
                    last_bad[ends] < ends - self.spec.length + 1
                ]
            if self.spec.group_cols and len(ends):
                runs = (
                    df.select(pl.struct(self.spec.group_cols).rle_id())
                    .to_series()
                    .to_numpy()
                )
                ends = ends[runs[ends] == runs[ends - self.spec.length + 1]]

            y = np.ascontiguousarray(target[ends])
            weight = None
            if self._source.sample_weight_col is not None:
                weight = np.ascontiguousarray(
                    df.get_column(self._source.sample_weight_col).to_numpy()[ends],
                    dtype=np.float32,
                )
                validate_sample_weight(
                    weight, len(weight), self._source.sample_weight_col
                )
            context: dict[str, np.ndarray] = {}
            context_bytes = 0
            for name in self.spec.context_cols:
                if name in {"date", "tag"}:
                    continue
                series = df.get_column(name).gather(ends)
                values = series.to_numpy()
                context[name] = values
                context_bytes += values.nbytes
                if values.dtype.hasobject:
                    context_bytes += series.estimated_size()
            day = _Day(
                x=x,
                ends=ends,
                y=y,
                weight=weight,
                context=context,
                date=str(dates[0]) if dates else date,
                tag=str(tags[0]) if tags else "",
                nbytes=x.nbytes
                + ends.nbytes
                + y.nbytes
                + (weight.nbytes if weight is not None else 0)
                + context_bytes,
            )

        self._counts[date] = len(day.ends)
        if self.spec.max_resident_bytes:
            while (
                self._cache
                and (
                    day.nbytes > self.spec.max_resident_bytes
                    or self._resident_bytes + day.nbytes
                    > self.spec.max_resident_bytes
                )
            ):
                _, evicted = self._cache.popitem(last=False)
                self._resident_bytes -= evicted.nbytes
            self._cache[date] = day
            self._resident_bytes += day.nbytes
        return day


def window_wrapper(spec: WindowSpec) -> Callable[[DataSource, str], Any]:
    """Build a ``Pipeline.data_source_wrapper`` for window models."""

    def wrap(source: DataSource, role: str) -> Any:
        if role == "fit":
            return source
        if role == "predict":
            raise NotImplementedError(
                "window models do not support Pipeline.predict_frame"
            )
        return WindowDataSource(source, spec, role)

    return wrap
