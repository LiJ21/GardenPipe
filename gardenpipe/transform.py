from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

from gardenpipe.data import Batch


@runtime_checkable
class Transform(Protocol):
    def fit(self, data: Any) -> "Transform": ...

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame: ...


class FitSource(Protocol):
    dates: list[str]
    target: str
    features: list[str]
    sample_weight_col: str | None

    def frame(self, select: bool = True) -> pl.LazyFrame: ...

    def batches(self, batch_size: int | None = None) -> Iterator[Batch]: ...

    def with_transform(self, transform: Transform) -> "FitSource": ...


@dataclass
class ComposeTransform:
    steps: list[Transform]

    def fit(self, data: Any) -> "ComposeTransform":
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for step in self.steps:
            lf = step.transform(lf)
        return lf


def compose_transform(left: Transform | None, right: Transform) -> Transform:
    if left is None:
        return right
    if isinstance(left, ComposeTransform):
        return ComposeTransform([*left.steps, right])
    return ComposeTransform([left, right])


@dataclass
class Passthrough:
    def fit(self, data: Any) -> "Passthrough":
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf


@dataclass
class FunctionTransform:
    fn: Callable[[pl.LazyFrame], pl.LazyFrame]

    def fit(self, data: Any) -> "FunctionTransform":
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return self.fn(lf)


@dataclass
class TargetScaler:
    """Divide the target by a (floored) feature column."""

    feature: str
    target: str | None = None
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError("feature must be non-empty")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    def fit(self, data: Any) -> "TargetScaler":
        if self.target is None:
            target = getattr(data, "target", None)
            if target is None:
                raise ValueError("TargetScaler needs target or source.target")
            self.target = str(target)
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.target is None:
            raise RuntimeError("TargetScaler must be fit before transform")
        denominator = pl.max_horizontal(
            pl.col(self.feature), pl.lit(float(self.eps))
        )
        return lf.with_columns(
            (pl.col(self.target) / denominator).alias(self.target)
        )


@dataclass
class Standardizer:
    cols: Sequence[str]
    means: dict[str, float] | None = None
    stds: dict[str, float] | None = None
    batch_size: int | None = 200_000

    def fit(self, data: Any) -> "Standardizer":
        if hasattr(data, "batches") and hasattr(data, "features"):
            return self._fit_source(data)
        return self._fit_lazy(data)

    def _fit_lazy(self, lf: pl.LazyFrame) -> "Standardizer":
        exprs = []
        for col in self.cols:
            exprs += [pl.col(col).mean().alias(f"{col}__mean"), pl.col(col).std().alias(f"{col}__std")]
        row = lf.select(exprs).collect(engine="streaming").to_dicts()[0]
        self.means = {col: float(row[f"{col}__mean"] or 0.0) for col in self.cols}
        self.stds = {col: float(row[f"{col}__std"] or 1.0) for col in self.cols}
        self.stds = {col: std if std != 0.0 else 1.0 for col, std in self.stds.items()}
        return self

    def _fit_source(self, src: Any) -> "Standardizer":
        idx = [src.features.index(col) for col in self.cols]
        n = np.zeros(len(self.cols), dtype=np.int64)
        means = np.zeros(len(self.cols), dtype=np.float64)
        m2 = np.zeros(len(self.cols), dtype=np.float64)

        for x, _, _ in src.batches(self.batch_size):
            # Column indexing makes a private copy, so centering in place
            # keeps the working set bounded without changing the source.
            vals = np.asarray(x[:, idx], dtype=np.float64)
            if vals.size == 0:
                continue
            valid = ~np.isnan(vals)
            batch_n = valid.sum(axis=0)
            batch_mean = np.divide(
                vals.sum(axis=0, where=valid), batch_n,
                out=np.zeros_like(means), where=batch_n > 0,
            )
            vals -= batch_mean
            np.square(vals, out=vals)
            batch_m2 = vals.sum(axis=0, where=valid)

            # Merge centered batch statistics (Chan's formula), avoiding
            # cancellation from subtracting sum(x)**2 / n from sum(x**2).
            total = n + batch_n
            ratio = np.divide(
                batch_n, total, out=np.zeros_like(means), where=total > 0
            )
            delta = batch_mean - means
            m2 += batch_m2 + delta * delta * n * ratio
            means += delta * ratio
            n = total

        variances = np.divide(
            np.maximum(m2, 0.0),
            n - 1,
            out=np.zeros_like(m2),
            where=n > 1,
        )
        stds = np.sqrt(variances)

        self.means = {col: float(mean) for col, mean in zip(self.cols, means)}
        self.stds = {
            col: float(std) if std != 0.0 else 1.0
            for col, std in zip(self.cols, stds)
        }
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.means is None or self.stds is None:
            raise RuntimeError("Standardizer must be fit before transform")
        return lf.with_columns(
            [((pl.col(col) - self.means[col]) / self.stds[col]).alias(col) for col in self.cols]
        )


@dataclass
class Chain:
    steps: list[Transform]

    def fit(self, data: Any) -> "Chain":
        fitted = []
        cur = data
        for step in self.steps:
            step = step.fit(cur)
            fitted.append(step)
            if hasattr(cur, "with_transform"):
                cur = cur.with_transform(step)
            else:
                cur = step.transform(cur)
        self.steps = fitted
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for step in self.steps:
            lf = step.transform(lf)
        return lf


def save_transform(
    transform: Transform,
    path: str | Path,
    name: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        _pickle_dump(transform, f)
    artifact_hash = _sha256_file(path)
    config = transform_to_config(transform)
    fingerprint = _fingerprint(config) if config is not None else artifact_hash
    return {
        "base_name": name or _transform_name(transform),
        "path": path.name,
        "format": "pickle",
        "sha256": artifact_hash,
        "fingerprint": fingerprint,
        "config": config,
    }


def load_transform(path: str | Path, meta: dict[str, Any] | None = None) -> Transform:
    path = Path(path)
    expected = (meta or {}).get("sha256")
    if expected is not None and _sha256_file(path) != expected:
        raise ValueError(f"transform hash mismatch: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def transform_to_config(transform: Transform) -> dict[str, Any] | None:
    if isinstance(transform, Passthrough):
        return {"type": "Passthrough"}
    if isinstance(transform, Standardizer):
        return {"type": "Standardizer", **_json_ready(asdict(transform))}
    if isinstance(transform, (Chain, ComposeTransform)):
        return {
            "type": type(transform).__name__,
            "steps": [transform_to_config(step) for step in transform.steps],
        }
    if isinstance(transform, FunctionTransform):
        fn = transform.fn
        return {
            "type": "FunctionTransform",
            "fn": _callable_info(fn),
        }
    if is_dataclass(transform):
        return {"type": type(transform).__name__, **_json_ready(asdict(transform))}
    return None


def _callable_info(fn: Callable[..., Any]) -> dict[str, Any]:
    info = {
        "name": getattr(fn, "__name__", None),
        "module": getattr(fn, "__module__", None),
        "qualname": getattr(fn, "__qualname__", None),
        "description": inspect.getdoc(fn),
    }
    try:
        info["source"] = inspect.getsource(fn)
    except (OSError, TypeError):
        info["source"] = None
    return info


def _transform_name(transform: Transform) -> str:
    if isinstance(transform, FunctionTransform):
        return _snake(getattr(transform.fn, "__name__", "function_transform"))
    return _snake(type(transform).__name__)


def _fingerprint(config: dict[str, Any]) -> str:
    blob = json.dumps(_json_ready(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pickle_dump(obj: Any, f: Any) -> None:
    try:
        import cloudpickle
    except ImportError:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        cloudpickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _snake(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).strip("_").lower()
