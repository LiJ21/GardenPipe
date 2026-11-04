from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any, Protocol

import numpy as np


class Dist(Protocol):
    def suggest(self, trial: Any, name: str) -> Any: ...

    def grid(self) -> list[Any] | None: ...


@dataclass(frozen=True)
class Uniform:
    lo: float
    hi: float

    def suggest(self, trial: Any, name: str) -> float:
        return trial.suggest_float(name, self.lo, self.hi)

    def grid(self) -> None:
        return None


@dataclass(frozen=True)
class LogUniform:
    lo: float
    hi: float

    def suggest(self, trial: Any, name: str) -> float:
        return trial.suggest_float(name, self.lo, self.hi, log=True)

    def grid(self) -> None:
        return None


@dataclass(frozen=True)
class Int:
    lo: int
    hi: int
    step: int = 1

    def suggest(self, trial: Any, name: str) -> int:
        return trial.suggest_int(name, self.lo, self.hi, step=self.step)

    def grid(self) -> list[int]:
        return list(range(self.lo, self.hi + 1, self.step))


@dataclass(frozen=True)
class Categorical:
    values: Sequence[Any]

    def suggest(self, trial: Any, name: str) -> Any:
        return trial.suggest_categorical(name, list(self.values))

    def grid(self) -> list[Any]:
        return list(self.values)


@dataclass(frozen=True)
class Grid:
    values: Sequence[Any]

    def suggest(self, trial: Any, name: str) -> Any:
        return trial.suggest_categorical(name, list(self.values))

    def grid(self) -> list[Any]:
        return list(self.values)


SearchSpace = dict[str, Dist]
ParamFn = Callable[[Any], dict[str, Any]]


def uniform(lo: float, hi: float) -> Uniform:
    return Uniform(lo, hi)


def loguniform(lo: float, hi: float) -> LogUniform:
    return LogUniform(lo, hi)


def int_(lo: int, hi: int, step: int = 1) -> Int:
    return Int(lo, hi, step)


def categorical(values: Sequence[Any]) -> Categorical:
    return Categorical(values)


def grid(values: Sequence[Any]) -> Grid:
    return Grid(values)


def expanding_folds(rolling: list[list[str]]) -> list[tuple[list[str], list[str]]]:
    if len(rolling) < 2:
        raise ValueError("rolling_dates must contain at least two chunks")
    return [(list(chain.from_iterable(rolling[: i + 1])), rolling[i + 1]) for i in range(len(rolling) - 1)]


def suggest_params(search_space: SearchSpace | ParamFn | None, trial: Any) -> dict[str, Any]:
    if search_space is None:
        return {}
    if callable(search_space):
        return dict(search_space(trial))
    return {name: dist.suggest(trial, name) for name, dist in search_space.items()}


@dataclass(frozen=True)
class BySizeRecency:
    by_size: bool = True
    halflife: float | None = 3.0

    def __call__(self, val_sizes: Sequence[int]) -> np.ndarray:
        n = len(val_sizes)
        if n == 0:
            return np.array([], dtype=float)
        weights = np.ones(n, dtype=float)
        if self.by_size:
            weights *= np.asarray(val_sizes, dtype=float)
        if self.halflife is not None:
            age = np.arange(n - 1, -1, -1, dtype=float)
            weights *= 0.5 ** (age / self.halflife)
        total = weights.sum()
        return weights / total if total else np.ones(n, dtype=float) / n


def weighted_mean(scores: Sequence[float], sizes: Sequence[int], weighting: Callable[[Sequence[int]], np.ndarray]) -> float:
    weights = weighting(sizes)
    return float(np.dot(np.asarray(scores, dtype=float), weights))


def create_study(
    sampler: str,
    search_space: SearchSpace | ParamFn | None,
    direction: str = "minimize",
    pruner: Any = None,
    seed: int | None = None,
):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Pipeline.train() requires optuna. Install project dependencies first.") from exc

    sampler_obj = _make_sampler(optuna, sampler, search_space, seed)
    pruner_obj = optuna.pruners.NopPruner() if pruner is None else pruner
    return optuna.create_study(
        direction=direction, sampler=sampler_obj, pruner=pruner_obj
    )


def _make_sampler(optuna: Any, sampler: str, search_space: SearchSpace | ParamFn | None, seed: int | None):
    sampler = sampler.lower()
    if sampler == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler == "grid":
        if callable(search_space):
            raise ValueError("grid sampler requires a declarative search_space")
        grid_space = {name: values for name, dist in (search_space or {}).items() if (values := dist.grid()) is not None}
        if search_space and len(grid_space) != len(search_space):
            missing = sorted(set(search_space) - set(grid_space))
            raise ValueError(f"grid sampler needs finite values for: {missing}")
        return optuna.samplers.GridSampler(grid_space)
    raise ValueError("sampler must be one of: tpe, random, grid")
