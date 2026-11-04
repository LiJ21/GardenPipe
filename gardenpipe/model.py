from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np

from gardenpipe.data import as_batch, validate_sample_weight

if TYPE_CHECKING:
    from gardenpipe.data import DataSource
    from gardenpipe.track import Tracker


@runtime_checkable
class ModelAdapter(Protocol):
    streaming: bool
    supports_sample_weight: bool

    def build(self, params: dict[str, Any]) -> Any: ...

    def fit(
        self,
        model: Any,
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> Any: ...

    def predict(self, model: Any, x: np.ndarray) -> np.ndarray: ...

    def evaluate(
        self,
        model: Any,
        src: "DataSource",
        score: Callable[..., float],
        fold: Any = None,
        keep_predictions: bool = False,
    ) -> tuple[float, dict[str, Any], np.ndarray | None]: ...

    def save_model(
        self, model: Any, path: str | Path, filename: str | None = None
    ) -> dict[str, Any]: ...

    def load_model(
        self, path: str | Path, meta: dict[str, Any] | None = None
    ) -> Any: ...


class BaseAdapter:
    supports_sample_weight: ClassVar[bool] = False

    def save_model(
        self, model: Any, path: str | Path, filename: str | None = None
    ) -> dict[str, Any]:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        artifact = path / f"{filename or 'model'}.pkl"
        with artifact.open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            "format": "pickle",
            "artifact": artifact.name,
            "sha256": _sha256_file(artifact),
        }

    def load_model(self, path: str | Path, meta: dict[str, Any] | None = None) -> Any:
        path = Path(path)
        artifact = (
            path / (meta or {}).get("artifact", "model.pkl") if path.is_dir() else path
        )
        _check_hash(artifact, meta)
        with artifact.open("rb") as f:
            return pickle.load(f)

    def evaluate(
        self,
        model: Any,
        src: "DataSource",
        score: Callable[..., float],
        fold: Any = None,
        keep_predictions: bool = False,
    ) -> tuple[float, dict[str, Any], np.ndarray | None]:
        from gardenpipe.pipeline import evaluate_model

        return evaluate_model(self, model, src, score, fold, keep_predictions)


@dataclass
class DummyAdapter(BaseAdapter):
    mode: str = "mean"
    streaming: bool = False
    supports_sample_weight: ClassVar[bool] = True

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"params": dict(params)}

    def fit(
        self,
        model: dict[str, Any],
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        batch = as_batch(train.materialize())
        x, y, weight = batch.x, batch.y, batch.weight
        if self.mode == "linear" and x.size:
            design = np.c_[np.ones(len(x)), x]
            if weight is not None:
                root_weight = np.sqrt(np.asarray(weight, dtype=float))
                design = design * root_weight[:, None]
                y = np.asarray(y) * root_weight
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            model.update({"intercept": coef[0], "coef": coef[1:]})
        else:
            model["mean"] = (
                float(np.average(y, weights=weight))
                if len(y) and weight is not None
                else float(np.mean(y)) if len(y) else 0.0
            )
        return model

    def predict(self, model: dict[str, Any], x: np.ndarray) -> np.ndarray:
        if "coef" in model:
            return np.asarray(model["intercept"] + x @ model["coef"])
        return np.full(len(x), model.get("mean", 0.0), dtype=float)


@dataclass
class XGBoostAdapter(BaseAdapter):
    num_boost_round: int = 100
    early_stopping_rounds: int | None = None
    batch_size: int | None = 200_000
    streaming: bool = True
    supports_sample_weight: ClassVar[bool] = True
    external_memory: bool = False
    cache_dir: str | Path = "/tmp/xgb_extmem"
    cache_prefix: str = "xgb"
    release_data: bool = True
    xgb_dtype: Any | None = np.float32
    callbacks: list[Any] = field(default_factory=list)
    pruning_metric: str | None = None
    quantiles: Sequence[float] | None = None
    # "cv_median": a refit without validation data trains the median of the
    # searched trial's early-stopped CV round counts; "fixed": num_boost_round.
    refit_rounds: str = "cv_median"
    fit_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    last_fit_history: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.refit_rounds not in {"cv_median", "fixed"}:
            raise ValueError("refit_rounds must be 'cv_median' or 'fixed'")

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        params = dict(params)
        if self.quantiles is not None:
            params["objective"] = "reg:quantileerror"
            params["quantile_alpha"] = np.asarray(self.quantiles, dtype=float)
        return params

    def fit(
        self,
        model: dict[str, Any],
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> Any:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("XGBoostAdapter requires xgboost.") from exc

        if self.fit_history and fit_context is None:
            warnings.warn(
                "XGBoostAdapter.fit() is being called again; a new model will be "
                "trained and last_fit_history will be replaced.",
                RuntimeWarning,
                stacklevel=2,
            )
        linear = model.get("booster", "gbtree") == "gblinear"
        if linear:
            warnings.warn(
                "gblinear returns the final iteration's coefficients. XGBoost "
                "early stopping still applies when configured with validation "
                "data, but best-iteration slicing is unavailable for gblinear.",
                RuntimeWarning,
                stacklevel=2,
            )
        history: dict[str, Any] = {}
        with self._cache_scope() as cache:
            dtrain: Any | None = None
            evals: list[tuple[Any, str]] = []
            has_val = val is not None
            try:
                dtrain = self._dmatrix(xgb, train, cache_prefix=cache.prefix("train"))
                evals.append((dtrain, "train"))
                if has_val:
                    evals.append(
                        (
                            self._dmatrix(
                                xgb,
                                val,
                                ref=dtrain,
                                cache_prefix=cache.prefix("val"),
                            ),
                            "val",
                        )
                    )
                callbacks = list(self.callbacks)
                if trial is not None and self.pruning_metric:
                    try:
                        from optuna.integration import XGBoostPruningCallback

                        callbacks.append(
                            XGBoostPruningCallback(trial, self.pruning_metric)
                        )
                    except ImportError:
                        warnings.warn(
                            "XGBoost pruning requested, but optuna-integration is "
                            "not installed; continuing without pruning callback.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                booster = xgb.train(
                    model,
                    dtrain,
                    num_boost_round=self._num_rounds(has_val, fit_context),
                    evals=evals,
                    evals_result=history,
                    early_stopping_rounds=(
                        self.early_stopping_rounds if has_val else None
                    ),
                    callbacks=callbacks,
                    verbose_eval=False,
                )
            finally:
                evals.clear()
                dtrain = None
        for name, metrics in history.items():
            for metric, values in metrics.items():
                if values:
                    tracker.log({f"xgb/{name}_{metric}": values[-1]})
        record = self._fit_record(booster, fit_context, history)
        self.last_fit_history = record
        self.fit_history.append(record)
        if trial is not None:
            self._record_trial_fit(trial, record)
        # xgb.train returns the last round's model; keep only the best rounds.
        best = record["best_num_boost_round"]
        if (
            not linear
            and has_val
            and self.early_stopping_rounds is not None
            and best is not None
        ):
            if best < booster.num_boosted_rounds():
                booster = booster[:best]
        return booster

    def _num_rounds(
        self, has_val: bool, fit_context: dict[str, Any] | None
    ) -> int:
        ctx = fit_context or {}
        if (
            has_val
            or self.refit_rounds != "cv_median"
            or self.early_stopping_rounds is None
            or ctx.get("role") != "refit"
            or ctx.get("search_trial") is None
        ):
            return self.num_boost_round
        rounds = [
            item["best_num_boost_round"]
            for item in self.fit_history
            if item.get("role") == "cv"
            and item.get("trial") == ctx["search_trial"]
            and item.get("best_num_boost_round") is not None
        ]
        return int(round(float(np.median(rounds)))) if rounds else self.num_boost_round

    def predict(self, model: Any, x: np.ndarray) -> np.ndarray:
        config = json.loads(model.save_config())
        if config["learner"]["gradient_booster"]["name"] == "gblinear":
            import xgboost as xgb

            return np.asarray(model.predict(xgb.DMatrix(x)))
        return np.asarray(model.inplace_predict(x))

    def save_model(
        self, model: Any, path: str | Path, filename: str | None = None
    ) -> dict[str, Any]:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        artifact = path / f"{filename or 'model'}.ubj"
        model.save_model(str(artifact))
        return {
            "format": "xgboost_ubj",
            "artifact": artifact.name,
            "sha256": _sha256_file(artifact),
        }

    def load_model(self, path: str | Path, meta: dict[str, Any] | None = None) -> Any:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("XGBoostAdapter requires xgboost.") from exc

        path = Path(path)
        artifact = (
            path / (meta or {}).get("artifact", "model.ubj") if path.is_dir() else path
        )
        _check_hash(artifact, meta)
        booster = xgb.Booster()
        booster.load_model(str(artifact))
        return booster

    def _dmatrix(
        self,
        xgb: Any,
        src: "DataSource",
        ref: Any | None = None,
        cache_prefix: str | None = None,
    ) -> Any:
        if not self.streaming:
            if self.external_memory:
                raise ValueError(
                    "XGBoostAdapter external_memory=True requires streaming=True"
                )
            batch = as_batch(src.materialize())
            x, y = batch.x, batch.y
            if self.xgb_dtype is not None:
                x = np.asarray(x, dtype=self.xgb_dtype)
            kwargs = {"label": y}
            if batch.weight is not None:
                kwargs["weight"] = batch.weight
            return xgb.DMatrix(x, **kwargs)

        if getattr(src, "is_shuffled", False):
            raise ValueError(
                "XGBoost streaming re-iterates the data source per pass and a "
                "shuffled source yields a different stream each time; use an "
                "unshuffled buffer config for XGBoost roles"
            )
        if self.external_memory and not getattr(src, "is_deterministic", True):
            raise ValueError(
                "external-memory XGBoost caches pages across iterations and "
                "needs a deterministic stream; use workers=1 and no shuffle"
            )
        batch_size = self.batch_size
        external_memory = self.external_memory
        if external_memory and not hasattr(xgb, "ExtMemQuantileDMatrix"):
            raise ImportError(
                "XGBoostAdapter external_memory=True requires "
                "xgboost.ExtMemQuantileDMatrix."
            )
        if external_memory and cache_prefix is None:
            raise ValueError("external-memory XGBoost needs a cache_prefix")
        xgb_dtype = self.xgb_dtype
        release_data = self.release_data

        class BatchIter(xgb.DataIter):
            def __init__(self):
                kwargs = (
                    {"cache_prefix": cache_prefix, "release_data": release_data}
                    if external_memory
                    else {}
                )
                super().__init__(**kwargs)
                self._iter = None

            def reset(self):
                self._iter = iter(src.batches(batch_size))

            def next(self, input_data):
                try:
                    batch = as_batch(next(self._iter))
                except StopIteration:
                    return 0
                x, y = batch.x, batch.y
                if xgb_dtype is not None:
                    x = np.asarray(x, dtype=xgb_dtype)
                kwargs = {"data": x, "label": y}
                if batch.weight is not None:
                    kwargs["weight"] = batch.weight
                input_data(**kwargs)
                return 1

        if external_memory:
            return xgb.ExtMemQuantileDMatrix(BatchIter(), ref=ref)
        return xgb.QuantileDMatrix(BatchIter(), ref=ref)

    def _cache_scope(self) -> "_XGBoostCacheScope":
        if not self.external_memory:
            return _XGBoostCacheScope(None)
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return _XGBoostCacheScope(
            TemporaryDirectory(prefix=f"{self.cache_prefix}-", dir=cache_dir)
        )

    def _record_trial_fit(
        self,
        trial: Any,
        record: dict[str, Any],
    ) -> None:
        records = list(trial.user_attrs.get("xgb_fits", []))
        records.append(record)
        trial.set_user_attr("xgb_fits", records)

        cv_rounds = [
            int(item["best_num_boost_round"])
            for item in records
            if item.get("role") == "cv" and item.get("best_num_boost_round") is not None
        ]
        if cv_rounds:
            trial.set_user_attr("xgb_cv_best_num_boost_rounds", cv_rounds)
            trial.set_user_attr(
                "xgb_cv_best_num_boost_round_median",
                float(np.median(cv_rounds)),
            )
            trial.set_user_attr("xgb_cv_best_num_boost_round_max", max(cv_rounds))

    def _fit_record(
        self,
        booster: Any,
        fit_context: dict[str, Any] | None,
        history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        best_iteration = _as_optional_int(_safe_attr(booster, "best_iteration"))
        best_score = _as_optional_float(_safe_attr(booster, "best_score"))
        return _fit_record(
            fit_context,
            history=_xgb_history(history) if history is not None else None,
            best_iteration=best_iteration,
            best_score=best_score,
            num_boosted_rounds=_num_boosted_rounds(booster),
        )


def _safe_attr(obj: Any, name: str) -> Any | None:
    try:
        return getattr(obj, name)
    except (AttributeError, ValueError):
        return None


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num_boosted_rounds(booster: Any) -> int | None:
    try:
        return int(booster.num_boosted_rounds())
    except (AttributeError, TypeError, ValueError):
        return None


def _fit_record(
    fit_context: dict[str, Any] | None,
    *,
    history: dict[str, dict[str, list[float]]] | None = None,
    best_iteration: int | None = None,
    best_score: float | None = None,
    num_boosted_rounds: int | None = None,
) -> dict[str, Any]:
    ctx = fit_context or {}
    record: dict[str, Any] = {
        "role": ctx.get("role"),
        "trial": ctx.get("trial"),
        "fold": ctx.get("fold"),
        "n_folds": ctx.get("n_folds"),
        "train_dates": ctx.get("train_dates"),
        "val_dates": ctx.get("val_dates"),
        "best_iteration": best_iteration,
        "best_num_boost_round": (
            best_iteration + 1 if best_iteration is not None else None
        ),
        "best_score": best_score,
        "num_boosted_rounds": num_boosted_rounds,
    }
    if history is not None:
        record["history"] = history
    return record


def _xgb_history(history: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    return {
        str(name): {
            str(metric): [float(value) for value in values]
            for metric, values in metrics.items()
        }
        for name, metrics in history.items()
    }


def _best_metric_index(metrics: dict[str, list[float]]) -> int | None:
    values = _first_metric_values(metrics)
    if values is None or not values:
        return None
    return int(np.nanargmin(np.asarray(values, dtype=float)))


def _best_metric_value(metrics: dict[str, list[float]]) -> float | None:
    idx = _best_metric_index(metrics)
    values = _first_metric_values(metrics)
    return float(values[idx]) if idx is not None and values is not None else None


def _first_metric_values(metrics: dict[str, list[float]]) -> list[float] | None:
    for values in metrics.values():
        return values
    return None


class _XGBoostCacheScope:
    def __init__(self, tempdir: TemporaryDirectory[str] | None):
        self._tempdir = tempdir
        self._path = Path(tempdir.name) if tempdir is not None else None

    def __enter__(self) -> "_XGBoostCacheScope":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()

    def prefix(self, name: str) -> str | None:
        if self._path is None:
            return None
        return str(self._path / name)


@dataclass
class LarsAdapter(BaseAdapter):
    alpha: float = 1.0
    method: str = "lasso"
    fit_intercept: bool = True
    max_iter: int = 500
    alpha_min: float = 0.0
    eps: float = float(np.finfo(float).eps)
    positive: bool = False
    batch_size: int | None = 200_000
    max_features: int | None = None
    stats_backend: str = "cpu"
    stats_dtype: Any = np.float64
    streaming: bool = True
    supports_sample_weight: ClassVar[bool] = True
    cache_path: bool = True
    vectorized_path_eval: bool = True
    path_eval_alphas: Sequence[float] | None = None
    _stats_cache: dict[Any, "_LinearStats"] = field(
        default_factory=dict, init=False, repr=False
    )
    _path_cache: dict[Any, tuple[np.ndarray, list[int], np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    _score_cache: dict[Any, dict[float, tuple[float, dict[str, Any]]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def build(self, params: dict[str, Any]) -> "LarsPathModel":
        cfg = {
            "alpha": self.alpha,
            "method": self.method,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "alpha_min": self.alpha_min,
            "eps": self.eps,
            "positive": self.positive,
            **params,
        }
        known = {
            "alpha",
            "method",
            "fit_intercept",
            "max_iter",
            "alpha_min",
            "eps",
            "positive",
        }
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unsupported LarsAdapter params: {sorted(unknown)}")
        return LarsPathModel(
            alpha=float(cfg["alpha"]),
            method=str(cfg["method"]),
            fit_intercept=bool(cfg["fit_intercept"]),
            max_iter=int(cfg["max_iter"]),
            alpha_min=float(cfg["alpha_min"]),
            eps=float(cfg["eps"]),
            positive=bool(cfg["positive"]),
            params=dict(params),
        )

    def fit(
        self,
        model: "LarsPathModel",
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> "LarsPathModel":
        try:
            from sklearn.linear_model import lars_path_gram
        except ImportError as exc:
            raise ImportError("LarsAdapter requires scikit-learn.") from exc

        stats = self._stats(train)
        gram, xy, x_mean, y_mean = stats.centered(model.fit_intercept)
        key = self._path_key(train, model)
        cached = self._path_cache.get(key) if key is not None else None
        if cached is None:
            alphas, active, coefs_path = lars_path_gram(
                xy,
                gram,
                n_samples=stats.n,
                method=model.method,
                max_iter=model.max_iter,
                alpha_min=model.alpha_min,
                eps=model.eps,
                positive=model.positive,
                return_path=True,
            )
            cached = (
                np.asarray(alphas),
                [int(i) for i in active],
                np.asarray(coefs_path),
            )
            if key is not None:
                self._path_cache[key] = cached
        model.alphas, model.active, model.coefs_path = cached
        model.path_key = key
        model.n_samples = stats.n
        model.x_mean = x_mean
        model.y_mean = y_mean
        model.coef = self._coef_at(model.alphas, model.coefs_path, model.alpha)
        model.intercept = (
            float(y_mean - x_mean @ model.coef) if model.fit_intercept else 0.0
        )
        if tracker is not None:
            tracker.log(
                {
                    "lars/alpha": model.alpha,
                    "lars/path_len": len(model.alphas),
                    "lars/n": stats.n,
                    "lars/n_features": stats.n_features,
                }
            )
        return model

    def predict(self, model: "LarsPathModel", x: np.ndarray) -> np.ndarray:
        return model.predict(x)

    def evaluate(
        self,
        model: "LarsPathModel",
        src: "DataSource",
        score: Callable[..., float],
        fold: Any = None,
        keep_predictions: bool = False,
    ) -> tuple[float, dict[str, Any], np.ndarray | None]:
        alphas = self._eval_alphas()
        if (
            keep_predictions
            or not self.vectorized_path_eval
            or not alphas
            or model.alphas is None
            or model.coefs_path is None
            or float(model.alpha) not in set(alphas)
        ):
            return super().evaluate(model, src, score, fold, keep_predictions)

        key = self._score_key(model, src, score, alphas)
        cached = self._score_cache.get(key) if key is not None else None
        if cached is None:
            cached = self._score_path(model, src, score, alphas)
            if key is not None:
                self._score_cache[key] = cached

        loss, ctx = cached[float(model.alpha)]
        ctx = dict(ctx)
        ctx["fold"] = fold
        return loss, ctx, None

    def path(self, model: "LarsPathModel") -> dict[str, Any]:
        if model.alphas is None or model.coefs_path is None:
            raise RuntimeError("LARS path is not available before fit()")
        return {
            "alphas": model.alphas.copy(),
            "active": list(model.active),
            "coefs_path": model.coefs_path.copy(),
        }

    def _stats(self, src: "DataSource") -> "_LinearStats":
        key = self._stats_key(src)
        if key is not None and key in self._stats_cache:
            return self._stats_cache[key]

        stats: _LinearStats | None = None
        for value in src.batches(self.batch_size):
            batch = as_batch(value)
            x = np.asarray(batch.x)
            y = np.asarray(batch.y).reshape(-1)
            if len(y) == 0:
                continue
            if stats is None:
                if self.max_features is not None and x.shape[1] > self.max_features:
                    raise MemoryError(
                        f"LarsAdapter got {x.shape[1]} features; max_features={self.max_features}"
                    )
                stats = _LinearStats.empty(
                    x.shape[1], self._stats_xp(), self.stats_dtype
                )
            stats.add(x, y, batch.weight)

        if stats is None or stats.n == 0:
            raise ValueError("cannot fit LarsAdapter on empty data")
        if stats.weight_sum <= 0.0:
            raise ValueError("cannot fit LarsAdapter with zero total sample weight")
        if key is not None:
            self._stats_cache[key] = stats
        return stats

    def _stats_key(self, src: "DataSource") -> Any | None:
        key = getattr(src, "cache_key", None)
        if key is None:
            return None
        return (
            key,
            tuple(src.features),
            src.target,
            getattr(src, "sample_weight_col", None),
            self.stats_backend,
            str(np.dtype(self.stats_dtype)),
        )

    def _path_key(self, src: "DataSource", model: "LarsPathModel") -> Any | None:
        key = self._stats_key(src)
        if not self.cache_path or key is None:
            return None
        return (
            key,
            model.method,
            model.fit_intercept,
            model.max_iter,
            model.alpha_min,
            model.eps,
            model.positive,
        )

    def _eval_alphas(self) -> tuple[float, ...]:
        if self.path_eval_alphas is None:
            return ()
        return tuple(dict.fromkeys(float(alpha) for alpha in self.path_eval_alphas))

    def _score_key(
        self,
        model: "LarsPathModel",
        src: "DataSource",
        score: Callable[..., float],
        alphas: tuple[float, ...],
    ) -> Any | None:
        src_key = getattr(src, "cache_key", None)
        if src_key is None:
            return None
        path_key = model.path_key
        if path_key is None:
            path_key = id(model.coefs_path)
        return (
            path_key,
            src_key,
            tuple(src.features),
            src.target,
            id(score),
            getattr(score, "__name__", None),
            alphas,
        )

    def _score_path(
        self,
        model: "LarsPathModel",
        src: "DataSource",
        score: Callable[..., float],
        alphas: tuple[float, ...],
    ) -> dict[float, tuple[float, dict[str, Any]]]:
        from gardenpipe.pipeline import call_score, merge_ctx

        coefs = np.column_stack(
            [self._coef_at(model.alphas, model.coefs_path, alpha) for alpha in alphas]
        )
        intercepts = (
            float(model.y_mean) - np.asarray(model.x_mean, dtype=float) @ coefs
            if model.fit_intercept
            else np.zeros(len(alphas), dtype=float)
        )
        states: list[Any] = [None] * len(alphas)
        ctx: dict[str, Any] = {"n": 0}

        for value in src.batches(self.batch_size):
            batch = as_batch(value)
            x = np.asarray(batch.x, dtype=float)
            y_true = np.asarray(batch.y)
            preds = x @ coefs + intercepts
            if len(y_true) != preds.shape[0]:
                raise ValueError(
                    f"score length mismatch: y_true={len(y_true)}, y_pred={preds.shape[0]}"
                )
            for i in range(len(alphas)):
                states[i] = call_score(
                    score,
                    y_true,
                    preds[:, i],
                    dict(batch.ctx),
                    combine_with=states[i],
                    sample_weight=batch.weight,
                )
            merge_ctx(ctx, batch.ctx)

        if any(state is None for state in states):
            raise ValueError("cannot score empty prediction stream")
        return {
            alpha: (float(state), dict(ctx)) for alpha, state in zip(alphas, states)
        }

    def _stats_xp(self) -> Any:
        name = self.stats_backend.lower()
        if name in {"cpu", "numpy"}:
            return np
        if name in {"cupy", "gpu"}:
            try:
                import cupy as cp
            except ImportError as exc:
                raise ImportError(
                    "LarsAdapter stats_backend='cupy' requires CuPy; for CUDA 12 install cupy-cuda12x, not cupy."
                ) from exc
            return cp
        raise ValueError("stats_backend must be one of: 'cpu', 'numpy', 'cupy', 'gpu'")

    @staticmethod
    def _coef_at(
        alphas: np.ndarray, coefs_path: np.ndarray, alpha: float
    ) -> np.ndarray:
        if len(alphas) == 1:
            return coefs_path[:, 0].copy()
        if alpha < alphas[-1]:
            warnings.warn(
                f"alpha={alpha} is below computed path minimum {alphas[-1]}; using path endpoint.",
                RuntimeWarning,
                stacklevel=2,
            )
        xp = alphas[::-1]
        return np.array(
            [np.interp(alpha, xp, row[::-1]) for row in coefs_path], dtype=float
        )


@dataclass
class LarsPathModel:
    alpha: float
    method: str = "lasso"
    fit_intercept: bool = True
    max_iter: int = 500
    alpha_min: float = 0.0
    eps: float = float(np.finfo(float).eps)
    positive: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    coef: np.ndarray | None = None
    intercept: float = 0.0
    alphas: np.ndarray | None = None
    active: list[int] = field(default_factory=list)
    coefs_path: np.ndarray | None = None
    path_key: Any | None = None
    n_samples: int = 0
    x_mean: np.ndarray | None = None
    y_mean: float = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("LarsPathModel is not fitted")
        return np.asarray(x, dtype=float) @ self.coef + self.intercept


@dataclass
class RidgeAdapter(BaseAdapter):
    alpha: float = 1.0
    fit_intercept: bool = True
    batch_size: int | None = 200_000
    max_features: int | None = None
    stats_backend: str = "cpu"
    stats_dtype: Any = np.float64
    streaming: bool = True
    supports_sample_weight: ClassVar[bool] = True
    cache_stats: bool = True
    _stats_cache: dict[Any, "_LinearStats"] = field(
        default_factory=dict, init=False, repr=False
    )

    def build(self, params: dict[str, Any]) -> "RidgeModel":
        cfg = {"alpha": self.alpha, "fit_intercept": self.fit_intercept, **params}
        unknown = set(cfg) - {"alpha", "fit_intercept"}
        if unknown:
            raise ValueError(f"unsupported RidgeAdapter params: {sorted(unknown)}")
        return RidgeModel(
            alpha=float(cfg["alpha"]),
            fit_intercept=bool(cfg["fit_intercept"]),
            params=dict(params),
        )

    def fit(
        self,
        model: "RidgeModel",
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> "RidgeModel":
        stats = self._stats(train)
        gram, xy, x_mean, y_mean = stats.centered(model.fit_intercept)
        reg = float(model.alpha) * np.eye(stats.n_features, dtype=gram.dtype)
        model.coef = _solve_linear(gram + reg, xy)
        model.intercept = (
            float(y_mean - x_mean @ model.coef) if model.fit_intercept else 0.0
        )
        model.n_samples = stats.n
        model.x_mean = x_mean
        model.y_mean = y_mean
        if tracker is not None:
            tracker.log(
                {
                    "ridge/alpha": model.alpha,
                    "ridge/n": stats.n,
                    "ridge/n_features": stats.n_features,
                }
            )
        return model

    def predict(self, model: "RidgeModel", x: np.ndarray) -> np.ndarray:
        return model.predict(x)

    def _stats(self, src: "DataSource") -> "_LinearStats":
        key = self._stats_key(src)
        if key is not None and key in self._stats_cache:
            return self._stats_cache[key]

        stats: _LinearStats | None = None
        for value in src.batches(self.batch_size):
            batch = as_batch(value)
            x = np.asarray(batch.x)
            y = np.asarray(batch.y).reshape(-1)
            if len(y) == 0:
                continue
            if stats is None:
                if self.max_features is not None and x.shape[1] > self.max_features:
                    raise MemoryError(
                        f"RidgeAdapter got {x.shape[1]} features; max_features={self.max_features}"
                    )
                stats = _LinearStats.empty(
                    x.shape[1], self._stats_xp(), self.stats_dtype
                )
            stats.add(x, y, batch.weight)

        if stats is None or stats.n == 0:
            raise ValueError("cannot fit RidgeAdapter on empty data")
        if stats.weight_sum <= 0.0:
            raise ValueError("cannot fit RidgeAdapter with zero total sample weight")
        if key is not None:
            self._stats_cache[key] = stats
        return stats

    def _stats_key(self, src: "DataSource") -> Any | None:
        if not self.cache_stats:
            return None
        key = getattr(src, "cache_key", None)
        if key is None:
            return None
        return (
            key,
            tuple(src.features),
            src.target,
            getattr(src, "sample_weight_col", None),
            self.stats_backend,
            str(np.dtype(self.stats_dtype)),
        )

    def _stats_xp(self) -> Any:
        name = self.stats_backend.lower()
        if name in {"cpu", "numpy"}:
            return np
        if name in {"cupy", "gpu"}:
            try:
                import cupy as cp
            except ImportError as exc:
                raise ImportError(
                    "RidgeAdapter stats_backend='cupy' requires CuPy; for CUDA 12 install cupy-cuda12x, not cupy."
                ) from exc
            return cp
        raise ValueError("stats_backend must be one of: 'cpu', 'numpy', 'cupy', 'gpu'")


@dataclass
class RidgeModel:
    alpha: float
    fit_intercept: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    coef: np.ndarray | None = None
    intercept: float = 0.0
    n_samples: int = 0
    x_mean: np.ndarray | None = None
    y_mean: float = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("RidgeModel is not fitted")
        return np.asarray(x, dtype=float) @ self.coef + self.intercept


def _solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


@dataclass
class _LinearStats:
    xp: Any
    dtype: Any
    xtx: Any
    xty: Any
    x_sum: Any
    y_sum: float = 0.0
    weight_sum: float = 0.0
    n: int = 0

    @classmethod
    def empty(
        cls, n_features: int, xp: Any = np, dtype: Any = np.float64
    ) -> "_LinearStats":
        dtype = xp.dtype(dtype)
        return cls(
            xp=xp,
            dtype=dtype,
            xtx=xp.zeros((n_features, n_features), dtype=dtype),
            xty=xp.zeros(n_features, dtype=dtype),
            x_sum=xp.zeros(n_features, dtype=dtype),
        )

    @property
    def n_features(self) -> int:
        return int(self.x_sum.size)

    def add(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        if x.shape[0] != len(y):
            raise ValueError(f"batch length mismatch: X={x.shape[0]} y={len(y)}")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"feature count changed: expected {self.n_features}, got {x.shape[1]}"
            )
        x = self.xp.asarray(x, dtype=self.dtype)
        y = self.xp.asarray(y, dtype=self.dtype).reshape(-1)
        if sample_weight is None:
            self.xtx += x.T @ x
            self.xty += x.T @ y
            self.x_sum += x.sum(axis=0)
            self.y_sum += float(self._cpu(y.sum()).item())
            self.weight_sum += len(y)
        else:
            validate_sample_weight(sample_weight, len(y))
            weight = self.xp.asarray(sample_weight, dtype=self.dtype).reshape(-1)
            weighted_x = x * weight[:, None]
            weighted_y = y * weight
            self.xtx += x.T @ weighted_x
            self.xty += x.T @ weighted_y
            self.x_sum += weighted_x.sum(axis=0)
            self.y_sum += float(self._cpu(weighted_y.sum()).item())
            self.weight_sum += float(self._cpu(weight.sum()).item())
        self.n += len(y)

    def centered(
        self, fit_intercept: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if not fit_intercept:
            return (
                self._cpu(self.xtx),
                self._cpu(self.xty),
                np.zeros(self.n_features),
                0.0,
            )
        if self.weight_sum <= 0.0:
            raise ValueError("cannot center data with zero total sample weight")
        x_mean = self.x_sum / self.weight_sum
        y_mean = self.y_sum / self.weight_sum
        gram = self.xtx - self.xp.outer(self.x_sum, self.x_sum) / self.weight_sum
        xy = self.xty - self.x_sum * y_mean
        return self._cpu(gram), self._cpu(xy), self._cpu(x_mean), float(y_mean)

    def _cpu(self, x: Any) -> np.ndarray:
        if self.xp is np:
            return np.asarray(x).copy()
        return self.xp.asnumpy(x)


@dataclass
class _TorchSnapshot:
    epoch: int
    score: float
    state_dict: dict[str, Any]


@dataclass
class TorchAdapter(BaseAdapter):
    module_builder: Callable[[dict[str, Any]], Any]
    loss_fn: Any | None = None
    optimizer_builder: Callable[[Any, dict[str, Any]], Any] | None = None
    epochs: int = 1
    batch_size: int | None = 8192
    device: str | None = None
    distributed: bool = False
    streaming: bool = True
    supports_sample_weight: ClassVar[bool] = True
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    restore_best: bool = True
    snapshot_mode: str = "off"
    snapshot_k: int = 1
    snapshot_monitor: str = "val_loss"
    snapshot_direction: str | None = None
    snapshot_start_epoch: int = 0
    snapshot_interval: int = 1
    fit_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    last_fit_history: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )

    def build(self, params: dict[str, Any]) -> Any:
        model = self.module_builder(params)
        setattr(model, "_pipeline_params", dict(params))
        return model

    def fit(
        self,
        model: Any,
        train: "DataSource",
        val: "DataSource | None",
        tracker: "Tracker",
        trial: Any | None = None,
        fit_context: dict[str, Any] | None = None,
    ) -> Any:
        import torch

        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        rank, world_size = self._rank_world(torch)
        model_params = getattr(model, "_pipeline_params", {})
        model = model.to(device)
        if self.distributed and world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model)
        loss_fn = self.loss_fn or torch.nn.MSELoss(reduction="none")
        optimizer = self._optimizer(torch, model, model_params)
        history: dict[str, dict[str, list[float]]] = {"train": {"loss": []}}
        if val is not None:
            history["val"] = {"loss": []}
        self._check_snapshot_config()
        monitor = self.snapshot_monitor.lower()
        direction = self._monitor_direction(monitor, fit_context)
        val_score = (fit_context or {}).get("val_score")
        if val is not None and monitor == "val_score":
            if val_score is None:
                raise ValueError(
                    "snapshot_monitor='val_score' requires fit_context['val_score']"
                )
            history.setdefault("val", {}).setdefault("score", [])
        monitor_enabled = self._monitor_enabled(val)
        if (
            monitor_enabled
            and val is None
            and monitor in {"val_loss", "val_score"}
        ):
            warnings.warn(
                "TorchAdapter early stopping/snapshots need validation data; "
                "continuing for the configured number of epochs.",
                RuntimeWarning,
                stacklevel=2,
            )
            monitor_enabled = False
        best_epoch: int | None = None
        best_score: float | None = None
        best_state: dict[str, Any] | None = None
        stopped_epoch: int | None = None
        stale_epochs = 0
        snapshots: list[_TorchSnapshot] = []

        for epoch in range(self.epochs):
            print(f"======== Torch Adapter -- Epoch {epoch}")
            model.train()
            loss_sum = loss_mass = 0.0
            for xb, yb, wb in self._loader(torch, train, rank, world_size):
                xb, yb = xb.to(device), yb.to(device)
                wb = wb.to(device) if wb is not None else None
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb).squeeze(-1)
                loss = self._reduce_loss(loss_fn, pred, yb.float(), wb)
                loss.backward()
                optimizer.step()
                mass = float(wb.sum().detach().cpu()) if wb is not None else len(yb)
                loss_sum += float(loss.detach().cpu()) * mass
                loss_mass += mass
            metrics = {"torch/train_loss": loss_sum / loss_mass if loss_mass else 0.0}
            history["train"]["loss"].append(metrics["torch/train_loss"])
            if val is not None:
                metrics["torch/val_loss"] = self._eval_loss(
                    torch, model, val, loss_fn, device
                )
                history["val"]["loss"].append(metrics["torch/val_loss"])
                if monitor == "val_score" and val_score is not None:
                    score_value, _ = self._eval_score(
                        torch,
                        model,
                        val,
                        val_score,
                        device,
                        fold=(fit_context or {}).get("fold"),
                    )
                    metrics["torch/val_score"] = score_value
                    history["val"]["score"].append(score_value)
            print(
                f"======== Torch Adapter -- train loss = {metrics['torch/train_loss']}"
                + (
                    f", val loss = {metrics['torch/val_loss']}"
                    if val is not None
                    else ""
                )
                + (
                    f", val score = {metrics['torch/val_score']}"
                    if "torch/val_score" in metrics
                    else ""
                )
            )
            tracker.log(self._log_metrics(metrics, fit_context), step=epoch)
            monitor_value = self._monitor_value(metrics, monitor)
            if monitor_enabled and monitor_value is not None:
                improved = self._is_better(
                    monitor_value,
                    best_score,
                    direction,
                    self.early_stopping_min_delta,
                )
                if improved:
                    best_epoch = epoch
                    best_score = monitor_value
                    best_state = self._cpu_state_dict(model)
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                self._maybe_snapshot(
                    snapshots,
                    model,
                    epoch,
                    monitor_value,
                    direction,
                )
                if (
                    self.early_stopping_patience is not None
                    and not improved
                    and stale_epochs >= self.early_stopping_patience
                ):
                    stopped_epoch = epoch
                    print(
                        "======== Torch Adapter -- early stopping at "
                        f"epoch {epoch}; best epoch = {best_epoch}"
                    )
                    break
        if self.restore_best and best_state is not None:
            self._module(model).load_state_dict(best_state)
        self._attach_snapshots(model, snapshots)
        record = _fit_record(
            fit_context,
            history=history,
            best_iteration=best_epoch,
            best_score=best_score,
            num_boosted_rounds=len(history["train"]["loss"]),
        )
        record.update(
            {
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "snapshot_mode": self.snapshot_mode,
                "snapshot_epochs": [item.epoch for item in snapshots],
                "snapshot_scores": [item.score for item in snapshots],
                "monitor": monitor,
                "monitor_direction": direction,
                "restored_best": bool(self.restore_best and best_state is not None),
            }
        )
        self.last_fit_history = record
        self.fit_history.append(record)
        if trial is not None:
            records = list(trial.user_attrs.get("torch_fits", []))
            records.append(record)
            trial.set_user_attr("torch_fits", records)
        return model

    def predict(self, model: Any, x: np.ndarray) -> np.ndarray:
        import torch

        device = next(model.parameters()).device
        model.eval()
        with torch.inference_mode():
            x = _writable_array(x, dtype=np.float32)
            xb = torch.as_tensor(x, dtype=torch.float32, device=device)
            module = self._module(model)
            snapshot_states = getattr(module, "_pipeline_snapshot_state_dicts", None)
            mode = getattr(module, "_pipeline_snapshot_mode", "off")
            if mode == "ensemble" and snapshot_states:
                current = self._cpu_state_dict(model)
                parts = []
                try:
                    for state in snapshot_states:
                        module.load_state_dict(state)
                        parts.append(self._predict_array(model, xb))
                finally:
                    module.load_state_dict(current)
                return np.mean(np.stack(parts, axis=0), axis=0)
            return self._predict_array(model, xb)

    def save_model(
        self, model: Any, path: str | Path, filename: str | None = None
    ) -> dict[str, Any]:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        artifact = path / f"{filename or 'model'}.pt"
        module = model.module if hasattr(model, "module") else model
        torch.save(
            {
                "params": getattr(module, "_pipeline_params", {}),
                "state_dict": module.state_dict(),
                "snapshot_mode": getattr(module, "_pipeline_snapshot_mode", "off"),
                "snapshot_epochs": getattr(module, "_pipeline_snapshot_epochs", []),
                "snapshot_scores": getattr(module, "_pipeline_snapshot_scores", []),
                "snapshot_state_dicts": getattr(
                    module, "_pipeline_snapshot_state_dicts", []
                ),
            },
            artifact,
        )
        return {
            "format": "torch_state_dict",
            "artifact": artifact.name,
            "sha256": _sha256_file(artifact),
        }

    def load_model(self, path: str | Path, meta: dict[str, Any] | None = None) -> Any:
        import torch

        path = Path(path)
        artifact = (
            path / (meta or {}).get("artifact", "model.pt") if path.is_dir() else path
        )
        _check_hash(artifact, meta)
        payload = torch.load(artifact, map_location=self.device or "cpu")
        model = self.build(payload.get("params", {}))
        model.load_state_dict(payload["state_dict"])
        if payload.get("snapshot_state_dicts"):
            setattr(
                model,
                "_pipeline_snapshot_mode",
                payload.get("snapshot_mode", "off"),
            )
            setattr(
                model,
                "_pipeline_snapshot_epochs",
                payload.get("snapshot_epochs", []),
            )
            setattr(
                model,
                "_pipeline_snapshot_scores",
                payload.get("snapshot_scores", []),
            )
            setattr(
                model,
                "_pipeline_snapshot_state_dicts",
                payload.get("snapshot_state_dicts", []),
            )
        if self.device is not None:
            model = model.to(self.device)
        return model

    def _loader(self, torch: Any, src: "DataSource", rank: int, world_size: int):
        if world_size > 1 and getattr(src, "is_shuffled", False):
            raise ValueError(
                "shuffled data sources cannot be sharded by batch index "
                "across ranks; disable shuffle or train on a single process"
            )
        batch_size = self.batch_size

        class SourceDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for i, value in enumerate(src.batches(batch_size)):
                    if i % world_size == rank:
                        batch = as_batch(value)
                        x = _writable_array(batch.x, dtype=np.float32)
                        y = _writable_array(batch.y)
                        weight = (
                            _writable_array(batch.weight, dtype=np.float32)
                            if batch.weight is not None
                            else None
                        )
                        yield (
                            torch.as_tensor(x, dtype=torch.float32),
                            torch.as_tensor(y),
                            torch.as_tensor(weight, dtype=torch.float32)
                            if weight is not None
                            else None,
                        )

        return torch.utils.data.DataLoader(SourceDataset(), batch_size=None)

    def _eval_loss(
        self, torch: Any, model: Any, src: "DataSource", loss_fn: Any, device: Any
    ) -> float:
        loss_sum = loss_mass = 0.0
        model.eval()
        with torch.inference_mode():
            for xb, yb, wb in self._loader(torch, src, 0, 1):
                pred = model(xb.to(device)).squeeze(-1)
                wb = wb.to(device) if wb is not None else None
                loss = self._reduce_loss(
                    loss_fn,
                    pred,
                    yb.to(device).float(),
                    wb,
                )
                mass = float(wb.sum().cpu()) if wb is not None else len(yb)
                loss_sum += float(loss.cpu()) * mass
                loss_mass += mass
        return loss_sum / loss_mass if loss_mass else 0.0

    @staticmethod
    def _reduce_loss(
        loss_fn: Any,
        pred: Any,
        target: Any,
        sample_weight: Any | None,
    ) -> Any:
        loss = loss_fn(pred, target)
        if sample_weight is None:
            return loss.mean() if loss.ndim else loss
        if loss.ndim == 0:
            raise ValueError(
                "sample weights require loss_fn to return per-sample losses; "
                "configure it with reduction='none'"
            )
        if loss.shape[0] != sample_weight.shape[0]:
            raise ValueError(
                "loss and sample weight batch lengths differ: "
                f"{loss.shape[0]} != {sample_weight.shape[0]}"
            )
        if loss.ndim > 1:
            loss = loss.reshape(loss.shape[0], -1).mean(dim=1)
        weight_sum = sample_weight.sum()
        if float(weight_sum.detach().cpu()) <= 0.0:
            return (loss * sample_weight).sum()
        return (loss * sample_weight).sum() / weight_sum

    def _eval_score(
        self,
        torch: Any,
        model: Any,
        src: "DataSource",
        score: Callable[..., float],
        device: Any,
        fold: Any = None,
    ) -> tuple[float, dict[str, Any]]:
        from gardenpipe.pipeline import call_score, merge_ctx

        state: Any = None
        ctx: dict[str, Any] = {"n": 0}
        model.eval()
        with torch.inference_mode():
            for value in src.batches(self.batch_size):
                batch = as_batch(value)
                x = _writable_array(batch.x, dtype=np.float32)
                xb = torch.as_tensor(x, dtype=torch.float32, device=device)
                y_pred = self._predict_array(model, xb)
                state = call_score(
                    score,
                    batch.y,
                    y_pred,
                    dict(batch.ctx),
                    combine_with=state,
                    sample_weight=batch.weight,
                )
                merge_ctx(ctx, batch.ctx)
        if state is None:
            raise ValueError("cannot score empty prediction stream")
        ctx["fold"] = fold
        return float(state), ctx

    def _optimizer(self, torch: Any, model: Any, params: dict[str, Any]) -> Any:
        if self.optimizer_builder is not None:
            return self.optimizer_builder(model.parameters(), params)
        return torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-3)))

    def _rank_world(self, torch: Any) -> tuple[int, int]:
        if (
            self.distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _check_snapshot_config(self) -> None:
        mode = self.snapshot_mode.lower()
        if mode not in {"off", "best", "top_k", "ensemble"}:
            raise ValueError("snapshot_mode must be one of: off, best, top_k, ensemble")
        self.snapshot_mode = mode
        self.snapshot_monitor = self.snapshot_monitor.lower()
        if self.snapshot_monitor not in {"train_loss", "val_loss", "val_score"}:
            raise ValueError(
                "snapshot_monitor must be one of: train_loss, val_loss, val_score"
            )
        if self.snapshot_k < 1:
            raise ValueError("snapshot_k must be positive")
        if self.snapshot_start_epoch < 0:
            raise ValueError("snapshot_start_epoch must be nonnegative")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 0
        ):
            raise ValueError("early_stopping_patience must be nonnegative")

    def _monitor_enabled(self, val: "DataSource | None") -> bool:
        return self.early_stopping_patience is not None or self.snapshot_mode != "off"

    def _log_metrics(
        self, metrics: dict[str, float], fit_context: dict[str, Any] | None
    ) -> dict[str, float]:
        prefix = self._metric_prefix(fit_context)
        if prefix is None:
            return metrics
        out: dict[str, float] = {}
        for key, value in metrics.items():
            name = key.removeprefix("torch/")
            out[f"{prefix}/{name}"] = value
        return out

    def _metric_prefix(self, fit_context: dict[str, Any] | None) -> str | None:
        if not fit_context:
            return None
        role = fit_context.get("role")
        if role == "cv":
            trial = fit_context.get("trial")
            fold = fit_context.get("fold")
            if trial is None or fold is None:
                return "torch/cv"
            return f"torch/cv/trial_{int(trial):03d}/fold_{int(fold):02d}"
        if role:
            return f"torch/{role}"
        return None

    def _monitor_direction(
        self, monitor: str, fit_context: dict[str, Any] | None
    ) -> str:
        direction = self.snapshot_direction
        if direction is None and monitor == "val_score":
            direction = (fit_context or {}).get("score_direction")
        direction = (direction or "minimize").lower()
        if direction not in {"minimize", "maximize"}:
            raise ValueError("snapshot_direction must be 'minimize' or 'maximize'")
        return direction

    def _monitor_value(self, metrics: dict[str, float], monitor: str) -> float | None:
        return metrics.get(f"torch/{monitor}")

    def _is_better(
        self,
        value: float,
        best: float | None,
        direction: str,
        min_delta: float,
    ) -> bool:
        if best is None:
            return True
        if direction == "maximize":
            return value > best + min_delta
        return value < best - min_delta

    def _maybe_snapshot(
        self,
        snapshots: list[_TorchSnapshot],
        model: Any,
        epoch: int,
        score: float,
        direction: str,
    ) -> None:
        if self.snapshot_mode == "off":
            return
        if epoch < self.snapshot_start_epoch:
            return
        if (epoch - self.snapshot_start_epoch) % self.snapshot_interval != 0:
            return
        limit = 1 if self.snapshot_mode == "best" else self.snapshot_k
        snapshots.append(_TorchSnapshot(epoch, score, self._cpu_state_dict(model)))
        reverse = direction == "maximize"
        snapshots.sort(key=lambda item: item.score, reverse=reverse)
        del snapshots[limit:]

    def _attach_snapshots(
        self, model: Any, snapshots: Sequence[_TorchSnapshot]
    ) -> None:
        module = self._module(model)
        setattr(module, "_pipeline_snapshot_mode", self.snapshot_mode)
        setattr(
            module,
            "_pipeline_snapshot_epochs",
            [item.epoch for item in snapshots],
        )
        setattr(
            module,
            "_pipeline_snapshot_scores",
            [item.score for item in snapshots],
        )
        setattr(
            module,
            "_pipeline_snapshot_state_dicts",
            [item.state_dict for item in snapshots],
        )

    def _cpu_state_dict(self, model: Any) -> dict[str, Any]:
        module = self._module(model)
        return {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
        }

    def _module(self, model: Any) -> Any:
        return model.module if hasattr(model, "module") else model

    def _predict_array(self, model: Any, xb: Any) -> np.ndarray:
        pred = model(xb).detach().cpu().numpy()
        return pred.reshape(-1) if pred.ndim == 2 and pred.shape[1] == 1 else pred


def _writable_array(x: Any, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    return arr if arr.flags.writeable else arr.copy()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_hash(path: Path, meta: dict[str, Any] | None) -> None:
    expected = (meta or {}).get("sha256")
    if expected is not None and _sha256_file(path) != expected:
        raise ValueError(f"model hash mismatch: {path}")
