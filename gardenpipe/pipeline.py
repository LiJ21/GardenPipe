from __future__ import annotations

import ast
import copy
import datetime as dt
import gc
import hashlib
import importlib
import io
import inspect
import json
import re
import resource
import threading
import time
import warnings
from contextlib import contextmanager
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from gardenpipe.data import (
    POLARS_ENGINES,
    Batch,
    DataSource,
    DateFrame,
    Loader,
    Raw,
    as_batch,
)
from gardenpipe.precision import check_precision, float_dtype
from gardenpipe.registry import (
    Registry,
    file_set_manifest,
    register_expr,
    register_pickle_feature,
)
from gardenpipe.search import (
    BySizeRecency,
    ParamFn,
    SearchSpace,
    create_study,
    expanding_folds,
    suggest_params,
    weighted_mean,
)
from gardenpipe.track import NullTracker, Tracker
from gardenpipe.transform import (
    Passthrough,
    Transform,
    compose_transform,
    load_transform,
    save_transform,
)
from gardenpipe.score import Score, rmse, _NO_COMBINE


def evaluate_model(
    adapter: Any,
    model: Any,
    src: DataSource,
    score: Score,
    fold: Any = None,
    keep_predictions: bool = False,
) -> tuple[float, dict[str, Any], np.ndarray | None]:
    if not getattr(adapter, "streaming", False):
        prediction = next(
            _iter_prediction_batches(adapter, model, src, streaming=False)
        )
        prediction.ctx["fold"] = fold
        return (
            score_arrays(
                score,
                prediction.y_true,
                prediction.y_pred,
                prediction.ctx,
                sample_weight=prediction.sample_weight,
            ),
            prediction.ctx,
            prediction.y_pred if keep_predictions else None,
        )
    loss, ctx, y_pred = score_stream(adapter, model, score, src, keep_predictions)
    ctx["fold"] = fold
    return loss, ctx, y_pred


def score_arrays(
    score: Score,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"score length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    return float(
        call_score(score, y_true, y_pred, ctx, sample_weight=sample_weight)
    )


def score_stream(
    adapter: Any,
    model: Any,
    score: Score,
    src: DataSource,
    keep_predictions: bool = False,
) -> tuple[float, dict[str, Any], np.ndarray | None]:
    state: Any = None
    ctx: dict[str, Any] = {"n": 0}
    pred_parts = [] if keep_predictions else None

    for prediction in _iter_prediction_batches(adapter, model, src, streaming=True):
        state = call_score(
            score,
            prediction.y_true,
            prediction.y_pred,
            prediction.ctx,
            combine_with=state,
            sample_weight=prediction.sample_weight,
        )
        merge_ctx(ctx, prediction.ctx)
        if pred_parts is not None:
            pred_parts.append(prediction.y_pred)

    if state is None:
        raise ValueError("cannot score empty prediction stream")
    y_pred = np.concatenate(pred_parts) if pred_parts else None
    return float(state), ctx, y_pred


@dataclass(frozen=True)
class _PredictionBatch:
    y_true: np.ndarray
    y_pred: np.ndarray
    ctx: dict[str, Any]
    sample_weight: np.ndarray | None = None
    frame: pl.DataFrame | None = None


def _iter_prediction_batches(
    adapter: Any,
    model: Any,
    src: Any,
    *,
    streaming: bool | None = None,
    frame_cols: Sequence[str] | None = None,
    batch_size: int | None = None,
):
    """Yield predictions with their exact label/context (and optional key frame).

    Evaluation consumes the ordinary ``Batch`` path. ``predict_frame`` requests
    ``frame_cols`` so keys and labels are collected in the same Polars batches as
    the features passed to the model; no timestamp join is needed afterward.
    """

    if frame_cols is None:
        use_stream = (
            getattr(adapter, "streaming", False)
            if streaming is None
            else streaming
        )
        values = (
            src.batches(adapter_batch_size(adapter))
            if use_stream
            else (src.materialize(),)
        )
        for value in values:
            batch = as_batch(value)
            y_pred = _validated_prediction(
                adapter.predict(model, batch.x), len(batch.y)
            )
            yield _PredictionBatch(
                y_true=batch.y,
                y_pred=y_pred,
                ctx=batch.ctx,
                sample_weight=batch.weight,
            )
        return

    size = adapter_batch_size(adapter) if batch_size is None else batch_size
    selected = _ordered_unique([*src.features, src.target, *frame_cols])
    for df in src.dataframe_batches(size, cols=selected):
        if df.height == 0:
            continue
        x = df.select(src.features).to_numpy()
        y_true = df.get_column(src.target).to_numpy()
        y_pred = _validated_prediction(adapter.predict(model, x), len(y_true))
        yield _PredictionBatch(
            y_true=y_true,
            y_pred=y_pred,
            ctx=_prediction_ctx(df),
            frame=df.select(frame_cols),
        )


def _validated_prediction(value: Any, expected_rows: int) -> np.ndarray:
    y_pred = np.asarray(value)
    if y_pred.ndim == 0:
        if expected_rows != 1:
            raise ValueError(
                "prediction length mismatch: "
                f"y_true={expected_rows}, y_pred is scalar"
            )
        y_pred = y_pred.reshape(1)
    if len(y_pred) != expected_rows:
        raise ValueError(
            f"score length mismatch: y_true={expected_rows}, y_pred={len(y_pred)}"
        )
    return y_pred


def _prediction_ctx(df: pl.DataFrame) -> dict[str, Any]:
    ctx: dict[str, Any] = {"n": df.height}
    for name in ("date", "tag"):
        if name not in df.columns:
            continue
        values = df.get_column(name).to_numpy()
        ctx[name] = values
        ctx[f"{name}s"] = _ordered_unique([str(v) for v in values.tolist()])
    return ctx


def call_score(
    score: Score,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any],
    combine_with: Any = _NO_COMBINE,
    sample_weight: np.ndarray | None = None,
) -> Any:
    sig = inspect.signature(score)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    has_varkw = any(p.kind == p.VAR_KEYWORD for p in params)
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    args: list[Any] = [y_true, y_pred]
    kwargs: dict[str, Any] = {}
    if has_varargs or (
        len(positional) >= 3
        and positional[2].name not in {"combine_with", "sample_weight"}
    ):
        args.append(ctx)
    elif "ctx" in sig.parameters:
        kwargs["ctx"] = ctx
    if sample_weight is not None and "sample_weight" in sig.parameters:
        kwargs["sample_weight"] = sample_weight
    if combine_with is not _NO_COMBINE:
        if "combine_with" not in sig.parameters and not has_varkw:
            raise TypeError("streaming scores must accept a combine_with argument")
        kwargs["combine_with"] = combine_with
    return score(*args, **kwargs)


def adapter_batch_size(adapter: Any) -> int | None:
    return getattr(adapter, "batch_size", None)


def merge_ctx(total: dict[str, Any], batch: dict[str, Any]) -> None:
    total["n"] = int(total.get("n", 0)) + int(batch.get("n", 0))
    for name in ("date", "tag"):
        values = batch.get(f"{name}s")
        if values is None and name in batch:
            values = [str(v) for v in np.asarray(batch[name]).tolist()]
        if values is not None:
            key = f"{name}s"
            total[key] = _ordered_unique(
                [*total.get(key, []), *[str(v) for v in values]]
            )
    for key, value in batch.items():
        if key not in {"n", "date", "dates", "tag", "tags"}:
            total[key] = value


def plot_train_val_loss(
    history: Any,
    *,
    trial: int | str | None = "best",
    fold: int | str | None = None,
    metric: str | None = None,
    ax: Any | None = None,
    axes: Any | None = None,
    max_cols: int = 2,
    title: str | None = None,
    score_direction: str | None = None,
    complete_trials: bool = True,
) -> Any:
    """Plot per-fit train/validation curves from Pipeline fit history.

    Accepts a ``Pipeline.train()`` result, a ``save_history()`` payload, raw
    ``pipeline.validation_history``, one validation record with ``fit_history``,
    or one record returned by ``pipeline.refit()``. By default, validation
    histories plot every fold from the best completed trial. Use ``trial=None``
    to plot all trials, ``fold=<n>`` to select one fold, or ``fold="best"`` to
    select the fold with the best available fit score. Pass
    ``complete_trials=False`` to allow automatic best-trial selection from
    pruned or otherwise incomplete trials.
    """

    def optional_float(value: Any) -> float:
        if value is None:
            return np.nan
        try:
            result = float(value)
        except (TypeError, ValueError):
            return np.nan
        return result if np.isfinite(result) else np.nan

    def optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def is_fit_record(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        nested = value.get("history")
        return isinstance(nested, Mapping) and isinstance(nested.get("train"), Mapping)

    def sequence_value(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        )

    def columnar_records(data: Mapping[str, Any]) -> list[dict[str, Any]]:
        columns = {
            key: list(value) if sequence_value(value) else [value]
            for key, value in data.items()
        }
        n = max((len(value) for value in columns.values()), default=0)
        return [
            {
                key: value[i] if i < len(value) else None
                for key, value in columns.items()
            }
            for i in range(n)
        ]

    def best_record(
        items: Sequence[dict[str, Any]], direction: str
    ) -> dict[str, Any] | None:
        scored = []
        for item in items:
            score = np.nan
            for key in ("weighted_score", "val_score", "best_score"):
                score = optional_float(item.get(key))
                if np.isfinite(score):
                    break
            if not np.isfinite(score):
                nested = item.get("history", {})
                val_metrics = nested.get("val") if isinstance(nested, Mapping) else None
                train_metrics = (
                    nested.get("train") if isinstance(nested, Mapping) else None
                )
                for metrics in (val_metrics, train_metrics):
                    if not isinstance(metrics, Mapping):
                        continue
                    for values in metrics.values():
                        curve_values = np.asarray(
                            [optional_float(value) for value in values],
                            dtype=float,
                        )
                        if curve_values.size:
                            score = float(np.nanmin(curve_values))
                            break
                    if np.isfinite(score):
                        break
            if np.isfinite(score):
                scored.append((score, item))
        if not scored:
            return items[0] if items else None
        return (max if direction == "maximize" else min)(
            scored, key=lambda item: item[0]
        )[1]

    def best_choice(value: Any) -> bool:
        return isinstance(value, str) and value.lower() in {"best", "optimal"}

    def same_trial(left: Any, right: Any) -> bool:
        if left == right:
            return True
        left_int = optional_int(left)
        right_int = optional_int(right)
        if left_int is not None and right_int is not None:
            return left_int == right_int
        return str(left) == str(right)

    def expected_folds(
        items: Sequence[dict[str, Any]],
        grouped: Mapping[Any, Sequence[dict[str, Any]]],
    ) -> int | None:
        meta_folds = optional_int(meta.get("n_folds"))
        if meta_folds and meta_folds > 0:
            return meta_folds
        record_folds = [
            value
            for item in items
            if (value := optional_int(item.get("n_folds"))) is not None and value > 0
        ]
        if record_folds:
            return max(record_folds)
        group_sizes = []
        for group in grouped.values():
            folds = {item.get("fold") for item in group if item.get("fold") is not None}
            if folds:
                group_sizes.append(len(folds))
        return max(group_sizes, default=None)

    def is_complete_trial(items: Sequence[dict[str, Any]], n_folds: int | None) -> bool:
        if n_folds is None or n_folds <= 0:
            return True
        folds = {item.get("fold") for item in items if item.get("fold") is not None}
        if folds:
            return len(folds) >= n_folds
        return len(items) >= n_folds

    def curve(record: dict[str, Any], split: str, metric_name: str) -> np.ndarray:
        nested = record.get("history", {})
        metrics = nested.get(split, {}) if isinstance(nested, Mapping) else {}
        values = metrics.get(metric_name, []) if isinstance(metrics, Mapping) else []
        return np.asarray([optional_float(value) for value in values], dtype=float)

    meta: dict[str, Any] = {}
    data = history
    if is_fit_record(history):
        records = [dict(history)]
    else:
        if isinstance(history, Mapping):
            meta = dict(history)
            data = history.get("validation_history", history.get("fit_history"))
            if data is None:
                has_columns = any(sequence_value(value) for value in history.values())
                data = columnar_records(history) if has_columns else [history]
        if is_fit_record(data):
            records = [dict(data)]
        else:
            if hasattr(data, "to_dicts"):
                data = data.to_dicts()
            elif hasattr(data, "to_dict") and not isinstance(data, Mapping):
                try:
                    data = data.to_dict("records")
                except TypeError:
                    pass
            if is_fit_record(data):
                records = [dict(data)]
            else:
                if isinstance(data, Mapping):
                    has_columns = any(sequence_value(value) for value in data.values())
                    data = columnar_records(data) if has_columns else [data]
                records = []
                for item in data or []:
                    if not isinstance(item, Mapping):
                        continue
                    if is_fit_record(item):
                        records.append(dict(item))
                        continue
                    fit_history = item.get("fit_history")
                    if not is_fit_record(fit_history):
                        continue
                    record = dict(fit_history)
                    for key in (
                        "trial",
                        "fold",
                        "val_score",
                        "weighted_score",
                        "n_folds",
                        "n",
                        "dates",
                        "tags",
                        "params",
                    ):
                        if key in item and key not in record:
                            record[key] = item[key]
                    records.append(record)

    direction = (score_direction or meta.get("score_direction") or "minimize").lower()
    if best_choice(trial):
        chosen_trial = meta.get("best_trial")
        if chosen_trial is not None:
            records = [
                item for item in records if same_trial(item.get("trial"), chosen_trial)
            ]
        final_by_trial: dict[Any, dict[str, Any]] = {}
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for item in records:
            if item.get("trial") is not None:
                grouped.setdefault(item["trial"], []).append(item)
                final_by_trial[item["trial"]] = item
        if chosen_trial is None:
            candidates = grouped
            if complete_trials and grouped:
                n_folds = expected_folds(records, grouped)
                complete = {
                    trial_id: items
                    for trial_id, items in grouped.items()
                    if is_complete_trial(items, n_folds)
                }
                if complete:
                    candidates = complete
            best = best_record(
                [final_by_trial[trial_id] for trial_id in candidates],
                direction,
            )
            chosen_trial = best.get("trial") if best is not None else None
            if chosen_trial is not None:
                records = [
                    item
                    for item in records
                    if same_trial(item.get("trial"), chosen_trial)
                ]
    elif trial is not None:
        records = [item for item in records if same_trial(item.get("trial"), trial)]

    if best_choice(fold):
        best = best_record(records, direction)
        records = [best] if best is not None else []
    elif fold is not None:
        records = [item for item in records if item.get("fold") == fold]
    if not records:
        raise ValueError("no fit-history records match the requested filters")

    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_train_val_loss() requires matplotlib.") from exc

    if ax is not None and axes is not None:
        raise ValueError("pass either ax or axes, not both")
    if ax is not None:
        if len(records) != 1:
            raise ValueError("ax can only be used when exactly one curve is selected")
        fig, flat_axes = ax.figure, np.asarray([ax], dtype=object)
    elif axes is not None:
        flat_axes = np.ravel(np.asarray(axes, dtype=object))
        if len(flat_axes) < len(records):
            raise ValueError(f"need at least {len(records)} axes, got {len(flat_axes)}")
        fig = flat_axes[0].figure
    else:
        if max_cols < 1:
            raise ValueError("max_cols must be positive")
        n_cols = min(max_cols, len(records))
        n_rows = int(np.ceil(len(records) / n_cols))
        fig, created_axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6.0 * n_cols, 3.8 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        flat_axes = np.ravel(created_axes)

    for record, item_ax in zip(records, flat_axes):
        nested = record.get("history", {})
        train_metrics = nested.get("train", {}) if isinstance(nested, Mapping) else {}
        val_metrics = nested.get("val", {}) if isinstance(nested, Mapping) else {}
        if metric is not None:
            metric_name = metric
        else:
            metric_name = None
            for name in val_metrics:
                if name in train_metrics:
                    metric_name = str(name)
                    break
            if metric_name is None:
                for name in train_metrics:
                    metric_name = str(name)
                    break
            if metric_name is None:
                for name in val_metrics:
                    metric_name = str(name)
                    break
            if metric_name is None:
                raise ValueError("fit history contains no train or validation metrics")
        train = curve(record, "train", metric_name)
        val = curve(record, "val", metric_name)
        if not train.size and not val.size:
            raise ValueError(f"metric {metric_name!r} has no train or val curve")
        if train.size:
            item_ax.plot(
                np.arange(1, len(train) + 1),
                train,
                color="#2563eb",
                linewidth=1.8,
                label=f"train {metric_name}",
            )
        if val.size:
            item_ax.plot(
                np.arange(1, len(val) + 1),
                val,
                color="#dc2626",
                linewidth=1.8,
                label=f"val {metric_name}",
            )
        best_iteration = record.get("best_iteration")
        if best_iteration is not None:
            try:
                item_ax.axvline(
                    int(best_iteration) + 1,
                    color="#f59e0b",
                    linestyle="--",
                    linewidth=1.0,
                    label="best",
                )
            except (TypeError, ValueError):
                pass

        parts = []
        if record.get("role") is not None:
            parts.append(str(record["role"]))
        if record.get("trial") is not None:
            parts.append(f"trial {record['trial']}")
        if record.get("fold") is not None:
            parts.append(f"fold {record['fold']}")
        item_title = " / ".join(parts) if parts else "fit"
        best_score = optional_float(record.get("best_score"))
        if np.isfinite(best_score):
            item_title += f" | best={best_score:.6g}"

        item_ax.set_title(item_title)
        item_ax.set_xlabel("round / epoch")
        item_ax.set_ylabel(metric_name)
        item_ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        item_ax.spines["top"].set_visible(False)
        item_ax.spines["right"].set_visible(False)
        item_ax.legend(frameon=False)

    for item_ax in flat_axes[len(records) :]:
        item_ax.set_visible(False)
    if title is not None:
        fig.suptitle(title)
    return fig


plot_validation_history = plot_train_val_loss


@dataclass
class Pipeline:
    rolling_dates: list[list[str]]
    adapter: Any
    target: str
    features: list[str]
    data_loader: Loader
    test_dates: list[str] | None = None
    search_space: SearchSpace | ParamFn | None = None
    val_score: Score = rmse
    score_direction: str = "minimize"
    transform: Transform = field(default_factory=Passthrough)
    train_sampler: tuple[pl.Expr, ...] = ()
    val_sampler: tuple[pl.Expr, ...] = ()
    test_sampler: tuple[pl.Expr, ...] = ()
    fold_weighting: BySizeRecency = field(default_factory=BySizeRecency)
    search_sampler: str = "tpe"
    n_trials: int = 50
    pruner: Any = None
    tracker: Tracker = field(default_factory=NullTracker)
    cache_arrays: bool = False
    seed: int | None = None
    polars_engine: str = "streaming"
    precision: str = "float64"
    refit_val_dates: list[str] | None = None
    data_source_wrapper: Callable[[DataSource, str], Any] | None = None
    sample_weight_col: str | None = None

    model: Any = field(default=None, init=False)
    best_params: dict[str, Any] | None = field(default=None, init=False)
    study: Any = field(default=None, init=False)
    fitted_transform: Transform | None = field(default=None, init=False)
    validation_history: list[dict[str, Any]] = field(default_factory=list, init=False)
    refit_history: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_refit_train_dates: list[str] | None = field(
        default=None, init=False, repr=False
    )
    last_refit_val_dates: list[str] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.score_direction = self.score_direction.lower()
        if self.score_direction not in {"minimize", "maximize"}:
            raise ValueError("score_direction must be 'minimize' or 'maximize'")
        self.polars_engine = self.polars_engine.lower()
        if self.polars_engine not in POLARS_ENGINES:
            raise ValueError(f"polars_engine must be one of: {sorted(POLARS_ENGINES)}")
        self.precision = check_precision(self.precision)
        if self.sample_weight_col == "":
            raise ValueError("sample_weight_col must be non-empty or None")
        self.train_sampler = tuple(self.train_sampler)
        self.val_sampler = tuple(self.val_sampler)
        self.test_sampler = tuple(self.test_sampler)
        if self.refit_val_dates is not None:
            self.refit_val_dates = (
                [self.refit_val_dates]
                if isinstance(self.refit_val_dates, str)
                else list(self.refit_val_dates)
            )
        self._transform_cache: dict[tuple[Any, ...], Transform] = {}
        self._array_cache: dict[tuple[Any, ...], Batch] = {}

    def train(
        self,
        verbose: int = 0,
        memory_log: bool = False,
        memory_interval: float = 0.05,
        no_refit: bool = False,
    ) -> dict[str, Any]:
        folds = expanding_folds(self.rolling_dates)
        fit_history = getattr(self.adapter, "fit_history", None)
        if self.study is not None or fit_history:
            warnings.warn(
                "Training is being called again; previous fit history is being "
                "cleared before starting a new search.",
                RuntimeWarning,
                stacklevel=2,
            )
        if fit_history is not None:
            fit_history.clear()
        if hasattr(self.adapter, "last_fit_history"):
            self.adapter.last_fit_history = None
        self.validation_history.clear()
        self.tracker.start_run(
            {
                "sampler": self.search_sampler,
                "n_trials": self.n_trials,
                "n_folds": len(folds),
                "score_direction": self.score_direction,
                "polars_engine": self.polars_engine,
                "precision": self.precision,
                "no_refit": no_refit,
            }
        )
        try:
            self.study = create_study(
                self.search_sampler,
                self.search_space,
                direction=self.score_direction,
                pruner=self.pruner,
                seed=self.seed,
            )
            import optuna

            if verbose > 0:
                print(f"======== Optuna study created. Launching optimization.")

            def objective(trial: Any) -> float:
                params = suggest_params(self.search_space, trial)
                trial.set_user_attr("params", params)
                if verbose > 0:
                    print(f"======== running params {params}")
                self.tracker.log_params({f"param/{k}": v for k, v in params.items()})
                scores: list[float] = []
                sizes: list[int] = []
                for fold, (train_dates, val_dates) in enumerate(folds):
                    if verbose > 1:
                        print(
                            f"======== fold: {fold}, with train = {train_dates} and val = {val_dates}"
                        )
                    with self._memory_log(
                        f"trial={trial.number} fold={fold} _fit_transform train_dates={train_dates}",
                        enabled=memory_log,
                        interval=memory_interval,
                    ):
                        fitted = self._fit_transform(train_dates)
                    train_src = self._src(
                        train_dates, self.train_sampler, fitted, "train"
                    )
                    val_src = self._src(val_dates, self.val_sampler, fitted, "val")
                    model = self.adapter.build(params)
                    if verbose > 2:
                        print("======== built model, start training...")
                    with self._memory_log(
                        f"trial={trial.number} fold={fold} _fit_model train_dates={train_dates} val_dates={val_dates}",
                        enabled=memory_log,
                        interval=memory_interval,
                    ):
                        model = self._fit_model(
                            model,
                            train_src,
                            val_src,
                            trial,
                            fit_context={
                                "role": "cv",
                                "trial": trial.number,
                                "fold": fold,
                                "n_folds": len(folds),
                                "train_dates": list(train_dates),
                                "val_dates": list(val_dates),
                                "val_score": self.val_score,
                                "score_direction": self.score_direction,
                                "score_name": _callable_name(self.val_score),
                            },
                        )
                    with self._memory_log(
                        f"trial={trial.number} fold={fold} _evaluate val_dates={val_dates}",
                        enabled=memory_log,
                        interval=memory_interval,
                    ):
                        loss, ctx, _ = self._evaluate(
                            model, val_src, self.val_score, fold
                        )
                    scores.append(loss)
                    sizes.append(int(ctx["n"]))
                    running = weighted_mean(scores, sizes, self.fold_weighting)
                    if verbose > 1:
                        print(f"======== loss = {loss}, running average = {running}")
                    record = {
                        "trial": trial.number,
                        "fold": fold,
                        "n_folds": len(folds),
                        "val_score": loss,
                        "weighted_score": running,
                        "n": int(ctx["n"]),
                        "dates": ctx.get("dates"),
                        "tags": ctx.get("tags"),
                        "params": params,
                    }
                    fit_history = getattr(self.adapter, "last_fit_history", None)
                    if fit_history is not None:
                        record["fit_history"] = fit_history
                    if verbose > 2:
                        print("======== record = \n", record)
                    self.validation_history.append(record)
                    self.tracker.log(
                        {
                            "val/fold_loss": loss,
                            "val/weighted_loss": running,
                            "val/fold": fold,
                            "val/n": int(ctx["n"]),
                        },
                        step=trial.number * len(folds) + fold,
                    )
                    trial.report(running, step=fold)
                    if fold + 1 < len(folds) and trial.should_prune():
                        raise optuna.TrialPruned()
                return weighted_mean(scores, sizes, self.fold_weighting)

            self.study.optimize(objective, n_trials=self.n_trials)
            if verbose > 0:
                action = (
                    "Skipping final refit."
                    if no_refit
                    else "Refitting with best params."
                )
                print(
                    f"======== optimization finished, best params extracted. {action}"
                )
            self.best_params = dict(
                self.study.best_trial.user_attrs.get("params", self.study.best_params)
            )
            self.model = None
            self.fitted_transform = None
            self.refit_history = None
            self.last_refit_train_dates = None
            self.last_refit_val_dates = None
            if not no_refit:
                self.refit(
                    dates=self._all_train_dates(),
                    val_dates=self.refit_val_dates,
                    params=self.best_params,
                    memory_log=memory_log,
                    memory_interval=memory_interval,
                )
            print("======== search done." if no_refit else "======== training done.")
            return {
                "best_params": self.best_params,
                "best_score": float(self.study.best_value),
                "best_trial": int(self.study.best_trial.number),
                "n_trials": len(self.study.trials),
                "n_folds": len(folds),
                "refit": not no_refit,
                "validation_history": self.validation_history,
                "refit_history": self.refit_history,
            }
        finally:
            self.tracker.finish()

    def test(
        self,
        score: Score = rmse,
        keep_predictions: bool = True,
        dates: list[str] | None = None,
        sampler: tuple[pl.Expr, ...] | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call train() and refit() before test()")
        transform = self.fitted_transform or self._fit_transform(
            self._all_train_dates()
        )
        dates = self.test_dates if dates is None else dates
        if dates is None:
            raise ValueError(
                "Test dates not specified either as pipeline parameter nor test function argument."
            )
        sampler = self.test_sampler if sampler is None else sampler
        src = self._src(dates, sampler, transform, "test")
        loss, ctx, y_pred = self._evaluate(
            self.model, src, score, "test", keep_predictions=keep_predictions
        )
        return {"test_score": loss, "n": int(ctx["n"]), "ctx": ctx, "y_pred": y_pred}

    def predict_frame(
        self,
        dates: Sequence[str] | str | None = None,
        *,
        sampler: Sequence[pl.Expr] | None = None,
        feature_source: Any = None,
        key_cols: Sequence[str] | str | None = None,
        prediction_names: Sequence[str] | str | None = None,
        start: Any = None,
        end: Any = None,
        timezone: str | None = None,
        time_col: str = "time",
        predicate: pl.Expr | None = None,
        batch_size: int | None = None,
    ) -> pl.DataFrame:
        """Predict a timestamp/key-aligned frame for inspection and plotting.

        The fitted pipeline transform and test sampler are applied before model
        inference. Bounds and ``predicate`` are ANDed with the test sampler, whose
        rules keep a row if any of them matches.

        ``feature_source`` may be a loader, a DataSource, a Polars frame, or a
        parquet path. A fitted model and transform are required; this method never
        fits or refits either one implicitly.
        """

        if self.model is None:
            raise RuntimeError("call refit() or load_pipeline() before predict_frame()")
        if self.fitted_transform is None:
            raise RuntimeError(
                "predict_frame() requires a fitted transform; "
                "call refit() or load_pipeline() first"
            )
        if not time_col:
            raise ValueError("time_col must be non-empty")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None")
        if predicate is not None and not isinstance(predicate, pl.Expr):
            raise TypeError("predicate must be a Polars expression or None")

        prediction_dates = _prediction_dates(dates, self.test_dates, feature_source)
        test_sampler = self.test_sampler if sampler is None else tuple(sampler)
        src = self._prediction_source(
            prediction_dates,
            test_sampler,
            self.fitted_transform,
            feature_source,
        )

        schema = src.frame(select=False).collect_schema()
        schema_names = set(schema.names())
        bounds = []
        if start is not None or end is not None:
            if time_col not in schema_names:
                raise ValueError(
                    f"time column {time_col!r} is not available in feature source"
                )
            start_value = _coerce_prediction_bound(start, schema[time_col], timezone)
            end_value = _coerce_prediction_bound(end, schema[time_col], timezone)
            if (
                start_value is not None
                and end_value is not None
                and start_value > end_value
            ):
                raise ValueError("start must be less than or equal to end")
            if start_value is not None:
                bounds.append(pl.col(time_col) >= start_value)
            if end_value is not None:
                bounds.append(pl.col(time_col) <= end_value)
        elif timezone is not None:
            _prediction_zone(timezone)
        if predicate is not None:
            bounds.append(predicate)
        if bounds:
            src.filters = _and_filter_groups(
                src.filters, *((bound,) for bound in bounds)
            )

        requested_keys = key_cols
        if requested_keys is None:
            requested_keys = ("date", time_col)
            resolved_keys = [name for name in requested_keys if name in schema_names]
        else:
            resolved_keys = _ordered_unique(
                [requested_keys]
                if isinstance(requested_keys, str)
                else list(requested_keys)
            )
            missing = [name for name in resolved_keys if name not in schema_names]
            if missing:
                raise ValueError(
                    "prediction key columns are not available in feature source: "
                    f"{missing}"
                )

        output_base = _ordered_unique([*resolved_keys, self.target])
        requested_names = _normalize_prediction_names(prediction_names)
        empty_names = _prediction_names_without_output(
            requested_names, self.adapter
        )
        empty_conflicts = [name for name in empty_names if name in output_base]
        if empty_conflicts:
            raise ValueError(
                f"prediction names conflict with output columns: {empty_conflicts}"
            )
        prediction_src = (
            self.data_source_wrapper(src, "predict")
            if self.data_source_wrapper is not None
            else src
        )
        parts: list[pl.DataFrame] = []
        resolved_names: list[str] | None = None
        for prediction in _iter_prediction_batches(
            self.adapter,
            self.model,
            prediction_src,
            frame_cols=output_base,
            batch_size=batch_size,
        ):
            values = _prediction_matrix(prediction.y_pred)
            if resolved_names is None:
                resolved_names = _resolve_prediction_names(
                    requested_names,
                    values.shape[1],
                    self.adapter,
                )
                conflicts = [name for name in resolved_names if name in output_base]
                if conflicts:
                    raise ValueError(
                        f"prediction names conflict with output columns: {conflicts}"
                    )
            elif len(resolved_names) != values.shape[1]:
                raise ValueError(
                    "prediction output width changed between batches: "
                    f"expected {len(resolved_names)}, got {values.shape[1]}"
                )
            assert prediction.frame is not None
            parts.append(
                prediction.frame.with_columns(
                    [
                        pl.Series(name, values[:, i])
                        for i, name in enumerate(resolved_names)
                    ]
                ).select([*output_base, *resolved_names])
            )

        if not parts:
            return _empty_prediction_frame(
                output_base,
                empty_names,
                schema,
                target=self.target,
                precision=self.precision,
            )
        return pl.concat(parts, how="vertical_relaxed")

    def _prediction_source(
        self,
        dates: Sequence[str],
        filters: Sequence[pl.Expr],
        transform: Transform,
        feature_source: Any,
    ) -> DataSource:
        loader: Any = self.data_loader
        source_filters: Sequence[pl.Expr] = ()
        source_transform: Transform | None = None
        if feature_source is not None:
            if isinstance(feature_source, (pl.DataFrame, pl.LazyFrame, str, Path)):
                loader = _prediction_frame_loader(feature_source)
            elif type(feature_source) is DataSource:
                loader = feature_source.loader
                source_filters = tuple(feature_source.filters)
                source_transform = feature_source.transform
            elif hasattr(feature_source, "frame") and hasattr(
                feature_source, "dates"
            ):
                raise TypeError(
                    "custom DataSource feature sources are not supported; "
                    "pass a plain DataSource or its loader"
                )
            elif callable(feature_source):
                loader = feature_source
            else:
                raise TypeError(
                    "feature_source must be a loader, DataSource, Polars frame, "
                    "parquet path, or None"
                )

        combined_transform = (
            transform
            if source_transform is None or source_transform is transform
            else compose_transform(source_transform, transform)
        )

        return DataSource(
            dates=list(dates),
            loader=loader,
            target=self.target,
            features=list(self.features),
            filters=_and_filter_groups(source_filters, filters),
            transform=combined_transform,
            sample_weight_col=self.sample_weight_col,
            polars_engine=self.polars_engine,
            precision=self.precision,
        )

    def get_model(self) -> Any:
        return self.model

    def save_model(
        self, path: str | Path, filename: str | None = None
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call train() before save_model()")
        return self.adapter.save_model(self.model, path, filename=filename)

    def load_model(
        self,
        path: str | Path,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        self.model = self.adapter.load_model(path, meta)
        return self.model

    def save_history(self, path: str | Path) -> dict[str, Any]:
        payload = self._history_payload()
        _write_json(Path(path), payload)
        return payload

    def save_pipeline(self, path: str | Path) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("call train() before save_pipeline()")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        run_hash, stamp, run_dir = self._new_run_dir(root)
        registry = Registry(root / "registry.jsonl")

        model_meta = self.save_model(root / "model", filename=f"model_{run_hash}")
        model_meta["path"] = f"model/{model_meta['artifact']}"

        transform_path = root / "transform" / f"transform_{run_hash}.pkl"
        transform_meta = save_transform(
            self.fitted_transform or self.transform, transform_path
        )
        transform_meta["path"] = f"transform/{transform_path.name}"

        self.save_history(run_dir / "history.json")

        filters = {
            role: self._register_filters(registry, exprs, f"{role}_filter", run_hash)
            for role, exprs in (
                ("train", self.train_sampler),
                ("val", self.val_sampler),
                ("test", self.test_sampler),
            )
        }
        features = self._register_features(registry, run_hash)
        feature_files = self._feature_file_manifest()

        manifest = {
            "version": 3,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "run": {"stamp": stamp, "hash": run_hash},
            "target": self.target,
            "features": list(self.features),
            "sample_weight_col": self.sample_weight_col,
            "feature_registry": features,
            "feature_files": feature_files,
            "rolling_dates": self.rolling_dates,
            "test_dates": self.test_dates,
            "best_params": self.best_params,
            "score_direction": self.score_direction,
            "val_score": _callable_name(self.val_score),
            "sampler": self.search_sampler,
            "n_trials": self.n_trials,
            "polars_engine": self.polars_engine,
            "precision": self.precision,
            "refit_val_dates": self.refit_val_dates,
            "last_refit_train_dates": self.last_refit_train_dates,
            "last_refit_val_dates": self.last_refit_val_dates,
            "adapter": _object_info(self.adapter),
            "data_loader": _object_info(self.data_loader),
            "model": model_meta,
            "transform": transform_meta,
            "filters": filters,
            "registry": {"path": "registry.jsonl"},
            "history": {"path": "history.json"},
        }
        _write_json(run_dir / "pipeline.json", manifest)
        return manifest

    def load_pipeline(self, path: str | Path) -> dict[str, Any]:
        run_dir = _resolve_pipeline_run(path)
        manifest = _read_json(run_dir / "pipeline.json")
        run_info = manifest.get("run")
        root = run_dir.parent if run_info is not None else run_dir

        self.target = manifest.get("target", self.target)
        self.features = list(manifest.get("features", self.features))
        self.sample_weight_col = manifest.get("sample_weight_col")
        self.rolling_dates = [
            list(x) for x in manifest.get("rolling_dates", self.rolling_dates)
        ]
        test_dates = manifest.get("test_dates", self.test_dates)
        self.test_dates = list(test_dates) if test_dates is not None else None
        self.refit_val_dates = manifest.get("refit_val_dates", self.refit_val_dates)
        self.best_params = manifest.get("best_params")
        self.score_direction = manifest.get("score_direction", self.score_direction)
        self.polars_engine = manifest.get("polars_engine", self.polars_engine)
        # Saves from before precision was recorded keep the current setting.
        self.precision = check_precision(manifest.get("precision", self.precision))

        history_path = manifest.get("history", {}).get("path", "history.json")
        history = _read_json(run_dir / history_path, {})
        self.validation_history = list(history.get("validation_history", []))
        self.refit_history = history.get("refit_history")
        last_train_dates = history.get(
            "last_refit_train_dates", manifest.get("last_refit_train_dates")
        )
        last_val_dates = history.get(
            "last_refit_val_dates", manifest.get("last_refit_val_dates")
        )
        self.last_refit_train_dates = (
            list(last_train_dates) if last_train_dates is not None else None
        )
        self.last_refit_val_dates = (
            list(last_val_dates) if last_val_dates is not None else None
        )

        transform_meta = manifest.get("transform")
        if transform_meta:
            loaded_transform = load_transform(
                root / transform_meta["path"], transform_meta
            )
            self.fitted_transform = loaded_transform
            # A loaded pipeline can be refit. _fit_transform() clones
            # self.transform, so keep the deserialized transform as its
            # template as well as the currently fitted transform.
            self.transform = copy.deepcopy(loaded_transform)
        model_meta = manifest.get("model")
        if model_meta:
            self.load_model(root / model_meta["path"], model_meta)

        registry_meta = manifest.get("registry")
        registry = Registry(root / registry_meta["path"]) if registry_meta else None
        filters = manifest.get("filters", {})
        self.train_sampler = tuple(_exprs_from_refs(filters.get("train", []), registry))
        self.val_sampler = tuple(_exprs_from_refs(filters.get("val", []), registry))
        self.test_sampler = tuple(_exprs_from_refs(filters.get("test", []), registry))
        return manifest

    @classmethod
    def from_saved(
        cls,
        path: str | Path,
        *,
        data_loader: Loader | None = None,
        adapter: Any | None = None,
        stamp: Any = None,
        run_hash: str | None = None,
    ) -> "Pipeline":
        """Construct and load a saved pipeline from its manifest.

        ``stamp`` and ``run_hash`` accept unique prefixes when ``path`` is a
        directory containing multiple ``pipeline_*`` runs. Adapters with a
        no-argument constructor are inferred from the manifest; pass
        ``adapter`` for custom adapters whose builders cannot be reconstructed.
        """

        run_dir = _resolve_pipeline_run(path, stamp=stamp, run_hash=run_hash)
        manifest = _read_json(run_dir / "pipeline.json")
        resolved_adapter = adapter or _adapter_from_manifest(manifest)
        test_dates = manifest.get("test_dates")
        pipeline = cls(
            rolling_dates=[
                list(block) for block in manifest.get("rolling_dates", [[]])
            ],
            test_dates=list(test_dates) if test_dates is not None else None,
            adapter=resolved_adapter,
            target=str(manifest.get("target", "")),
            features=list(manifest.get("features", [])),
            data_loader=data_loader or _unavailable_saved_loader,
            sample_weight_col=manifest.get("sample_weight_col"),
            score_direction=manifest.get("score_direction", "minimize"),
            polars_engine=manifest.get("polars_engine", "streaming"),
            precision=manifest.get("precision", "float64"),
            refit_val_dates=manifest.get("refit_val_dates"),
        )
        pipeline.load_pipeline(run_dir)
        return pipeline

    def _new_run_dir(self, root: Path) -> tuple[str, str, Path]:
        while True:
            now = dt.datetime.now(dt.UTC)
            run_hash = hashlib.sha256(
                now.isoformat(timespec="milliseconds").encode()
            ).hexdigest()[:8]
            stamp = now.strftime("%Y%m%dT%H%M")
            run_dir = root / f"pipeline_{stamp}_{run_hash}"
            try:
                run_dir.mkdir(parents=True)
            except FileExistsError:
                continue
            return run_hash, stamp, run_dir

    def _register_filters(
        self,
        registry: Registry,
        exprs: Sequence[pl.Expr],
        prefix: str,
        run_hash: str,
    ) -> list[dict[str, Any]]:
        return [
            _register_expr(registry, "filter", f"{prefix}_{i}", expr, run_hash)
            for i, expr in enumerate(exprs)
        ]

    def _register_features(
        self, registry: Registry, run_hash: str
    ) -> dict[str, dict[str, Any]]:
        exprs = _feature_exprs(self.data_loader)
        stateful = _stateful_features(self.data_loader)
        refs: dict[str, dict[str, Any]] = {}
        names = [*self.features]
        if self.sample_weight_col is not None:
            names.append(self.sample_weight_col)
        for name in _ordered_unique(names):
            expr = exprs.get(name)
            if expr is not None:
                refs[name] = register_expr(registry, "feature", name, expr, run_hash)
                continue
            feature = stateful.get(name)
            if feature is not None:
                refs[name] = register_pickle_feature(registry, name, feature, run_hash)
        return refs

    def _feature_file_manifest(self) -> dict[str, Any]:
        try:
            paths = _loader_feature_files(self.data_loader, self._feature_file_dates())
            return file_set_manifest(paths)
        except Exception as exc:
            return {
                "algorithm": "sha256",
                "hash": None,
                "files": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _feature_file_dates(self) -> list[str]:
        dates = [date for chunk in self.rolling_dates for date in chunk]
        if self.test_dates is not None:
            dates.extend(self.test_dates)
        if self.refit_val_dates is not None:
            dates.extend(self.refit_val_dates)
        return _ordered_unique(dates)

    def refit(
        self,
        dates: Sequence[str] | None = None,
        val_dates: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
        memory_log: bool = False,
        memory_interval: float = 0.05,
    ) -> Any:
        if params is None:
            if self.best_params is None:
                raise RuntimeError("call train() before refit() or pass params")
            params = self.best_params
        if dates is None:
            train_dates = self._all_train_dates()
        elif isinstance(dates, str):
            train_dates = [dates]
        else:
            train_dates = list(dates)
        if val_dates is None:
            final_val_dates = None
        elif isinstance(val_dates, str):
            final_val_dates = [val_dates]
        else:
            final_val_dates = list(val_dates)
        if final_val_dates:
            val_set = set(final_val_dates)
            train_dates = [date for date in train_dates if date not in val_set]
            if not train_dates:
                raise ValueError("refit validation dates leave no training dates")
        with self._memory_log(
            f"refit _fit_transform train_dates={train_dates}",
            enabled=memory_log,
            interval=memory_interval,
        ):
            self.fitted_transform = self._fit_transform(train_dates)
        train_src = self._src(
            train_dates, self.train_sampler, self.fitted_transform, "final_train"
        )
        val_src = (
            self._src(
                final_val_dates,
                self.val_sampler,
                self.fitted_transform,
                "final_val",
            )
            if final_val_dates
            else None
        )
        model = self.adapter.build(params)
        with self._memory_log(
            f"refit _fit_model train_dates={train_dates} val_dates={final_val_dates}",
            enabled=memory_log,
            interval=memory_interval,
        ):
            self.model = self._fit_model(
                model,
                train_src,
                val_src,
                None,
                fit_context={
                    "role": "refit",
                    "trial": None,
                    # The search trial these params came from, so adapters
                    # can reuse what its CV fits learned (e.g. XGBoost rounds).
                    "search_trial": (
                        int(self.study.best_trial.number)
                        if self.study is not None and params == self.best_params
                        else None
                    ),
                    "fold": None,
                    "train_dates": list(train_dates),
                    "val_dates": list(final_val_dates) if final_val_dates else None,
                    "val_score": self.val_score,
                    "score_direction": self.score_direction,
                    "score_name": _callable_name(self.val_score),
                },
            )
        self.refit_history = getattr(self.adapter, "last_fit_history", None)
        self.last_refit_train_dates = list(train_dates)
        self.last_refit_val_dates = (
            list(final_val_dates) if final_val_dates else None
        )
        return self.refit_history

    def _refit(
        self,
        params: dict[str, Any],
        memory_log: bool = False,
        memory_interval: float = 0.05,
    ) -> Any:
        self.refit(
            dates=self._all_train_dates(),
            params=params,
            memory_log=memory_log,
            memory_interval=memory_interval,
        )
        return self.model

    def _fit_transform(self, dates: Sequence[str]) -> Transform:
        src = self._src(dates, self.train_sampler, None, "fit")
        sampler = _sampler_key(self.train_sampler)
        if sampler is None:  # unserializable sampler: never reuse a fit
            return copy.deepcopy(self.transform).fit(src)
        key = (
            tuple(dates),
            sampler,
            self.polars_engine,
            self.precision,
            self.sample_weight_col,
        )
        if key not in self._transform_cache:
            transform = copy.deepcopy(self.transform)
            self._transform_cache[key] = transform.fit(src)
        return self._transform_cache[key]

    def _src(
        self,
        dates: Sequence[str],
        filters: Sequence[pl.Expr],
        transform: Transform | None,
        role: str,
    ) -> Any:
        # Array and adapter caches (Ridge/Lars stats, Lars path scores) reuse
        # results by this key, so it must name the sampler too; without a
        # serializable sampler nothing is cached.
        sampler = _sampler_key(filters)
        key = (
            (
                role,
                tuple(dates),
                sampler,
                id(transform),
                self.polars_engine,
                self.precision,
                self.sample_weight_col,
            )
            if sampler is not None
            else None
        )
        src = DataSource(
            dates=list(dates),
            loader=self.data_loader,
            target=self.target,
            features=self.features,
            filters=tuple(filters),
            transform=transform,
            sample_weight_col=self.sample_weight_col,
            cache=self._array_cache if self.cache_arrays else None,
            cache_key=key,
            polars_engine=self.polars_engine,
            precision=self.precision,
        )
        if self.data_source_wrapper is not None:
            return self.data_source_wrapper(src, role)
        return src

    def _fit_model(
        self,
        model: Any,
        train: DataSource,
        val: DataSource | None,
        trial: Any | None,
        fit_context: dict[str, Any] | None = None,
    ) -> Any:
        if self.sample_weight_col is not None and not getattr(
            self.adapter, "supports_sample_weight", False
        ):
            raise TypeError(
                f"{type(self.adapter).__name__} does not support sample weights"
            )
        sig = inspect.signature(self.adapter.fit)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        kwargs: dict[str, Any] = {}
        accepts_trial = "trial" in sig.parameters or accepts_kwargs
        if accepts_trial:
            kwargs["trial"] = trial
        if "fit_context" in sig.parameters or accepts_kwargs:
            kwargs["fit_context"] = fit_context
        return self.adapter.fit(model, train, val, self.tracker, **kwargs)

    def _evaluate(
        self,
        model: Any,
        src: DataSource,
        score: Callable[..., float],
        fold: Any,
        keep_predictions: bool = False,
    ) -> tuple[float, dict[str, Any], np.ndarray | None]:
        evaluate = getattr(self.adapter, "evaluate", None)
        if evaluate is None:
            return evaluate_model(
                self.adapter, model, src, score, fold, keep_predictions
            )
        return evaluate(model, src, score, fold=fold, keep_predictions=keep_predictions)

    def _all_train_dates(self) -> list[str]:
        return [date for chunk in self.rolling_dates for date in chunk]

    def _history_payload(self) -> dict[str, Any]:
        best_score = None
        best_trial = None
        n_trials = 0
        if self.study is not None:
            n_trials = len(getattr(self.study, "trials", []))
            try:
                best_score = float(self.study.best_value)
            except (ValueError, AttributeError):
                pass
            try:
                best_trial = int(self.study.best_trial.number)
            except (ValueError, AttributeError):
                pass
        return {
            "best_params": self.best_params,
            "best_score": best_score,
            "best_trial": best_trial,
            "score_direction": self.score_direction,
            "val_score": _callable_name(self.val_score),
            "n_trials": n_trials,
            "n_folds": max(0, len(self.rolling_dates) - 1),
            "refit_val_dates": self.refit_val_dates,
            "last_refit_train_dates": self.last_refit_train_dates,
            "last_refit_val_dates": self.last_refit_val_dates,
            "sample_weight_col": self.sample_weight_col,
            "validation_history": self.validation_history,
            "refit_history": self.refit_history,
        }

    @contextmanager
    def _memory_log(
        self,
        label: str,
        enabled: bool = False,
        interval: float = 0.05,
    ):
        if not enabled:
            yield
            return

        try:
            import psutil
        except ImportError as exc:
            raise ImportError("memory_log=True requires psutil.") from exc

        proc = psutil.Process()
        gc.collect()
        start = proc.memory_info().rss
        peak = start
        running = True

        def poll() -> None:
            nonlocal peak
            while running:
                peak = max(peak, proc.memory_info().rss)
                time.sleep(interval)

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        t0 = time.perf_counter()
        print(f"[mem] {label}: start rss={_gb(start):.2f} GB", flush=True)
        try:
            yield
        finally:
            running = False
            thread.join()
            end = proc.memory_info().rss
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            print(
                f"[mem] {label}: end rss={_gb(end):.2f} GB, "
                f"delta={_gb(end - start):+.2f} GB, "
                f"peak={_gb(peak):.2f} GB, "
                f"ru_maxrss={_gb(maxrss):.2f} GB, "
                f"time={time.perf_counter() - t0:.1f}s",
                flush=True,
            )


def _sampler_key(exprs: Sequence[pl.Expr]) -> tuple[str, ...] | None:
    """A hashable identity for sampler expressions, or None if one can't be
    serialized (callers then skip caching rather than risk a stale hit)."""
    try:
        return tuple(expr.meta.serialize(format="json") for expr in exprs)
    except Exception:
        return None


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _and_filter_groups(*groups: Sequence[pl.Expr]) -> tuple[pl.Expr, ...]:
    """AND filter groups while preserving OR semantics inside each group."""

    conjunctions: list[pl.Expr] = []
    for group in groups:
        expressions = list(group)
        if not expressions:
            continue
        expression = expressions[0]
        for item in expressions[1:]:
            expression = expression | item
        conjunctions.append(expression)
    if not conjunctions:
        return ()
    combined = conjunctions[0]
    for expression in conjunctions[1:]:
        combined = combined & expression
    return (combined,)


def _prediction_dates(
    dates: Sequence[str] | str | None,
    default: Sequence[str] | None,
    feature_source: Any,
) -> list[str]:
    values: Sequence[str] | str | None = dates
    if values is None and hasattr(feature_source, "dates"):
        values = feature_source.dates
    if values is None:
        values = default
    if values is None:
        raise ValueError(
            "prediction dates were not specified either on the pipeline, "
            "predict_frame(), or feature DataSource"
        )
    result = [values] if isinstance(values, str) else list(values)
    if not result:
        raise ValueError("prediction dates must contain at least one date")
    return [str(value) for value in result]


def _prediction_frame_loader(source: Any) -> Loader:
    lf = (
        pl.scan_parquet(source)
        if isinstance(source, (str, Path))
        else source.lazy() if isinstance(source, pl.DataFrame) else source
    )
    if not isinstance(lf, pl.LazyFrame):
        raise TypeError("feature frame must be a Polars DataFrame or LazyFrame")
    schema = lf.collect_schema()
    names = set(schema.names())

    def load(dates: list[str]) -> list[DateFrame]:
        if len(dates) > 1 and "date" not in names:
            raise ValueError(
                "a feature frame without a date column can only serve one date"
            )
        frames: list[DateFrame] = []
        for date in dates:
            date_lf = lf
            if "date" in names:
                date_expr = pl.col("date")
                if _prediction_base_type(schema["date"]) == pl.Datetime:
                    date_expr = date_expr.dt.date()
                date_lf = date_lf.filter(date_expr.cast(pl.String) == date)
            frames.append(DateFrame(date=date, tag="prediction", lf=date_lf))
        return frames

    return load


def _normalize_prediction_names(
    prediction_names: Sequence[str] | str | None,
) -> list[str] | None:
    if prediction_names is None:
        return None
    names = (
        [prediction_names]
        if isinstance(prediction_names, str)
        else list(prediction_names)
    )
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("prediction_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("prediction_names must be unique")
    return names


def _prediction_matrix(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(
            "predictions must be one- or two-dimensional, "
            f"got shape {values.shape}"
        )
    if values.shape[1] == 0:
        raise ValueError("predictions must contain at least one output column")
    return values


def _resolve_prediction_names(
    requested: Sequence[str] | None,
    width: int,
    adapter: Any,
) -> list[str]:
    if requested is not None:
        if len(requested) != width:
            raise ValueError(
                "prediction_names length does not match prediction width: "
                f"{len(requested)} != {width}"
            )
        return list(requested)
    if width == 1:
        return ["prediction"]

    quantiles = getattr(adapter, "quantiles", None)
    if quantiles is None or len(quantiles) != width:
        raise ValueError(
            f"model produced {width} prediction columns; pass prediction_names"
        )
    names = [_quantile_prediction_name(value) for value in quantiles]
    if len(set(names)) != len(names):
        raise ValueError("adapter quantiles do not produce unique prediction names")
    return names


def _prediction_names_without_output(
    requested: Sequence[str] | None,
    adapter: Any,
) -> list[str]:
    if requested is not None:
        return list(requested)
    quantiles = getattr(adapter, "quantiles", None)
    if quantiles is not None and len(quantiles) > 1:
        return _resolve_prediction_names(None, len(quantiles), adapter)
    return ["prediction"]


def _empty_prediction_frame(
    output_base: Sequence[str],
    prediction_names: Sequence[str],
    schema: Any,
    *,
    target: str,
    precision: str,
) -> pl.DataFrame:
    columns: dict[str, pl.Series] = {}
    for name in output_base:
        dtype = float_dtype(precision) if name == target else schema[name]
        columns[name] = pl.Series(name, [], dtype=dtype)
    for name in prediction_names:
        columns[name] = pl.Series(name, [], dtype=pl.Float64)
    return pl.DataFrame(columns)


def _quantile_prediction_name(value: Any) -> str:
    quantile = float(value)
    label = f"{quantile * 100.0:.12g}".replace("-", "m").replace(".", "p")
    return f"prediction_q{label}"


def _prediction_base_type(dtype: Any) -> Any:
    base_type = getattr(dtype, "base_type", None)
    return base_type() if callable(base_type) else dtype


def _prediction_zone(name: str) -> dt.tzinfo:
    return dt.timezone.utc if name.upper() == "UTC" else ZoneInfo(name)


def _coerce_prediction_bound(value: Any, dtype: Any, timezone: str | None) -> Any:
    if value is None:
        return None
    base_type = _prediction_base_type(dtype)
    if base_type == pl.Date:
        parsed = _parse_prediction_datetime(value)
        return parsed.date() if isinstance(parsed, dt.datetime) else parsed
    if base_type != pl.Datetime:
        return value

    parsed = _parse_prediction_datetime(value)
    if not isinstance(parsed, dt.datetime):
        raise TypeError(f"cannot coerce {value!r} to a datetime bound")
    source_tz = getattr(dtype, "time_zone", None)
    source_zone = _prediction_zone(source_tz) if source_tz is not None else None
    if parsed.tzinfo is None:
        if timezone is not None:
            parsed = parsed.replace(tzinfo=_prediction_zone(timezone))
            if source_zone is not None:
                return parsed.astimezone(source_zone)
            return parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
        if source_zone is not None:
            return parsed.replace(tzinfo=source_zone)
        return parsed
    if source_zone is not None:
        return parsed.astimezone(source_zone)
    return parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _parse_prediction_datetime(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    if isinstance(value, np.datetime64):
        value = np.datetime_as_string(value, unit="us")
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        return to_pydatetime()
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return dt.datetime.fromisoformat(text)
    return value


def _gb(n_bytes: int) -> float:
    return n_bytes / 1024**3


def _register_expr(
    registry: Registry, kind: str, name: str, expr: pl.Expr, run_hash: str
) -> dict[str, Any]:
    return register_expr(registry, kind, name, expr, run_hash)


def _exprs_from_refs(
    configs: Sequence[dict[str, Any]], registry: Registry | None
) -> list[pl.Expr]:
    exprs = []
    for cfg in configs:
        if "expr" in cfg:
            exprs.append(
                pl.Expr.deserialize(
                    io.StringIO(cfg["expr"]), format=cfg.get("format", "json")
                )
            )
            continue
        if registry is None:
            raise RuntimeError(
                f"cannot resolve filter {cfg.get('name')!r}: no registry available"
            )
        entry = registry.resolve("filter", cfg["name"], cfg["version"])
        if entry["fingerprint"] != cfg.get("fingerprint", entry["fingerprint"]):
            raise ValueError(
                f"registry fingerprint mismatch for filter {cfg['name']!r} "
                f"v{cfg['version']}"
            )
        exprs.append(
            pl.Expr.deserialize(
                io.StringIO(entry["expr"]), format=entry.get("format", "json")
            )
        )
    return exprs


def _feature_exprs(loader: Any) -> Mapping[str, pl.Expr]:
    seen: set[int] = set()
    while loader is not None and id(loader) not in seen:
        seen.add(id(loader))
        exprs = getattr(loader, "feature_exprs", None)
        if exprs:
            return exprs
        loader = getattr(loader, "loader", None)
    return {}


def _stateful_features(loader: Any) -> Mapping[str, Any]:
    seen: set[int] = set()
    while loader is not None and id(loader) not in seen:
        seen.add(id(loader))
        features = getattr(loader, "stateful_features", None)
        if features:
            return {feature.name: feature for feature in features}
        loader = getattr(loader, "loader", None)
    return {}


def _loader_feature_files(loader: Any, dates: Sequence[str]) -> list[Path]:
    seen: set[int] = set()
    while loader is not None and id(loader) not in seen:
        seen.add(id(loader))
        files = getattr(loader, "feature_files", None)
        if callable(files):
            return _paths(files(list(dates)))
        if files is not None:
            return _paths(files)

        path = getattr(loader, "path", None)
        if path is not None and Path(path).is_file():
            return [Path(path)]
        if path is not None and "{" in str(path):
            source = getattr(loader, "source", None)
            return [Path(Raw.resolve_path(date, path, source)[0]) for date in dates]
        loader = getattr(loader, "loader", None)
    return []


def _paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]


_PIPELINE_RUN_RE = re.compile(
    r"^pipeline_(?P<stamp>\d{8}T\d{4})_(?P<hash>[0-9a-fA-F]+)$"
)


def _pipeline_selector_key(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _resolve_pipeline_run(
    path: str | Path,
    *,
    stamp: Any = None,
    run_hash: str | None = None,
) -> Path:
    root = Path(path)
    if (root / "pipeline.json").exists():
        candidates = [root]
    else:
        candidates = sorted(
            candidate
            for candidate in root.glob("pipeline_*")
            if (candidate / "pipeline.json").exists()
        )
    stamp_prefix = _pipeline_selector_key(stamp)
    hash_prefix = str(run_hash or "").lower()
    if stamp_prefix or hash_prefix:
        selected = []
        for candidate in candidates:
            manifest = _read_json(candidate / "pipeline.json", {})
            run = manifest.get("run") or {}
            candidate_stamp = str(run.get("stamp") or "")
            candidate_hash = str(run.get("hash") or "")
            match = _PIPELINE_RUN_RE.fullmatch(candidate.name)
            if match is not None:
                parts = match.groupdict()
                candidate_stamp = candidate_stamp or parts["stamp"]
                candidate_hash = candidate_hash or parts["hash"]
            if stamp_prefix and not _pipeline_selector_key(candidate_stamp).startswith(
                stamp_prefix
            ):
                continue
            if hash_prefix and not candidate_hash.lower().startswith(hash_prefix):
                continue
            selected.append(candidate)
        candidates = selected
    if not candidates:
        raise FileNotFoundError(
            f"no pipeline manifest found under {root} for "
            f"stamp={stamp!r}, hash={run_hash!r}"
        )
    if stamp_prefix or hash_prefix:
        if len(candidates) != 1:
            raise ValueError(
                "pipeline stamp/hash matched multiple runs: "
                f"{[candidate.name for candidate in candidates]}"
            )
        return candidates[0]
    return candidates[-1]


def _adapter_from_manifest(manifest: Mapping[str, Any]) -> Any:
    adapter_meta = manifest.get("adapter") or {}
    module_name = adapter_meta.get("module")
    class_name = adapter_meta.get("class")
    if not module_name or not class_name:
        raise ValueError("saved pipeline manifest does not identify its adapter")
    if class_name == "TorchAdapter":
        return _saved_mlp_adapter(manifest, module_name, class_name)
    try:
        adapter_type = getattr(importlib.import_module(module_name), class_name)
        adapter = adapter_type()
    except (AttributeError, ImportError, TypeError) as exc:
        raise ValueError(
            f"cannot construct saved adapter {module_name}.{class_name}; "
            "pass adapter= explicitly"
        ) from exc

    if class_name == "XGBoostAdapter" and getattr(adapter, "quantiles", None) is None:
        text = str(adapter_meta.get("repr") or "")
        match = re.search(r"quantiles=(\[[^\]]*\]|None)", text)
        if match is not None and match.group(1) != "None":
            adapter.quantiles = ast.literal_eval(match.group(1))
    return adapter


def _saved_mlp_adapter(
    manifest: Mapping[str, Any], module_name: str, class_name: str
) -> Any:
    model_meta = manifest.get("model") or {}
    if model_meta.get("format") != "torch_state_dict":
        raise ValueError(
            "cannot infer a Torch adapter for a non-state-dict model; "
            "pass adapter= explicitly"
        )
    quantiles = _manifest_quantiles(manifest)
    if not quantiles:
        raise ValueError(
            "cannot infer saved MLP outputs from the manifest; pass adapter= explicitly"
        )
    try:
        adapter_type = getattr(importlib.import_module(module_name), class_name)
        import torch
    except (AttributeError, ImportError) as exc:
        raise ValueError(
            f"cannot construct saved adapter {module_name}.{class_name}; "
            "pass adapter= explicitly"
        ) from exc

    n_features = len(manifest.get("features") or [])

    def build_saved_mlp(params: dict[str, Any]) -> Any:
        torch.manual_seed(int(params.get("seed", 0)))
        n_layers = int(params.get("hidden_layers", 0))
        hidden_sizes = [
            int(params[f"hidden_units_l{idx}"]) for idx in range(1, n_layers + 1)
        ]
        activation_name = str(params.get("activation", "relu")).lower()
        activation_types = {
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "silu": torch.nn.SiLU,
            "tanh": torch.nn.Tanh,
        }
        if activation_name not in activation_types:
            raise ValueError(f"unsupported saved MLP activation: {activation_name!r}")
        dropout = float(params.get("dropout", 0.0))
        layers: list[Any] = []
        in_features = n_features
        for width in hidden_sizes:
            layers.append(torch.nn.Linear(in_features, width))
            layers.append(activation_types[activation_name]())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            in_features = width
        layers.append(torch.nn.Linear(in_features, len(quantiles)))
        model = torch.nn.Sequential(*layers)
        setattr(model, "_hidden_sizes", hidden_sizes)
        setattr(model, "_quantiles", tuple(quantiles))
        return model

    device = "cuda" if torch.cuda.is_available() else None
    adapter = adapter_type(module_builder=build_saved_mlp, device=device)
    adapter.quantiles = quantiles
    return adapter


def _manifest_quantiles(manifest: Mapping[str, Any]) -> list[float] | None:
    score_name = str(manifest.get("val_score") or "")
    if not score_name.startswith("pinball_"):
        return None
    try:
        return [float(value) for value in score_name.removeprefix("pinball_").split("_")]
    except ValueError:
        return None


def _unavailable_saved_loader(dates: list[str]) -> Any:
    raise RuntimeError(
        "this loaded pipeline has no data loader; pass data_loader= to "
        "Pipeline.from_saved() or supply feature_source to predict_frame()"
    )


def _read_json(path: Path, default: Any | None = None) -> Any:
    if default is not None and not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(_json_ready(obj), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _callable_name(fn: Any) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


def _object_info(obj: Any) -> dict[str, Any]:
    cls = type(obj)
    return {"class": cls.__name__, "module": cls.__module__, "repr": repr(obj)}
