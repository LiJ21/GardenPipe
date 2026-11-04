from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Any
import inspect
import numpy as np

Score = Callable[..., float]
_NO_COMBINE = object()


@dataclass(frozen=True)
class ScoreValue:
    value: float
    n: int
    state: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.value


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    err = np.asarray(y_true) - np.asarray(y_pred)
    sse = float(np.dot(err, err))
    n = int(len(err))
    if combine_with is _NO_COMBINE:
        return float(np.sqrt(sse / n)) if n else 0.0
    if combine_with is not None:
        sse += float(getattr(combine_with, "state", {}).get("sse", 0.0))
        n += int(getattr(combine_with, "n", 0))
    return ScoreValue(float(np.sqrt(sse / n)) if n else 0.0, n, {"sse": sse})


def weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    sample_weight: np.ndarray | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if sample_weight is None:
        weight = np.ones(len(err), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.ndim != 1 or len(weight) != len(err):
            raise ValueError(
                "sample weight must be one-dimensional and match y_true"
            )
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("sample weight must be finite and nonnegative")
    sse = float(np.dot(weight, err * err))
    weight_sum = float(weight.sum())
    n = int(len(err))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        state = getattr(combine_with, "state", {})
        sse += float(state.get("sse", 0.0))
        weight_sum += float(
            state.get("weight_sum", getattr(combine_with, "n", 0))
        )
        n += int(getattr(combine_with, "n", 0))
    value = float(np.sqrt(sse / weight_sum)) if weight_sum else 0.0
    if combine_with is _NO_COMBINE:
        return value
    return ScoreValue(value, n, {"sse": sse, "weight_sum": weight_sum})


def weighted(score: Score) -> Score:
    if score is rmse:
        return weighted_rmse
    raise ValueError(
        f"no weighted implementation registered for {getattr(score, '__name__', score)!r}"
    )


def on_column(score: Score, index: int) -> Score:
    """Apply a scalar-prediction ``score`` to one column of a 2-D prediction.

    Multi-output models (quantile heads) predict ``(n, k)``; scores such as
    ``rmse`` expect one number per row. 1-D predictions pass through, so the
    wrapper stays valid if the head later collapses to a single output.
    Sample weights are forwarded only when ``score`` explicitly accepts them.
    """
    accepts_weight = "sample_weight" in inspect.signature(score).parameters

    def wrapped(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ctx: dict[str, Any] | None = None,
        combine_with: Any = _NO_COMBINE,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> float | ScoreValue:
        p = np.asarray(y_pred)
        if p.ndim > 1:
            p = p[:, index]
        if accepts_weight and sample_weight is not None:
            return score(
                y_true, p, ctx, combine_with=combine_with, sample_weight=sample_weight
            )
        return score(y_true, p, ctx, combine_with=combine_with)

    wrapped.__name__ = f"{getattr(score, '__name__', 'score')}_col{index}"
    return wrapped


def r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    err = y - np.asarray(y_pred, dtype=float)
    sse = float(np.dot(err, err))
    y_sum = float(y.sum())
    y2_sum = float(np.dot(y, y))
    n = int(len(y))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        state = getattr(combine_with, "state", {})
        sse += float(state.get("sse", 0.0))
        y_sum += float(state.get("y_sum", 0.0))
        y2_sum += float(state.get("y2_sum", 0.0))
        n += int(getattr(combine_with, "n", 0))
    score = _r2_from_stats(sse, y_sum, y2_sum, n)
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, {"sse": sse, "y_sum": y_sum, "y2_sum": y2_sum})


def _r2_from_stats(sse: float, y_sum: float, y2_sum: float, n: int) -> float:
    if n == 0:
        return 0.0
    sst = y2_sum - y_sum * y_sum / n
    if sst <= 0.0:
        return 1.0 if sse <= 0.0 else 0.0
    return float(1.0 - sse / sst)


def correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    state = {
        "y_sum": float(y.sum()),
        "p_sum": float(p.sum()),
        "y2_sum": float(np.dot(y, y)),
        "p2_sum": float(np.dot(p, p)),
        "yp_sum": float(np.dot(y, p)),
    }
    n = int(len(y))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        old = getattr(combine_with, "state", {})
        state = {k: v + float(old.get(k, 0.0)) for k, v in state.items()}
        n += int(getattr(combine_with, "n", 0))
    score = _corr_from_stats(n, **state)
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, state)


def _corr_from_stats(
    n: int,
    y_sum: float,
    p_sum: float,
    y2_sum: float,
    p2_sum: float,
    yp_sum: float,
) -> float:
    if n < 2:
        return 0.0
    cov = yp_sum - y_sum * p_sum / n
    y_var = y2_sum - y_sum * y_sum / n
    p_var = p2_sum - p_sum * p_sum / n
    denom = y_var * p_var
    return float(cov / np.sqrt(denom)) if denom > 0.0 else 0.0


def pinball(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: dict[str, Any] | None = None,
    quantiles: Sequence[float] = (0.5,),
    combine_with: Any = _NO_COMBINE,
) -> float | ScoreValue:
    y = np.asarray(y_true, dtype=float).reshape(-1, 1)
    p = np.asarray(y_pred, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    q = np.asarray(quantiles, dtype=float)
    if p.shape[1] != q.size:
        raise ValueError(
            f"y_pred has {p.shape[1]} columns for {q.size} quantiles"
        )
    err = y - p
    loss = float(np.sum(np.maximum(q * err, (q - 1.0) * err)))
    n = int(len(y))
    if combine_with is not _NO_COMBINE and combine_with is not None:
        loss += float(getattr(combine_with, "state", {}).get("loss", 0.0))
        n += int(getattr(combine_with, "n", 0))
    score = loss / (n * q.size) if n else 0.0
    if combine_with is _NO_COMBINE:
        return score
    return ScoreValue(score, n, {"loss": loss})


def get_pinball(quantiles: Sequence[float]) -> Score:
    q = tuple(float(x) for x in quantiles)

    def score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ctx: dict[str, Any] | None = None,
        combine_with: Any = _NO_COMBINE,
    ) -> float | ScoreValue:
        return pinball(y_true, y_pred, ctx, q, combine_with)

    score.__name__ = f"pinball_{'_'.join(f'{x:g}' for x in q)}"
    return score
