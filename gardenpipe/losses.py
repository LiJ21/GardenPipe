from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


class PinballLoss(nn.Module):
    """Per-element pinball (quantile) loss over a ``(batch, quantile)`` output.

    Mirrors ``gardenpipe.score.pinball``: with ``reduction='none'`` the returned
    ``(B, Q)`` tensor averages -- over both axes -- to the same value the score
    reports, so the training loss and the validation metric are the same number.
    """

    def __init__(
        self, quantiles: Sequence[float] = QUANTILES, reduction: str = "none"
    ) -> None:
        super().__init__()
        values = tuple(float(q) for q in quantiles)
        if not values:
            raise ValueError("quantiles must be non-empty")
        if any(not 0.0 < q < 1.0 for q in values):
            raise ValueError("quantiles must lie strictly between 0 and 1")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.reduction = reduction
        self.register_buffer("quantiles", torch.tensor(values, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 2 or pred.shape[1] != self.quantiles.numel():
            raise ValueError(
                f"expected (B,{self.quantiles.numel()}) predictions, "
                f"got {tuple(pred.shape)}"
            )
        # The loss module is never moved with the model, so align it here.
        q = self.quantiles.to(device=pred.device, dtype=pred.dtype)
        err = target.reshape(-1, 1) - pred
        loss = torch.maximum(q * err, (q - 1.0) * err)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
