from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    def start_run(self, config: dict[str, Any]) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def finish(self) -> None: ...


@dataclass
class NullTracker:
    def start_run(self, config: dict[str, Any]) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


@dataclass
class TensorBoardTracker:
    log_dir: str | Path = "runs/tensorboard"
    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    flush_secs: int = 5
    use_explicit_step: bool = False
    step_key: str = "source_step"
    _writer: Any = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def start_run(self, config: dict[str, Any]) -> None:
        writer_cls = _summary_writer_cls()
        run_name = self.name or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(self.log_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._writer = writer_cls(log_dir=str(run_dir), flush_secs=self.flush_secs)
        self._step = 0
        merged = {**self.config, **config}
        if merged:
            self._writer.add_text("config", _json_text(merged), 0)
            self._writer.flush()

    def log_params(self, params: dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.add_text("params", _json_text(params), self._step)
            self._writer.flush()

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._writer is None:
            return
        log_step = step if self.use_explicit_step and step is not None else self._step
        if step is not None and not self.use_explicit_step:
            self._writer.add_scalar(self.step_key, step, log_step)
        for key, value in metrics.items():
            value = _scalar_value(value)
            if value is not None:
                self._writer.add_scalar(key, value, log_step)
        self._writer.flush()
        self._step += 1

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


@dataclass
class WandbTracker:
    project: str
    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    use_explicit_step: bool = False
    step_key: str = "source_step"
    _run: Any = field(default=None, init=False, repr=False)

    def start_run(self, config: dict[str, Any]) -> None:
        import wandb

        merged = {**self.config, **config}
        self._run = wandb.init(project=self.project, name=self.name, config=merged, **self.kwargs)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(params, allow_val_change=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            payload = dict(metrics)
            if step is not None and not self.use_explicit_step:
                payload[self.step_key] = step
                step = None
            self._run.log(payload, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def _summary_writer_cls() -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter

            return SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoardTracker requires tensorboard or tensorboardX. "
                "Install tensorboard to view logs locally."
            ) from exc


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _scalar_value(value: Any) -> float | int | None:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    return None
