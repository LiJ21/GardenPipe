"""Minimal GardenPipe run: per-day parquet files -> streamed features -> ridge.

    python examples/minimal.py                     # small, a few seconds
    python examples/minimal.py --rows 4000000 --data /tmp/gp   # write big days
    python examples/minimal.py --data /tmp/gp                  # train on them

Peak memory is set by ``BATCH_ROWS`` and ``BUFFER_BYTES``, not by the number of
days or rows on disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import resource
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from gardenpipe.buffered import BufferConfig, buffered_wrapper
from gardenpipe.data import FeatureLoader, expand_dates
from gardenpipe.features import EwmaFeature
from gardenpipe.model import RidgeAdapter
from gardenpipe.pipeline import Pipeline
from gardenpipe.score import r2

BATCH_ROWS = 100_000  # rows per streamed batch
BUFFER_BYTES = 64 << 20  # prefetch budget across worker threads


def write_day(path: Path, day: str, rows: int, seed: int) -> None:
    """A mean-reverting sensor reading sampled evenly over one day."""

    rng = np.random.default_rng(seed)
    # AR(1): level[i] = 0.995 * level[i - 1] + noise[i]
    noise = pl.Series(rng.normal(size=rows))
    level = noise.ewm_mean(alpha=0.005, adjust=False) / 0.005
    start = dt.datetime.fromisoformat(day)
    step = dt.timedelta(seconds=86_400 / rows)
    pl.DataFrame(
        {
            "time": pl.datetime_range(
                start, start + step * (rows - 1), step, eager=True, time_unit="ns"
            ),
            "level": level,
        }
    ).write_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000, help="rows per day")
    parser.add_argument("--days", type=int, default=10, help="number of days")
    parser.add_argument(
        "--data", type=Path, help="keep generated files here and reuse them"
    )
    args = parser.parse_args()

    dates = expand_dates("20260105-20261231")[: args.days]
    k = max(1, len(dates) // 5)  # 2/5 first train, 1/5 per val fold, rest test
    with tempfile.TemporaryDirectory() as tmp:
        root = args.data or Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        for i, day in enumerate(dates):
            path = root / f"sensor_{day}.parquet"
            if not path.exists():  # reuse files kept with --data
                write_day(path, day, args.rows, seed=i)

        # 1. How to read one date and what to compute from it.
        loader = FeatureLoader(
            path=f"{root}/sensor_{{date}}.parquet",
            feature_exprs={"level": pl.col("level")},
            stateful_features=(
                EwmaFeature("level_ewm_1m", pl.col("level"), half_life="1m"),
            ),
            targets={"change_5m": pl.col("level")},  # level(t + 5m) - level(t)
            horizons="5m",
        )
        keep = pl.col("change_5m").is_not_null()

        # 2. Walk-forward folds over dates, a streaming model, bounded prefetch.
        pipe = Pipeline(
            rolling_dates=[dates[: 2 * k], dates[2 * k : 3 * k], dates[3 * k : 4 * k]],
            test_dates=dates[4 * k :],
            adapter=RidgeAdapter(batch_size=BATCH_ROWS),
            target="change_5m",
            features=["level", "level_ewm_1m"],
            data_loader=loader,
            train_sampler=(keep,),
            val_sampler=(keep,),
            test_sampler=(keep,),
            val_score=r2,
            score_direction="maximize",
            n_trials=1,
            data_source_wrapper=buffered_wrapper(
                BufferConfig(workers=2, max_buffer_bytes=BUFFER_BYTES)
            ),
        )
        pipe.train()
        result = pipe.test(score=r2, keep_predictions=False)

        disk = sum(p.stat().st_size for p in root.glob("sensor_*.parquet"))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"test R^2: {result['test_score']:.3f} on {result['n']:,} rows")
        print(f"data on disk: {disk / 2**20:.0f} MiB, peak RSS: {peak:.0f} MiB")


if __name__ == "__main__":
    main()
