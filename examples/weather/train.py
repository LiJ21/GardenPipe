"""Tune and test streaming Ridge on both feature recipes; write predictions.

    python examples/weather/train.py

Each recipe gets the same folds, eligibility rule and alpha grid. The fitted
pipelines are saved, and one prediction frame holds both baselines and both
models on identical issue times.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ALPHAS,
    BATCH_ROWS,
    DATA_DIR,
    HORIZON,
    RECIPES,
    ROLLING_DATES,
    RUNS_DIR,
    SEED,
    TARGET,
    TEST_DATES,
    eligible,
    lag_col,
    make_loader,
)

from gardenpipe.model import RidgeAdapter  # noqa: E402
from gardenpipe.pipeline import Pipeline  # noqa: E402
from gardenpipe.score import rmse  # noqa: E402
from gardenpipe.search import grid  # noqa: E402
from gardenpipe.transform import Standardizer  # noqa: E402

KEYS = ["time", "T_raw", lag_col("T", "18h")]


def build_pipeline(
    recipe: str,
    loader,
    rolling_dates: list[list[str]] = ROLLING_DATES,
    test_dates: list[str] = TEST_DATES,
    batch_rows: int = BATCH_ROWS,
    alphas: list[float] = ALPHAS,
) -> Pipeline:
    features = RECIPES[recipe]
    rule = (eligible(),)
    return Pipeline(
        rolling_dates=rolling_dates,
        test_dates=test_dates,
        adapter=RidgeAdapter(batch_size=batch_rows),
        target=TARGET,
        features=features,
        data_loader=loader,
        transform=Standardizer(features),  # fitted per fold on training rows
        train_sampler=rule,
        val_sampler=rule,
        test_sampler=rule,
        search_space={"alpha": grid(alphas)},
        search_sampler="grid",
        n_trials=len(alphas),
        val_score=rmse,
        score_direction="minimize",
        seed=SEED,
        polars_engine="in-memory",  # batches are small; files are still streamed
    )


def prediction_frame(pipes: dict[str, Pipeline], dates: list[str]) -> pl.DataFrame:
    """Baselines and model predictions as temperatures at the target time."""

    frames = {
        name: pipe.predict_frame(dates, key_cols=KEYS, time_col="time")
        for name, pipe in pipes.items()
    }
    base, *rest = frames.values()
    for other in rest:  # one eligibility rule, so identical issue times
        if not other.select(KEYS).equals(base.select(KEYS)):
            raise RuntimeError("recipes were scored on different rows")
    out = base.select(
        pl.col("time").alias("issue_time"),
        (pl.col("time") + pl.duration(hours=int(HORIZON.removesuffix("h")))).alias(
            "target_time"
        ),
        pl.col("T_raw").alias("T_now"),
        (pl.col("T_raw") + pl.col(TARGET)).alias("T_observed"),
        pl.col("T_raw").alias("pred_persistence"),
        pl.col(lag_col("T", "18h")).alias("pred_seasonal"),
    )
    return out.with_columns(
        (pl.col("T_now") + frame.get_column("prediction")).alias(f"pred_{name}")
        for name, frame in frames.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    args = parser.parse_args()
    if not (args.data_dir / "manifest.json").exists():
        raise SystemExit(f"no prepared data in {args.data_dir}; run prepare.py first")
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(args.data_dir)
    pipes: dict[str, Pipeline] = {}
    validation: dict[str, dict] = {}
    durations: dict[str, float] = {}
    for recipe in RECIPES:
        t0 = time.perf_counter()
        pipe = build_pipeline(recipe, loader, batch_rows=args.batch_rows)
        result = pipe.train()
        durations[f"train_{recipe}"] = time.perf_counter() - t0
        pipe.save_pipeline(args.runs_dir / "pipelines" / recipe)
        pipes[recipe] = pipe
        validation[recipe] = {
            "best_alpha": result["best_params"]["alpha"],
            "val_rmse": result["best_score"],
            "trials": [
                {"alpha": t.params["alpha"], "val_rmse": t.value}
                for t in sorted(pipe.study.trials, key=lambda t: t.params["alpha"])
            ],
        }

    selected = min(validation, key=lambda r: validation[r]["val_rmse"])
    t0 = time.perf_counter()
    preds = prediction_frame(pipes, TEST_DATES)
    durations["predict_test"] = time.perf_counter() - t0
    preds.write_parquet(args.runs_dir / "predictions.parquet")

    write = lambda name, obj: (args.runs_dir / name).write_text(  # noqa: E731
        json.dumps(obj, indent=2) + "\n"
    )
    write(
        "run_config.json",
        {
            "rolling_dates": ROLLING_DATES,
            "test_dates": TEST_DATES,
            "horizon": HORIZON,
            "target": TARGET,
            "recipes": RECIPES,
            "alphas": ALPHAS,
            "batch_rows": args.batch_rows,
            "seed": SEED,
            "eligibility": str(eligible()),
        },
    )
    write("validation.json", {"selected_recipe": selected, "recipes": validation})
    write(
        "runtime.json",
        {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "numpy": np.__version__,
            "seconds": durations,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )
    for recipe, v in validation.items():
        print(f"{recipe}: alpha={v['best_alpha']:g}, validation RMSE {v['val_rmse']:.3f}")
    print(f"validation selects the {selected} recipe; {preds.height:,} test issue times")
    print(f"wrote predictions and run records to {args.runs_dir}")


if __name__ == "__main__":
    main()
