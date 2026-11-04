"""Score the test predictions and write the report.

    python examples/weather/report.py

All models are scored on the same issue times. Errors are in degrees Celsius
at the target time; skill is relative to persistence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA_DIR, RUNS_DIR, TEST_DATES  # noqa: E402

MODELS = {
    "persistence": "Persistence, T(t+6h) = T(t)",
    "seasonal": "Daily persistence, T(t+6h) = T(t-18h)",
    "calendar": "Ridge, calendar recipe",
    "history": "Ridge, history recipe",
}
# Fixed in advance, not chosen after looking at results.
WEEKS = {"winter": ("2015-01-12", "2015-01-19"), "summer": ("2015-07-13", "2015-07-20")}


def scores(preds: pl.DataFrame, by: str | None = None) -> pl.DataFrame:
    """MAE, RMSE and bias per model, optionally per group."""

    exprs = []
    for m in MODELS:
        err = pl.col(f"pred_{m}") - pl.col("T_observed")
        exprs += [
            err.abs().mean().alias(f"{m}_mae"),
            (err**2).mean().sqrt().alias(f"{m}_rmse"),
            err.mean().alias(f"{m}_bias"),
        ]
    exprs.append(pl.len().alias("n"))
    if by is None:
        return preds.select(exprs)
    return preds.group_by(by, maintain_order=True).agg(exprs).sort(by)


def summary(preds: pl.DataFrame) -> dict:
    row = scores(preds).row(0, named=True)
    out = {"n": row["n"], "models": {}}
    for m, label in MODELS.items():
        out["models"][m] = {
            "label": label,
            "mae": row[f"{m}_mae"],
            "rmse": row[f"{m}_rmse"],
            "bias": row[f"{m}_bias"],
            "mae_skill": 1 - row[f"{m}_mae"] / row["persistence_mae"],
            "rmse_skill": 1 - row[f"{m}_rmse"] / row["persistence_rmse"],
        }
    return out


def plot_weeks(preds: pl.DataFrame, out_dir: Path) -> list[str]:
    try:
        import plotly  # noqa: F401
    except ImportError:
        print("plotly is not installed (uv sync --extra viz); skipping HTML plots")
        return []
    from gardenpipe.viewer import DataViewer, Panel, Series

    written = []
    for name, (start, end) in WEEKS.items():
        bounds = [pl.lit(day).str.to_datetime("%Y-%m-%d") for day in (start, end)]
        week = preds.filter(pl.col("target_time").is_between(*bounds))
        viewer = DataViewer(
            panels=[
                Panel.from_source(
                    week,
                    [
                        Series("T_observed", kind="line", name="observed"),
                        Series("pred_history", kind="line", name="Ridge, history"),
                        Series("pred_calendar", kind="line", name="Ridge, calendar"),
                        Series("pred_persistence", kind="line", name="persistence"),
                    ],
                    x="target_time",
                    title=f"{name} week: temperature (°C) at the target time, "
                    "forecast issued six hours earlier",
                )
            ],
            x_label="target time",
        )
        path = out_dir / f"week_{name}.html"
        viewer.to_html(path)
        written.append(path.name)
    return written


def markdown(
    metrics: dict, monthly: pl.DataFrame, validation: dict, runtime: dict
) -> str:
    lines = [
        "# Jena six-hour temperature forecast: test results",
        "",
        f"Test years {', '.join(TEST_DATES)}; {metrics['test']['n']:,} hourly issue "
        f"times, {metrics['coverage']:.1%} of all hourly issue times in those years. "
        "Every model is scored on the same issue times.",
        "",
        "| Model | MAE (°C) | RMSE (°C) | Bias (°C) | MAE skill | RMSE skill |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in metrics["test"]["models"].values():
        lines.append(
            f"| {m['label']} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['bias']:+.3f} "
            f"| {m['mae_skill']:+.1%} | {m['rmse_skill']:+.1%} |"
        )
    lines += [
        "",
        "Skill is `1 - error / persistence error`. Hourly forecast errors are "
        "strongly correlated, so these are point estimates, not independent samples.",
        "",
        "## Validation (2013 and 2014 folds)",
        "",
        "| Recipe | Best alpha | Validation RMSE (°C) |",
        "| --- | ---: | ---: |",
    ]
    for recipe, v in validation["recipes"].items():
        lines.append(f"| {recipe} | {v['best_alpha']:g} | {v['val_rmse']:.3f} |")
    lines += [
        "",
        f"Validation selects the **{validation['selected_recipe']}** recipe; "
        "both are reported on the test years.",
        "",
        "## MAE by month (°C)",
        "",
        "| Month | n | " + " | ".join(MODELS) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in MODELS) + " |",
    ]
    for row in monthly.iter_rows(named=True):
        cells = " | ".join(f"{row[f'{m}_mae']:.2f}" for m in MODELS)
        lines.append(f"| {row['month']} | {row['n']:,} | {cells} |")
    lines += ["", "## Runtime", ""]
    for stage, r in runtime.items():
        seconds = r["seconds"]
        total = sum(seconds.values()) if isinstance(seconds, dict) else seconds
        lines.append(f"- {stage}: {total:.1f} s, peak RSS {r['peak_rss_mib']:.0f} MiB")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    preds = pl.read_parquet(args.runs_dir / "predictions.parquet")
    validation = json.loads((args.runs_dir / "validation.json").read_text())
    quality = json.loads((args.data_dir / "quality.json").read_text())
    hourly = sum(quality["per_year"][y]["hourly_issue_times"] for y in TEST_DATES)
    runtime = {
        stage: json.loads(path.read_text())
        for stage, path in (
            ("preparation", args.data_dir / "runtime.json"),
            ("training and prediction", args.runs_dir / "runtime.json"),
        )
        if path.exists()
    }

    issue = pl.col("issue_time")
    monthly = scores(preds.with_columns(issue.dt.month().alias("month")), "month")
    yearly = scores(preds.with_columns(issue.dt.year().alias("year")), "year")
    metrics = {
        "test": summary(preds),
        "coverage": preds.height / hourly,
        "issue_times": [str(preds["issue_time"].min()), str(preds["issue_time"].max())],
        "target_times": [str(preds["target_time"].min()), str(preds["target_time"].max())],
        "by_year": yearly.to_dicts(),
        "by_month": monthly.to_dicts(),
        "selected_recipe": validation["selected_recipe"],
    }
    (args.runs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = markdown(metrics, monthly, validation, runtime)
    (args.runs_dir / "report.md").write_text(report)
    plots = plot_weeks(preds, args.runs_dir)

    print(report.split("\n## ")[0])
    print(f"wrote metrics.json, report.md{''.join(', ' + p for p in plots)} to {args.runs_dir}")


if __name__ == "__main__":
    main()
