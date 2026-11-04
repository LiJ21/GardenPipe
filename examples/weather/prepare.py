"""Download, clean and partition the Jena archive into yearly parquet files.

    python examples/weather/prepare.py            # cached after the first run
    python examples/weather/prepare.py --force    # rebuild from the download

Everything that needs the whole series at once lives here: sorting,
duplicates and sentinels. The pipeline does the rest, lags included.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import resource
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    COLUMNS,
    DATA_DIR,
    SHA256,
    URL,
    WIND_SENTINEL,
    YEARS,
)

STEP = dt.timedelta(minutes=10)
END = dt.datetime(2017, 1, 1)  # the experiment covers 2009-01-01 .. 2016-12-31


def download(raw_dir: Path) -> Path:
    path = raw_dir / URL.rsplit("/", 1)[-1]
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {SHA256}")
    return path


def read_raw(zip_path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        (name,) = [n for n in zf.namelist() if n.endswith(".csv")]
        df = pl.read_csv(io.BytesIO(zf.read(name)))
    return df.rename(COLUMNS).with_columns(
        pl.col("time").str.to_datetime("%d.%m.%Y %H:%M:%S")
    )


def clean(raw: pl.DataFrame, end: dt.datetime = END) -> tuple[pl.DataFrame, dict]:
    """Sort, drop exact duplicates, reject conflicts, null the wind sentinels."""

    df = raw.sort("time", maintain_order=True)
    deduped = df.unique(keep="first", maintain_order=True)
    conflicts = deduped.filter(pl.col("time").is_duplicated())
    if conflicts.height:
        times = conflicts.get_column("time").unique().sort().to_list()
        raise ValueError(f"conflicting records at {len(times)} timestamps: {times[:5]}")
    dropped = deduped.filter(pl.col("time") >= end)
    out = deduped.filter(pl.col("time") < end).with_columns(
        pl.when(pl.col(c) == WIND_SENTINEL).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("wv", "max_wv")
    )
    stats = {
        "raw_rows": raw.height,
        "raw_in_time_order": bool(raw.get_column("time").is_sorted()),
        "duplicate_rows_removed": df.height - deduped.height,
        "conflicting_timestamps": 0,
        "wind_sentinels": {
            c: int((raw.get_column(c) == WIND_SENTINEL).sum()) for c in ("wv", "max_wv")
        },
        "dropped_after_end": [str(t) for t in dropped.get_column("time").to_list()],
        "rows": out.height,
        "first_time": str(out.get_column("time").min()),
        "last_time": str(out.get_column("time").max()),
    }
    return out, stats


def gaps(df: pl.DataFrame) -> list[dict]:
    t = df.get_column("time")
    step = t.diff()
    out = []
    for i in (step > STEP).arg_true().to_list():
        missing = int(step[i] / STEP) - 1
        out.append({"before": str(t[i - 1]), "after": str(t[i]), "missing_rows": missing})
    return out


def quality(df: pl.DataFrame, gap_list: list[dict]) -> dict:
    per_year = {}
    for year in YEARS:
        part = df.filter(pl.col("time").dt.year() == int(year))
        start = dt.datetime(int(year), 1, 1)
        slots = int((dt.datetime(int(year) + 1, 1, 1) - start) / STEP)
        per_year[year] = {
            "rows": part.height,
            # The archive starts at 00:10, so 2009 lacks one slot by design.
            "missing_rows": slots - part.height,
            "hourly_issue_times": part.filter(pl.col("time").dt.minute() == 0).height,
            "nulls": {c: n for c, n in part.null_count().row(0, named=True).items() if n},
        }
    return {"gaps": gap_list, "per_year": per_year}


def clock_evidence(raw: pl.DataFrame, df: pl.DataFrame, gap_list: list[dict]) -> dict:
    """Is there summer time, and which offset fits the diurnal cycle?"""

    def last_sunday(year: int, month: int) -> dt.date:
        day = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
        return day - dt.timedelta(days=(day.weekday() + 1) % 7)

    years = range(2009, 2017)
    spring = {last_sunday(y, 3) for y in years}
    autumn = {last_sunday(y, 10) for y in years}
    gap_days = {dt.date.fromisoformat(g["before"][:10]) for g in gap_list}
    repeated = raw.filter(pl.col("time").is_duplicated()).get_column("time").dt.date()

    by_slot = (
        df.with_columns(
            pl.col("time").dt.month().alias("month"),
            (pl.col("time").dt.hour().cast(pl.Int32) * 60 + pl.col("time").dt.minute())
            .alias("minute_of_day"),
        )
        .group_by("month", "minute_of_day")
        .agg(pl.col("T").mean())
    )

    def extreme(month: int, fn: str) -> str:
        rows = by_slot.filter(pl.col("month") == month).sort("T", descending=fn == "max")
        minute = int(rows.get_column("minute_of_day")[0])
        return f"{minute // 60:02d}:{minute % 60:02d}"

    return {
        "spring_forward_gaps": len(spring & gap_days),
        "autumn_repeated_hours": int(repeated.is_in(sorted(autumn)).sum()),
        "diurnal_minimum": {"june": extreme(6, "min"), "december": extreme(12, "min")},
        "diurnal_maximum": {"june": extreme(6, "max"), "december": extreme(12, "max")},
        "conclusion": (
            "No summer-time transitions; the diurnal cycle fits CET (UTC+1) better "
            "than UTC. This is inference, not metadata: timestamps are kept naive."
        ),
    }


def write_partitions(df: pl.DataFrame, out_dir: Path, key: pl.Expr) -> list[str]:
    """Write one parquet file per partition key, ``jena_<key>.parquet``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    keyed = df.with_columns(key.alias("__key"))
    keys = keyed.get_column("__key").unique(maintain_order=True).to_list()
    for k in keys:
        keyed.filter(pl.col("__key") == k).drop("__key").write_parquet(
            out_dir / f"jena_{k}.parquet"
        )
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="rebuild cached files")
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    files = [args.data_dir / f"jena_{y}.parquet" for y in YEARS]
    if not args.force and manifest_path.exists() and all(f.exists() for f in files):
        if json.loads(manifest_path.read_text()).get("sha256") == SHA256:
            print(f"prepared data is cached in {args.data_dir} (use --force to rebuild)")
            return

    t0 = time.perf_counter()
    raw = read_raw(download(args.data_dir / "raw"))
    df, stats = clean(raw)
    gap_list = gaps(df)
    keys = write_partitions(df, args.data_dir, pl.col("time").dt.year().cast(pl.String))
    if keys != YEARS:
        raise ValueError(f"expected partitions {YEARS}, got {keys}")

    report = quality(df, gap_list)
    (args.data_dir / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "url": URL,
        "sha256": SHA256,
        "interval": "2009-01-01 to 2016-12-31, ten-minute observations",
        "policy": [
            "stable sort by timestamp",
            "remove exact duplicate records; reject conflicting records",
            f"wind sentinels {WIND_SENTINEL:g} in wv and max_wv become null",
            "drop observations on or after 2017-01-01",
            "no imputation",
        ],
        "counts": stats,
        "gaps": gap_list,
        "clock": clock_evidence(raw, df, gap_list),
        "files": [f.name for f in files],
        "license": "CC BY 4.0, Max Planck Institute for Biogeochemistry, Jena",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    runtime = {
        "seconds": time.perf_counter() - t0,  # includes the download on a first run
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (args.data_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"wrote {len(files)} yearly files, {df.height:,} rows, to {args.data_dir}")


if __name__ == "__main__":
    main()
