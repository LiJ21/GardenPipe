"""Shared definitions for the Jena weather example: paths, data, features, folds."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from gardenpipe.data import FeatureLoader
from gardenpipe.features import EwmaFeature, LookbackFeature

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "jena"
RUNS_DIR = ROOT / "runs" / "weather"

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
SHA256 = "63d757501e92284a7de7cdbef0337f03b24e13ead7ac2b5b8c86a18d8e38ba5b"
YEARS = [str(y) for y in range(2009, 2017)]

# Short names for the fourteen measurements in the archive.
COLUMNS = {
    "Date Time": "time",
    "p (mbar)": "p",
    "T (degC)": "T",
    "Tpot (K)": "Tpot",
    "Tdew (degC)": "Tdew",
    "rh (%)": "rh",
    "VPmax (mbar)": "VPmax",
    "VPact (mbar)": "VPact",
    "VPdef (mbar)": "VPdef",
    "sh (g/kg)": "sh",
    "H2OC (mmol/mol)": "H2OC",
    "rho (g/m**3)": "rho",
    "wv (m/s)": "wv",
    "max. wv (m/s)": "max_wv",
    "wd (deg)": "wd",
}
WIND_SENTINEL = -9999.0

HORIZON = "6h"
TARGET = "change_6h"  # T(t + 6h) - T(t)

# Expanding folds over yearly partitions, and the untouched test years.
ROLLING_DATES = [["2009", "2010", "2011", "2012"], ["2013"], ["2014"]]
TEST_DATES = ["2015", "2016"]

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
BATCH_ROWS = 20_000  # several streamed batches per yearly partition
SEED = 0


def lag_col(col: str, lag: str) -> str:
    return f"{col}_lag_{lag}"


def exact_lag(col: str, lag: str) -> pl.Expr:
    """``col`` exactly ``lag`` earlier; null when that observation is missing."""

    then = pl.col("time") - pl.duration(hours=int(lag.removesuffix("h")))
    at = pl.col("time").search_sorted(then)
    return pl.when(pl.col("time").gather(at) == then).then(pl.col(col).gather(at))


def _minutes_of_day() -> pl.Expr:
    t = pl.col("time")
    return t.dt.hour().cast(pl.Int32) * 60 + t.dt.minute().cast(pl.Int32)


def feature_exprs() -> dict[str, pl.Expr]:
    """Row-local features of both recipes, plus an untransformed copy of T."""

    tau = 2 * 3.141592653589793
    day = _minutes_of_day() / 1440
    year = (pl.col("time").dt.ordinal_day() - 1) / 365.25
    wd = pl.col("wd").radians()
    return {
        # T is a model feature, so predictions come back with T standardized;
        # T_raw is not a feature and keeps degrees Celsius for the baselines.
        "T_raw": pl.col("T"),
        "u": pl.col("wv") * wd.cos(),
        "v": pl.col("wv") * wd.sin(),
        "tod_sin": (day * tau).sin(),
        "tod_cos": (day * tau).cos(),
        "doy_sin": (year * tau).sin(),
        "doy_cos": (year * tau).cos(),
    }


def lag_features() -> tuple[LookbackFeature, ...]:
    """Differences against earlier observations; each carries ``period`` of history."""

    def lag(name: str, expr: pl.Expr, period: str) -> LookbackFeature:
        return LookbackFeature(name, expr, period=period)

    return (
        lag("dT_1h", pl.col("T") - exact_lag("T", "1h"), "1h"),
        lag("dT_3h", pl.col("T") - exact_lag("T", "3h"), "3h"),
        lag("dT_24h", pl.col("T") - exact_lag("T", "24h"), "24h"),
        lag("T18_minus_T", exact_lag("T", "18h") - pl.col("T"), "18h"),
        lag("dp_3h", pl.col("p") - exact_lag("p", "3h"), "3h"),
        # Not a model feature: the seasonal baseline's forecast.
        lag(lag_col("T", "18h"), exact_lag("T", "18h"), "18h"),
    )


def ewm_features() -> tuple[EwmaFeature, ...]:
    return tuple(
        EwmaFeature(f"{col}_ewm_{hl}", pl.col(col), half_life=hl)
        for col in ("T", "p")
        for hl in ("1h", "6h")
    )


def stateful_features() -> tuple[LookbackFeature | EwmaFeature, ...]:
    return (*lag_features(), *ewm_features())


RECIPES = {
    "calendar": ["T", "p", "rh", "u", "v", "tod_sin", "tod_cos", "doy_sin", "doy_cos"],
}
RECIPES["history"] = [
    *RECIPES["calendar"],
    "dT_1h",
    "dT_3h",
    "dT_24h",
    "T18_minus_T",
    "dp_3h",
    *(f.name for f in ewm_features()),
]

# Inputs any compared model or baseline needs.
REQUIRED = [*RECIPES["history"], "T_raw", lag_col("T", "18h")]


def eligible() -> pl.Expr:
    """One row-local rule, used unchanged as the train, val and test sampler.

    Sampler tuples OR their entries, so this must stay a single expression.
    """

    t = pl.col("time")
    return (
        (t.dt.minute() == 0)
        & (t.dt.second() == 0)
        & (t.dt.ordinal_day() > 1)  # lag and EWMA warm-up after the yearly restart
        & pl.col(TARGET).is_not_null()
        & pl.all_horizontal(pl.col(c).is_not_null() for c in REQUIRED)
    )


def make_loader(data_dir: Path = DATA_DIR) -> FeatureLoader:
    return FeatureLoader(
        path=f"{data_dir}/jena_{{date}}.parquet",  # {date} is "2009" .. "2016"
        feature_exprs=feature_exprs(),
        stateful_features=stateful_features(),
        targets={TARGET: pl.col("T")},
        horizons=HORIZON,
        target_tolerance="0s",  # exact timestamps only; gaps give null labels
    )
