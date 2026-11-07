"""Several streams per date: each computed alone, joined per time window."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from gardenpipe.data import (
    DataSource,
    FeatureLoader,
    LoadHint,
    MultiSourceLoader,
    _epoch_ns,
    _stateful_internal_cols,
    _time_span,
    _time_windows,
    _union_window_ends,
    scan_labels,
)
from gardenpipe.features import CarryForwardFeature, EwmaFeature, LookbackFeature
from gardenpipe.model import RidgeAdapter
from gardenpipe.pipeline import Pipeline
from gardenpipe.score import rmse
from gardenpipe.search import grid
from gardenpipe.transform import Standardizer

T0 = dt.datetime(2026, 1, 5)
DATES = ["2026-01-05", "2026-01-06", "2026-01-07"]


def ticks(seed: int, labels: dict[str, int], span: int = 60) -> pl.DataFrame:
    """Irregular whole-second ticks: ties inside a label and across labels."""
    rng = np.random.default_rng(seed)
    parts = []
    for label, n in labels.items():
        secs = np.sort(rng.integers(0, span, size=n))
        x = rng.normal(size=n)
        x[rng.random(n) < 0.1] = np.nan
        parts.append(
            pl.DataFrame(
                {
                    "time": [T0 + dt.timedelta(seconds=int(s)) for s in secs],
                    "label": [label] * n,
                    "x": x,
                }
            ).with_columns(pl.col("x").fill_nan(None))
        )
    return (
        pl.concat(parts)
        .with_columns(pl.col("time").cast(pl.Datetime("ns")))
        .sort("time", maintain_order=True)
    )


def public(df: pl.DataFrame, loader: FeatureLoader) -> pl.DataFrame:
    """Drop what the library keeps internal: helper columns, sub-feature outputs."""
    internal = set(_stateful_internal_cols(loader.stateful_features))
    return df.select(c for c in df.columns if c not in internal)


def collect_windows(loader, date: str, batch_size: int, **hint) -> list[pl.DataFrame]:
    frames = loader.iter_date_frames([date], hint=LoadHint(batch_size=batch_size, **hint))
    return [frame.lf.collect() for frame in frames]


def wide_reference(streams: dict[str, pl.DataFrame], tick: str | None = None) -> pl.DataFrame:
    """Chained full joins of each stream's last row per timestamp, then a fill."""
    out = None
    for label, df in streams.items():
        df = df.drop("label", strict=False).group_by("time", maintain_order=True).last()
        if tick is not None:
            df = df.with_columns(pl.lit(True).alias(tick))
        df = df.rename({c: f"{label.lower()}__{c}" for c in df.columns if c != "time"})
        out = df if out is None else out.join(df, on="time", how="full", coalesce=True)
    out = out.sort("time")
    ticks_ = [c for c in out.columns if tick is not None and c.endswith(f"__{tick}")]
    fill = [c for c in out.columns if c != "time" and c not in ticks_]
    out = out.with_columns(pl.col(fill).forward_fill())
    return out.with_columns(pl.col(ticks_).fill_null(False)) if ticks_ else out


def assert_same(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    assert sorted(actual.columns) == sorted(expected.columns), (
        actual.columns,
        expected.columns,
    )
    assert_frame_equal(actual, expected.select(actual.columns))


class TimeWindowTest(unittest.TestCase):
    def windows(self, times: list[int], batch_size: int, chunk: int):
        chunks = [pl.Series(times[i : i + chunk]) for i in range(0, len(times), chunk)]
        return list(_time_windows(chunks, batch_size, "t"))

    def test_windows_hold_whole_timestamps(self) -> None:
        times = [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6]
        for batch_size in range(1, 8):
            for chunk in (1, 2, 4, 16):
                with self.subTest(batch_size=batch_size, chunk=chunk):
                    windows = self.windows(times, batch_size, chunk)
                    self.assertEqual(windows[0][0], 0)
                    self.assertEqual(windows[-1][1], len(times))
                    for (_, stop, _), (start, _, _) in zip(windows, windows[1:]):
                        self.assertEqual(stop, start)
                    for start, stop, end in windows:
                        rows = times[start:stop]
                        self.assertEqual(end, rows[-1])
                        if stop < len(times):  # the next window starts later
                            self.assertLess(rows[-1], times[stop])
                        self.assertTrue(len(rows) <= batch_size or len(set(rows)) == 1)

    def test_unsorted_or_null_times_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "not sorted"):
            self.windows([0, 2, 1, 3], 2, 4)
        with self.assertRaisesRegex(ValueError, "not sorted"):
            self.windows([0, 2, 1, 3], 2, 2)  # across chunks
        with self.assertRaisesRegex(ValueError, "null"):
            list(_time_windows([pl.Series([1, None, 3])], 2, "t"))

    def test_union_windows_bound_rows_and_keep_timestamps_whole(self) -> None:
        rng = np.random.default_rng(3)
        files = [np.sort(rng.integers(0, 1_000, size=n)) for n in (300, 120, 500)]
        files[1] = np.sort(np.concatenate([files[1], np.full(40, 777)]))  # a burst
        union = np.sort(np.concatenate(files))
        for batch_size in (7, 30, 100, 1_000):
            with self.subTest(batch_size=batch_size):
                ends = _union_window_ends(
                    [pl.LazyFrame({"t": f}) for f in files], batch_size, 1, "auto"
                )
                self.assertEqual(ends, sorted(set(ends)))
                self.assertEqual(ends[-1], union[-1])
                lo = -1
                for end in ends:
                    rows = union[(union > lo) & (union <= end)]
                    self.assertGreater(len(rows), 0)
                    self.assertTrue(len(rows) <= batch_size or len(set(rows)) == 1)
                    lo = end


class TimeSpanTest(unittest.TestCase):
    """The window filter of both loaders compares the bare column, in its own
    type, with bounds given in epoch ns. A bound between two ticks of a coarse
    column is rounded, and the rows must still be those of an exact comparison:
    before 1970 too, where rounding down and rounding towards zero differ."""

    KINDS = {
        "ns": pl.Datetime("ns"),
        "us": pl.Datetime("us"),
        "ms": pl.Datetime("ms"),
        "ms, UTC": pl.Datetime("ms", "UTC"),
        "us, Asia/Tokyo": pl.Datetime("us", "Asia/Tokyo"),
        "date": pl.Date,
        "int64": pl.Int64,
    }

    def column(self, ns: pl.Series, dtype: pl.DataType) -> pl.Series:
        if dtype == pl.Int64:
            return ns.cast(pl.Int64)
        if getattr(dtype, "time_zone", None):
            zoned = ns.dt.replace_time_zone("UTC").dt.convert_time_zone(dtype.time_zone)
            return zoned.dt.cast_time_unit(dtype.time_unit)
        return ns.cast(dtype)

    def test_rows_match_an_exact_comparison(self) -> None:
        day = 86_400 * 10**9
        rng = np.random.default_rng(5)
        for era, start in (
            ("2026", T0),
            ("1960", dt.datetime(1960, 1, 5)),
            ("across 1970", dt.datetime(1969, 12, 28)),
        ):
            t0 = (start - dt.datetime(1970, 1, 1)) // dt.timedelta(microseconds=1) * 1_000
            ns = pl.Series(t0 + np.sort(rng.integers(0, 10 * day, 200))).cast(pl.Datetime("ns"))
            for kind, dtype in self.KINDS.items():
                with self.subTest(era=era, kind=kind):
                    lf = pl.LazyFrame({"time": self.column(ns, dtype)}).with_row_index("i")
                    exact = lf.select(_epoch_ns("time")).collect().to_series().to_numpy()
                    # On a tick, a nanosecond either side, half a millisecond
                    # either side, and anywhere in the span or just outside it.
                    on = rng.choice(exact, 40)
                    bounds = np.concatenate(
                        [on + step for step in (0, -1, 1, -500_000, 500_000)]
                        + [rng.integers(exact[0] - day, exact[-1] + day, 40)]
                    )
                    for _ in range(100):
                        lo, hi = (int(b) for b in np.sort(rng.choice(bounds, 2)))
                        after = None if rng.random() < 0.2 else lo
                        rows = lf.filter(_time_span(lf, "time", after, hi)).collect()
                        keep = exact <= hi if after is None else (exact > after) & (exact <= hi)
                        self.assertEqual(
                            rows["i"].to_list(), np.flatnonzero(keep).tolist(), (after, hi)
                        )


class SharedFileTest(unittest.TestCase):
    """One file per date holds every label; each label is its own stream."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        days = [
            ticks(1, {"A": 40, "B": 35, "C": 30}),
            ticks(2, {"A": 40, "B": 35}),  # C has no rows
            ticks(3, {"A": 30, "B": 30, "C": 20, "D": 10}),
        ]
        # C goes quiet for a stretch, so it is absent from some windows.
        days[0] = days[0].filter(
            ~((pl.col("label") == "C") & pl.col("time").dt.second().is_between(20, 45))
        )
        for date, day in zip(DATES, days):
            day.write_parquet(root / f"{date}.parquet")
        self.loader = FeatureLoader(
            path=f"{root}/{{date}}.parquet",
            feature_exprs={"x2": pl.col("x") * 2},
            stateful_features=(
                EwmaFeature("x_ewm", pl.col("x"), half_life="5s"),
                # Needs the previous row of the same label, carried as pad.
                EwmaFeature("dx_ewm", pl.col("x").diff(), half_life="5s"),
                CarryForwardFeature("x_last", pl.col("x")),
                EwmaFeature(
                    "tree",
                    pl.col("x"),
                    half_life="3s",
                    sub_features=(CarryForwardFeature("tree_last", pl.col("x")),),
                ),
                LookbackFeature("x_lag4", pl.col("x").shift(4), rows=4),
            ),
            targets={"fwd": pl.coalesce("x", "x_last")},
            horizons="4s",
            filters=(pl.col("x") > -1.5,),
            front_pad=1,
        )

    def alone(self, date: str, label: str) -> pl.DataFrame:
        """The label's rows through the base loader alone, in one pass."""
        only = replace(
            self.loader, filters=((pl.col("x") > -1.5) & (pl.col("label") == label),)
        )
        return public(only.load_date(date).lf.collect(), only)

    def labels(self, date: str) -> list[str]:
        return scan_labels(self.loader, [date], "label")

    def test_long_streams_match_each_label_alone(self) -> None:
        loader = MultiSourceLoader(["A", "B", "C"], self.loader, label_col="label", layout="long")
        for date in DATES[:2]:
            expected = pl.concat(
                [self.alone(date, label) for label in self.labels(date) if label != "D"]
            ).sort(["time", "label"], maintain_order=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # C has no rows
                whole = loader.load_dates([date])[0].lf.collect()
            assert_same(whole, expected)
            for batch_size in (4, 9, 25):
                for workers in (1, 4):
                    with self.subTest(date=date, batch_size=batch_size, workers=workers):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            windows = collect_windows(
                                replace(loader, workers=workers), date, batch_size
                            )
                        self.assertGreater(len(windows), 2)
                        for before, after in zip(windows, windows[1:]):
                            self.assertLess(before["time"].max(), after["time"].min())
                        assert_same(pl.concat(windows), expected)

    def test_wide_matches_a_join_of_each_label_alone(self) -> None:
        loader = MultiSourceLoader(["A", "B", "C"], self.loader, label_col="label", tick_col="tick")
        schemas = []
        for date in DATES[:2]:
            empty = pl.DataFrame(schema=self.alone(DATES[0], "A").schema)
            streams = {
                label: self.alone(date, label) if label in self.labels(date) else empty
                for label in ("A", "B", "C")
            }
            expected = wide_reference(streams, tick="tick")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                whole = loader.load_dates([date])[0].lf.collect()
            assert_same(whole, expected)
            schemas.append(whole.schema)
            for batch_size in (4, 9, 25):
                with self.subTest(date=date, batch_size=batch_size):
                    windows = collect_windows(loader, date, batch_size)
                    self.assertGreater(len(windows), 2)
                    assert_same(pl.concat(windows), expected)
        # C has no rows on the second date: typed null columns, same schema.
        self.assertEqual(schemas[0], schemas[1])

    def test_cross_features_and_tick_markers(self) -> None:
        cross = {
            "difference": pl.col("a__x") - pl.col("b__x"),
            "mean_x": pl.mean_horizontal("a__x", "b__x", "c__x"),
        }
        loader = MultiSourceLoader(
            ["A", "B", "C"], self.loader, label_col="label", tick_col="tick", feature_exprs=cross
        )
        streams = {label: self.alone(DATES[0], label) for label in ("A", "B", "C")}
        expected = wide_reference(streams, tick="tick").with_columns(
            expr.alias(name) for name, expr in cross.items()
        )
        for batch_size in (None, 5, 13):
            with self.subTest(batch_size=batch_size):
                frames = (
                    [loader.load_dates([DATES[0]])[0].lf.collect()]
                    if batch_size is None
                    else collect_windows(loader, DATES[0], batch_size)
                )
                out = pl.concat(frames)
                assert_same(out, expected)
        # A marker is true exactly on the label's own (kept) timestamps.
        own = set(streams["A"]["time"].to_list())
        marked = set(out.filter(pl.col("a__tick"))["time"].to_list())
        self.assertEqual(marked, own)
        self.assertTrue((out.filter(~pl.col("a__tick"))["a__x"].null_count()) < out.height)

    def test_long_layout_joins_cross_features_back(self) -> None:
        cross = {"difference": pl.col("a__x") - pl.col("b__x")}
        loader = MultiSourceLoader(
            ["A", "B"], self.loader, label_col="label", layout="long", feature_exprs=cross
        )
        streams = {label: self.alone(DATES[0], label) for label in ("A", "B")}
        wide = wide_reference(streams).with_columns(cross["difference"].alias("difference"))
        expected = (
            pl.concat(list(streams.values()))
            .sort(["time", "label"], maintain_order=True)
            .join(wide.select("time", "difference"), on="time", how="left", maintain_order="left")
        )
        assert_same(loader.load_dates([DATES[0]])[0].lf.collect(), expected)
        assert_same(pl.concat(collect_windows(loader, DATES[0], 6)), expected)

    def test_long_layout_without_sources_loads_every_label(self) -> None:
        loader = MultiSourceLoader([], self.loader, label_col="label", layout="long")
        date = DATES[2]  # D appears only here
        expected = pl.concat(
            [self.alone(date, label) for label in ("A", "B", "C", "D")]
        ).sort(["time", "label"], maintain_order=True)
        assert_same(loader.load_dates([date])[0].lf.collect(), expected)
        assert_same(pl.concat(collect_windows(loader, date, 7)), expected)

    def test_listed_label_without_rows_warns_once(self) -> None:
        loader = MultiSourceLoader(["A", "C"], self.loader, label_col="label")
        with self.assertWarnsRegex(RuntimeWarning, "'C' has no rows on 2026-01-06"):
            loader.load_dates([DATES[1]])[0].lf.collect()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            collect_windows(loader, DATES[1], 5)  # already reported

    def test_layout_needs_listed_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "sources must list"):
            MultiSourceLoader([], self.loader, label_col="label")
        with self.assertRaisesRegex(ValueError, "duplicate column prefixes"):
            MultiSourceLoader(["a", "A"], self.loader, label_col="label")

    def test_pipeline_fits_the_same_model_windowed_or_whole(self) -> None:
        loader = MultiSourceLoader(
            ["A", "B", "C"],
            self.loader,
            label_col="label",
            tick_col="tick",
            feature_exprs={"difference": pl.col("a__x") - pl.col("b__x")},
        )
        features = ["a__x_ewm", "b__x_ewm", "b__dx_ewm", "difference"]
        sampler = (
            pl.col("a__tick")
            & pl.col("a__fwd").is_not_null()
            & pl.all_horizontal(pl.col(features).is_not_null()),
        )

        def fit(batch_size: int | None) -> Pipeline:
            pipe = Pipeline(
                rolling_dates=[[DATES[0]], [DATES[1]]],
                test_dates=[DATES[2]],
                adapter=RidgeAdapter(batch_size=batch_size),
                target="a__fwd",
                features=features,
                data_loader=loader,
                transform=Standardizer(features),
                train_sampler=sampler,
                val_sampler=sampler,
                test_sampler=sampler,
                search_space={"alpha": grid([1.0])},
                search_sampler="grid",
                n_trials=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pipe.train()
            return pipe

        windowed, whole = fit(5), fit(None)
        np.testing.assert_allclose(windowed.model.coef, whole.model.coef)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertAlmostEqual(
                windowed.test(score=rmse)["test_score"], whole.test(score=rmse)["test_score"]
            )
            frame = windowed.predict_frame([DATES[2]], key_cols=["time"])
        self.assertEqual(frame.height, whole.test(score=rmse)["n"])


class PerFileTest(unittest.TestCase):
    """Each source has its own file, schema and loader; windows span them all."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        for i, date in enumerate(DATES[:2]):
            station_a = ticks(10 + i, {"A": 50}).select(
                "time",
                pl.col("x").alias("temperature"),
                (pl.col("x").abs() * 10).alias("weight"),
            )
            station_b = ticks(20 + i, {"B": 70}).select(
                "time",
                (pl.col("x") - 0.1).alias("temperature_low"),
                (pl.col("x") + 0.1).alias("temperature_high"),
            )
            station_a.write_parquet(root / f"station_a_{date}.parquet")
            station_b.write_parquet(root / f"station_b_{date}.parquet")
        self.loaders = {
            "station_a": FeatureLoader(
                path=f"{root}/station_a_{{date}}.parquet",
                feature_exprs={"weighted_temperature": pl.col("temperature") * pl.col("weight")},
                stateful_features=(
                    EwmaFeature("temperature_ewm", pl.col("temperature"), half_life="5s"),
                    EwmaFeature("temperature_change_ewm", pl.col("temperature").diff(), half_life="5s"),
                ),
                targets={"fwd": pl.col("temperature")},
                horizons="3s",
                front_pad=1,
            ),
            "station_b": FeatureLoader(
                path=f"{root}/station_b_{{date}}.parquet",
                feature_exprs={
                    "temperature_mean": (pl.col("temperature_low") + pl.col("temperature_high")) / 2
                },
                stateful_features=(
                    EwmaFeature(
                        "temperature_mean_ewm",
                        (pl.col("temperature_low") + pl.col("temperature_high")) / 2,
                        half_life="5s",
                    ),
                ),
            ),
        }

    def test_windows_match_whole_date_and_each_feed_alone(self) -> None:
        cross = {
            "temperature_difference": pl.col("station_a__temperature")
            - pl.col("station_b__temperature_mean")
        }
        loader = MultiSourceLoader(
            ["station_a", "station_b"], self.loaders, tick_col="tick", feature_exprs=cross
        )
        for date in DATES[:2]:
            streams = {
                name: public(feed.load_date(date).lf.collect(), feed)
                for name, feed in self.loaders.items()
            }
            expected = wide_reference(streams, tick="tick").with_columns(
                cross["temperature_difference"].alias("temperature_difference")
            )
            assert_same(loader.load_dates([date])[0].lf.collect(), expected)
            for batch_size in (6, 20, 200):
                with self.subTest(date=date, batch_size=batch_size):
                    windows = collect_windows(loader, date, batch_size)
                    if batch_size < 100:
                        self.assertGreater(len(windows), 2)
                    assert_same(pl.concat(windows), expected)

    def test_long_layout_adds_the_source(self) -> None:
        loader = MultiSourceLoader(["station_a", "station_b"], self.loaders, layout="long")
        out = pl.concat(collect_windows(loader, DATES[0], 9))
        self.assertEqual(sorted(set(out["source"])), ["station_a", "station_b"])
        self.assertEqual(out.height, 50 + 70)
        self.assertTrue(out["time"].is_sorted())

    def test_data_source_streams_windows(self) -> None:
        loader = MultiSourceLoader(["station_a", "station_b"], self.loaders)
        source = DataSource(
            dates=DATES[:2],
            loader=loader,
            target="station_a__fwd",
            features=["station_a__temperature_ewm", "station_b__temperature_mean_ewm"],
            filters=(pl.col("station_a__fwd").is_not_null(),),
        )
        whole = pl.concat(list(source.dataframe_batches()))
        for engine in ("streaming", "in-memory"):
            for multicollect in (-1, 1):
                with self.subTest(engine=engine, multicollect=multicollect):
                    batches = replace(source, polars_engine=engine).dataframe_batches(
                        10, multicollect=multicollect
                    )
                    batched = list(batches)
                    self.assertGreater(len(batched), 4)
                    assert_frame_equal(pl.concat(batched), whole)



class MixedUnitTest(unittest.TestCase):
    """One file per stream, each in its own time unit. Window ends come from
    every stream, so an end from the nanosecond stream falls between the ticks
    of the others, and their bounds are rounded: no row may be lost or repeated,
    before 1970 too."""

    UNITS = {"A": "ns", "B": "us", "C": "ms"}

    def write(self, root: Path, start: dt.datetime, rng: np.random.Generator) -> None:
        t0 = (start - dt.datetime(1970, 1, 1)) // dt.timedelta(microseconds=1) * 1_000
        for label, unit in self.UNITS.items():
            # 300 rows in one second, so a coarse stream often has a row
            # just after an end that came from a finer one.
            ns = t0 + np.sort(rng.integers(0, 10**9, 300))
            pl.DataFrame(
                {
                    "time": pl.Series(ns).cast(pl.Datetime("ns")).cast(pl.Datetime(unit)),
                    "level": rng.normal(size=300).cumsum(),
                }
            ).write_parquet(root / f"{label}_{DATES[0]}.parquet", row_group_size=32)

    def test_windows_hold_every_row_of_each_stream(self) -> None:
        rng = np.random.default_rng(11)
        for era, start in (("2026", T0), ("1960", dt.datetime(1960, 1, 5))):
            with tempfile.TemporaryDirectory() as tmp:
                self.write(Path(tmp), start, rng)
                for horizon in ("100ms", "1500us"):
                    base = FeatureLoader(
                        path=f"{tmp}/{{source}}_{{date}}.parquet",
                        feature_exprs={"level": pl.col("level")},
                        targets={"y": pl.col("level")},
                        horizons=horizon,
                    )
                    loader = MultiSourceLoader(list(self.UNITS), base, layout="long")
                    whole = loader.load_dates([DATES[0]])[0].lf.collect()
                    for batch_size in (7, 50):
                        with self.subTest(era=era, horizon=horizon, batch_size=batch_size):
                            windows = collect_windows(loader, DATES[0], batch_size)
                            self.assertGreater(len(windows), 2)
                            out = pl.concat(windows)
                            # The long layout puts every stream in the coarsest
                            # unit, where rows of different streams can tie, so
                            # only the order within each stream is compared.
                            for label in self.UNITS:
                                part = out.filter(pl.col("source") == label)
                                self.assertEqual(part.height, 300)
                                assert_frame_equal(part, whole.filter(pl.col("source") == label))


class StreamEdgeTest(unittest.TestCase):
    """Windows that a stream's filters empty, and joins that windows cannot keep."""

    def frame(self, rows: dict[str, list[float]], seconds: list[int]) -> pl.DataFrame:
        return pl.concat(
            pl.DataFrame(
                {"time": [T0 + dt.timedelta(seconds=s) for s in seconds], "label": label, "x": x}
            )
            for label, x in rows.items()
        ).with_columns(pl.col("time").cast(pl.Datetime("ns"))).sort("time", maintain_order=True)

    def assert_windows_match_whole(self, loader: MultiSourceLoader, batch_size: int) -> None:
        whole = loader.load_dates([DATES[0]])[0].lf.collect()
        assert_frame_equal(pl.concat(collect_windows(loader, DATES[0], batch_size)), whole)

    def test_filtered_out_window_keeps_nested_state(self) -> None:
        # B has rows in the middle window but the filter drops them all; its
        # EWMA must not lose its own state while its look-back keeps history.
        df = self.frame(
            {"A": [1.0] * 6, "B": [10.0, 20.0, -99.0, -99.0, 40.0, 50.0]}, list(range(6))
        )
        base = FeatureLoader(
            frames={DATES[0]: df},
            filters=(pl.col("x") > 0,),
            stateful_features=(
                EwmaFeature(
                    "e",
                    pl.col("x") + 0 * pl.col("lag").fill_null(0),
                    half_life="1s",
                    sub_features=(LookbackFeature("lag", pl.col("x").shift(1), rows=1),),
                ),
            ),
        )
        loader = MultiSourceLoader(["A", "B"], base, label_col="label", forward_fill=False)
        self.assert_windows_match_whole(loader, 4)

    def test_carried_pad_rows_are_not_filtered_again(self) -> None:
        # A filter over a timestamp keeps the larger of each pair; alone in
        # the next window, the carried pad row would fail it.
        df = self.frame({"A": [1.0, 5.0, 3.0, 2.0, 8.0, 4.0, 6.0, 7.0]}, [0, 0, 1, 1, 2, 2, 3, 3])
        base = FeatureLoader(
            frames={DATES[0]: df},
            front_pad=1,
            filters=(pl.col("x") > pl.col("x").min().over("time"),),
            stateful_features=(EwmaFeature("e", pl.col("x").diff(), half_life="1s"),),
        )
        loader = MultiSourceLoader(["A"], base, label_col="label")
        self.assert_windows_match_whole(loader, 2)

    def test_windows_need_time_in_on(self) -> None:
        df = self.frame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, [0, 1]).with_columns(id=pl.lit(1))
        loader = MultiSourceLoader(
            ["A", "B"], FeatureLoader(frames={DATES[0]: df}), label_col="label", on=("id",)
        )
        with self.assertRaisesRegex(ValueError, "must start with the time column"):
            collect_windows(loader, DATES[0], 2)

    def test_rows_run_in_the_order_of_on(self) -> None:
        # The file lists id before time, and ids run against time: sorted by
        # id, forward fill would copy A's value at 2s back to 1s.
        df = pl.DataFrame(
            {"id": [2, 1], "time": [T0 + dt.timedelta(seconds=s) for s in (1, 2)], "label": ["B", "A"], "x": [1.0, 2.0]}
        ).with_columns(pl.col("time").cast(pl.Datetime("ns")))
        loader = MultiSourceLoader(
            ["A", "B"], FeatureLoader(frames={DATES[0]: df}), label_col="label", on=("time", "id")
        )
        whole = loader.load_dates([DATES[0]])[0].lf.collect()
        self.assertEqual(whole["id"].to_list(), [2, 1])
        self.assertEqual(whole["a__x"].to_list(), [None, 2.0])
        self.assert_windows_match_whole(loader, 2)
        with self.assertRaisesRegex(ValueError, "must start with the time column"):
            collect_windows(replace(loader, on=("id", "time")), DATES[0], 2)

    def test_prefixed_columns_must_differ(self) -> None:
        df = self.frame({"A": [1.0], "A__B": [2.0]}, [0]).with_columns(b__x=pl.col("x"))
        loader = MultiSourceLoader(["A", "A__B"], FeatureLoader(frames={DATES[0]: df}), label_col="label")
        with self.assertRaisesRegex(ValueError, "same prefixed columns"):
            loader.load_dates([DATES[0]])[0].lf.collect()


if __name__ == "__main__":
    unittest.main()
