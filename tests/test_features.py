from __future__ import annotations

import datetime as dt
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from gardenpipe.data import (
    DataSource,
    DateFrame,
    FeatureLoader,
    LoadHint,
    MultiSourceLoader,
    Raw,
    expand_dates,
)
from gardenpipe.features import (
    CarryForwardFeature,
    EwmaFeature,
    ExprFeature,
    LookbackFeature,
    compute_features,
    to_features,
)
from gardenpipe.filters import time_of_day
from gardenpipe.targets import add_forward_change, add_forward_value
from gardenpipe.transform import FunctionTransform, Standardizer

T0 = dt.datetime(2026, 1, 5)


def series(values: list[float | None], step_s: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [T0 + dt.timedelta(seconds=i * step_s) for i in range(len(values))],
            "x": values,
        }
    ).with_columns(pl.col("time").cast(pl.Datetime("ns")))


class ToFeaturesTest(unittest.TestCase):
    def test_frame_keeps_kind_and_filters_before_features(self) -> None:
        df = series([1.0, -2.0, 3.0])
        exprs = {"x2": pl.col("x") * 2}

        eager = to_features(df, exprs, filters=(pl.col("x") > 0,))
        lazy = to_features(df.lazy(), exprs, meta_cols=("time",))

        self.assertIsInstance(eager, pl.DataFrame)
        self.assertEqual(eager["x2"].to_list(), [2.0, 6.0])
        self.assertIsInstance(lazy, pl.LazyFrame)
        self.assertEqual(lazy.collect().columns, ["time", "x2"])

    def test_path_yields_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.parquet"
            series([float(i) for i in range(10)]).write_parquet(path)

            batches = list(to_features(path, {"x2": pl.col("x") * 2}, batch_size=4))

        self.assertEqual(sum(b.height for b in batches), 10)
        self.assertEqual(pl.concat(batches)["x2"].to_list()[-1], 18.0)

    def test_compute_features_selects_time_and_features(self) -> None:
        out = compute_features(series([1.0, 2.0]).lazy(), {"y": pl.col("x") + 1})
        self.assertEqual(out.collect().columns, ["time", "y"])


class TargetsTest(unittest.TestCase):
    def test_forward_value_is_asof_and_null_past_the_end(self) -> None:
        out = add_forward_value(series([1.0, 2.0, 3.0, 4.0]), pl.col("x"), "1500ms")
        self.assertEqual(out["forward_value"].to_list(), [2.0, 3.0, None, None])

    def test_forward_value_averages_weighted_horizons(self) -> None:
        out = add_forward_value(
            series([0.0, 10.0, 20.0, 30.0]), pl.col("x"), ["1s", "2s"], [3.0, 1.0]
        )
        self.assertEqual(out["forward_value"].to_list()[:2], [12.5, 22.5])

    def test_forward_value_stays_within_groups(self) -> None:
        df = pl.concat(
            [
                series([1.0, 2.0]).with_columns(pl.lit("a").alias("g")),
                series([10.0, 20.0]).with_columns(pl.lit("b").alias("g")),
            ]
        )
        out = add_forward_value(df, pl.col("x"), "1s", by=("g",))
        self.assertEqual(out["forward_value"].to_list(), [2.0, None, 20.0, None])

    def test_forward_change_kinds(self) -> None:
        df = series([2.0, 4.0])
        diff = add_forward_change(df, pl.col("x"), "1s")
        pct = add_forward_change(df, pl.col("x"), "1s", kind="pct", scale=100.0)
        log = add_forward_change(df, pl.col("x"), "1s", kind="log")

        self.assertEqual(diff["forward_change"][0], 2.0)
        self.assertEqual(pct["forward_change"][0], 100.0)
        self.assertAlmostEqual(log["forward_change"][0], math.log(2.0))
        self.assertIsNone(diff["forward_change"][1])
        with self.assertRaises(ValueError):
            add_forward_change(df, pl.col("x"), "1s", kind="ratio")


    def test_zero_tolerance_nulls_inexact_horizons(self) -> None:
        df = pl.DataFrame(
            {
                "time": [T0 + dt.timedelta(seconds=s) for s in (0, 1, 2, 5, 6)],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("ns")))
        asof = add_forward_value(df, pl.col("x"), "1s")
        exact = add_forward_change(df, pl.col("x"), "1s", tolerance="0s")

        self.assertEqual(asof["forward_value"].to_list(), [2.0, 3.0, 3.0, 5.0, None])
        self.assertEqual(exact["forward_change"].to_list(), [1.0, 1.0, None, 1.0, None])


class DatesAndPathsTest(unittest.TestCase):
    def test_expand_dates(self) -> None:
        self.assertEqual(
            expand_dates("20260102-20260105"),
            ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        )
        self.assertEqual(
            expand_dates("20260102-20260105", exclude="weekends"),
            ["2026-01-02", "2026-01-05"],
        )
        self.assertEqual(
            expand_dates("20260102-20260105", exclude=lambda d: d.day == 2, end_date=False),
            ["2026-01-03", "2026-01-04"],
        )

    def test_resolve_path_extracts_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sensor_a").mkdir()
            (Path(tmp) / "sensor_a" / "20260105_warm.run.parquet").touch()
            template = f"{tmp}/{{source_lower}}/{{date_compact}}_{{tag}}.run.parquet"

            path, tag = Raw.resolve_path("2026-01-05", template, "SENSOR_A")

            self.assertEqual(Path(path).name, "20260105_warm.run.parquet")
            self.assertEqual(tag, "warm")
            with self.assertRaises(ValueError):
                Raw.resolve_path("2026-01-05", template)


class FeatureLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        values = [math.sin(i / 3) if i % 5 else None for i in range(23)]
        for date, tag in (("2026-01-05", "warm"), ("2026-01-06", "cold")):
            series(values, step_s=0.5).write_parquet(
                Path(self._tmp.name) / f"{date}_{tag}.parquet"
            )
        self.loader = FeatureLoader(
            path=f"{self._tmp.name}/{{date}}_{{tag}}.parquet",
            feature_exprs={"x2": pl.col("x") * 2},
            stateful_features=(
                EwmaFeature("x_ewm", pl.col("x"), half_life="2s"),
                CarryForwardFeature("x_last", pl.col("x")),
            ),
            # A forward-filled x: the history crosses batches, so it comes
            # from the carry-forward feature rather than x.forward_fill().
            targets={"fwd": pl.coalesce(pl.col("x"), pl.col("x_last"))},
            horizons="1500ms",
        )
        self.source = DataSource(
            dates=["2026-01-05", "2026-01-06"],
            loader=self.loader,
            target="fwd",
            features=["x2", "x_ewm"],
        )

    def test_frames_carry_tags(self) -> None:
        frames = self.loader(["2026-01-05", "2026-01-06"])
        self.assertEqual([f.tag for f in frames], ["warm", "cold"])

    def test_windowed_batches_match_single_pass(self) -> None:
        full = pl.concat(list(self.source.dataframe_batches()))
        for batch_size in (3, 4, 5, 7):
            for front_pad in (0, 3):
                with self.subTest(batch_size=batch_size, front_pad=front_pad):
                    batched = list(
                        self.source.dataframe_batches(batch_size, front_pad=front_pad)
                    )
                    self.assertGreater(len(batched), 2)
                    assert_frame_equal(pl.concat(batched), full)

    def test_requires_path(self) -> None:
        with self.assertRaises(ValueError):
            FeatureLoader()


class SelectedFeatureCarryoverTest(unittest.TestCase):
    """Model inputs need not include every stateful feature the loader computes."""

    def source(self, **kwargs) -> DataSource:
        loader = FeatureLoader(
            frames={"2026-01-05": series([float(i) for i in range(1, 18)])},
            stateful_features=(
                EwmaFeature("e1", pl.col("x"), half_life="2s"),
                EwmaFeature("e2", pl.col("e1"), half_life="2s"),
            ),
            targets={"fwd": pl.col("e2")},
            horizons="1s",
        )
        return DataSource(
            dates=["2026-01-05"], loader=loader, target="fwd", features=["e2"],
            **kwargs,
        )

    def test_unselected_intermediate_keeps_its_state(self) -> None:
        full = self.source().frame().collect()
        for engine in ("streaming", "in-memory"):
            source = self.source(polars_engine=engine)
            for batch_size in (2, 5):
                for multicollect in (-1, 1):
                    with self.subTest(
                        engine=engine, batch_size=batch_size, multicollect=multicollect
                    ):
                        batches = list(source.dataframe_batches(batch_size, multicollect))
                        self.assertGreater(len(batches), 1)
                        self.assertTrue(all(batch.height <= batch_size for batch in batches))
                        assert_frame_equal(pl.concat(batches), full)
                        self.assertNotIn("e1", batches[0].columns)

    def test_intermediate_state_survives_sampler_and_transform(self) -> None:
        # Drop complete windows and rewrite both stateful columns. Carryover
        # still comes from every row's original feature values.
        source = self.source(
            filters=(~pl.col("x").is_between(3.0, 8.0),),
            transform=Standardizer(
                cols=["e1", "e2"],
                means={"e1": 10.0, "e2": 20.0},
                stds={"e1": 2.0, "e2": 3.0},
            ),
        )
        full = source.frame().collect()
        for multicollect in (-1, 1):
            with self.subTest(multicollect=multicollect):
                batches = list(source.dataframe_batches(2, multicollect))
                self.assertTrue(all(batch.height for batch in batches))
                assert_frame_equal(pl.concat(batches), full)

    def test_target_only_projection_still_keeps_feature_state(self) -> None:
        source = self.source()
        full = source.frame().select("fwd").collect()
        for multicollect in (-1, 1):
            with self.subTest(multicollect=multicollect):
                batches = source.dataframe_batches(3, multicollect, cols=["fwd"])
                assert_frame_equal(pl.concat(list(batches)), full)

    def test_transform_can_drop_the_intermediate_output(self) -> None:
        source = self.source(transform=FunctionTransform(lambda lf: lf.drop("e1")))
        full = source.frame().collect()
        for multicollect in (-1, 1):
            with self.subTest(multicollect=multicollect):
                batches = source.dataframe_batches(3, multicollect)
                assert_frame_equal(pl.concat(list(batches)), full)

    def test_intermediate_only_projection_preserves_rows_and_dtype(self) -> None:
        source = self.source(precision="float32")
        full = source.frame(select=False).select("e1").collect()
        for multicollect in (-1, 1):
            with self.subTest(multicollect=multicollect):
                batches = source.dataframe_batches(3, multicollect, cols=["e1"])
                assert_frame_equal(pl.concat(list(batches)), full)
                # Only model inputs and the target use the source precision;
                # an explicitly requested intermediate keeps its own dtype.
                self.assertEqual(full.schema["e1"], pl.Float64)
                public = next(source.dataframe_batches(3, multicollect))
                self.assertEqual(public.schema["e2"], pl.Float32)


class ExactTargetWindowTest(unittest.TestCase):
    def test_zero_tolerance_batches_match_single_pass(self) -> None:
        # A window ending just before the gap looks up a missing timestamp;
        # nothing may stand in for it at any time unit.
        seconds = [*range(10), *range(20, 30)]
        for unit in ("ns", "us", "ms"):
            with self.subTest(unit=unit), tempfile.TemporaryDirectory() as tmp:
                pl.DataFrame(
                    {
                        "time": [T0 + dt.timedelta(seconds=s) for s in seconds],
                        "x": [float(s) for s in seconds],
                    }
                ).with_columns(pl.col("time").cast(pl.Datetime(unit))).write_parquet(
                    Path(tmp) / "2026-01-05.parquet"
                )
                source = DataSource(
                    dates=["2026-01-05"],
                    loader=FeatureLoader(
                        path=f"{tmp}/{{date}}.parquet",
                        feature_exprs={"x2": pl.col("x") * 2},
                        targets={"fwd": pl.col("x")},
                        horizons="1s",
                        target_tolerance="0s",
                    ),
                    target="fwd",
                    features=["x2"],
                )
                full = pl.concat(list(source.dataframe_batches()))
                self.assertEqual(full["fwd"].null_count(), 2)
                for batch_size in (3, 5, 7):
                    batched = pl.concat(list(source.dataframe_batches(batch_size)))
                    assert_frame_equal(batched, full)


class SamplerCarryoverTest(unittest.TestCase):
    """A sampler picks rows; it must not change the state carried between batches."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        values = [math.sin(i / 3) if i % 5 else None for i in range(23)]
        series(values, step_s=0.5).write_parquet(
            Path(self._tmp.name) / "2026-01-05.parquet"
        )
        self.loader = FeatureLoader(
            path=f"{self._tmp.name}/{{date}}.parquet",
            feature_exprs={"x2": pl.col("x") * 2},
            stateful_features=(
                EwmaFeature("x_ewm", pl.col("x"), half_life="2s"),
                CarryForwardFeature("x_last", pl.col("x")),
            ),
            # A row-local target keeps windowed targets exact on their own.
            targets={"fwd": pl.col("x")},
            horizons="1500ms",
        )

    def test_sampled_batches_match_single_pass(self) -> None:
        gap = pl.col("time").is_between(
            T0 + dt.timedelta(seconds=2), T0 + dt.timedelta(seconds=6)
        )
        for name, sampler in (
            ("drops window ends", pl.col("x") > -0.5),
            ("drops whole windows", ~gap),
        ):
            source = DataSource(
                dates=["2026-01-05"],
                loader=self.loader,
                target="fwd",
                features=["x2", "x_ewm", "x_last"],
                filters=(sampler,),
            )
            full = pl.concat(list(source.dataframe_batches()))
            for batch_size, multicollect in ((3, -1), (5, -1), (7, -1), (3, 1)):
                with self.subTest(
                    sampler=name, batch_size=batch_size, multicollect=multicollect
                ):
                    batches = list(
                        source.dataframe_batches(batch_size, multicollect=multicollect)
                    )
                    self.assertTrue(all(batch.height for batch in batches))
                    assert_frame_equal(pl.concat(batches), full)


class TransformCarryoverTest(unittest.TestCase):
    """A transform rewrites feature columns; it must not change the state carried between batches."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        values = [math.sin(i / 3) if i % 5 else None for i in range(23)]
        series(values, step_s=0.5).write_parquet(
            Path(self._tmp.name) / "2026-01-05.parquet"
        )
        self.loader = FeatureLoader(
            path=f"{self._tmp.name}/{{date}}.parquet",
            feature_exprs={"x2": pl.col("x") * 2},
            stateful_features=(
                EwmaFeature("x_ewm", pl.col("x"), half_life="2s"),
                CarryForwardFeature("x_last", pl.col("x")),
            ),
            targets={"fwd": pl.col("x")},
            horizons="1500ms",
        )
        self.features = ["x2", "x_ewm", "x_last"]

    def source(self, **kwargs) -> DataSource:
        return DataSource(
            dates=["2026-01-05"],
            loader=self.loader,
            target="fwd",
            features=self.features,
            **kwargs,
        )

    def test_standardized_batches_match_single_pass(self) -> None:
        # Standardizing the stateful outputs is what a pipeline transform does;
        # the state carried to the next batch must still be the raw value.
        standardizer = Standardizer(cols=self.features).fit(self.source())
        for name, sampler in (("no sampler", ()), ("sampler", (pl.col("x") > -0.5,))):
            source = self.source(transform=standardizer, filters=sampler)
            full = pl.concat(list(source.dataframe_batches()))
            raw = pl.concat(list(self.source(filters=sampler).dataframe_batches()))
            self.assertNotEqual(
                full["x_ewm"].drop_nulls().to_list(), raw["x_ewm"].drop_nulls().to_list()
            )
            for batch_size, multicollect in ((3, -1), (5, -1), (7, -1), (3, 1)):
                with self.subTest(
                    sampler=name, batch_size=batch_size, multicollect=multicollect
                ):
                    batches = list(
                        source.dataframe_batches(batch_size, multicollect=multicollect)
                    )
                    assert_frame_equal(pl.concat(batches), full)


class FrontPadTest(unittest.TestCase):
    """front_pad gives each batch a few earlier rows for row-dependent inputs."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        values = [math.sin(i / 3) if i % 5 else None for i in range(23)]
        series(values, step_s=0.5).write_parquet(
            Path(self._tmp.name) / "2026-01-05.parquet"
        )

    def source(self, feature_exprs, stateful_features=(), **loader_kwargs):
        loader = FeatureLoader(
            path=f"{self._tmp.name}/{{date}}.parquet",
            feature_exprs=feature_exprs,
            stateful_features=stateful_features,
            targets={"fwd": pl.col("x")},
            horizons="1500ms",
            **loader_kwargs,
        )
        return DataSource(
            dates=["2026-01-05"],
            loader=loader,
            target="fwd",
            features=[*feature_exprs, *(f.name for f in stateful_features)],
        )

    def assert_batches_match_single_pass(self, source: DataSource) -> None:
        full = pl.concat(list(source.dataframe_batches()))
        for batch_size in (3, 4, 5, 7):
            with self.subTest(batch_size=batch_size):
                # No explicit pad here: this is the path training takes.
                batched = list(source.dataframe_batches(batch_size))
                assert_frame_equal(pl.concat(batched), full)

    def test_loader_pad_reaches_stateful_inputs(self) -> None:
        dx_ewm = EwmaFeature("dx_ewm", pl.col("x").diff(), half_life="2s")
        source = self.source({"x2": pl.col("x") * 2}, (dx_ewm,), front_pad=1)
        self.assert_batches_match_single_pass(source)

    def test_seed_survives_pad_rows_dropped_by_loader_filters(self) -> None:
        stateful = (
            EwmaFeature("x_ewm", pl.col("x"), half_life="2s"),
            CarryForwardFeature("x_last", pl.col("x")),
        )
        source = self.source(
            {"x2": pl.col("x") * 2},
            stateful,
            front_pad=3,
            filters=(pl.col("x") > -0.5,),
        )
        self.assert_batches_match_single_pass(source)

    def test_plain_look_back_uses_the_pad_without_stateful_features(self) -> None:
        source = self.source({"dx": pl.col("x").diff()}, front_pad=1)
        self.assert_batches_match_single_pass(source)

    def test_unrelated_stateful_feature_does_not_change_plain_features(self) -> None:
        plain = {"dx": pl.col("x").diff()}
        alone = self.source(plain)
        with_ewm = self.source(
            plain, (EwmaFeature("x_ewm", pl.col("x"), half_life="2s"),)
        )
        for batch_size in (3, 4, 5, 7):
            with self.subTest(batch_size=batch_size):
                assert_frame_equal(
                    pl.concat(list(alone.dataframe_batches(batch_size))).select("dx"),
                    pl.concat(list(with_ewm.dataframe_batches(batch_size))).select(
                        "dx"
                    ),
                )

    def test_front_pad_must_be_nonnegative(self) -> None:
        with self.assertRaises(ValueError):
            FeatureLoader(path="x.parquet", front_pad=-1)


class TimeWindowLoaderTest(unittest.TestCase):
    """Windows hold whole timestamps, and targets match a full pass."""

    def write(self, tmp: str, seconds: list[float], **cols) -> str:
        pl.DataFrame(
            {"time": [T0 + dt.timedelta(seconds=s) for s in seconds], **cols}
        ).with_columns(pl.col("time").cast(pl.Datetime("ns"))).write_parquet(
            Path(tmp) / "2026-01-05.parquet"
        )
        return f"{tmp}/{{date}}.parquet"

    def test_ties_stay_in_one_window(self) -> None:
        # A count per timestamp is batch-local: exact only if no window
        # splits a timestamp.
        seconds = [0, 1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 5, 6, 7]
        with tempfile.TemporaryDirectory() as tmp:
            loader = FeatureLoader(
                path=self.write(tmp, seconds, x=[float(i % 4) for i in seconds]),
                feature_exprs={"n_at_t": pl.len().over("time")},
                stateful_features=(
                    EwmaFeature("e", pl.col("x"), half_life="2s"),
                    EwmaFeature("s", pl.col("x"), half_life="2s", normalized=False),
                ),
                targets={"fwd": pl.col("x")},
                horizons="1s",
            )
            source = DataSource(
                dates=["2026-01-05"], loader=loader, target="fwd", features=["n_at_t", "e", "s"]
            )
            full = pl.concat(list(source.dataframe_batches()))
            for batch_size in (1, 2, 3, 5):
                with self.subTest(batch_size=batch_size):
                    windows = [
                        frame.lf.collect()
                        for frame in loader.iter_date_frames(
                            ["2026-01-05"], hint=LoadHint(batch_size=batch_size)
                        )
                    ]
                    self.assertEqual(sum(w.height for w in windows), len(seconds))
                    for before, after in zip(windows, windows[1:]):
                        self.assertLess(before["time"].max(), after["time"].min())
                    for window in windows:
                        self.assertTrue(
                            window.height <= batch_size or window["time"].n_unique() == 1
                        )
                    assert_frame_equal(
                        pl.concat(list(source.dataframe_batches(batch_size))), full
                    )

    def test_target_over_a_timestamp_matches_single_pass(self) -> None:
        # A target over the rows of a timestamp must see only real rows at
        # the date's last timestamp, also when the target is named after the
        # column it reads.
        seconds = [i // 2 for i in range(12)]
        values = [float(i) for i in range(12)]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, seconds, x=values, v=values)
            for name in ("fwd", "x"):
                loader = FeatureLoader(
                    path=path,
                    targets={name: pl.col("x").mean().over("time")},
                    horizons="1s",
                )
                source = DataSource(dates=["2026-01-05"], loader=loader, target=name, features=["v"])
                full = pl.concat(list(source.dataframe_batches()))
                for batch_size in (2, 4, 6, 100):
                    with self.subTest(name, batch_size=batch_size):
                        assert_frame_equal(
                            pl.concat(list(source.dataframe_batches(batch_size))), full
                        )

    def test_targets_can_read_earlier_targets(self) -> None:
        # Exact within one window. A window looks ahead one horizon and a
        # target of a target needs two, so near a window's end it is null,
        # never a value made from the last row loaded.
        with tempfile.TemporaryDirectory() as tmp:
            loader = FeatureLoader(
                path=self.write(tmp, list(range(8)), x=[float(i * i) for i in range(8)]),
                targets={"y1": pl.col("x"), "y2": pl.col("y1")},
                horizons="1s",
            )
            source = DataSource(dates=["2026-01-05"], loader=loader, target="y2", features=["x"])
            full = pl.concat(list(source.dataframe_batches()))
            self.assertGreater(full["y2"].drop_nulls().len(), 3)
            assert_frame_equal(pl.concat(list(source.dataframe_batches(100))), full)
            batched = pl.concat(list(source.dataframe_batches(2)))["y2"]
            self.assertGreater(batched.null_count(), full["y2"].null_count())
            known = batched.is_not_null()
            self.assertEqual(batched.filter(known).to_list(), full["y2"].filter(known).to_list())

    def test_multicollect_with_a_pad_matches_single_pass(self) -> None:
        # Every window must scan the file afresh: Polars 1.41 mixed up the
        # columns of multicollect queries sharing one scan across windows,
        # when the file has columns that no feature reads.
        seconds = [i // 2 for i in range(30)]
        with tempfile.TemporaryDirectory() as tmp:
            loader = FeatureLoader(
                path=self.write(
                    tmp, seconds, unused=[1] * 30, x=[math.sin(i) for i in range(30)]
                ),
                feature_exprs={"x2": pl.col("x") * 2},
                stateful_features=(EwmaFeature("e", pl.col("x"), half_life="2s"),),
                targets={"fwd": pl.col("x")},
                horizons="3s",
                front_pad=1,
            )
            source = DataSource(
                dates=["2026-01-05"], loader=loader, target="fwd", features=["x2", "e"]
            )
            full = pl.concat(list(source.dataframe_batches()))
            for multicollect in (-1, 1, 2):
                with self.subTest(multicollect=multicollect):
                    batched = source.dataframe_batches(7, multicollect=multicollect)
                    assert_frame_equal(pl.concat(list(batched)), full)

    def test_unsorted_rows_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = DataSource(
                dates=["2026-01-05"],
                loader=FeatureLoader(
                    path=self.write(tmp, [0, 2, 1, 3], x=[1.0, 2.0, 3.0, 4.0]),
                    targets={"fwd": pl.col("x")},
                    horizons="1s",
                ),
                target="fwd",
                features=["x"],
            )
            with self.assertRaisesRegex(ValueError, "not sorted by time"):
                list(source.dataframe_batches(2))

    def test_targets_ignore_rows_the_loader_filters_drop(self) -> None:
        # The last rows of the file fail the filter: a full pass has no
        # target past the last kept row, so no window may borrow one.
        with tempfile.TemporaryDirectory() as tmp:
            source = DataSource(
                dates=["2026-01-05"],
                loader=FeatureLoader(
                    path=self.write(
                        tmp, list(range(12)), x=[float(i) for i in range(12)],
                        keep=[True] * 9 + [False] * 3,
                    ),
                    targets={"fwd": pl.col("x")},
                    horizons="2s",
                    filters=(pl.col("keep"),),
                ),
                target="fwd",
                features=["x"],
            )
            full = pl.concat(list(source.dataframe_batches()))
            self.assertEqual(full["fwd"].null_count(), 2)
            for batch_size in (3, 4):
                with self.subTest(batch_size=batch_size):
                    assert_frame_equal(
                        pl.concat(list(source.dataframe_batches(batch_size))), full
                    )


class LookbackFeatureTest(unittest.TestCase):
    """History longer than a batch is carried, so look-backs match a full pass."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        seconds = [0.5 * i + (0.2 if i % 7 == 0 else 0.0) for i in range(40)]
        pl.DataFrame(
            {
                "time": [T0 + dt.timedelta(seconds=s) for s in seconds],
                "x": [math.sin(i / 3) if i % 5 else None for i in range(40)],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime("ns"))).write_parquet(
            Path(self._tmp.name) / "2026-01-05.parquet"
        )
        self.features = (
            LookbackFeature("x_lag5", pl.col("x").shift(5), rows=5),
            LookbackFeature("x_mean10", pl.col("x").rolling_mean(10, min_samples=1), rows=9),
            LookbackFeature("x_mean4s", pl.col("x").rolling_mean_by("time", "4s"), period="4s"),
            EwmaFeature(
                "lag_ewm",
                pl.col("x_lag3"),
                half_life="2s",
                sub_features=(LookbackFeature("x_lag3", pl.col("x").shift(3), rows=3),),
            ),
        )
        self.names = ["x_lag5", "x_mean10", "x_mean4s", "lag_ewm"]

    def source(self, **kwargs) -> DataSource:
        loader = FeatureLoader(
            path=f"{self._tmp.name}/{{date}}.parquet",
            stateful_features=self.features,
            targets={"fwd": pl.col("x")},
            horizons="1s",
            **kwargs,
        )
        return DataSource(
            dates=["2026-01-05"], loader=loader, target="fwd", features=self.names
        )

    def test_batches_shorter_than_the_look_back_match_single_pass(self) -> None:
        for pad, filters in ((0, ()), (2, ()), (3, (pl.col("x") > -0.6,))):
            source = self.source(front_pad=pad, filters=filters)
            full = pl.concat(list(source.dataframe_batches()))
            self.assertGreater(full["x_lag5"].drop_nulls().len(), 10)
            for batch_size in (2, 3, 7, 11):
                with self.subTest(front_pad=pad, filters=bool(filters), batch_size=batch_size):
                    batched = pl.concat(list(source.dataframe_batches(batch_size)))
                    assert_frame_equal(batched, full)

    def test_standardized_batches_match_single_pass(self) -> None:
        standardizer = Standardizer(cols=self.names).fit(self.source())
        source = replace(self.source(), transform=standardizer)
        full = pl.concat(list(source.dataframe_batches()))
        for batch_size in (3, 4):
            with self.subTest(batch_size=batch_size):
                assert_frame_equal(
                    pl.concat(list(source.dataframe_batches(batch_size))), full
                )

    def test_in_memory_engine_matches_single_pass(self) -> None:
        # In memory, each window's rows are read first and the whole batch is
        # computed from them; the batches must not change.
        for pad, filters in ((0, ()), (3, (pl.col("x") > -0.6,))):
            streaming = self.source(front_pad=pad, filters=filters)
            in_memory = replace(streaming, polars_engine="in-memory")
            full = pl.concat(list(streaming.dataframe_batches()))
            for batch_size, multicollect in ((3, -1), (7, -1), (7, 2)):
                with self.subTest(front_pad=pad, batch_size=batch_size, multicollect=multicollect):
                    assert_frame_equal(
                        pl.concat(list(in_memory.dataframe_batches(batch_size, multicollect))),
                        full,
                    )

    def test_pad_rows_take_stored_inputs(self) -> None:
        # The loader's pad rows recompute a sub-feature without the rows
        # before them; the look-back must read its stored values instead.
        cases = {
            "nested shifts": (
                1,
                LookbackFeature(
                    "f",
                    pl.col("a").shift(1),
                    rows=1,
                    sub_features=(LookbackFeature("a", pl.col("x").shift(1), rows=1),),
                ),
            ),
            "diff sub-feature": (
                1,
                LookbackFeature(
                    "f",
                    pl.col("d").shift(2),
                    rows=2,
                    sub_features=(ExprFeature("d", pl.col("x").diff()),),
                ),
            ),
            "EWMA sub-feature": (
                2,
                LookbackFeature(
                    "f",
                    pl.col("e") - pl.col("e").shift(3),
                    rows=3,
                    sub_features=(EwmaFeature("e", pl.col("x"), half_life="2s"),),
                ),
            ),
        }
        for name, (pad, feature) in cases.items():
            loader = FeatureLoader(
                path=f"{self._tmp.name}/{{date}}.parquet",
                stateful_features=(feature,),
                targets={"fwd": pl.col("x")},
                horizons="1s",
                front_pad=pad,
            )
            source = DataSource(dates=["2026-01-05"], loader=loader, target="fwd", features=["f"])
            full = pl.concat(list(source.dataframe_batches()))
            self.assertGreater(full["f"].drop_nulls().len(), 10)
            for batch_size in (2, 3, 7):
                with self.subTest(name, batch_size=batch_size):
                    assert_frame_equal(
                        pl.concat(list(source.dataframe_batches(batch_size))), full
                    )

    def test_needs_rows_or_period(self) -> None:
        for kwargs in ({}, {"rows": 2, "period": "1s"}, {"rows": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                LookbackFeature("bad", pl.col("x").shift(1), **kwargs)


class FrameLoaderTest(unittest.TestCase):
    """Frames in memory load like the same rows in files."""

    def test_frames_match_files(self) -> None:
        values = [math.sin(i / 3) if i % 5 else None for i in range(23)]
        frames = {d: series(values, step_s=0.5) for d in ("2026-01-05", "2026-01-06")}
        kwargs = dict(
            feature_exprs={"x2": pl.col("x") * 2},
            stateful_features=(EwmaFeature("x_ewm", pl.col("x"), half_life="2s"),),
            targets={"fwd": pl.col("x")},
            horizons="1500ms",
            front_pad=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            for d, frame in frames.items():
                frame.write_parquet(Path(tmp) / f"{d}.parquet")
            by_file = FeatureLoader(path=f"{tmp}/{{date}}.parquet", **kwargs)
            in_memory = FeatureLoader(
                frames=frames,
                **kwargs,
            )
            sources = [
                DataSource(dates=list(frames), loader=loader, target="fwd", features=["x2", "x_ewm"])
                for loader in (by_file, in_memory)
            ]
            for batch_size in (None, 4):
                with self.subTest(batch_size=batch_size):
                    files, memory = (
                        pl.concat(list(src.dataframe_batches(batch_size))) for src in sources
                    )
                    assert_frame_equal(memory.drop("tag"), files.drop("tag"))
            with self.assertRaisesRegex(ValueError, "no frame"):
                in_memory.load_date("2026-01-07")

    def test_needs_path_or_frames_not_both(self) -> None:
        with self.assertRaisesRegex(ValueError, "not both"):
            FeatureLoader(path="x_{date}.parquet", frames={})

    def test_frames_must_be_dataframes(self) -> None:
        loader = FeatureLoader(frames={"2026-01-05": series([1.0]).lazy()})
        with self.assertRaisesRegex(TypeError, "DataFrames"):
            loader.load_date("2026-01-05")


class MultiSourceLoaderTest(unittest.TestCase):
    def test_outer_joins_and_prefixes_sources(self) -> None:
        frames = {
            "A": series([1.0, 2.0], step_s=2.0),
            "B": series([10.0, 20.0, 30.0], step_s=1.0),
        }

        def load(dates: list[str], source: str):
            return [
                DateFrame(date=d, tag=source.lower(), lf=frames[source].lazy())
                for d in dates
            ]

        (item,) = MultiSourceLoader(["A", "B"], load)(["2026-01-05"])
        out = item.lf.collect()

        self.assertEqual(item.tag, "a+b")
        self.assertEqual(out.columns, ["time", "a__x", "b__x"])
        self.assertEqual(out["a__x"].to_list(), [1.0, 1.0, 2.0])
        self.assertEqual(out["b__x"].to_list(), [10.0, 20.0, 30.0])


class TimeOfDayTest(unittest.TestCase):
    def test_wraps_past_midnight(self) -> None:
        df = pl.DataFrame(
            {"time": [dt.datetime(2026, 1, 5, h) for h in (1, 12, 23)]}
        )
        day = df.filter(time_of_day("09:00", "17:00"))
        night = df.filter(time_of_day("22:00", "02:00"))
        self.assertEqual([t.hour for t in day["time"]], [12])
        self.assertEqual([t.hour for t in night["time"]], [1, 23])


if __name__ == "__main__":
    unittest.main()
