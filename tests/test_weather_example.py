from __future__ import annotations

import datetime as dt
import math
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "weather"))

import common  # noqa: E402
import prepare  # noqa: E402
import train  # noqa: E402

from gardenpipe.data import DataSource  # noqa: E402
from gardenpipe.pipeline import Pipeline  # noqa: E402

START = dt.datetime(2015, 3, 2)
STEP = dt.timedelta(minutes=10)
DAYS = 12
# Two-day partitions: lag state restarts with each file, and the 24-hour lag
# needs a full day of it.
KEYS = [f"b{i}" for i in range(DAYS // 2)]
GAP = (dt.datetime(2015, 3, 4, 6), dt.datetime(2015, 3, 4, 8))  # [start, end)
CUTOFF = dt.datetime(2015, 3, 11, 12)  # second day of the first test partition


def partition(t: dt.datetime) -> str:
    return f"b{(t - START).days // 2}"


def raw_fixture(perturb_after: dt.datetime | None = None) -> pl.DataFrame:
    """Archive-shaped rows: a gap, exact duplicates, a sentinel, out of order."""

    rows = []
    for i in range(DAYS * 144):
        t = START + i * STEP
        if GAP[0] <= t < GAP[1]:
            continue
        temp = 5 + 5 * math.sin(2 * math.pi * i / 144) + 0.01 * i + math.sin(i / 7)
        if perturb_after is not None and t > perturb_after:
            temp += 3.0
        rows.append(
            {
                "time": t,
                "p": 1000 + math.sin(i / 50),
                "T": temp,
                "rh": 70 + 10 * math.cos(i / 30),
                "wv": 2 + math.sin(i / 11),
                "max_wv": 4.0,
                "wd": float(i * 7 % 360),
                **{c: 1.0 for c in ("Tpot", "Tdew", "VPmax", "VPact", "VPdef", "sh", "H2OC", "rho")},
            }
        )
    df = pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Datetime("us")))
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 50).then(common.WIND_SENTINEL).otherwise("wv").alias("wv")
    )
    return pl.concat([df.slice(300), df.slice(0, 300), df.slice(100, 5)])


def prepare_into(root: Path, raw: pl.DataFrame) -> pl.DataFrame:
    df, _ = prepare.clean(raw, end=START + dt.timedelta(days=DAYS))
    key = pl.format("b{}", (pl.col("time") - START).dt.total_days() // 2)
    prepare.write_partitions(df, root, key)
    return df


class PrepareTest(unittest.TestCase):
    def test_clean_sorts_dedupes_and_nulls_sentinels(self) -> None:
        raw = raw_fixture()
        df, stats = prepare.clean(raw, end=START + dt.timedelta(days=DAYS))

        self.assertTrue(df.get_column("time").is_sorted())
        self.assertEqual(df.get_column("time").n_unique(), df.height)
        self.assertEqual(stats["duplicate_rows_removed"], 5)
        self.assertFalse(stats["raw_in_time_order"])
        self.assertEqual(stats["wind_sentinels"]["wv"], 1)
        self.assertEqual(df.get_column("wv").null_count(), 1)

    def test_conflicting_duplicate_is_rejected(self) -> None:
        raw = raw_fixture()
        conflict = raw.slice(0, 1).with_columns(pl.col("T") + 1)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            prepare.clean(pl.concat([raw, conflict]))

    def test_preparation_is_deterministic(self) -> None:
        first = prepare.clean(raw_fixture())[0]
        second = prepare.clean(raw_fixture())[0]
        assert_frame_equal(first, second)


class ExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.clean = prepare_into(self.root / "data", raw_fixture())
        self.loader = common.make_loader(self.root / "data")

    def frame(self, dates: list[str], loader=None, batch_size: int | None = None) -> pl.DataFrame:
        src = DataSource(
            dates=dates,
            loader=loader or self.loader,
            target=common.TARGET,
            features=[*common.REQUIRED],
        )
        return pl.concat(list(src.dataframe_batches(batch_size, cols=["time", *src.features, common.TARGET])))

    def test_lags_are_exact_and_stay_in_their_partition(self) -> None:
        df = self.frame(KEYS, batch_size=50)
        values = dict(zip(self.clean.get_column("time"), self.clean.get_column("T")))
        for t, t_now, lag_18h, d_1h in zip(df["time"], df["T_raw"], df["T_lag_18h"], df["dT_1h"]):
            for hours, got, expected in ((18, lag_18h, lambda v: v), (1, d_1h, lambda v: t_now - v)):
                then = t - dt.timedelta(hours=hours)
                if then in values and partition(then) == partition(t):
                    self.assertAlmostEqual(got, expected(values[then]))
                else:
                    self.assertIsNone(got, (t, hours))  # never an earlier observation
        self.assertGreater(df["dT_1h"].drop_nulls().len(), 1000)

    def test_labels_are_exact_and_stay_in_their_partition(self) -> None:
        df = self.frame(KEYS)
        values = dict(zip(self.clean.get_column("time"), self.clean.get_column("T")))
        for t, label in zip(df["time"], df[common.TARGET]):
            target = t + dt.timedelta(hours=6)
            same_partition = partition(target) == partition(t)
            if target in values and same_partition:
                self.assertAlmostEqual(label, values[target] - values[t])
            else:
                self.assertIsNone(label, (t, target))

    def test_batched_equals_full_pass(self) -> None:
        # A 7-row batch on the gap day ends at 01:00, whose 07:00 target is
        # missing: nothing loaded later may stand in for it. Batches
        # cost a fixed charge each, so the smallest stays near the gap.
        for batch_size, keys in ((7, KEYS[:2]), (17, KEYS), (50, KEYS)):
            with self.subTest(batch_size=batch_size):
                assert_frame_equal(self.frame(keys, batch_size=batch_size), self.frame(keys))

    def test_end_to_end(self) -> None:
        train_keys, val_keys, test_keys = KEYS[:3], KEYS[3:4], KEYS[4:]
        pipes = {
            recipe: train.build_pipeline(
                recipe, self.loader, [train_keys, val_keys], test_keys, batch_rows=100
            )
            for recipe in common.RECIPES
        }
        for pipe in pipes.values():
            pipe.train()
        preds = train.prediction_frame(pipes, test_keys)  # raises if rows differ
        self.assertGreater(preds.height, 20)

        with self.subTest("training-only standardizer"):
            history = pipes["history"]
            rows = DataSource(
                dates=train_keys + val_keys,
                loader=self.loader,
                target=common.TARGET,
                features=history.features,
                filters=(common.eligible(),),
            ).frame().collect()
            for col in history.features:
                self.assertAlmostEqual(history.fitted_transform.means[col], rows[col].mean())
                self.assertAlmostEqual(history.fitted_transform.stds[col], rows[col].std())

        with self.subTest("reloaded pipeline reproduces predictions"):
            pipes["history"].save_pipeline(self.root / "saved")
            reloaded = Pipeline.from_saved(self.root / "saved", data_loader=self.loader)
            key = ["time"]
            assert_frame_equal(
                reloaded.predict_frame(test_keys, key_cols=key),
                pipes["history"].predict_frame(test_keys, key_cols=key),
            )

        with self.subTest("causality"):
            perturbed = prepare_into(self.root / "perturbed", raw_fixture(CUTOFF))
            self.assertFalse(perturbed.equals(self.clean))
            loader = common.make_loader(self.root / "perturbed")
            before = self.frame(test_keys)
            after = self.frame(test_keys, loader=loader)
            early = pl.col("time") <= CUTOFF
            features = [c for c in before.columns if c != common.TARGET]
            assert_frame_equal(before.filter(early).select(features), after.filter(early).select(features))
            settled = pl.col("time") + pl.duration(hours=6) <= CUTOFF
            assert_frame_equal(before.filter(settled), after.filter(settled))
            self.assertFalse(before.filter(~settled & early).equals(after.filter(~settled & early)))

            model = pipes["history"]
            kept = model.predict_frame(test_keys, key_cols=key).filter(early)
            moved = model.predict_frame(test_keys, key_cols=key, feature_source=loader)
            moved = moved.filter(early).filter(pl.col("time").is_in(kept["time"].implode()))
            assert_frame_equal(kept.select("time", "prediction"), moved.select("time", "prediction"))


if __name__ == "__main__":
    unittest.main()
