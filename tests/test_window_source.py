from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from gardenpipe.data import DataSource, DateFrame
from gardenpipe.window import WindowDataSource, WindowSpec, window_wrapper


DATES = ["2026-01-02", "2026-01-05"]


def loader(dates):
    frames = []
    for offset, date in enumerate(dates):
        row = np.arange(8) + 10 * offset
        frames.append(
            DateFrame(
                date,
                "default",
                pl.DataFrame(
                    {
                        "x": row,
                        "y": row + 100,
                        "endpoint": [True] * 8,
                        "group": ["a"] * 4 + ["b"] * 4,
                        "row_nr": row,
                    }
                ).lazy(),
            )
        )
    return frames


def source(dates=DATES):
    return DataSource(list(dates), loader, "y", ["x"], precision="float32")


class AlignedLoader:
    alignment_cols = ("aligned",)

    def __init__(self, aligned=True):
        self.aligned = aligned
        self.calls = 0

    def __call__(self, dates):
        return [
            DateFrame(
                frame.date,
                frame.tag,
                frame.lf.with_columns(pl.lit(self.aligned).alias("aligned")),
            )
            for frame in loader(dates)
        ]

    def assert_aligned(self, df):
        self.calls += 1
        if not df.get_column("aligned").all():
            raise ValueError("misaligned")


class WindowSourceTests(unittest.TestCase):
    def test_windows_align_to_endpoints_and_stay_in_group(self):
        src = WindowDataSource(
            source([DATES[0]]),
            WindowSpec(
                length=3,
                endpoint_col="endpoint",
                group_cols=("group",),
                context_cols=("row_nr",),
                val_stride=1,
            ),
            "val",
        )
        batches = list(src.batches(2))
        x = np.concatenate([batch.x for batch in batches])
        y = np.concatenate([batch.y for batch in batches])
        keys = np.concatenate([batch.ctx["row_nr"] for batch in batches])

        self.assertEqual(x.shape, (4, 3, 1))
        np.testing.assert_array_equal(
            x[:, :, 0],
            [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]],
        )
        np.testing.assert_array_equal(y, [102, 103, 106, 107])
        np.testing.assert_array_equal(keys, [2, 3, 6, 7])
        self.assertEqual(x.dtype, np.float32)

    def test_validation_stride_is_fixed(self):
        spec = WindowSpec(
            length=3,
            endpoint_col="endpoint",
            group_cols=("group",),
            context_cols=("row_nr",),
            val_stride=2,
        )
        batch = next(WindowDataSource(source([DATES[0]]), spec, "val").batches())
        np.testing.assert_array_equal(batch.ctx["row_nr"], [2, 6])
        repeated = next(
            WindowDataSource(source([DATES[0]]), spec, "val").batches()
        )
        np.testing.assert_array_equal(batch.x, repeated.x)

    def test_train_sampling_is_fresh_and_reproducible(self):
        spec = WindowSpec(
            length=2,
            endpoint_col="endpoint",
            train_samples=5,
            context_cols=("row_nr",),
            seed=7,
        )
        left = WindowDataSource(source(), spec, "train")
        right = WindowDataSource(source(), spec, "train")

        def keys(src):
            return np.concatenate([batch.ctx["row_nr"] for batch in src.batches(2)])

        first = keys(left)
        self.assertEqual(len(first), spec.train_samples)
        np.testing.assert_array_equal(first, keys(right))
        second = keys(left)
        np.testing.assert_array_equal(second, keys(right))
        self.assertFalse(np.array_equal(first, second))

    def test_lru_budget_and_oversized_day(self):
        spec = WindowSpec(
            length=2,
            endpoint_col="endpoint",
            context_cols=("row_nr",),
            val_stride=1,
            max_resident_bytes=180,
        )
        src = WindowDataSource(source(), spec, "val")
        list(src.batches(4))
        self.assertLessEqual(src.resident_bytes, spec.max_resident_bytes)
        self.assertEqual(len(src.resident_dates), 1)

        oversized = WindowDataSource(
            source([DATES[0]]),
            WindowSpec(length=2, endpoint_col="endpoint", max_resident_bytes=1),
            "test",
        )
        list(oversized.batches(4))
        self.assertEqual(oversized.resident_dates, (DATES[0],))
        self.assertGreater(oversized.resident_bytes, 1)

        disabled = WindowDataSource(
            source([DATES[0]]),
            WindowSpec(length=2, endpoint_col="endpoint", max_resident_bytes=0),
            "test",
        )
        list(disabled.batches(4))
        self.assertEqual(disabled.resident_dates, ())

    def test_nonfinite_feature_invalidates_the_whole_window(self):
        def bad_loader(dates):
            frames = loader(dates)
            return [
                DateFrame(
                    frame.date,
                    frame.tag,
                    frame.lf.with_columns(
                        pl.when(pl.col("row_nr") % 10 == 1)
                        .then(float("nan"))
                        .otherwise(pl.col("x"))
                        .alias("x")
                    ),
                )
                for frame in frames
            ]

        src = DataSource(
            [DATES[0]], bad_loader, "y", ["x"], precision="float32"
        )
        spec = WindowSpec(
            length=3,
            endpoint_col="endpoint",
            group_cols=("group",),
            context_cols=("row_nr",),
            val_stride=1,
        )
        rows = np.concatenate(
            [
                batch.ctx["row_nr"]
                for batch in WindowDataSource(src, spec, "val").batches(2)
            ]
        )
        np.testing.assert_array_equal(rows, [6, 7])

    def test_alignment_projection_and_validation(self):
        good = AlignedLoader()
        src = DataSource(
            [DATES[0]], good, "y", ["x"], precision="float32"
        )
        list(
            WindowDataSource(
                src,
                WindowSpec(length=2, endpoint_col="endpoint"),
                "test",
            ).batches(4)
        )
        self.assertEqual(good.calls, 1)

        bad = DataSource(
            [DATES[0]], AlignedLoader(False), "y", ["x"], precision="float32"
        )
        with self.assertRaisesRegex(ValueError, "misaligned"):
            list(
                WindowDataSource(
                    bad,
                    WindowSpec(length=2, endpoint_col="endpoint"),
                    "test",
                ).batches(4)
            )

    def test_role_wrapper(self):
        wrap = window_wrapper(WindowSpec(endpoint_col="endpoint"))
        bare = source([DATES[0]])
        self.assertIs(wrap(bare, "fit"), bare)
        self.assertIsInstance(wrap(bare, "final_train"), WindowDataSource)
        self.assertIsInstance(wrap(bare, "final_val"), WindowDataSource)
        with self.assertRaises(NotImplementedError):
            wrap(bare, "predict")


if __name__ == "__main__":
    unittest.main()
