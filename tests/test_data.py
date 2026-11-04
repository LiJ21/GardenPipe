from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from gardenpipe.data import DataSource, DateFrame


def _loader(_dates: list[str]) -> list[DateFrame]:
    df = pl.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "f3": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    return [DateFrame(date="2026-01-02", tag="test", lf=df.lazy())]


def _source() -> DataSource:
    return DataSource(
        dates=["2026-01-02"],
        loader=_loader,
        target="y",
        features=["f1", "f2", "f3"],
    )


def _concat_batches(batches):
    x = np.vstack([batch[0] for batch in batches])
    y = np.concatenate([batch[1] for batch in batches])
    dates = np.concatenate([batch[2]["date"] for batch in batches])
    tags = np.concatenate([batch[2]["tag"] for batch in batches])
    return x, y, dates, tags


class EmptyFrameTest(unittest.TestCase):
    def test_whole_frames_skip_dates_without_sampled_rows(self) -> None:
        def loader(dates: list[str]) -> list[DateFrame]:
            df = pl.DataFrame({"f": [1.0, 2.0], "y": [0.1, -0.2]})
            return [DateFrame(date=d, tag="t", lf=df.lazy()) for d in dates]

        source = DataSource(
            dates=["2026-01-02", "2026-01-03"],
            loader=loader,
            target="y",
            features=["f"],
            filters=(pl.col("y") > 1.0,),
        )
        for multicollect in (-1, 1):
            with self.subTest(multicollect=multicollect):
                self.assertEqual(list(source.batches(None, multicollect=multicollect)), [])


class RecordingStatefulFeature:
    name = "sf"

    def __init__(self) -> None:
        self.seen_columns: list[tuple[str, ...]] = []

    def internal_cols(self) -> list[str]:
        return ["__sf_internal"]

    def get_carryover(self, df: pl.DataFrame):
        self.seen_columns.append(tuple(df.columns))
        if "__sf_internal" not in df.columns:
            raise AssertionError("stateful carryover read without internal column")
        return {"value": df.get_column("__sf_internal")[-1]}


def _stateful_source(feature: RecordingStatefulFeature) -> DataSource:
    def loader(_dates: list[str]) -> list[DateFrame]:
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0],
                "sf": [10.0, 20.0, 30.0],
                "__sf_internal": [100.0, 200.0, 300.0],
                "y": [0.1, 0.2, 0.3],
            }
        )
        return [
            DateFrame(
                date="2026-01-02",
                tag="test",
                lf=df.lazy(),
                stateful_features=(feature,),
            )
        ]

    return DataSource(
        dates=["2026-01-02"],
        loader=loader,
        target="y",
        features=["f1", "sf"],
    )


class DataSourceMulticollectTest(unittest.TestCase):
    def test_multicollect_matches_standard_collect(self) -> None:
        standard = list(_source().batches(batch_size=None))
        multi = list(_source().batches(batch_size=None, multicollect=2))

        for left, right in zip(_concat_batches(standard), _concat_batches(multi)):
            np.testing.assert_array_equal(left, right)

    def test_multicollect_matches_standard_collect_batches(self) -> None:
        standard = list(_source().batches(batch_size=2))
        multi = list(_source().batches(batch_size=2, multicollect=1))

        for left, right in zip(_concat_batches(standard), _concat_batches(multi)):
            np.testing.assert_array_equal(left, right)

    def test_dataframe_batches_selects_ordered_columns(self) -> None:
        batches = list(
            _source().dataframe_batches(
                batch_size=2,
                multicollect=2,
                cols=["date", "f2", "y"],
            )
        )

        out = pl.concat(batches, how="vertical")

        self.assertEqual(out.columns, ["date", "f2", "y"])
        self.assertEqual(out["f2"].to_list(), [10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(out["date"].to_list(), ["2026-01-02"] * 5)

    def test_zero_multicollect_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            list(_source().batches(batch_size=2, multicollect=0))

    def test_precision_float32_materializes_arrays(self) -> None:
        x, y, _ = DataSource(
            dates=["2026-01-02"],
            loader=_loader,
            target="y",
            features=["f1", "f2", "f3"],
            precision="float32",
        ).materialize()

        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)

    def test_precision_float32_multicollect_arrays(self) -> None:
        batches = list(
            DataSource(
                dates=["2026-01-02"],
                loader=_loader,
                target="y",
                features=["f1", "f2", "f3"],
                precision="float32",
            ).batches(batch_size=2, multicollect=2)
        )
        x, y, _, _ = _concat_batches(batches)

        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)

    def test_multicollect_updates_carryover_from_feature_group(self) -> None:
        feature = RecordingStatefulFeature()

        batches = list(_stateful_source(feature).batches(batch_size=2, multicollect=1))

        self.assertEqual(sum(batch[2]["n"] for batch in batches), 3)
        self.assertGreaterEqual(len(feature.seen_columns), 1)
        for cols in feature.seen_columns:
            self.assertIn("sf", cols)
            self.assertIn("__sf_internal", cols)
            self.assertNotIn("y", cols)

    def test_dataframe_batches_uses_internal_cols_only_for_carryover(self) -> None:
        feature = RecordingStatefulFeature()

        batches = list(
            _stateful_source(feature).dataframe_batches(
                batch_size=2,
                multicollect=1,
                cols=["f1", "sf", "y"],
            )
        )

        self.assertEqual(pl.concat(batches, how="vertical").columns, ["f1", "sf", "y"])
        self.assertGreaterEqual(len(feature.seen_columns), 1)
        for cols in feature.seen_columns:
            self.assertIn("__sf_internal", cols)


if __name__ == "__main__":
    unittest.main()
