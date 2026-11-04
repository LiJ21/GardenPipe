from __future__ import annotations

import datetime as dt
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import polars as pl

from gardenpipe.data import DataSource, DateFrame
from gardenpipe.model import DummyAdapter
from gardenpipe.pipeline import Pipeline
from gardenpipe.score import rmse
from gardenpipe.transform import FunctionTransform, Passthrough


DAY = "2026-07-11"


class RecordingAdapter:
    streaming = True
    batch_size = 2

    def __init__(self, *, multioutput: bool = False) -> None:
        self.multioutput = multioutput
        self.calls: list[np.ndarray] = []
        if multioutput:
            self.quantiles = [0.1, 0.9]

    def predict(self, _model, x: np.ndarray) -> np.ndarray:
        self.calls.append(x.copy())
        values = x[:, 0] * 2.0
        if self.multioutput:
            return np.column_stack((values, -values))
        return values


def feature_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [DAY] * 5,
            "time": [
                dt.datetime(2026, 7, 11, 7, 59),
                dt.datetime(2026, 7, 11, 8, 1),
                dt.datetime(2026, 7, 11, 8, 2),
                dt.datetime(2026, 7, 11, 8, 2),
                dt.datetime(2026, 7, 11, 8, 5),
            ],
            "row_nr": [10, 11, 12, 13, 14],
            "step": [100, 101, 102, 103, 104],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [100.0, 101.0, 102.0, 103.0, 104.0],
            "keep_a": [True, False, False, True, True],
            "keep_b": [False, False, True, False, False],
        }
    )


def frame_loader(dates: list[str]) -> list[DateFrame]:
    df = feature_frame()
    return [
        DateFrame(
            date=date,
            tag="test",
            lf=df.lazy().filter(pl.col("date") == date),
        )
        for date in dates
    ]


def add_unit_weight(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(pl.lit(1.0).alias("weight"))


def pipeline(adapter: RecordingAdapter) -> Pipeline:
    pipe = Pipeline(
        rolling_dates=[["2026-07-10"]],
        adapter=adapter,
        target="y",
        features=["x"],
        data_loader=frame_loader,
        test_dates=[DAY],
        test_sampler=(pl.col("keep_a"), pl.col("keep_b")),
    )
    pipe.model = object()
    pipe.fitted_transform = FunctionTransform(
        lambda lf: lf.with_columns((pl.col("x") + 10.0).alias("x"))
    )
    return pipe


class PredictFrameTest(unittest.TestCase):
    def test_scalar_prediction_is_key_aligned_and_window_is_conjunctive(self) -> None:
        adapter = RecordingAdapter()
        pipe = pipeline(adapter)

        result = pipe.predict_frame(
            start="2026-07-11T10:02:00",
            end="2026-07-11T10:02:00",
            timezone="Europe/Zurich",
        )

        self.assertEqual(
            result.columns,
            [
                "date",
                "time",
                "y",
                "prediction",
            ],
        )
        self.assertEqual(
            result["time"].to_list(),
            [dt.datetime(2026, 7, 11, 8, 2)] * 2,
        )
        self.assertEqual(result["y"].to_list(), [102.0, 103.0])
        self.assertEqual(result["prediction"].to_list(), [24.0, 26.0])
        self.assertEqual([len(x) for x in adapter.calls], [2])

    def test_multioutput_names_are_inferred_and_frame_source_overrides_loader(
        self,
    ) -> None:
        adapter = RecordingAdapter(multioutput=True)

        def unused_loader(_dates: list[str]) -> list[DateFrame]:
            raise AssertionError("pipeline loader should not be used")

        pipe = Pipeline(
            rolling_dates=[["2026-07-10"]],
            adapter=adapter,
            target="y",
            features=["x"],
            data_loader=unused_loader,
        )
        pipe.model = object()
        pipe.fitted_transform = Passthrough()

        result = pipe.predict_frame(
            dates=DAY,
            feature_source=feature_frame(),
            sampler=(pl.col("row_nr") >= 12,),
            key_cols=("time", "row_nr", "step"),
        )

        self.assertEqual(
            result.columns,
            [
                "time",
                "row_nr",
                "step",
                "y",
                "prediction_q10",
                "prediction_q90",
            ],
        )
        self.assertEqual(result["row_nr"].to_list(), [12, 13, 14])
        self.assertEqual(result["prediction_q10"].to_list(), [4.0, 6.0, 8.0])
        self.assertEqual(result["prediction_q90"].to_list(), [-4.0, -6.0, -8.0])
        self.assertEqual([len(x) for x in adapter.calls], [2, 1])

    def test_test_contract_uses_shared_prediction_batches(self) -> None:
        adapter = RecordingAdapter()
        pipe = pipeline(adapter)

        with patch(
            "gardenpipe.pipeline._iter_prediction_batches",
            wraps=__import__(
                "gardenpipe.pipeline", fromlist=["_iter_prediction_batches"]
            )._iter_prediction_batches,
        ) as prediction_batches:
            result = pipe.test(score=rmse, keep_predictions=False)

        self.assertTrue(prediction_batches.called)
        self.assertEqual(result["n"], 4)
        self.assertIsNone(result["y_pred"])
        self.assertEqual(result["ctx"]["fold"], "test")
        self.assertEqual([len(x) for x in adapter.calls], [2, 2])

    def test_predict_frame_never_fits_transform_implicitly(self) -> None:
        pipe = pipeline(RecordingAdapter())
        pipe.fitted_transform = None

        with self.assertRaisesRegex(RuntimeError, "fitted transform"):
            pipe.predict_frame()

    def test_empty_window_is_schemaful_and_does_not_call_model(self) -> None:
        for batch_size in (None, 2):
            with self.subTest(batch_size=batch_size):
                adapter = RecordingAdapter()
                pipe = pipeline(adapter)

                result = pipe.predict_frame(
                    start="2026-07-12T00:00:00",
                    key_cols="row_nr",
                    batch_size=batch_size,
                )

                self.assertEqual(result.shape, (0, 3))
                self.assertEqual(result.columns, ["row_nr", "y", "prediction"])
                self.assertEqual(result.schema["row_nr"], pl.Int64)
                self.assertEqual(result.schema["y"], pl.Float64)
                self.assertEqual(result.schema["prediction"], pl.Float64)
                self.assertEqual(adapter.calls, [])

    def test_feature_source_dates_filters_and_transform_are_preserved(self) -> None:
        source_day = "2026-07-12"
        loaded_dates: list[list[str]] = []

        def source_loader(dates: list[str]) -> list[DateFrame]:
            loaded_dates.append(list(dates))
            lf = feature_frame().drop("date").lazy()
            return [DateFrame(date, "source", lf) for date in dates]

        source = DataSource(
            dates=[source_day],
            loader=source_loader,
            target="y",
            features=["x"],
            filters=(pl.col("row_nr") == 12,),
            transform=FunctionTransform(
                lambda lf: lf.with_columns((pl.col("x") * 3.0).alias("x"))
            ),
        )
        adapter = RecordingAdapter()
        pipe = pipeline(adapter)

        result = pipe.predict_frame(feature_source=source, key_cols="row_nr")

        self.assertTrue(loaded_dates)
        self.assertEqual(set(map(tuple, loaded_dates)), {(source_day,)})
        self.assertEqual(result["row_nr"].to_list(), [12])
        self.assertEqual(result["prediction"].to_list(), [32.0])

    def test_prediction_data_source_wrapper_receives_predict_role(self) -> None:
        roles: list[str] = []
        dataframe_calls: list[int | None] = []

        class KeyedWrapper:
            def __init__(self, source: DataSource) -> None:
                self.source = source

            def __getattr__(self, name: str):
                return getattr(self.source, name)

            def dataframe_batches(self, batch_size=None, **kwargs):
                dataframe_calls.append(batch_size)
                yield from self.source.dataframe_batches(batch_size, **kwargs)

        def wrap(source: DataSource, role: str):
            roles.append(role)
            return KeyedWrapper(source)

        pipe = pipeline(RecordingAdapter())
        pipe.data_source_wrapper = wrap

        result = pipe.predict_frame(key_cols="row_nr")

        self.assertGreater(result.height, 0)
        self.assertEqual(roles, ["predict"])
        self.assertEqual(dataframe_calls, [2])

    def test_refit_date_provenance_round_trips_manifest_and_history(self) -> None:
        adapter = DummyAdapter()
        pipe = Pipeline(
            rolling_dates=[["2026-07-09"]],
            adapter=adapter,
            target="y",
            features=["x"],
            data_loader=frame_loader,
            test_dates=[DAY],
        )
        pipe.model = {"mean": 0.0}
        pipe.fitted_transform = Passthrough()
        pipe.last_refit_train_dates = ["2026-07-09", "2026-07-10"]
        pipe.last_refit_val_dates = [DAY]

        with TemporaryDirectory() as tmp:
            manifest = pipe.save_pipeline(tmp)
            loaded = Pipeline(
                rolling_dates=[["placeholder"]],
                adapter=DummyAdapter(),
                target="other",
                features=["other"],
                data_loader=frame_loader,
            )
            loaded.load_pipeline(tmp)

        self.assertEqual(
            manifest["last_refit_train_dates"],
            ["2026-07-09", "2026-07-10"],
        )
        self.assertEqual(manifest["last_refit_val_dates"], [DAY])
        self.assertEqual(
            loaded.last_refit_train_dates,
            ["2026-07-09", "2026-07-10"],
        )
        self.assertEqual(loaded.last_refit_val_dates, [DAY])

    def test_loaded_transform_is_reused_as_refit_template(self) -> None:
        pipe = Pipeline(
            rolling_dates=[[DAY]],
            adapter=DummyAdapter(),
            target="y",
            features=["x"],
            data_loader=frame_loader,
            test_dates=[DAY],
            sample_weight_col="weight",
            transform=FunctionTransform(add_unit_weight),
        )
        pipe.model = {"mean": 0.0}
        pipe.fitted_transform = FunctionTransform(add_unit_weight)

        with TemporaryDirectory() as tmp:
            pipe.save_pipeline(tmp)
            loaded = Pipeline(
                rolling_dates=[[DAY]],
                adapter=DummyAdapter(),
                target="y",
                features=["x"],
                data_loader=frame_loader,
                test_dates=[DAY],
            )
            loaded.load_pipeline(tmp)
            loaded.refit(dates=[DAY], params={})

        self.assertIsInstance(loaded.transform, FunctionTransform)
        self.assertEqual(loaded.sample_weight_col, "weight")
        self.assertEqual(loaded.last_refit_train_dates, [DAY])

    def test_from_saved_constructs_pipeline_and_selects_date_hash_prefixes(self) -> None:
        train_sampler = (pl.col("x") > 0,)
        test_sampler = (pl.col("keep_a"), pl.col("keep_b"))
        pipe = Pipeline(
            rolling_dates=[[DAY]],
            adapter=DummyAdapter(),
            target="y",
            features=["x"],
            data_loader=frame_loader,
            test_dates=[DAY],
            train_sampler=train_sampler,
            test_sampler=test_sampler,
        )
        pipe.model = {"mean": 2.0}
        pipe.fitted_transform = Passthrough()

        with TemporaryDirectory() as tmp:
            manifest = pipe.save_pipeline(tmp)
            loaded = Pipeline.from_saved(
                tmp,
                stamp=manifest["run"]["stamp"][:8],
                run_hash=manifest["run"]["hash"][:4],
                data_loader=frame_loader,
            )

        self.assertIsInstance(loaded.adapter, DummyAdapter)
        self.assertEqual(loaded.target, "y")
        self.assertEqual(loaded.features, ["x"])
        self.assertEqual(loaded.model, {"mean": 2.0})
        for saved, original in (
            (loaded.train_sampler, train_sampler),
            (loaded.test_sampler, test_sampler),
        ):
            self.assertEqual(len(saved), len(original))
            for left, right in zip(saved, original):
                self.assertTrue(left.meta.eq(right))
        self.assertEqual(loaded.val_sampler, ())

    def test_from_saved_without_test_dates_keeps_precision(self) -> None:
        pipe = Pipeline(
            rolling_dates=[[DAY]],
            adapter=DummyAdapter(),
            target="y",
            features=["x"],
            data_loader=frame_loader,
            precision="float32",
        )
        pipe.model = {"mean": 2.0}
        pipe.fitted_transform = Passthrough()

        with TemporaryDirectory() as tmp:
            pipe.save_pipeline(tmp)
            loaded = Pipeline.from_saved(tmp, data_loader=frame_loader)

        self.assertIsNone(loaded.test_dates)
        self.assertEqual(loaded.precision, "float32")
        self.assertEqual(loaded.model, {"mean": 2.0})


if __name__ == "__main__":
    unittest.main()
