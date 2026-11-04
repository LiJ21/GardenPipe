from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

from gardenpipe.buffered import BufferConfig, BufferedDataSource
from gardenpipe.data import Batch, DataSource, DateFrame
from gardenpipe.model import (
    DummyAdapter,
    LarsAdapter,
    RidgeAdapter,
    TorchAdapter,
    XGBoostAdapter,
)
from gardenpipe.pipeline import Pipeline, call_score
from gardenpipe.score import rmse, weighted
from gardenpipe.track import NullTracker


def _source(
    x=(0.0, 1.0, 2.0),
    y=(0.0, 1.0, 10.0),
    weight=(1.0, 1.0, 0.0),
    *,
    weighted_source: bool = True,
    precision: str = "float64",
) -> DataSource:
    df = pl.DataFrame({"x": x, "y": y, "w": weight})

    def loader(_dates: list[str]) -> list[DateFrame]:
        return [DateFrame("2026-01-02", "test", df.lazy())]

    return DataSource(
        dates=["2026-01-02"],
        loader=loader,
        target="y",
        features=["x"],
        sample_weight_col="w" if weighted_source else None,
        precision=precision,
    )


class BatchAndDataSourceTest(unittest.TestCase):
    def test_none_preserves_legacy_batch_interface(self) -> None:
        batch = _source(weighted_source=False).materialize()

        x, y, ctx = batch

        self.assertIsInstance(batch, Batch)
        self.assertIsNone(batch.weight)
        self.assertIs(x, batch.x)
        self.assertIs(y, batch.y)
        self.assertIs(ctx, batch.ctx)
        self.assertEqual(len(batch), 3)

    def test_weight_is_first_class_and_not_context(self) -> None:
        batch = _source(precision="float32").materialize()

        np.testing.assert_array_equal(batch.weight, [1.0, 1.0, 0.0])
        self.assertEqual(batch.weight.dtype, np.float32)
        self.assertNotIn("w", batch.ctx)

    def test_multicollect_preserves_weight(self) -> None:
        batches = list(_source().batches(batch_size=2, multicollect=1))

        np.testing.assert_array_equal(
            np.concatenate([batch.weight for batch in batches]),
            [1.0, 1.0, 0.0],
        )

    def test_invalid_weights_fail_at_materialization(self) -> None:
        cases = (
            ((1.0, -1.0, 1.0), "nonnegative"),
            ((1.0, np.nan, 1.0), "finite"),
            ((0.0, 0.0, 0.0), "positive"),
        )
        for values, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    _source(weight=values).materialize()


class BufferedWeightTest(unittest.TestCase):
    def test_row_shuffle_keeps_weight_aligned(self) -> None:
        y = np.arange(12, dtype=float)
        src = _source(x=y * 2.0, y=y, weight=y + 1.0)
        wrapped = BufferedDataSource(
            src,
            BufferConfig(
                workers=1,
                row_shuffle_rows=5,
                train_ctx="minimal",
                seed=7,
            ),
        )

        batches = list(wrapped.batches(3))

        for batch in batches:
            np.testing.assert_array_equal(batch.x[:, 0], batch.y * 2.0)
            np.testing.assert_array_equal(batch.weight, batch.y + 1.0)
            self.assertEqual(set(batch.ctx), {"n"})
        np.testing.assert_array_equal(
            np.sort(np.concatenate([batch.y for batch in batches])), y
        )


class AdapterWeightTest(unittest.TestCase):
    def test_dummy_uses_weighted_mean(self) -> None:
        src = _source(x=(0.0, 0.0), y=(0.0, 10.0), weight=(9.0, 1.0))
        adapter = DummyAdapter()

        model = adapter.fit(adapter.build({}), src, None, NullTracker())

        self.assertAlmostEqual(model["mean"], 1.0)

    def test_ridge_matches_weighted_least_squares(self) -> None:
        src = _source()
        adapter = RidgeAdapter(alpha=0.0, batch_size=None)
        model = adapter.fit(adapter.build({}), src, None, NullTracker())

        batch = src.materialize()
        design = np.c_[np.ones(len(batch.x)), batch.x]
        root_weight = np.sqrt(batch.weight)
        expected, *_ = np.linalg.lstsq(
            design * root_weight[:, None],
            batch.y * root_weight,
            rcond=None,
        )

        self.assertAlmostEqual(model.intercept, expected[0])
        np.testing.assert_allclose(model.coef, expected[1:])

    def test_lars_statistics_are_weighted(self) -> None:
        src = _source()
        stats = LarsAdapter(batch_size=None)._stats(src)

        gram, xy, x_mean, y_mean = stats.centered(True)
        batch = src.materialize()
        weight_sum = float(batch.weight.sum())
        expected_x_mean = (batch.x * batch.weight[:, None]).sum(axis=0) / weight_sum
        expected_y_mean = float(np.dot(batch.weight, batch.y) / weight_sum)
        centered_x = batch.x - expected_x_mean
        centered_y = batch.y - expected_y_mean

        np.testing.assert_allclose(
            gram,
            centered_x.T @ (centered_x * batch.weight[:, None]),
        )
        np.testing.assert_allclose(xy, centered_x.T @ (centered_y * batch.weight))
        np.testing.assert_allclose(x_mean, expected_x_mean)
        self.assertAlmostEqual(y_mean, expected_y_mean)

    def test_xgboost_stream_passes_native_weight(self) -> None:
        calls: list[dict[str, np.ndarray]] = []

        class DataIter:
            def __init__(self, **_kwargs):
                pass

        def quantile_dmatrix(iterator, ref=None):
            iterator.reset()
            while iterator.next(lambda **kwargs: calls.append(kwargs)):
                pass
            return ref

        xgb = SimpleNamespace(
            DataIter=DataIter,
            QuantileDMatrix=quantile_dmatrix,
        )

        XGBoostAdapter(batch_size=None, xgb_dtype=None)._dmatrix(xgb, _source())

        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0]["weight"], [1.0, 1.0, 0.0])

    def test_torch_applies_per_sample_weight(self) -> None:
        src = _source(
            x=(1.0, 1.0),
            y=(0.0, 2.0),
            weight=(1.0, 3.0),
        )

        def build(_params):
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.zero_()
            return model

        def optimizer(parameters, _params):
            return torch.optim.SGD(parameters, lr=0.5)

        adapter = TorchAdapter(
            module_builder=build,
            optimizer_builder=optimizer,
            epochs=1,
            batch_size=None,
            device="cpu",
        )

        model = adapter.fit(adapter.build({}), src, None, NullTracker())

        self.assertAlmostEqual(float(model.weight.detach()[0, 0]), 1.5, places=6)

    def test_torch_rejects_scalar_custom_loss_with_weights(self) -> None:
        adapter = TorchAdapter(
            module_builder=lambda _params: torch.nn.Linear(1, 1),
            loss_fn=torch.nn.MSELoss(),
            epochs=1,
            batch_size=None,
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "per-sample losses"):
            adapter.fit(adapter.build({}), _source(), None, NullTracker())


class WeightedScoreTest(unittest.TestCase):
    def test_unwrapped_score_ignores_available_weight(self) -> None:
        value = call_score(
            rmse,
            np.asarray([0.0, 2.0]),
            np.asarray([0.0, 0.0]),
            {"n": 2},
            combine_with=None,
            sample_weight=np.asarray([1.0, 3.0]),
        )

        self.assertAlmostEqual(float(value), np.sqrt(2.0))

    def test_kwargs_wrapper_does_not_imply_weight_support(self) -> None:
        def wrapped(y_true, y_pred, ctx=None, **kwargs):
            return rmse(y_true, y_pred, ctx, **kwargs)

        value = call_score(
            wrapped,
            np.asarray([0.0, 2.0]),
            np.asarray([0.0, 0.0]),
            {"n": 2},
            combine_with=None,
            sample_weight=np.asarray([1.0, 3.0]),
        )

        self.assertAlmostEqual(float(value), np.sqrt(2.0))

    def test_weighted_rmse_uses_weight_and_combines_streams(self) -> None:
        score = weighted(rmse)
        state = call_score(
            score,
            np.asarray([0.0]),
            np.asarray([0.0]),
            {"n": 1},
            combine_with=None,
            sample_weight=np.asarray([1.0]),
        )
        state = call_score(
            score,
            np.asarray([2.0]),
            np.asarray([0.0]),
            {"n": 1},
            combine_with=state,
            sample_weight=np.asarray([3.0]),
        )

        self.assertAlmostEqual(float(state), np.sqrt(3.0))


class PipelineWeightTest(unittest.TestCase):
    def test_pipeline_propagates_weight_column(self) -> None:
        src = _source()
        pipeline = Pipeline(
            rolling_dates=[["2026-01-02"]],
            adapter=DummyAdapter(),
            target="y",
            features=["x"],
            data_loader=src.loader,
            sample_weight_col="w",
        )

        built = pipeline._src(["2026-01-02"], (), None, "train")

        self.assertEqual(built.sample_weight_col, "w")
        np.testing.assert_array_equal(built.materialize().weight, [1.0, 1.0, 0.0])

    def test_pipeline_rejects_adapter_without_weight_support(self) -> None:
        class UnsupportedAdapter:
            def fit(self, model, train, val, tracker):
                return model

        pipeline = Pipeline(
            rolling_dates=[["2026-01-02"]],
            adapter=UnsupportedAdapter(),
            target="y",
            features=["x"],
            data_loader=_source().loader,
            sample_weight_col="w",
        )

        with self.assertRaisesRegex(TypeError, "does not support sample weights"):
            pipeline._fit_model(None, _source(), None, None)

    def test_pipeline_manifest_round_trips_weight_column(self) -> None:
        src = _source()
        pipeline = Pipeline(
            rolling_dates=[["2026-01-02"]],
            adapter=DummyAdapter(),
            target="y",
            features=["x"],
            data_loader=src.loader,
            test_dates=["2026-01-02"],
            sample_weight_col="w",
        )
        pipeline.model = {"mean": 0.0}

        with TemporaryDirectory() as tmp:
            manifest = pipeline.save_pipeline(tmp)
            loaded = Pipeline(
                rolling_dates=[["placeholder"]],
                adapter=DummyAdapter(),
                target="other",
                features=["other"],
                data_loader=src.loader,
                test_dates=["placeholder"],
            )
            loaded.load_pipeline(tmp)

        self.assertEqual(manifest["version"], 3)
        self.assertEqual(manifest["sample_weight_col"], "w")
        self.assertEqual(loaded.sample_weight_col, "w")


if __name__ == "__main__":
    unittest.main()
