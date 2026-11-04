from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import polars as pl

from gardenpipe.data import DateFrame
from gardenpipe.pipeline import Pipeline, call_score
from gardenpipe.score import on_column, rmse, weighted
from gardenpipe.transform import Passthrough


class ColumnScoreTest(unittest.TestCase):
    def test_weighted_column_score_accepts_scalar_and_multioutput_predictions(self):
        y = np.array([0.0, 0.0])
        p = np.array([0.0, 10.0])
        weight = np.array([1.0, 0.0])
        score = on_column(weighted(rmse), 1)
        for predictions in (p, np.column_stack((p + 20, p))):
            with self.subTest(ndim=predictions.ndim):
                value = call_score(score, y, predictions, {}, sample_weight=weight)
                self.assertEqual(value, 0.0)

    def test_unweighted_column_score_ignores_available_weights(self):
        # **kwargs alone does not opt a score into accepting sample weights.
        def kwargs_score(y_true, y_pred, ctx=None, **kwargs):
            return rmse(y_true, y_pred, ctx, **kwargs)

        for base_score in (rmse, kwargs_score):
            with self.subTest(score=base_score.__name__):
                value = call_score(
                    on_column(base_score, 0),
                    np.array([0.0, 0.0]),
                    np.array([[0.0], [10.0]]),
                    {},
                    sample_weight=np.array([1.0, 0.0]),
                )
                self.assertAlmostEqual(value, np.sqrt(50.0))

    def test_pipeline_combines_weighted_column_scores_across_batches(self):
        # The first batch has no positive weights and must contribute no error.
        frame = pl.DataFrame({
            "x": [100.0, 200.0, 2.0, 4.0],
            "y": [0.0] * 4,
            "w": [0.0, 0.0, 1.0, 3.0],
        })
        calls = []

        def predict(_model, x):
            calls.append(len(x))
            return np.column_stack((x[:, 0] + 50, x[:, 0]))

        pipe = Pipeline(
            rolling_dates=[["2026-01-05"]],
            test_dates=["2026-01-06"],
            adapter=SimpleNamespace(streaming=True, batch_size=2, predict=predict),
            data_loader=lambda dates: [DateFrame(d, "test", frame.lazy()) for d in dates],
            target="y",
            features=["x"],
            sample_weight_col="w",
        )
        pipe.model = object()
        pipe.fitted_transform = Passthrough()

        result = pipe.test(score=on_column(weighted(rmse), 1), keep_predictions=False)

        self.assertAlmostEqual(result["test_score"], np.sqrt(13.0))
        self.assertEqual(result["n"], 4)
        self.assertEqual(calls, [2, 2])
        self.assertIsNone(result["y_pred"])


if __name__ == "__main__":
    unittest.main()
