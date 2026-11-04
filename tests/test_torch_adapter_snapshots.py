import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gardenpipe.data import Batch
from gardenpipe.model import TorchAdapter
from gardenpipe.track import NullTracker


class ArraySource:
    def __init__(self, x, y, weight=None):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float32)

    def batches(self, batch_size=None):
        batch_size = batch_size or len(self.y)
        for start in range(0, len(self.y), batch_size):
            stop = start + batch_size
            yield Batch(
                self.x[start:stop],
                self.y[start:stop],
                {"n": len(self.y[start:stop]), "dates": ["d0"]},
                None if self.weight is None else self.weight[start:stop],
            )


def linear_model(_params):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    return model


def sgd(lr):
    def build(parameters, _params):
        return torch.optim.SGD(parameters, lr=lr)

    return build


class TorchAdapterSnapshotTest(unittest.TestCase):
    def test_cv_log_metrics_are_trial_and_fold_scoped(self):
        adapter = TorchAdapter(module_builder=linear_model)

        metrics = adapter._log_metrics(
            {"torch/train_loss": 1.0, "torch/val_loss": 2.0},
            {"role": "cv", "trial": 7, "fold": 2},
        )

        self.assertEqual(
            metrics,
            {
                "torch/cv/trial_007/fold_02/train_loss": 1.0,
                "torch/cv/trial_007/fold_02/val_loss": 2.0,
            },
        )

    def test_unscoped_log_metrics_preserve_old_tags(self):
        adapter = TorchAdapter(module_builder=linear_model)
        metrics = {"torch/train_loss": 1.0}

        self.assertEqual(adapter._log_metrics(metrics, None), metrics)

    def test_early_stopping_stops_after_patience(self):
        src = ArraySource([[0.0]], [0.0])
        adapter = TorchAdapter(
            module_builder=linear_model,
            optimizer_builder=sgd(0.0),
            epochs=5,
            batch_size=None,
            device="cpu",
            early_stopping_patience=1,
            snapshot_mode="off",
        )

        model = adapter.fit(adapter.build({}), src, src, NullTracker())

        self.assertIsNotNone(model)
        self.assertEqual(adapter.last_fit_history["best_epoch"], 0)
        self.assertEqual(adapter.last_fit_history["stopped_epoch"], 1)
        self.assertEqual(adapter.last_fit_history["num_boosted_rounds"], 2)
        self.assertEqual(adapter.last_fit_history["snapshot_epochs"], [])

    def test_losses_use_row_mass_for_unequal_batches(self):
        for weight, expected in ((None, 3.0), ([1.0, 1.0, 8.0], 7.2)):
            with self.subTest(weight=weight):
                src = ArraySource(
                    [[0.0], [0.0], [0.0]], [0.0, 0.0, 3.0], weight
                )
                adapter = TorchAdapter(
                    module_builder=linear_model,
                    optimizer_builder=sgd(0.0),
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    snapshot_mode="best",
                )

                adapter.fit(adapter.build({}), src, src, NullTracker())

                history = adapter.last_fit_history["history"]
                self.assertAlmostEqual(history["train"]["loss"][0], expected)
                self.assertAlmostEqual(history["val"]["loss"][0], expected)
                self.assertEqual(adapter.last_fit_history["snapshot_epochs"], [0])
                self.assertAlmostEqual(
                    adapter.last_fit_history["snapshot_scores"][0], expected
                )

    def test_ensemble_snapshots_average_predictions(self):
        src = ArraySource([[1.0]], [1.0])
        adapter = TorchAdapter(
            module_builder=linear_model,
            optimizer_builder=sgd(0.25),
            epochs=3,
            batch_size=None,
            device="cpu",
            snapshot_mode="ensemble",
            snapshot_k=2,
        )
        model = adapter.fit(adapter.build({}), src, src, NullTracker())

        pred = adapter.predict(model, np.asarray([[1.0]], dtype=np.float32))

        self.assertEqual(adapter.last_fit_history["snapshot_epochs"], [2, 1])
        self.assertAlmostEqual(float(pred[0]), 0.8125, places=6)

    def test_save_load_preserves_ensemble_snapshots(self):
        src = ArraySource([[1.0]], [1.0])
        adapter = TorchAdapter(
            module_builder=linear_model,
            optimizer_builder=sgd(0.25),
            epochs=3,
            batch_size=None,
            device="cpu",
            snapshot_mode="ensemble",
            snapshot_k=2,
        )
        model = adapter.fit(adapter.build({}), src, src, NullTracker())
        expected = adapter.predict(model, np.asarray([[1.0]], dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmp:
            meta = adapter.save_model(model, Path(tmp))
            loaded = adapter.load_model(Path(tmp), meta)

        actual = adapter.predict(loaded, np.asarray([[1.0]], dtype=np.float32))
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
