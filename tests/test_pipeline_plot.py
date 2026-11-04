from __future__ import annotations

import unittest

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - exercised only in minimal envs
    plt = None

from gardenpipe.pipeline import plot_train_val_loss
from gardenpipe.search import create_study


def _record(
    trial: int,
    fold: int,
    weighted_score: float,
    *,
    n_folds: int | None = 2,
) -> dict:
    record = {
        "trial": trial,
        "fold": fold,
        "weighted_score": weighted_score,
        "fit_history": {
            "role": "cv",
            "fold": fold,
            "history": {
                "train": {"loss": [1.0, 0.8]},
                "val": {"loss": [1.1, 0.9]},
            },
        },
    }
    if n_folds is not None:
        record["n_folds"] = n_folds
    return record


@unittest.skipIf(plt is None, "matplotlib is not installed")
class PlotTrainValLossTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_best_trial_skips_incomplete_trial(self) -> None:
        history = {
            "score_direction": "minimize",
            "n_folds": 2,
            "validation_history": [
                _record(0, 0, 0.8),
                _record(0, 1, 0.6),
                _record(1, 0, 0.1),
            ],
        }

        fig = plot_train_val_loss(history)

        titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
        self.assertEqual(len(titles), 2)
        self.assertTrue(all("trial 0" in title for title in titles))

    def test_best_trial_infers_complete_trials_for_old_history(self) -> None:
        history = [
            _record(0, 0, 0.8, n_folds=None),
            _record(0, 1, 0.6, n_folds=None),
            _record(1, 0, 0.1, n_folds=None),
        ]

        fig = plot_train_val_loss(history)

        titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
        self.assertEqual(len(titles), 2)
        self.assertTrue(all("trial 0" in title for title in titles))

    def test_best_trial_prefers_train_result_metadata(self) -> None:
        history = {
            "score_direction": "minimize",
            "best_trial": 2,
            "n_folds": 2,
            "validation_history": [
                _record(0, 0, 0.4),
                _record(0, 1, 0.3),
                _record(2, 0, 0.8),
                _record(2, 1, 0.7),
            ],
        }

        fig = plot_train_val_loss(history)

        titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
        self.assertEqual(len(titles), 2)
        self.assertTrue(all("trial 2" in title for title in titles))

    def test_best_trial_can_include_incomplete_trials(self) -> None:
        history = {
            "score_direction": "minimize",
            "n_folds": 2,
            "validation_history": [
                _record(0, 0, 0.8),
                _record(0, 1, 0.6),
                _record(1, 0, 0.1),
            ],
        }

        fig = plot_train_val_loss(history, complete_trials=False)

        titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
        self.assertEqual(titles, ["cv / trial 1 / fold 0"])


class CreateStudyTest(unittest.TestCase):
    def test_none_pruner_means_no_pruning(self) -> None:
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna is not installed")

        study = create_study("random", None, pruner=None, seed=7)

        self.assertIsInstance(study.pruner, optuna.pruners.NopPruner)

    def test_explicit_pruner_is_preserved(self) -> None:
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna is not installed")

        pruner = optuna.pruners.MedianPruner()
        study = create_study("random", None, pruner=pruner, seed=7)

        self.assertIs(study.pruner, pruner)


if __name__ == "__main__":
    unittest.main()
