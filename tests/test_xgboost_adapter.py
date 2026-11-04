from __future__ import annotations

import importlib.util
import unittest
import warnings
from unittest import mock

import numpy as np
import polars as pl

from gardenpipe.data import DataSource, DateFrame
from gardenpipe.model import XGBoostAdapter
from gardenpipe.pipeline import Pipeline
from gardenpipe.track import NullTracker

HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
FEATURES = [f"f{i}" for i in range(5)]


def _frames() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(0)
    frames = {}
    for date in DATES:
        x = rng.normal(size=(400, len(FEATURES)))
        # Weak signal, heavy noise: deep trees overfit within a few rounds.
        y = x[:, 0] + rng.normal(size=len(x)) * 2
        frames[date] = pl.DataFrame(
            {**{name: x[:, i] for i, name in enumerate(FEATURES)}, "y": y}
        )
    return frames


FRAMES = _frames()


def _loader(dates: list[str], hint=None) -> list[DateFrame]:
    return [DateFrame(date, "t", FRAMES[date].lazy()) for date in dates]


def _source(dates: list[str]) -> DataSource:
    return DataSource(dates=dates, loader=_loader, target="y", features=FEATURES)


PARAMS = {"eta": 0.3, "max_depth": 6}


@unittest.skipUnless(HAS_XGBOOST, "xgboost is not installed")
class XGBoostEarlyStoppingTest(unittest.TestCase):
    def test_model_is_cut_to_the_best_round(self) -> None:
        adapter = XGBoostAdapter(num_boost_round=300, early_stopping_rounds=10)
        train, val = _source(DATES[:1]), _source(DATES[1:2])
        booster = adapter.fit(adapter.build(PARAMS), train, val, NullTracker())
        record = adapter.last_fit_history
        best = record["best_num_boost_round"]
        self.assertLess(best, record["num_boosted_rounds"])
        self.assertEqual(booster.num_boosted_rounds(), best)

        x = FRAMES[DATES[1]].select(FEATURES).to_numpy()
        with self.assertWarnsRegex(RuntimeWarning, "fit.*called again"):
            full = adapter.fit(adapter.build(PARAMS), train, None, NullTracker())
        # Without validation data nothing is cut: the full model is returned.
        self.assertEqual(full.num_boosted_rounds(), 300)
        np.testing.assert_allclose(
            adapter.predict(booster, x),
            full.inplace_predict(x, iteration_range=(0, best)),
        )

    def test_gblinear_warns_and_keeps_final_coefficients(self) -> None:
        import xgboost as xgb

        adapter = XGBoostAdapter(
            num_boost_round=100, early_stopping_rounds=5, streaming=False
        )
        train, val = _source(DATES[:1]), _source(DATES[1:2])
        params = {"booster": "gblinear", "eta": 0.3, "updater": "coord_descent"}
        with self.assertWarnsRegex(RuntimeWarning, "gblinear returns the final"):
            booster = adapter.fit(params, train, val, NullTracker())
        record = adapter.last_fit_history
        self.assertLess(record["best_num_boost_round"], record["num_boosted_rounds"])
        self.assertEqual(booster.num_boosted_rounds(), record["num_boosted_rounds"])
        self.assertLess(booster.num_boosted_rounds(), adapter.num_boost_round)
        reference = xgb.train(
            params,
            xgb.DMatrix(train.materialize().x, label=train.materialize().y),
            num_boost_round=record["num_boosted_rounds"],
        )
        x = val.materialize().x
        np.testing.assert_allclose(
            adapter.predict(booster, x), reference.predict(xgb.DMatrix(x))
        )

    def test_refit_rounds_must_be_known(self) -> None:
        with self.assertRaises(ValueError):
            XGBoostAdapter(refit_rounds="bogus")


@unittest.skipUnless(HAS_XGBOOST, "xgboost is not installed")
class XGBoostRefitRoundsTest(unittest.TestCase):
    def _pipeline(self, **adapter_kwargs) -> Pipeline:
        return Pipeline(
            rolling_dates=[[d] for d in DATES[:3]],
            adapter=XGBoostAdapter(
                num_boost_round=300,
                early_stopping_rounds=10,
                batch_size=None,
                **adapter_kwargs,
            ),
            target="y",
            features=FEATURES,
            data_loader=_loader,
            search_space=lambda trial: dict(PARAMS),
            n_trials=1,
        )

    def _cv_rounds(self, pipe: Pipeline) -> list[int]:
        return [
            item["best_num_boost_round"]
            for item in pipe.adapter.fit_history
            if item["role"] == "cv"
        ]

    def test_refit_without_val_trains_median_cv_rounds(self) -> None:
        pipe = self._pipeline()
        pipe.train()
        expected = int(round(float(np.median(self._cv_rounds(pipe)))))
        self.assertLess(expected, 300)
        self.assertEqual(pipe.refit_history["num_boosted_rounds"], expected)
        self.assertEqual(pipe.model.num_boosted_rounds(), expected)

    def test_repeated_train_clears_history_before_fitting(self) -> None:
        pipe = self._pipeline()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            pipe.train()
        self.assertFalse(any("called again" in str(w.message) for w in caught))
        previous_rounds = self._cv_rounds(pipe)
        pipe.search_space = lambda trial: {"eta": 0.01, "max_depth": 2}
        original_fit = pipe.adapter.fit
        first_fit = True

        def checked_fit(*args, **kwargs):
            nonlocal first_fit
            if first_fit:
                self.assertEqual(pipe.adapter.fit_history, [])
                self.assertIsNone(pipe.adapter.last_fit_history)
                first_fit = False
            return original_fit(*args, **kwargs)

        with mock.patch.object(pipe.adapter, "fit", side_effect=checked_fit):
            with self.assertWarnsRegex(RuntimeWarning, "previous fit history.*cleared"):
                pipe.train()
        rounds = self._cv_rounds(pipe)
        self.assertEqual(len(rounds), 2)
        self.assertNotEqual(rounds, previous_rounds)
        expected = int(round(float(np.median(rounds))))
        self.assertEqual(pipe.model.num_boosted_rounds(), expected)
        self.assertEqual(len(pipe.adapter.fit_history), 3)
        history = list(pipe.adapter.fit_history)
        pipe.refit()
        self.assertEqual(pipe.adapter.fit_history[:-1], history)
        self.assertEqual(pipe.model.num_boosted_rounds(), expected)

    def test_fixed_refit_rounds_keeps_num_boost_round(self) -> None:
        pipe = self._pipeline(refit_rounds="fixed")
        pipe.train()
        self.assertEqual(pipe.model.num_boosted_rounds(), 300)

    def test_refit_with_val_dates_early_stops_and_is_cut(self) -> None:
        pipe = self._pipeline()
        pipe.refit_val_dates = [DATES[2]]
        pipe.train()
        best = pipe.refit_history["best_num_boost_round"]
        self.assertLess(best, pipe.refit_history["num_boosted_rounds"])
        self.assertEqual(pipe.model.num_boosted_rounds(), best)


if __name__ == "__main__":
    unittest.main()
