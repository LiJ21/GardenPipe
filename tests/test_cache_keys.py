from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import polars as pl

from gardenpipe.data import DateFrame
from gardenpipe.model import DummyAdapter, LarsAdapter, RidgeAdapter
from gardenpipe.pipeline import Pipeline
from gardenpipe.score import rmse
from gardenpipe.transform import Standardizer

DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
POSITIVE = (pl.col("a") > 0,)
NEGATIVE = (pl.col("a") <= 0,)


def _frames() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(0)
    frames = {}
    for date in DATES:
        a, b = rng.normal(size=500), rng.normal(size=500)
        # The slope on b flips sign with a, so the two samplers fit opposite models.
        y = np.where(a > 0, 3 * b, -3 * b) + rng.normal(size=500) * 0.1
        frames[date] = pl.DataFrame({"a": a, "b": b, "y": y})
    return frames


FRAMES = _frames()


def _loader(dates: list[str], hint=None) -> list[DateFrame]:
    return [DateFrame(date, "t", FRAMES[date].lazy()) for date in dates]


def _pipeline(adapter, **kwargs) -> Pipeline:
    return Pipeline(
        rolling_dates=[DATES[:2], DATES[2:3]],
        test_dates=DATES[3:],
        adapter=adapter,
        target="y",
        features=["a", "b"],
        data_loader=_loader,
        n_trials=1,
        **kwargs,
    )


def _ridge() -> RidgeAdapter:
    return RidgeAdapter(alpha=0.0, batch_size=100)


class SamplerCacheKeyTest(unittest.TestCase):
    """Changing a sampler must never reuse results computed for another one."""

    def test_array_cache_follows_the_test_sampler(self) -> None:
        pipe = _pipeline(DummyAdapter(mode="linear"), cache_arrays=True)
        pipe.train()
        pipe.test(sampler=POSITIVE)
        fresh = _pipeline(DummyAdapter(mode="linear"))
        fresh.train()
        self.assertEqual(
            pipe.test(sampler=NEGATIVE)["n"], fresh.test(sampler=NEGATIVE)["n"]
        )

    def test_ridge_stats_follow_the_train_sampler(self) -> None:
        pipe = _pipeline(_ridge(), train_sampler=POSITIVE)
        pipe.train()
        pipe.train_sampler = NEGATIVE
        pipe.train()
        fresh = _pipeline(_ridge(), train_sampler=NEGATIVE)
        fresh.train()
        np.testing.assert_allclose(pipe.model.coef, fresh.model.coef)
        self.assertLess(pipe.model.coef[1], 0)

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") is not None, "scikit-learn is not installed"
    )
    def test_lars_path_scores_follow_the_test_sampler(self) -> None:
        def lars():
            return LarsAdapter(batch_size=100, alpha=0.0, path_eval_alphas=[0.0])

        pipe = _pipeline(lars())
        pipe.train()
        pipe.test(score=rmse, sampler=POSITIVE, keep_predictions=False)
        fresh = _pipeline(lars())
        fresh.train()
        self.assertAlmostEqual(
            pipe.test(score=rmse, sampler=NEGATIVE, keep_predictions=False)[
                "test_score"
            ],
            fresh.test(score=rmse, sampler=NEGATIVE, keep_predictions=False)[
                "test_score"
            ],
        )

    def test_transform_is_refit_for_a_new_train_sampler(self) -> None:
        pipe = _pipeline(
            RidgeAdapter(batch_size=100),
            train_sampler=POSITIVE,
            transform=Standardizer(["a", "b"]),
        )
        pipe.train()
        self.assertGreater(pipe.fitted_transform.means["a"], 0)
        pipe.train_sampler = NEGATIVE
        pipe.train()
        self.assertLess(pipe.fitted_transform.means["a"], 0)

    def test_unchanged_sampler_still_hits_the_cache(self) -> None:
        pipe = _pipeline(_ridge(), train_sampler=POSITIVE)
        pipe.train()
        cached = dict(pipe.adapter._stats_cache)
        pipe.refit()
        self.assertEqual(pipe.adapter._stats_cache, cached)


if __name__ == "__main__":
    unittest.main()
