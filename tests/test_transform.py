from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from gardenpipe.data import Batch, DataSource, DateFrame
from gardenpipe.transform import (
    Chain,
    TargetScaler,
    Standardizer,
    load_transform,
    save_transform,
    transform_to_config,
)


def _loader(_dates: list[str]) -> list[DateFrame]:
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "rv": [2.0, 4.0, 0.0],
            "ret": [10.0, 20.0, 30.0],
        }
    )
    return [DateFrame(date="2026-01-02", tag="test", lf=df.lazy())]


def _source(transform=None) -> DataSource:
    return DataSource(
        dates=["2026-01-02"],
        loader=_loader,
        target="ret",
        features=["x"],
        transform=transform,
    )


class StandardizerTest(unittest.TestCase):
    def source(self, frames: list[pl.DataFrame]) -> DataSource:
        by_date = {f"2026-01-{i + 1:02d}": frame for i, frame in enumerate(frames)}
        return DataSource(
            dates=list(by_date),
            loader=lambda dates: [DateFrame(d, "test", by_date[d].lazy()) for d in dates],
            target="y",
            features=[name for name in frames[0].columns if name != "y"],
        )

    def test_large_offset_variance_matches_numpy_across_batches_and_dates(self) -> None:
        frame = pl.DataFrame({
            "x": 1e9 + np.arange(10),
            "z": -1e9 + np.arange(10) * 3.0,
            "centered": np.linspace(-1.0, 1.0, 10),
            "y": np.zeros(10),
        })
        for frames in ([frame], [frame.head(3), frame.slice(3)]):
            for batch_size in (1, 3, 10, None):
                with self.subTest(dates=len(frames), batch_size=batch_size):
                    fitted = Standardizer(["z", "centered", "x"], batch_size=batch_size).fit(
                        self.source(frames)
                    )
                    for name in ("z", "centered", "x"):
                        values = frame[name].to_numpy()
                        np.testing.assert_allclose(
                            fitted.means[name], values.mean(), rtol=0, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            fitted.stds[name], values.std(ddof=1), rtol=1e-10
                        )

    def test_missing_constant_and_singleton_columns(self) -> None:
        frame = pl.DataFrame({
            "x": [None, None, 1e9, 1e9 + 1, None, None, 1e9 + 2, 1e9 + 3],
            "z": [1e9, 1e9 + 1, None, None, 1e9 + 2, 1e9 + 3, None, None],
            "constant": [5.0, None, 5.0, 5.0, None, 5.0, 5.0, 5.0],
            "singleton": [None, None, None, 7.0, None, None, None, None],
            "missing": [float("nan")] * 8,
            "y": np.zeros(8),
        })
        for batch_size in (1, 2, 3, 8):
            with self.subTest(batch_size=batch_size):
                fitted = Standardizer(frame.columns[:-1], batch_size=batch_size).fit(
                    self.source([frame])
                )
                for name in frame.columns[:-1]:
                    values = frame[name].to_numpy()
                    values = values[~np.isnan(values)]
                    mean = values.mean() if len(values) else 0.0
                    std = values.std(ddof=1) if len(values) > 1 else 0.0
                    np.testing.assert_allclose(fitted.means[name], mean, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(fitted.stds[name], std or 1.0)

    def test_fit_uses_one_batch_pass_without_mutating_inputs(self) -> None:
        values = (1e9 + np.arange(10)).reshape(-1, 1)

        class BatchOnlySource:
            features = ["x"]
            calls = 0

            def batches(self, batch_size):
                self.calls += 1
                if self.calls != 1:
                    raise AssertionError("standardization must use a single pass")
                yield Batch(values[:0], np.empty(0), {})
                for start in range(0, len(values), batch_size):
                    x = values[start : start + batch_size]
                    yield Batch(x, np.zeros(len(x)), {})

        source = BatchOnlySource()
        fitted = Standardizer(["x"], batch_size=3).fit(source)
        self.assertEqual(source.calls, 1)
        np.testing.assert_array_equal(values[:, 0], 1e9 + np.arange(10))
        np.testing.assert_allclose(fitted.stds["x"], values.std(ddof=1))


class TargetScalerTest(unittest.TestCase):
    def test_fit_infers_target_from_source(self) -> None:
        transform = TargetScaler("rv")

        fitted = transform.fit(_source())

        self.assertIs(fitted, transform)
        self.assertEqual(transform.target, "ret")

    def test_materialize_normalizes_target_without_changing_features(self) -> None:
        transform = TargetScaler("rv").fit(_source())

        x, y, _ = _source(transform).materialize()

        np.testing.assert_allclose(x[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(y, [5.0, 5.0, 30.0 / 1e-12])

    def test_explicit_target_supports_lazy_frame_transform(self) -> None:
        lf = pl.DataFrame({"rv": [2.0], "ret": [8.0]}).lazy()
        transform = TargetScaler("rv", target="ret")

        out = transform.transform(lf).collect()

        self.assertEqual(out["ret"].to_list(), [4.0])

    def test_transform_requires_fitted_or_explicit_target(self) -> None:
        with self.assertRaises(RuntimeError):
            TargetScaler("rv").transform(pl.DataFrame({"rv": [1.0]}).lazy())

    def test_chain_normalizes_by_raw_feature_before_standardizing(self) -> None:
        transform = Chain(
            [
                TargetScaler("rv"),
                Standardizer(["rv"], batch_size=None),
            ]
        ).fit(
            DataSource(
                dates=["2026-01-02"],
                loader=_loader,
                target="ret",
                features=["rv"],
            )
        )

        _, y, _ = DataSource(
            dates=["2026-01-02"],
            loader=_loader,
            target="ret",
            features=["rv"],
            transform=transform,
        ).materialize()

        np.testing.assert_allclose(y, [5.0, 5.0, 30.0 / 1e-12])

    def test_config_and_pickle_round_trip(self) -> None:
        transform = TargetScaler("rv").fit(_source())

        config = transform_to_config(transform)
        self.assertEqual(config["type"], "TargetScaler")
        self.assertEqual(config["feature"], "rv")
        self.assertEqual(config["target"], "ret")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transform.pkl"
            meta = save_transform(transform, path)
            loaded = load_transform(path, meta)

        self.assertIsInstance(loaded, TargetScaler)
        self.assertEqual(loaded.feature, "rv")
        self.assertEqual(loaded.target, "ret")


if __name__ == "__main__":
    unittest.main()
