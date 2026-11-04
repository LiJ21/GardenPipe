from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from gardenpipe.features import EwmaFeature
from gardenpipe.registry import (
    Registry,
    file_set_manifest,
    register_expr,
    register_pickle_feature,
)


class RegistryTest(unittest.TestCase):
    def test_register_expr_reuses_existing_version_for_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.jsonl"

            first = register_expr(
                Registry(path),
                "filter",
                "train_filter_0",
                pl.col("x") > 0,
                run="a",
            )
            second = register_expr(
                Registry(path),
                "filter",
                "train_filter_0",
                pl.col("x") > 0,
                run="b",
            )
            changed = register_expr(
                Registry(path),
                "filter",
                "train_filter_0",
                pl.col("x") >= 0,
                run="c",
            )

            self.assertEqual(first["version"], 1)
            self.assertEqual(second["version"], 1)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(changed["version"], 2)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_register_pickle_feature_reuses_config_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.jsonl"

            first = register_pickle_feature(
                Registry(path),
                "x_ewma_hl1s",
                EwmaFeature("x_ewma_hl1s", pl.col("x"), half_life="1s"),
                run="a",
            )
            second = register_pickle_feature(
                Registry(path),
                "x_ewma_hl1s",
                EwmaFeature("x_ewma_hl1s", pl.col("x"), half_life="1s"),
                run="b",
            )

            self.assertEqual(first["version"], 1)
            self.assertEqual(second["version"], 1)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            entry = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(entry["storage"], "pickle")
            self.assertIn("pickle_sha256", entry)

    def test_file_set_manifest_hashes_files_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.txt"
            b = root / "b.txt"
            a.write_text("alpha")
            b.write_text("bravo")

            first = file_set_manifest([b, a])
            second = file_set_manifest([a, b])
            b.write_text("changed")
            changed = file_set_manifest([a, b])

            self.assertEqual(first["hash"], second["hash"])
            self.assertEqual(len(first["files"]), 2)
            self.assertNotEqual(first["hash"], changed["hash"])


if __name__ == "__main__":
    unittest.main()
