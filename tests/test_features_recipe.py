from __future__ import annotations

import unittest

import polars as pl

from gardenpipe.features_recipe import (
    CarryForwardCarryover,
    CarryForwardFeature,
    EwmaCarryover,
    EwmaFeature,
    SELF_CARRYOVER,
)


class CarryForwardFeatureTest(unittest.TestCase):
    def test_emits_last_non_null_value_strictly_before_row(self) -> None:
        df = pl.DataFrame({"x": [None, 1.0, None, -1.0]})
        feature = CarryForwardFeature("prev_x", pl.col("x"))

        out = feature.apply(df.lazy(), None).collect()

        self.assertEqual(out["prev_x"].to_list(), [None, None, 1.0, 1.0])
        self.assertEqual(out["__prev_x_current"].to_list(), [None, 1.0, 1.0, -1.0])
        self.assertEqual(
            feature.get_carryover(out),
            {SELF_CARRYOVER: CarryForwardCarryover(value=-1.0)},
        )

    def test_carryover_seeds_first_rows_and_tracks_current_input(self) -> None:
        df = pl.DataFrame({"x": [None, 3.0]})
        feature = CarryForwardFeature("prev_x", pl.col("x"))

        out = feature.apply(
            df.lazy(),
            {SELF_CARRYOVER: CarryForwardCarryover(value=2.0)},
        ).collect()

        self.assertEqual(out["prev_x"].to_list(), [2.0, 2.0])
        self.assertEqual(out["__prev_x_current"].to_list(), [2.0, 3.0])
        self.assertEqual(
            feature.get_carryover(out),
            {SELF_CARRYOVER: CarryForwardCarryover(value=3.0)},
        )

    def test_front_pad_seeds_first_active_row(self) -> None:
        df = pl.DataFrame({"x": [9.0, None, 3.0]})
        feature = CarryForwardFeature("prev_x", pl.col("x"))

        out = feature.apply(
            df.lazy(),
            {SELF_CARRYOVER: CarryForwardCarryover(value=2.0)},
            front_pad=2,
        ).collect()

        self.assertEqual(out["prev_x"].to_list(), [None, None, 2.0])
        self.assertEqual(out["__prev_x_current"].to_list(), [None, 2.0, 3.0])

    def test_precision_float32(self) -> None:
        df = pl.DataFrame({"x": [None, 1.0, None]})
        feature = CarryForwardFeature("prev_x", pl.col("x"), precision="float32")

        out = feature.apply(df.lazy(), None).collect()

        self.assertEqual(out.schema["prev_x"], pl.Float32)
        self.assertEqual(out.schema["__prev_x_input"], pl.Float32)
        self.assertEqual(out.schema["__prev_x_current"], pl.Float32)


class EwmaFeatureTest(unittest.TestCase):
    def test_normalized_ewma_carries_on_null_input(self) -> None:
        df = pl.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000],
                "x": [1.0, None, None],
            }
        )
        feature = EwmaFeature(
            "ewm",
            pl.col("x"),
            half_life="1s",
            time="time",
            normalized=True,
        )

        out = feature.apply(df.lazy(), None).collect()

        self.assertEqual(out["__ewm_input"].to_list(), [1.0, None, None])
        self.assertEqual(out["__ewm_state_time_ns"].to_list(), [0, 0, 0])
        self.assertEqual(out["ewm"].to_list(), [1.0, 1.0, 1.0])
        self.assertEqual(
            feature.get_carryover(out),
            {SELF_CARRYOVER: EwmaCarryover(time_ns=0, value=1.0)},
        )

    def test_fractional_duration_string_matches_numeric_half_life(self) -> None:
        df = pl.DataFrame(
            {
                "time": [0, 100_000_000, 250_000_000, 600_000_000],
                "x": [1.0, 2.0, -1.0, 3.0],
            }
        )
        for normalized in (True, False):
            from_str = EwmaFeature(
                "ewm",
                pl.col("x"),
                half_life="0.1s",
                time="time",
                normalized=normalized,
            )
            from_seconds = EwmaFeature(
                "ewm",
                pl.col("x"),
                half_life=0.1,
                time="time",
                normalized=normalized,
            )

            str_out = from_str.apply(df.lazy(), None).collect()
            seconds_out = from_seconds.apply(df.lazy(), None).collect()

            for left, right in zip(
                str_out["ewm"], seconds_out["ewm"], strict=True
            ):
                self.assertAlmostEqual(left, right)

    def test_unnormalized_ewma_decays_on_null_input(self) -> None:
        df = pl.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000],
                "x": [1.0, None, None],
            }
        )
        feature = EwmaFeature(
            "ewm",
            pl.col("x"),
            half_life="1s",
            time="time",
            normalized=False,
        )

        out = feature.apply(df.lazy(), None).collect()

        self.assertEqual(out["__ewm_input"].to_list(), [1.0, 0.0, 0.0])
        self.assertEqual(
            out["__ewm_state_time_ns"].to_list(),
            [0, 1_000_000_000, 2_000_000_000],
        )
        self.assertEqual(out["ewm"].to_list(), [1.0, 0.5, 0.25])

    def test_normalized_carryover_carries_on_null_input(self) -> None:
        df = pl.DataFrame(
            {
                "time": [1_000_000_000],
                "x": [None],
            }
        )
        feature = EwmaFeature(
            "ewm",
            pl.col("x"),
            half_life="1s",
            time="time",
        )

        out = feature.apply(
            df.lazy(),
            {SELF_CARRYOVER: EwmaCarryover(time_ns=0, value=1.0)},
        ).collect()

        self.assertEqual(out["__ewm_input"].to_list(), [None])
        self.assertEqual(out["__ewm_state_time_ns"].to_list(), [0])
        self.assertEqual(out["ewm"].to_list(), [1.0])
        self.assertEqual(
            feature.get_carryover(out),
            {SELF_CARRYOVER: EwmaCarryover(time_ns=0, value=1.0)},
        )

    def test_normalized_carryover_uses_elapsed_time_to_next_observation(self) -> None:
        df = pl.DataFrame(
            {
                "time": [1_000_000_000, 2_000_000_000],
                "x": [None, 0.0],
            }
        )
        feature = EwmaFeature(
            "ewm",
            pl.col("x"),
            half_life="1s",
            time="time",
        )

        out = feature.apply(
            df.lazy(),
            {SELF_CARRYOVER: EwmaCarryover(time_ns=0, value=1.0)},
        ).collect()

        self.assertEqual(out["__ewm_input"].to_list(), [None, 0.0])
        self.assertEqual(out["__ewm_state_time_ns"].to_list(), [0, 2_000_000_000])
        self.assertEqual(out["ewm"].to_list(), [1.0, 0.25])

    def test_normalized_batch_with_null_gap_matches_single_pass(self) -> None:
        df = pl.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000],
                "x": [1.0, None, 0.0],
            }
        )
        feature = EwmaFeature(
            "ewm",
            pl.col("x"),
            half_life="1s",
            time="time",
        )

        single = feature.apply(df.lazy(), None).collect()
        head = feature.apply(df.head(1).lazy(), None).collect()
        tail = feature.apply(
            df.tail(2).lazy(),
            feature.get_carryover(head),
        ).collect()

        self.assertEqual(single["ewm"].to_list(), [1.0, 1.0, 0.25])
        self.assertEqual(tail["ewm"].to_list(), single["ewm"].tail(2).to_list())

    def test_unnormalized_carryover_decays_on_null_input(self) -> None:
        df = pl.DataFrame(
            {
                "time": [1_000_000_000],
                "x": [None],
            }
        )
        feature = EwmaFeature(
            "ewm",
            pl.col("x"),
            half_life="1s",
            time="time",
            normalized=False,
        )

        out = feature.apply(
            df.lazy(),
            {SELF_CARRYOVER: EwmaCarryover(time_ns=0, value=1.0)},
        ).collect()

        self.assertEqual(out["__ewm_input"].to_list(), [0.0])
        self.assertEqual(out["__ewm_state_time_ns"].to_list(), [1_000_000_000])
        self.assertEqual(out["ewm"].to_list(), [0.5])

    def test_precision_float32(self) -> None:
        df = pl.DataFrame(
            {
                "time": [0, 1_000_000_000],
                "x": [1.0, None],
            }
        )
        feature = EwmaFeature(
            "ewm",
            pl.col("x"),
            half_life="1s",
            time="time",
            normalized=False,
            precision="float32",
        )

        out = feature.apply(df.lazy(), None).collect()

        self.assertEqual(out.schema["__ewm_input"], pl.Float32)
        self.assertEqual(out.schema["ewm"], pl.Float32)
        self.assertAlmostEqual(out["ewm"][1], 0.5)


if __name__ == "__main__":
    unittest.main()
