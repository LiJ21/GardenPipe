from __future__ import annotations

import unittest

import polars as pl

from gardenpipe.viewer import DataViewer, Layer, Panel, Series


def _viewer(
    frame: pl.DataFrame,
    series: list[Series],
    *,
    max_points: int | None = 200_000,
    resample: str = "auto",
) -> DataViewer:
    return DataViewer(
        panels=[Panel(layers=[Layer(source=frame, x="x", series=series)])],
        max_points=max_points,
        resample=resample,
    )


class LineSeriesTest(unittest.TestCase):
    def test_line_is_straight_without_markers(self) -> None:
        frame = pl.DataFrame({"x": range(4), "prediction": [0.1, 0.2, 0.3, 0.4]})

        trace = _viewer(
            frame,
            [
                Series(
                    "prediction",
                    kind="line",
                    line_shape="hv",
                    point_size=12,
                )
            ],
        ).figure().data[0]

        self.assertEqual(trace.mode, "lines")
        self.assertEqual(trace.line.shape, "linear")

    def test_same_length_lines_share_uniform_sampled_timestamps(self) -> None:
        frame = pl.DataFrame(
            {
                "x": range(10),
                "prediction": [float(value) for value in range(10)],
                "forward_change": [float(-value) for value in range(10)],
            }
        )

        traces = _viewer(
            frame,
            [
                Series("prediction", kind="line"),
                Series("forward_change", kind="line"),
            ],
            max_points=4,
        ).figure().data

        self.assertEqual(list(traces[0].x), [0, 3, 6, 9])
        self.assertEqual(list(traces[1].x), list(traces[0].x))
        self.assertEqual(list(traces[0].y), [0.0, 3.0, 6.0, 9.0])
        self.assertEqual(list(traces[1].y), [0.0, -3.0, -6.0, -9.0])

    def test_resample_none_keeps_all_line_points(self) -> None:
        frame = pl.DataFrame({"x": range(5), "prediction": range(5)})

        trace = _viewer(
            frame,
            [Series("prediction", kind="line", resample="none")],
            max_points=2,
        ).figure().data[0]

        self.assertEqual(list(trace.x), [0, 1, 2, 3, 4])


class ExistingSeriesBehaviorTest(unittest.TestCase):
    def test_level_retains_change_points_and_step_style(self) -> None:
        frame = pl.DataFrame({"x": range(6), "level": [1, 1, 2, 2, 3, 3]})

        trace = _viewer(
            frame,
            [Series("level", kind="level")],
            max_points=4,
        ).figure().data[0]

        self.assertEqual(list(trace.x), [0, 2, 4, 5])
        self.assertEqual(trace.mode, "lines+markers")
        self.assertEqual(trace.line.shape, "hv")

    def test_event_retains_uniform_sampling_and_marker_style(self) -> None:
        frame = pl.DataFrame({"x": range(10), "event": range(10)})

        trace = _viewer(
            frame,
            [Series("event", kind="event")],
            max_points=4,
        ).figure().data[0]

        self.assertEqual(list(trace.x), [0, 3, 6, 9])
        self.assertEqual(trace.mode, "markers")


if __name__ == "__main__":
    unittest.main()
