from datetime import time

import polars as pl


def time_of_day(
    start: str | time,
    end: str | time,
    ts: str = "time",
    timezone: str | None = None,
    closed: str = "left",
) -> pl.Expr:
    """Keep rows whose wall-clock time lies between ``start`` and ``end``.

    ``end`` before ``start`` wraps past midnight. ``closed`` picks which bounds
    are inclusive.
    """

    start_t, end_t = _parse_time(start), _parse_time(end)
    t = pl.col(ts)
    if timezone is not None:
        t = t.dt.convert_time_zone(timezone)
    t = t.dt.time()
    left, right = _bounds(t, start_t, end_t, closed)
    return (left & right) if start_t <= end_t else (left | right)


def _bounds(t: pl.Expr, start: time, end: time, closed: str) -> tuple[pl.Expr, pl.Expr]:
    if closed == "left":
        return t >= start, t < end
    if closed == "right":
        return t > start, t <= end
    if closed == "both":
        return t >= start, t <= end
    if closed == "none":
        return t > start, t < end
    raise ValueError("closed must be one of: 'left', 'right', 'both', 'none'")


def _parse_time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("time must be HH:MM or HH:MM:SS")
    hour, minute = map(int, parts[:2])
    second = int(parts[2]) if len(parts) == 3 else 0
    return time(hour, minute, second)
