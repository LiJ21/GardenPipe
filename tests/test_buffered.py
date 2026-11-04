from __future__ import annotations

import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import polars as pl

import gardenpipe.buffered as buffered
from gardenpipe.buffered import (
    BufferConfig,
    BufferDeadlockError,
    BufferedDataSource,
    CreditQueue,
    buffered_wrapper,
    required_buffer_bytes,
)
from gardenpipe.data import DataSource, DateFrame
from gardenpipe.model import DummyAdapter, RidgeAdapter, TorchAdapter, XGBoostAdapter
from gardenpipe.pipeline import Pipeline
from gardenpipe.transform import FunctionTransform

DATES = ["2026-01-05", "2026-01-06", "2026-01-07"]
FEATURES = ["f1", "f2"]
ROWS_PER_DATE = 100
DATE_ID_STRIDE = 10_000


def _frames(dates=DATES, rows=ROWS_PER_DATE) -> dict[str, pl.DataFrame]:
    frames = {}
    for i, date in enumerate(dates):
        ids = np.arange(rows, dtype=np.float64) + i * DATE_ID_STRIDE
        frames[date] = pl.DataFrame(
            {"f1": ids * 2.0, "f2": -ids, "y": ids}
        )
    return frames


def _make_loader(frames: dict[str, pl.DataFrame]):
    def loader(dates: list[str]) -> list[DateFrame]:
        return [
            DateFrame(date=d, tag="default", lf=frames[d].lazy()) for d in dates
        ]

    return loader


def _source(dates=DATES, frames=None, **kwargs) -> DataSource:
    frames = frames if frames is not None else _frames(dates)
    return DataSource(
        dates=list(dates),
        loader=_make_loader(frames),
        target="y",
        features=list(FEATURES),
        **kwargs,
    )


def _config(**kwargs) -> BufferConfig:
    kwargs.setdefault("watchdog_interval", 2.0)
    kwargs.setdefault("join_timeout", 2.0)
    return BufferConfig(**kwargs)


def _by_date(batches):
    grouped: dict[str, list] = {}
    for x, y, ctx in batches:
        grouped.setdefault(ctx["dates"][0], []).append((x, y))
    return grouped


def _buffered_threads():
    return [
        t
        for t in threading.enumerate()
        if t.name.startswith("buffered-") and t.is_alive()
    ]


def _wait_for_no_buffered_threads(timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _buffered_threads():
            return True
        time.sleep(0.02)
    return not _buffered_threads()


def _drain(batches, timeout: float = 10.0) -> list:
    """Consume batches in a thread so a stall fails the test instead of hanging."""
    out: list = []
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            out.extend(batches)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
            errors.append(exc)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise AssertionError(f"batch stream did not finish within {timeout}s")
    if errors:
        raise errors[0]
    return out


class CreditQueueTests(unittest.TestCase):
    def test_reserve_blocks_until_release(self):
        q = CreditQueue(100)
        first = q.reserve(60)
        results = []
        thread = threading.Thread(target=lambda: results.append(q.reserve(60)))
        thread.start()
        thread.join(timeout=0.2)
        self.assertTrue(thread.is_alive())
        first.release()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(q.used_bytes, 60)
        results[0].release()
        self.assertEqual(q.used_bytes, 0)

    def test_oversized_reservation_admitted_when_empty(self):
        q = CreditQueue(100)
        big = q.reserve(1000)
        self.assertEqual(q.used_bytes, 1000)
        big.release()
        self.assertEqual(q.used_bytes, 0)

    def test_sentinel_bypasses_full_budget(self):
        q = CreditQueue(100)
        held = q.reserve(100)
        q.put("sentinel")
        item, reservation = q.get(timeout=0.5)
        self.assertEqual(item, "sentinel")
        self.assertIsNone(reservation)
        held.release()

    def test_release_is_idempotent(self):
        q = CreditQueue(100)
        reservation = q.reserve(50)
        reservation.release()
        reservation.release()
        self.assertEqual(q.used_bytes, 0)

    def test_resize(self):
        q = CreditQueue(100)
        reservation = q.reserve(50)
        reservation.resize(20)
        self.assertEqual(q.used_bytes, 20)
        reservation.release()
        self.assertEqual(q.used_bytes, 0)
        with self.assertRaises(RuntimeError):
            reservation.resize(10)

    def test_reserve_aborts_on_stop(self):
        q = CreditQueue(100)
        held = q.reserve(60)
        stop = threading.Event()
        stop.set()
        self.assertIsNone(q.reserve(60, stop=stop, poll=0.01))
        held.release()

    def test_get_timeout_raises_empty(self):
        q = CreditQueue(100)
        with self.assertRaises(queue.Empty):
            q.get(timeout=0.05)

    def test_drain_releases_queued_credits(self):
        q = CreditQueue(100)
        q.put("a", q.reserve(30))
        q.put("b", q.reserve(30))
        outside = q.reserve(20)
        q.drain()
        self.assertEqual(q.used_bytes, 20)
        outside.release()
        self.assertEqual(q.used_bytes, 0)

    def test_multi_producer_stress_settles_to_zero(self):
        q = CreditQueue(64)
        n_producers, per_producer, cost = 4, 25, 8

        def produce():
            for _ in range(per_producer):
                q.put("item", q.reserve(8))

        threads = [threading.Thread(target=produce) for _ in range(n_producers)]
        for t in threads:
            t.start()
        held = None
        for _ in range(n_producers * per_producer):
            item, reservation = q.get(timeout=5.0)
            self.assertEqual(item, "item")
            if held is not None:
                held.release()
            held = reservation
        for t in threads:
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive())
        self.assertEqual(q.used_bytes, cost)
        held.release()
        self.assertEqual(q.used_bytes, 0)

    def test_get_reports_stall_when_no_waiting_claim_fits(self):
        q = CreditQueue(100)
        q.start_producers(1)
        held = q.reserve(80)
        waiter = threading.Thread(target=q.reserve, args=(50,), daemon=True)
        waiter.start()
        with self.assertRaises(buffered._CreditDeadlock):
            q.get(timeout=5.0)
        held.release()
        waiter.join(timeout=5.0)
        self.assertFalse(waiter.is_alive())

    def test_waiting_claim_that_fits_is_not_a_stall(self):
        # A producer woken by a release still counts as waiting until it runs.
        q = CreditQueue(100)
        q.start_producers(1)
        q._waiting[50] += 1
        with self.assertRaises(queue.Empty):
            q.get(timeout=0.05)


class BufferConfigTests(unittest.TestCase):
    def test_row_shuffle_requires_minimal_ctx(self):
        with self.assertRaises(ValueError):
            BufferConfig(row_shuffle_rows=16, train_ctx="full")
        BufferConfig(row_shuffle_rows=16, train_ctx="minimal")

    def test_validation(self):
        with self.assertRaises(ValueError):
            BufferConfig(workers=0)
        with self.assertRaises(ValueError):
            BufferConfig(max_buffer_bytes=0)
        with self.assertRaises(ValueError):
            BufferConfig(train_ctx="none")


class BufferedSourceTests(unittest.TestCase):
    def test_equivalence_with_bare_source(self):
        bare = list(_source().batches(32))
        wrapped = BufferedDataSource(_source(), _config(workers=3))
        got = list(wrapped.batches(32))
        self.assertTrue(_wait_for_no_buffered_threads())

        want_by_date = _by_date(bare)
        got_by_date = _by_date(got)
        self.assertEqual(sorted(want_by_date), sorted(got_by_date))
        for date, want in want_by_date.items():
            have = got_by_date[date]
            self.assertEqual(len(want), len(have))
            for (wx, wy), (hx, hy) in zip(want, have):
                np.testing.assert_array_equal(wx, hx)
                np.testing.assert_array_equal(wy, hy)

    def test_equivalence_with_filters_and_empty_date(self):
        filters = (pl.col("y") >= DATE_ID_STRIDE,)
        bare = list(_source(filters=filters).batches(32))
        wrapped = BufferedDataSource(_source(filters=filters), _config(workers=3))
        got = list(wrapped.batches(32))
        self.assertEqual(
            sorted(np.concatenate([b[1] for b in bare]).tolist()),
            sorted(np.concatenate([b[1] for b in got]).tolist()),
        )

    def test_empty_dates_yields_nothing(self):
        wrapped = BufferedDataSource(_source(dates=[]), _config(workers=2))
        self.assertEqual(list(wrapped.batches(32)), [])

    def test_single_worker_preserves_global_order(self):
        bare = list(_source().batches(32))
        wrapped = BufferedDataSource(_source(), _config(workers=1))
        got = list(wrapped.batches(32))
        self.assertEqual(len(bare), len(got))
        for (wx, wy, _), (hx, hy, _) in zip(bare, got):
            np.testing.assert_array_equal(wx, hx)
            np.testing.assert_array_equal(wy, hy)

    def test_delegation_and_materialize(self):
        src = _source()
        wrapped = BufferedDataSource(src, _config())
        self.assertEqual(wrapped.features, src.features)
        self.assertEqual(wrapped.target, "y")
        self.assertIsNone(wrapped.cache_key)
        self.assertEqual(wrapped.count(), len(DATES) * ROWS_PER_DATE)
        wx, wy, _ = wrapped.materialize()
        bx, by, _ = _source().materialize()
        np.testing.assert_array_equal(wx, bx)
        np.testing.assert_array_equal(wy, by)

    def test_with_transform_rewraps(self):
        wrapped = BufferedDataSource(_source(), _config())
        shifted = wrapped.with_transform(
            FunctionTransform(
                lambda lf: lf.with_columns((pl.col("f1") + 1.0).alias("f1"))
            )
        )
        self.assertIsInstance(shifted, BufferedDataSource)
        self.assertEqual(shifted.config, wrapped.config)
        x0, _, _ = wrapped.materialize()
        x1, _, _ = shifted.materialize()
        np.testing.assert_allclose(x1[:, 0], x0[:, 0] + 1.0)

    def test_producer_error_propagates(self):
        class BoomError(Exception):
            pass

        frames = _frames()

        def loader(dates: list[str]) -> list[DateFrame]:
            if "2026-01-06" in dates:
                raise BoomError("boom")
            return [
                DateFrame(date=d, tag="default", lf=frames[d].lazy())
                for d in dates
            ]

        src = DataSource(
            dates=list(DATES), loader=loader, target="y", features=list(FEATURES)
        )
        wrapped = BufferedDataSource(src, _config(workers=2))
        with self.assertRaises(BoomError):
            list(wrapped.batches(32))
        self.assertTrue(_wait_for_no_buffered_threads())

    def test_abandoned_iterator_stops_producers(self):
        wrapped = BufferedDataSource(
            _source(), _config(workers=2, max_buffer_bytes=1 << 14)
        )
        it = wrapped.batches(16)
        next(it)
        it.close()
        self.assertTrue(_wait_for_no_buffered_threads())

    def test_watchdog_raises_instead_of_hanging(self):
        with mock.patch.object(buffered._Run, "_worker", lambda self: None):
            wrapped = BufferedDataSource(
                _source(), _config(workers=1, watchdog_interval=0.2)
            )
            with self.assertRaises(RuntimeError):
                list(wrapped.batches(32))

    def test_minimal_ctx(self):
        wrapped = BufferedDataSource(
            _source(), _config(workers=1, train_ctx="minimal")
        )
        batches = list(wrapped.batches(32))
        self.assertEqual(
            sum(b[2]["n"] for b in batches), len(DATES) * ROWS_PER_DATE
        )
        for _, _, ctx in batches:
            self.assertEqual(set(ctx), {"n"})


class DateShuffleTests(unittest.TestCase):
    DATES6 = [f"2026-01-{d:02d}" for d in range(5, 11)]

    def _wrapped(self, seed=11, **kwargs):
        return BufferedDataSource(
            _source(dates=self.DATES6),
            _config(workers=1, shuffle_dates=True, seed=seed, **kwargs),
        )

    def test_orders_reproducible_and_epoch_varying(self):
        def date_sequence(wrapped):
            return [
                ctx["dates"][0] for _, _, ctx in wrapped.batches(64)
            ]

        a, b = self._wrapped(), self._wrapped()
        epoch_a0, epoch_a1, epoch_a2 = (date_sequence(a) for _ in range(3))
        epoch_b0 = date_sequence(b)
        self.assertEqual(epoch_a0, epoch_b0)
        self.assertEqual(sorted(set(epoch_a0)), sorted(self.DATES6))
        self.assertTrue(epoch_a0 != epoch_a1 or epoch_a1 != epoch_a2)

    def test_bucketed_order_with_sizes(self):
        sizes = {d: float(i) for i, d in enumerate(self.DATES6)}
        wrapped = BufferedDataSource(
            _source(dates=self.DATES6),
            _config(
                workers=1,
                shuffle_dates=True,
                seed=3,
                size_buckets=3,
                date_sizes=sizes.__getitem__,
            ),
        )
        order = wrapped._date_order(seed=7)
        self.assertEqual(sorted(order), sorted(self.DATES6))
        buckets = [order[0:2], order[2:4], order[4:6]]
        self.assertEqual({sizes[d] for d in buckets[0]}, {5.0, 4.0})
        self.assertEqual({sizes[d] for d in buckets[1]}, {3.0, 2.0})
        self.assertEqual({sizes[d] for d in buckets[2]}, {1.0, 0.0})


class RowShuffleTests(unittest.TestCase):
    def _wrapped(self, seed=123, **kwargs):
        kwargs.setdefault("workers", 1)
        kwargs.setdefault("row_shuffle_rows", 64)
        kwargs.setdefault("train_ctx", "minimal")
        return BufferedDataSource(_source(), _config(seed=seed, **kwargs))

    def test_rows_preserved_and_dates_not_mixed(self):
        wrapped = self._wrapped()
        batches = list(wrapped.batches(32))
        ys = np.concatenate([b[1] for b in batches])
        bare_ys = np.concatenate([b[1] for b in _source().batches(32)])
        np.testing.assert_array_equal(np.sort(ys), np.sort(bare_ys))
        for x, y, _ in batches:
            self.assertEqual(len({int(v) // DATE_ID_STRIDE for v in y}), 1)
            np.testing.assert_array_equal(x[:, 0], y * 2.0)
            np.testing.assert_array_equal(x[:, 1], -y)

    def test_shuffle_changes_batch_composition(self):
        batches = list(self._wrapped().batches(32))
        bare = list(_source().batches(32))
        shuffled_any = any(
            not np.array_equal(np.sort(h[1]), np.sort(w[1]))
            for h, w in zip(batches, bare)
        )
        self.assertTrue(shuffled_any)

    def test_same_seed_reproduces_stream(self):
        a = list(self._wrapped(seed=42).batches(32))
        b = list(self._wrapped(seed=42).batches(32))
        self.assertEqual(len(a), len(b))
        for (ax, ay, _), (bx, by, _) in zip(a, b):
            np.testing.assert_array_equal(ax, bx)
            np.testing.assert_array_equal(ay, by)

    def test_epochs_differ(self):
        wrapped = self._wrapped(seed=7)
        epoch0 = np.concatenate([b[1] for b in wrapped.batches(32)])
        epoch1 = np.concatenate([b[1] for b in wrapped.batches(32)])
        np.testing.assert_array_equal(np.sort(epoch0), np.sort(epoch1))
        self.assertFalse(np.array_equal(epoch0, epoch1))

    def test_window_smaller_than_batch_size(self):
        wrapped = self._wrapped(row_shuffle_rows=16)
        batches = list(wrapped.batches(32))
        ys = np.concatenate([b[1] for b in batches])
        self.assertEqual(len(ys), len(DATES) * ROWS_PER_DATE)


class GuardTests(unittest.TestCase):
    def _shuffled(self):
        return BufferedDataSource(
            _source(), _config(workers=1, shuffle_dates=True)
        )

    def _fake_torch(self):
        return SimpleNamespace(
            utils=SimpleNamespace(
                data=SimpleNamespace(
                    IterableDataset=object,
                    DataLoader=lambda dataset, batch_size=None: dataset,
                )
            )
        )

    def test_flags(self):
        plain = BufferedDataSource(_source(), _config(workers=1))
        self.assertFalse(plain.is_shuffled)
        self.assertTrue(plain.is_deterministic)
        shuffled = self._shuffled()
        self.assertTrue(shuffled.is_shuffled)
        self.assertFalse(shuffled.is_deterministic)
        multi = BufferedDataSource(_source(), _config(workers=2))
        self.assertFalse(multi.is_shuffled)
        self.assertFalse(multi.is_deterministic)

    def test_torch_loader_rejects_shuffled_shards(self):
        adapter = TorchAdapter(module_builder=lambda params: None)
        with self.assertRaises(ValueError):
            adapter._loader(self._fake_torch(), self._shuffled(), 0, 2)
        adapter._loader(self._fake_torch(), self._shuffled(), 0, 1)
        adapter._loader(self._fake_torch(), _source(), 0, 2)

    def test_xgboost_rejects_shuffled_stream(self):
        adapter = XGBoostAdapter(streaming=True)
        fake_xgb = SimpleNamespace(
            DataIter=object,
            QuantileDMatrix=lambda it, ref=None: ("qdm", it),
        )
        with self.assertRaises(ValueError):
            adapter._dmatrix(fake_xgb, self._shuffled())
        adapter._dmatrix(fake_xgb, _source())
        adapter._dmatrix(
            fake_xgb, BufferedDataSource(_source(), _config(workers=1))
        )

    def test_external_memory_requires_deterministic(self):
        adapter = XGBoostAdapter(streaming=True, external_memory=True)
        nondeterministic = BufferedDataSource(_source(), _config(workers=2))
        with self.assertRaises(ValueError):
            adapter._dmatrix(SimpleNamespace(), nondeterministic)


class PipelineHookTests(unittest.TestCase):
    def _pipeline(self, wrapper):
        return Pipeline(
            rolling_dates=[[d] for d in DATES],
            adapter=DummyAdapter(),
            target="y",
            features=list(FEATURES),
            data_loader=_make_loader(_frames()),
            data_source_wrapper=wrapper,
        )

    def test_wrapper_applied_by_role(self):
        pipe = self._pipeline(
            buffered_wrapper(train=_config(workers=2), other=None)
        )
        train_src = pipe._src(DATES, (), None, "train")
        self.assertIsInstance(train_src, BufferedDataSource)
        self.assertEqual(train_src.config.workers, 2)
        final_src = pipe._src(DATES, (), None, "final_train")
        self.assertIsInstance(final_src, BufferedDataSource)
        for role in ("val", "final_val", "test", "fit"):
            self.assertIsInstance(pipe._src(DATES, (), None, role), DataSource)

    def test_default_other_config_is_sequential(self):
        pipe = self._pipeline(buffered_wrapper(train=None))
        self.assertIsInstance(pipe._src(DATES, (), None, "train"), DataSource)
        val_src = pipe._src(DATES, (), None, "val")
        self.assertIsInstance(val_src, BufferedDataSource)
        self.assertEqual(val_src.config.workers, 1)
        self.assertTrue(val_src.is_deterministic)

    def test_no_wrapper_returns_plain_source(self):
        pipe = self._pipeline(None)
        self.assertIsInstance(pipe._src(DATES, (), None, "train"), DataSource)

    def test_wrapped_train_end_to_end(self):
        pipe = self._pipeline(
            buffered_wrapper(train=_config(workers=2), other=None)
        )
        src = pipe._src(DATES, (), None, "train")
        x, y, _ = src.materialize()
        self.assertEqual(len(y), len(DATES) * ROWS_PER_DATE)
        total = sum(b[2]["n"] for b in src.batches(64))
        self.assertEqual(total, len(DATES) * ROWS_PER_DATE)


class BudgetCheckTests(unittest.TestCase):
    def test_counted_row_bytes_match_real_batches(self):
        frames = {d: f.with_columns(w=pl.lit(1.0)) for d, f in _frames().items()}
        for precision, weight, train_ctx in (
            ("float64", False, "full"),
            ("float64", False, "minimal"),
            ("float32", True, "full"),
            ("float32", True, "minimal"),
        ):
            with self.subTest(precision=precision, weight=weight, ctx=train_ctx):
                src = _source(
                    frames=frames,
                    precision=precision,
                    sample_weight_col="w" if weight else None,
                )
                config = _config(workers=1, train_ctx=train_ctx)
                row = buffered._counted_row_bytes(
                    len(FEATURES), precision, weight, full_ctx=train_ctx == "full"
                )
                it = BufferedDataSource(src, config).batches(32)
                batch = next(it)
                it.close()
                self.assertEqual(len(batch.y), 32)
                self.assertEqual(buffered._batch_nbytes(batch), 32 * row)
                run = buffered._Run(
                    source=src,
                    config=config,
                    dates=list(DATES),
                    batch_size=32,
                    multicollect=-1,
                    workers=1,
                    base_seed=0,
                    run_index=0,
                )
                self.assertEqual(run._estimated_cost(), 2 * 32 * row)
                if train_ctx == "minimal":  # row size the shuffle window counts
                    window_row = batch.x.itemsize * batch.x.shape[1]
                    window_row += batch.y.itemsize
                    if weight:
                        window_row += batch.weight.itemsize
                    self.assertEqual(window_row, row)

    def test_rejects_one_byte_short_and_runs_at_the_requirement(self):
        expected = len(DATES) * ROWS_PER_DATE
        for case in (
            dict(workers=1),
            dict(workers=2),
            dict(workers=1, row_shuffle_rows=64, train_ctx="minimal"),
            dict(workers=2, row_shuffle_rows=64, train_ctx="minimal"),
            dict(workers=1, row_shuffle_rows=16, train_ctx="minimal"),
        ):
            with self.subTest(**case):
                need = required_buffer_bytes(_config(**case), 32, len(FEATURES))
                short = BufferedDataSource(
                    _source(), _config(max_buffer_bytes=need - 1, **case)
                )
                before = set(_buffered_threads())
                with self.assertRaisesRegex(ValueError, "max_buffer_bytes"):
                    short.batches(32)
                self.assertLessEqual(set(_buffered_threads()), before)
                self.assertEqual(short._run_index, 0)

                exact = BufferedDataSource(
                    _source(), _config(max_buffer_bytes=need, **case)
                )
                batches = _drain(exact.batches(32))
                self.assertEqual(sum(len(b.y) for b in batches), expected)

    def test_error_names_the_requirement_and_largest_fitting_batch(self):
        need = required_buffer_bytes(_config(), 32, len(FEATURES))
        self.assertEqual(need, 3 * 32 * 40)  # 3 batches of 32 rows x 40 B
        wrapped = BufferedDataSource(_source(), _config(max_buffer_bytes=need - 1))
        with self.assertRaises(ValueError) as caught:
            wrapped.batches(32)
        message = str(caught.exception)
        self.assertIn(f"at least {need:,}", message)
        self.assertIn("batch_size to at most 31", message)

    def test_batch_size_must_be_a_positive_integer(self):
        wrapped = BufferedDataSource(_source(), _config())
        for bad in (0, -1, True, 2.5):
            with self.subTest(batch_size=bad), self.assertRaises(ValueError):
                wrapped.batches(bad)

    def test_pipeline_train_fails_fast(self):
        pipe = Pipeline(
            rolling_dates=[[d] for d in DATES],
            adapter=RidgeAdapter(batch_size=32),
            target="y",
            features=list(FEATURES),
            data_loader=_make_loader(_frames()),
            n_trials=1,
            data_source_wrapper=buffered_wrapper(
                _config(workers=2, max_buffer_bytes=1_000)
            ),
        )
        start = time.monotonic()
        with self.assertRaisesRegex(ValueError, "max_buffer_bytes"):
            pipe.train()
        self.assertLess(time.monotonic() - start, 5.0)


class StallDetectionTests(unittest.TestCase):
    # batch_size=None: each date is one 100-row batch of 4,000 counted bytes.

    def test_whole_date_larger_than_budget_raises_promptly(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):
                wrapped = BufferedDataSource(
                    _source(),
                    _config(
                        workers=workers,
                        max_buffer_bytes=3_000,
                        watchdog_interval=30.0,
                    ),
                )
                start = time.monotonic()
                with self.assertRaisesRegex(BufferDeadlockError, "batch_size=None"):
                    _drain(wrapped.batches(None))
                self.assertLess(time.monotonic() - start, 5.0)
                self.assertTrue(_wait_for_no_buffered_threads())

    def test_whole_dates_that_fit_still_stream(self):
        wrapped = BufferedDataSource(
            _source(), _config(workers=2, max_buffer_bytes=8_000)
        )
        got = np.concatenate([b.y for b in _drain(wrapped.batches(None))])
        bare = np.concatenate([b.y for b in _source().batches(None)])
        np.testing.assert_array_equal(np.sort(got), np.sort(bare))


if __name__ == "__main__":
    unittest.main()
