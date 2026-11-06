# GardenPipe

GardenPipe is a training and testing framework for large-scale sequence-like data.
The main design point is (almost) strictly bounded-memory modeling, implemented through 
a batched processing workflow and versatile stateful feature design, all based on Polars. 
It is built for a workstation, not a cluster.

## Why "GardenPipe"

When training machine learning models, many people share the experience (and nightmare)
of OOM kills, particularly when working with limited computational resources. You
start a promising training run, go to bed, and check the results the next morning,
only to find that the task was terminated 22 minutes after it started due to an out-of-memory error.

Avoiding OOM errors is an art, sometimes requiring the allocation of excessive resources on a cluster,
more than sufficient for a specific task. This is unfortunately not possible when you
work with limited resources on a local workstation or have tight cloud budgets.

That's where the name comes from. With a fire main, you don't worry much about flow: there is capacity to spare,
and a surge simply goes somewhere. A garden hose is different. You are watering
beds you care about, so you don't open the tap all the way and flood them; you
want a steady stream whose rate you control.

Data work on a single PC is the garden case. The dataset on disk can be many
times larger than RAM, and one careless `collect()` floods the machine. So
GardenPipe treats memory as a hard bound you set, not a side effect:

- data is read **one date at a time** and streamed in **fixed-size row batches**;
- stateful features (EWMAs, carried-forward values) **carry state across batch
  boundaries**, so streaming gives the same numbers as a single full pass;
- background prefetch runs against a **byte budget**, never an unbounded queue;
- models train from **streamed sufficient statistics** or mini-batches.

Growing the dataset adds time, not memory.

## Install

```sh
uv sync                     # core: polars, numpy, optuna, tqdm
uv sync --extra all         # + xgboost, scikit-learn, torch, tracking, plotting
```

Optional Rust EWMA kernel (falls back to pure Polars when absent):

```sh
cargo build --release --manifest-path gardenpipe/feature_rs/Cargo.toml
```

## Features first

A loader computes every feature and target of a date in one pass, before any
pipeline exists. `FeatureLoader(frames=...)` takes frames already in memory,
keyed by partition, in place of a path template:

```python
loader = FeatureLoader(
    frames={"2026-01-05": df},                   # columns: time, level
    feature_exprs={"level": pl.col("level")},
    stateful_features=(EwmaFeature("level_ewm_1m", pl.col("level"), half_life="1m"),),
    targets={"change_5m": pl.col("level")},
    horizons="5m",
)
frame = loader.load_date("2026-01-05").lf.collect()   # inspect, plot, iterate
```

Pass a loader to `Pipeline` through `data_loader`. `Pipeline.predict_frame`
and `gardenpipe.viewer` show predictions against the data.

## Minimal example

Given three daily Parquet files containing finite numeric columns `x`
(feature) and `y` (precomputed target):

```python
from gardenpipe.data import FeatureLoader
from gardenpipe.model import RidgeAdapter
from gardenpipe.pipeline import Pipeline

pipe = Pipeline(
    data_loader=FeatureLoader(path="data/{date}.parquet"),
    features=["x"],
    target="y",
    rolling_dates=[["2026-01-05"], ["2026-01-06"]],
    test_dates=["2026-01-07"],
    adapter=RidgeAdapter(),
    n_trials=1,
)
pipe.train()
print(pipe.test(keep_predictions=False)["test_score"])
```

This trains on January 5, validates on January 6, refits on both dates, and
reports test RMSE for January 7. Ridge streams batches by default.

Samplers choose the rows each role trains on or is scored on; a row is used if
any rule matches. Features are always computed on every row; to remove rows
from the data itself, use `FeatureLoader(filters=...)`.

API migration: `train_filters`, `val_filters`, and `test_filters` are now
`train_sampler`, `val_sampler`, and `test_sampler`. The `filters=` argument
to `test()` and `predict_frame()` is now `sampler=`, and the pipeline
constructor's Optuna `sampler=` argument is now `search_sampler=`.

To tune hyper-parameters, add `search_space={"alpha": LogUniform(1e-3, 1e3)}`
(from `gardenpipe.search`) and raise `n_trials`.

For generated data, stateful features, forward targets, and buffered loading,
see [`examples/minimal.py`](examples/minimal.py):

```sh
uv run python examples/minimal.py                                   # seconds
uv run python examples/minimal.py --rows 4000000 --days 30 --data /tmp/gp  # write
uv run python examples/minimal.py --days 30 --data /tmp/gp          # train
```

Peak RSS of the training run (Linux, Polars 2.0, pure-Polars EWMA, 4M rows
per day, files in the page cache; Polars 1.41 needs about 30 MiB less):

| Days | Parquet on disk | Test rows | Peak RSS |
| ---: | ---: | ---: | ---: |
| 10 | 409 MiB | 8.0M | ~440 MiB |
| 30 | 1228 MiB | 23.9M | ~440 MiB |

Tripling the data leaves memory flat.

`examples/weather/` runs the library on real data: six-hour temperature
forecasts from the Jena weather archive, with exact-timestamp labels, yearly
folds and two baselines (see its README).

## The memory model

Peak memory is roughly

    interpreter + libraries
    + one window's working set (its rows, and the look-ahead its targets need)
    + max(batch_size, largest same-timestamp burst) × columns × 8 bytes
    + max_buffer_bytes (prefetch)
    + model state

A loader cuts each date into windows of about `batch_size` rows when it has
targets or a front pad, and `MultiSourceLoader` always does; otherwise the
streaming scan can hold most of the date, whose working set then replaces the
window's. A window never splits a timestamp, so one timestamp with more rows
than `batch_size` makes a larger batch. With several streams per date
(below), the joined wide window adds union timestamps × streams × columns × 8
bytes.

None of these grow with the number of dates. The knobs:

| Knob | Where | Effect |
| --- | --- | --- |
| `batch_size` | adapter (`RidgeAdapter`, `LarsAdapter`, `XGBoostAdapter`, `TorchAdapter`) | rows per streamed batch |
| `max_buffer_bytes`, `workers` | `BufferConfig` | prefetch budget and parallelism |
| `precision="float32"` | `Pipeline`, `DataSource` | halves batch arrays |
| `polars_engine="in-memory"` | `Pipeline`, `DataSource` | files are still read and batches still collected streaming, but each time window's features, carried state and targets are computed in memory; speed and peak memory depend on the workload; a whole date (`batch_size=None`) is then computed in memory too |
| file layout | your data | the largest date without windows bounds its working set; split big days into smaller files if needed |

The prefetch budget must leave room to make progress: the batch the model is
working on plus a 2× claim for the next one, so three batches by default (with
row shuffle, also each worker's window). `batches()` raises `ValueError` up
front when `max_buffer_bytes` is smaller; `required_buffer_bytes()` in
`gardenpipe.buffered` computes the figure. With `batch_size=None` each batch is
a whole date, which cannot be checked in advance: a date larger than the
budget raises `BufferDeadlockError` instead of stalling.

Options that trade the bound for speed or convenience:

- `Pipeline(cache_arrays=True)` keeps materialized arrays across trials;
- `test(keep_predictions=True)` (the default) keeps every prediction;
- non-streaming adapters (`streaming=False`) call `materialize()` on the whole
  source;
- `XGBoostAdapter` streams batches but holds a quantized matrix in memory unless
  `external_memory=True`.

## Concepts

| Module | What it provides |
| --- | --- |
| `data` | `DateFrame` (date, tag, lazy frame), `Raw` path templating, `FeatureLoader` (files or frames), `MultiSourceLoader` (several streams per date, joined per window), `scan_labels`, `DataSource` (batched streaming, carryover state), `expand_dates` |
| `features` / `features_recipe` | `StatefulFeature` protocol; `ExprFeature`, `EwmaFeature` (normalized or unnormalized, irregular time), `CarryForwardFeature`, `LookbackFeature` (shifts and rolling windows across batches); composable sub-feature trees |
| `targets` | `add_forward_value` / `add_forward_change`: as-of lookups at `time + horizon`, weighted across horizons, per group |
| `transform` | `Standardizer`, `TargetScaler`, `Chain`, `FunctionTransform` (fit by streaming) |
| `model` | `RidgeAdapter`, `LarsAdapter`, `XGBoostAdapter`, `TorchAdapter` |
| `pipeline` | `Pipeline`: expanding-window folds, Optuna search, refit, test, `predict_frame`, save/load |
| `buffered` | Threaded prefetch under a byte budget; date and row shuffling |
| `window` | Endpoint-aligned `(batch, time, feature)` windows for sequence models |
| `score` / `losses` | RMSE, R², correlation, pinball (combinable across batches); `PinballLoss` |
| `registry` / `track` / `viewer` | Fingerprinted artifact registry, TensorBoard/W&B trackers, Plotly time-series viewer |

### Path templates

`FeatureLoader(path=...)` resolves exactly one file per date. The file name may
use `{date}` (`YYYY-MM-DD`), `{date_compact}`, `{source}`, `{source_lower}` and
`{tag}`. `{tag}` matches any text and becomes the frame's tag, which flows into
batch context and per-tag diagnostics:

```python
FeatureLoader(path="data/{source_lower}/{date_compact}_{tag}.parquet", source="PUMP_A")
```

### Stateful features

A stateful feature implements `apply(lf, carryover, front_pad=0)`,
`get_carryover(df)` and `internal_cols()`. When a date is streamed in batches,
the carryover from the end of one batch seeds the next, so batched output
matches a single full pass (`tests/test_features.py` checks this).

A date is cut into windows of whole timestamps: every row of a timestamp is
computed in the same window, so rows must be sorted by time (an unsorted file
raises). Every window is computed from its own rows, so plain feature
expressions, target expressions, loader `filters` and samplers must be
batch-local: row-local, or over the rows of one timestamp, such as
`pl.len().over("time")`. Anything that needs history across batches belongs in
a stateful feature: a forward-filled `x`, for example, is
`pl.coalesce("x", "x_last")` with `CarryForwardFeature("x_last", pl.col("x"))`.
For a few previous rows, such as `x.shift(1)` inside a stateful feature's
input, set `FeatureLoader(front_pad=1)`; every batch then gets that many
earlier rows. A pad row is filtered again in the batch it pads, without the
rest of its timestamp, so with a front pad `filters` should be row-local. For
a longer look-back, `LookbackFeature` carries the history itself:
`LookbackFeature("x_mean_5m", pl.col("x").rolling_mean_by("time", "5m"),
period="5m")`, or `rows=k` for row counts.

A target may read an earlier target, as in
`targets={"y1": pl.col("x"), "y2": pl.col("y1")}`, but in batches `y2` is null
near the end of each window: a window looks ahead one horizon, and `y2` needs
two.

### Several streams per date

`MultiSourceLoader` computes several streams per date, each on its own, and
joins them per window. A stream is one label's rows of a shared file
(`label_col`), or one source with its own file and loader (feeds with
different columns):

```python
panel = MultiSourceLoader(
    ["NORTH", "SOUTH"],
    base,                                   # shared file: time, station, temperature
    label_col="station",
    tick_col="observed",                     # north__observed: NORTH has a reading
    feature_exprs={
        "temperature_difference": pl.col("north__temperature") - pl.col("south__temperature")
    },
)
```

Each stream runs the base loader's features, carried state and targets on
its own rows only, exactly as a loader reading that stream alone would. Given
a batch size, every stream shares the same time windows, carries its own
state and pad rows from window to window, and the streams are joined per
window, so memory is one window, not one date. The shared file must be sorted
by time; it is read once per window and split by label.

The wide layout (default) has one row per timestamp, columns prefixed by
label, forward-filled across streams; its columns are fixed by the listed
labels, which `scan_labels` can find once. `feature_exprs` are evaluated on
these rows, for differences, ratios or averages across stations. The long layout
(`layout="long"`) keeps every stream's own rows with the label as a column,
for one pooled model, and joins the cross-stream features back by time.

Every stream costs a fixed charge per window: about 2 ms with the default
eight `workers` and 3.5 ms on one thread, for five EWMAs and a target on 1,000
rows per stream and window (Polars on four threads). The label
count times the window count drives the run time, so use large batches; a few
hundred labels with irregular observations is the practical range.

## Tests

```sh
uv run pytest
```

## License

MIT; see [LICENSE](LICENSE). The weather dataset has its own
[attribution and license](examples/weather/README.md#data-and-license).
