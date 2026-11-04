# Jena weather: six-hour temperature forecast

How much do recent weather observations add to a six-hour temperature
forecast, beyond what the time of day and the season already predict?

This example forecasts the temperature change over the next six hours at
every full hour, from the Jena weather station archive (2009–2016, fourteen
measurements every ten minutes). It compares two streaming Ridge models with
two baselines on identical issue times.

## Run it

```sh
uv sync --extra viz                       # viz is only needed for the HTML plots
uv run python examples/weather/prepare.py # download (13.6 MB), clean, partition
uv run python examples/weather/train.py   # tune, refit and predict both recipes
uv run python examples/weather/report.py  # scores, report.md, HTML plots
```

The whole run takes about ten seconds after the download. Prepared data goes
to `data/jena/` and results to `runs/weather/`; `prepare.py` reuses its files
unless you pass `--force`.

## Results

Test years 2015–2016, 17,377 hourly issue times (99.5% of all hourly issue
times in those years):

| Model | MAE (°C) | RMSE (°C) | MAE skill vs persistence |
| --- | ---: | ---: | ---: |
| Persistence, `T(t+6h) = T(t)` | 3.26 | 4.30 | – |
| Daily persistence, `T(t+6h) = T(t−18h)` | 2.62 | 3.42 | +19.6% |
| Ridge, calendar recipe | 1.97 | 2.57 | +39.6% |
| Ridge, history recipe | 1.70 | 2.21 | +47.7% |

The time of day and season do most of the work against persistence; recent
observations cut the remaining error by about 13%. `report.md` adds monthly
errors, validation scores and runtime. Hourly forecast errors are strongly
correlated, so treat these as point estimates.

## What the library does, and what it doesn't

`prepare.py` handles everything that needs the whole series at once. It
verifies the download's SHA-256, sorts the rows (the archive is not in time
order), removes 327 exact duplicate timestamps, rejects conflicting ones,
turns the `-9999` wind sentinels into nulls, drops the lone 2017 row, and
writes one parquet file per calendar year. `data/jena/manifest.json`
records the counts, the five gaps in the record, and the evidence about the
clock.

GardenPipe does the rest (`common.py`):

- **Features** are computed on every ten-minute row: wind components,
  calendar sine and cosine terms, differences against earlier observations
  (temperature 1, 3, 18 and 24 hours earlier, pressure 3 hours earlier), and
  EWMAs of temperature and pressure (`EwmaFeature`, half-lives of one and six
  hours). A difference reads the observation at exactly that earlier time and
  is null when it is missing; `LookbackFeature(period=...)` carries that much
  history across streamed batches, as `EwmaFeature` carries its state.
- **The label** `change_6h = T(t+6h) − T(t)` comes from the loader's forward
  target with `target_tolerance="0s"`, so it needs an observation at exactly
  `t + 6h`; across a gap the label is null rather than borrowed from an
  earlier reading.
- **Eligibility** is one sampler expression used for training, validation and
  testing: the issue time is on the hour, it is not on 1 January, the label is
  present, and every input of every compared model is present.
- **Folds** are the yearly partitions: train 2009–2012 and validate on 2013,
  then train 2009–2013 and validate on 2014. The pipeline searches a six-point
  alpha grid over both folds, refits on 2009–2014 and tests once on 2015–2016.
  The standardizer is fitted on each fold's training rows only.

Because partitions are calendar years, labels never cross into a later fold.
The cost is one restart per year: the last six hours of each year have no
label, and 1 January is excluded while the lags and EWMAs warm up.

## Outputs in `runs/weather/`

| File | Contents |
| --- | --- |
| `predictions.parquet` | Issue and target time, observed temperatures, every model's forecast |
| `metrics.json`, `report.md` | Scores overall, by year and by month; coverage; validation |
| `validation.json` | Validation RMSE for every alpha, best alpha per recipe |
| `run_config.json`, `runtime.json` | Folds, recipes, grid, eligibility rule; versions, durations, memory |
| `pipelines/<recipe>/` | Saved pipelines; reload with `Pipeline.from_saved(path, data_loader=make_loader())` |
| `week_winter.html`, `week_summer.html` | Forecasts against observations for two fixed weeks of 2015 |

## Notes

- Timestamps are naive and kept as published. The data show no summer-time
  changes, and the daily temperature cycle fits CET (UTC+1) better than UTC;
  this is inferred, not documented by the provider.
- Memory is not the point of this example: the data fit in memory and
  preparation sorts them in one go. Measured peak RSS on Linux:

  | Stage | Peak RSS | Time |
  | --- | ---: | ---: |
  | Imports only (polars, numpy, optuna, GardenPipe) | 64 MiB | – |
  | `prepare.py`, whole series in memory | 584 MiB | 1.7 s |
  | `train.py`, 5,000-row batches | 346 MiB | 21.4 s |
  | `train.py`, 20,000-row batches (default) | 329 MiB | 7.3 s |
  | `train.py`, one batch per year | 308 MiB | 4.1 s |

  Training's peak hardly depends on the batch size, because the batches are
  tiny; it is mostly fixed overhead. Smaller batches only cost time, since
  each re-reads its year's file with six hours of lookahead for the labels.
  `train.py --batch-rows N` changes the size. See `examples/minimal.py` for
  memory that stays flat as the data grow.

## Data and license

Weather data: Max Planck Institute for Biogeochemistry, Jena
([weather.bgc-jena.mpg.de](https://weather.bgc-jena.mpg.de/weather_data.html)),
CC BY 4.0. The archive used here is the copy hosted for the TensorFlow time
series tutorial, pinned by its SHA-256 in `common.py`.
