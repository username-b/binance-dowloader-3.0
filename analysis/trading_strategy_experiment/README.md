# Trading strategy experiment

Runs a fixed, cost-aware comparison of the selected point, Student-t GARCH,
and CatBoost MultiQuantile models for 10, 20, and 30 minute horizons.

```powershell
python analysis/trading_strategy_experiment/run_trading_strategy_experiment.py
```

Recommended command for a 64-vCPU / 256-GB machine (three independent
horizon processes, 20 native threads each, four CPUs left for the OS and S3):

```powershell
python analysis/trading_strategy_experiment/run_trading_strategy_experiment.py `
  --max-parallel-horizons 3 `
  --threads 20
```

The three workers load their datasets independently, so parallel mode trades
additional RAM for lower wall-clock time. The run configuration records both
the worker count and the maximum requested compute-thread count.

Smoke-check the resolved configuration without reading or writing S3:

```powershell
python analysis/trading_strategy_experiment/run_trading_strategy_experiment.py --dry-run
```

The default S3 run prefix is:

```text
s3://binance-data-downloader/trading_strategy_experiment/<run_id>/
```

Every horizon contains serialized point, GARCH, and CatBoost models; model
metadata; aligned predictions; individual trades; equity curves; metrics; and
the validation-selected probability threshold. `latest.json` points to the
latest immutable run rather than duplicating large model files.

The experiment uses the existing target log return as an execution proxy. It
charges 0.05% at entry and 0.05% at exit, allows only one non-overlapping
position at a time, and excludes slippage and funding.

The 20-minute CatBoost model uses `dataset_target_20/with_price_hmm_n4`,
matching the original ML comparison; the selected linear model continues to
use the base `dataset_target_20` feature set stored in its stage-3 metadata.
