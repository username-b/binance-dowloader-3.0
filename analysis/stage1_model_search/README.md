# Stage 1 Minimal Model Benchmark

Fixed-config benchmark for target return horizons. The runner supports two suites:

- `fast`: reliable models that should finish quickly across all horizons.
- `long`: heavier models that should run with no or minimal model-level parallelism.

There is no grid search; model parameters are intentionally close to defaults and use
500 boosting/tree iterations where applicable.

## Suites

Fast suite:

- CatBoost RMSE
- LightGBM RMSE
- XGBoost RMSE
- CatBoost `RMSEWithUncertainty`
- CatBoost `MultiQuantile`
- LightGBM `Quantile`
- Optional HistGradientBoosting `Quantile` with `--include-histgradient-quantile`

Long suite:

- Extra Trees
- Random Forest
- NGBoost Normal

## Recommended Runs For 64 vCPU / 256 GB RAM

Fast run for all horizons:

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py `
  --horizon all `
  --suite fast
```

Fast run with optional HistGradientBoosting Quantile:

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py `
  --horizon all `
  --suite fast `
  --include-histgradient-quantile
```

Long run for all horizons:

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py `
  --horizon all `
  --suite long
```

Default resources:

- `fast`: `--max-parallel-models 5 --threads-per-model 12`
- `long`: `--max-parallel-models 1 --threads-per-model 16`

Override these flags explicitly if the host is under memory pressure or mostly idle.

## Smoke Checks

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py --dry-run --horizon all --suite fast
python analysis\stage1_model_search\train_stage1_architecture_search.py --dry-run --horizon all --suite long
```

Expected queue sizes per horizon:

- `fast`: 6 jobs.
- `fast --include-histgradient-quantile`: 7 jobs.
- `long`: 3 jobs.

## Metrics

Point metrics are written for every model:

- `MAE`
- `RMSE`
- `Direction_Accuracy`
- `Direction_Accuracy_0.25%`
- `R2`
- `OOS_R2`
- `train_time`
- `predict_time`

Probabilistic metrics are filled where the model exposes quantiles or a normal
distribution:

- `PinballLoss_05`, `PinballLoss_25`, `PinballLoss_50`, `PinballLoss_75`,
  `PinballLoss_95`
- `CRPS`
- `Coverage90`
- `IntervalWidth90`
- `NLL` when normal uncertainty is available

## S3 Layout

New outputs are stored near the bucket root:

`s3://binance-data-downloader/stage1_model_benchmarks/<suite>/horizon_<N>/<run_id>/`

Latest pointers are per suite and horizon:

`s3://binance-data-downloader/stage1_model_benchmarks/<suite>/horizon_<N>/latest/`

Each run contains:

- `run_config.json`
- `job_queue.parquet`
- `jobs/<job_id>/model.joblib`
- `jobs/<job_id>/metadata.json`
- `jobs/<job_id>/metrics.json`
- `jobs/<job_id>/predictions.parquet`
- `jobs/<job_id>/feature_importance.parquet`
- `stage1_results.parquet`
- `stage1_results.csv`
- `leaderboard_top10.parquet`
- `leaderboard_top10.csv`
- `run_summary.json`

The script resumes by default: if `jobs/<job_id>/metrics.json` already exists, that job
is skipped. Use `--overwrite` to retrain existing jobs.

## HMM Features

HMM posterior probability features are required only for datasets whose prefix contains
`with_price_hmm`. This lets `--horizon all` run horizon 10 and 30 datasets without HMM
columns while still validating horizon 20's HMM-enriched dataset.

## Dependencies

Install/update dependencies before a real run:

```powershell
pip install -r requirements.txt
```
