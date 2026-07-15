# Stage 1 Minimal Model Benchmark

Fixed-config benchmark for target return horizons. The default run is for
`target_log_return_20m` from `dataset_target_20/with_price_hmm_n4`.

The runner trains 5 point ML models:

- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost

And 4 probabilistic models by default:

- CatBoost `RMSEWithUncertainty`
- CatBoost `MultiQuantile`
- LightGBM `Quantile`
- NGBoost Normal

Optionally add HistGradientBoosting Quantile with `--include-histgradient-quantile`.
There is no grid search; model parameters are intentionally close to defaults and use
500 boosting/tree iterations where applicable.

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

## Smoke Checks

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py --dry-run --horizon 20
python analysis\stage1_model_search\train_stage1_architecture_search.py --dry-run --horizon 20 --include-histgradient-quantile
```

Expected queue sizes:

- Base run: 9 jobs.
- With HistGradientBoosting Quantile: 10 jobs.

## Recommended Run For 64 vCPU / 256 GB RAM

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py `
  --horizon 20 `
  --max-parallel-models 5 `
  --threads-per-model 12
```

This uses up to about 60 model worker threads and leaves a small reserve for Python,
S3 I/O, and system work. If RAM pressure appears, reduce `--max-parallel-models` to
3 or 4 while keeping `--threads-per-model 12`.

To include the optional fifth probabilistic model:

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py `
  --horizon 20 `
  --max-parallel-models 5 `
  --threads-per-model 12 `
  --include-histgradient-quantile
```

## HMM Features

The HMM posterior probability features are included by default:

- `price_hmm_n4_prob_state_0`
- `price_hmm_n4_prob_state_1`
- `price_hmm_n4_prob_state_2`
- `price_hmm_n4_prob_state_3`

The runner requires these columns unless `--allow-missing-hmm` is passed.

## S3 Layout

For a run under
`s3://binance-data-downloader/dataset_target_20/with_price_hmm_n4/stage1_architecture_search/<run_id>/`:

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

## Dependencies

Install/update dependencies before a real run:

```powershell
pip install -r requirements.txt
```
