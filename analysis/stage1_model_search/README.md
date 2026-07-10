# Stage 1 Architecture Search

First-stage model search for target return horizons. The default production run is for
`target_log_return_20m` from `dataset_target_20/with_price_hmm_n4` and excludes LightGBM
quantile models so the overnight run stays bounded.

The HMM posterior probability features are included by default:

- `price_hmm_n4_prob_state_0`
- `price_hmm_n4_prob_state_1`
- `price_hmm_n4_prob_state_2`
- `price_hmm_n4_prob_state_3`

The runner requires these columns unless `--allow-missing-hmm` is passed.

## Smoke checks

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py --dry-run --horizon 20
python analysis\stage1_model_search\train_stage1_architecture_search.py --dry-run --horizon 20 --include-lightgbm-quantile
```

Expected queue sizes:

- Base run: 135 jobs.
- With LightGBM Quantile: 270 jobs.

## Recommended first run

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py `
  --horizon 20 `
  --max-parallel-models 6 `
  --threads-per-model 4
```

This uses about 24 worker threads and leaves CPU/RAM reserve on a 32 vCPU, 128 GB host.

## Optional extended run

```powershell
python analysis\stage1_model_search\train_stage1_architecture_search.py `
  --horizon 20 `
  --max-parallel-models 6 `
  --threads-per-model 4 `
  --include-lightgbm-quantile
```

## S3 layout

For a run under `s3://binance-data-downloader/dataset_target_20/with_price_hmm_n4/stage1_architecture_search/<run_id>/`:

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

The script resumes by default: if `jobs/<job_id>/metrics.json` already exists, that job is
skipped. Use `--overwrite` to retrain existing jobs.

## Results analysis

Open `stage1_results_overview.ipynb` after a run. It reads the `latest` S3 prefix by
default and resolves it to the concrete run id from `run_config.json`. If the final
`stage1_results.parquet` is not present yet, it analyzes partial results from
`jobs/*/metrics.json`.

The notebook is designed to prune the hyperparameter space, not just pick a winner. It
builds Top-10 tables for RMSE/MAE/direction/composite score, summarizes every
hyperparameter with distribution plots and Top-10 counts, draws pairwise heatmaps,
estimates hyperparameter importance with a RandomForest surrogate, shows partial
dependence, cost-quality correlations, Pareto frontier, stability of Top-N models,
early-stop diagnostics, HMM feature-importance checks, and an automatically justified
`GRID_STAGE2`.

## Dependencies

Install/update dependencies before a real CatBoost run:

```powershell
pip install -r requirements.txt
```
