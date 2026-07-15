# ML Stage 1/2 vs Statistical Baselines

This note prepares the comparison requested for the report. The statistical baseline rows are transcribed from the provided screenshot. Stage 1 and Stage 2 are available only for the 20 minute horizon, because both runs target `target_log_return_20m` from `dataset_target_20/with_price_hmm_n4`.

## Sequential Model Search

1. Statistical baseline/reference search covered horizons 10m, 20m, and 30m. It selected the best linear/statistical candidates by objective: MAE, RMSE, and direction accuracy. Model families in that search include baseline zero/mean, OLS, Ridge, Lasso, ElasticNet, and Huber, with staged feature blocks from AR through calendar/session features.

2. Stage 1 architecture search trained ML models for 20m with `price_hmm_n4` posterior probability features. The planned base grid had 135 jobs: 36 CatBoost RMSE, 36 CatBoost RMSEWithUncertainty, 36 CatBoost MultiQuantile, and 27 LightGBM RMSE. The completed final table has 99 rows: CatBoost RMSE, CatBoost RMSEWithUncertainty, and LightGBM RMSE. CatBoost MultiQuantile was not evaluated in the final Stage 1 conclusion.

3. Stage 2 regularization search continued from the best Stage 1 CatBoost RMSEWithUncertainty region. It planned 288 CatBoost jobs from 8 architecture seeds crossed with `bagging_temperature`, `rsm`, and `random_strength`. The available overview is partial: 176 of 288 jobs were loaded, with no final `stage2_results.parquet` at notebook creation time.

## Comparison-Ready Tables

- `model_search_inventory.csv` lists which models and parameter grids were evaluated or planned at each stage.
- `ml_vs_statistical_comparison.csv` is the merged comparison table for the report.

For Stage 1/2 normalized metrics, the comparison uses the same convention as the screenshot:

- `NRMSE = RMSE / RMSE_zero_baseline`
- `OOS_R2 = 1 - NRMSE^2`
- `DA lift = Direction_Accuracy_0.25% - baseline Direction_Accuracy_0.25%`

The baseline values used for 20m ML rows are `baseline_rmse_used = 0.007169` and `baseline_da_025_used = 0.492556`, inferred from the target-20 baseline outputs in the existing overview notebook. Treat these as report-ready estimates unless you refresh the exact S3 result table.

## Key Readout

On the 20m horizon, the best statistical RMSE row in the screenshot remains stronger than Stage 1/2 ML by NRMSE/OOS R2:

- Statistical RMSE-(OLS): NRMSE `0.987`, OOS R2 `0.026`, DA lift `1.34 pp`.
- Stage 1 best RMSE CatBoost uncertainty: NRMSE `0.998881`, OOS R2 `0.002236`, DA lift `2.349 pp`.
- Stage 2 best RMSE CatBoost uncertainty: NRMSE `0.997768`, OOS R2 `0.004459`, DA lift `2.458 pp`.

For direction accuracy, the statistical DA-(Huber) row is also stronger:

- Statistical DA-(Huber): NRMSE `0.992`, OOS R2 `0.016`, DA lift `3.18 pp`.
- Stage 1 best DA CatBoost uncertainty: NRMSE `0.999861`, OOS R2 `0.000279`, DA lift `2.589 pp`.
- Stage 2 best DA CatBoost uncertainty: NRMSE `0.999721`, OOS R2 `0.000558`, DA lift `2.544 pp`.

Conclusion for the report: Stage 2 improves Stage 1 RMSE slightly, but the linear/statistical 20m reference models from the screenshot are still the better comparison winners on both OOS R2 and DA lift.
