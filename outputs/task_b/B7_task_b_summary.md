# Task B Summary (Pilot Dataset)

## Scope
- Built AI composite target from MHISS/Atthakomol/Strickland/MUES with per-patient weight renormalization.
- Trained and compared multiple regressors on the current pilot dataset.
- This is a pilot Task B run; final thesis-grade validation still requires full retrospective/prospective datasets.

## Selected Features
- `time_injury_to_ed_abx_hrs`
- `age`
- `time_to_debride_hrs`
- `nail_bed_damage_digits`
- `hand_skin_loss_volar`
- `lig_injury_type_d1`
- `mechanism`
- `fx_type_d4`
- `fdp_injured_d3`
- `fx_type_d1`
- `open_fracture`
- `skin_loss_dorsum_d1`

## Model Comparison
| model                 |   cv_r2_mean |   cv_mae_mean |   cv_rmse_mean |   test_r2 |   test_mae |   test_rmse |
|:----------------------|-------------:|--------------:|---------------:|----------:|-----------:|------------:|
| Ridge                 |    0.145338  |       7.2977  |        8.92309 | 0.32846   |    6.38084 |     9.53654 |
| RandomForestRegressor |    0.094116  |       7.51862 |        9.52782 | 0.277941  |    7.24931 |     9.88875 |
| XGBRegressor          |   -0.0582257 |       7.8112  |       10.3445  | 0.130017  |    8.08263 |    10.8545  |
| LinearRegression      |   -0.107575  |       8.22141 |       10.2835  | 0.0785867 |    6.92883 |    11.1707  |

## Best Model
- `Ridge`

## Spearman (Test Set) : AI Prediction vs Score Systems
- `mhiss_total`: 0.2810
- `attha_score`: 0.2825
- `mues_score`: 0.4589
- `strick_max`: 0.3520

## Output Files
- `outputs/task_b/B1_model_comparison.csv`
- `outputs/task_b/B2_best_model.pkl`
- `outputs/task_b/B3_feature_weights.csv`
- `outputs/task_b/B4_predictions_pilot.csv`
- `outputs/task_b/B5_scatter_ai_vs_ips.png`
- `outputs/task_b/B6_bland_altman_ai_vs_ips.png`
- `outputs/task_b/B7_task_b_summary.md`