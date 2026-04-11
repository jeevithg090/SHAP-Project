# Task A Final Report (Sheet + Dictionary Only)

## Scope
This report covers **Task A only** (feature importance), using:
- `Version 2- MIT calculation table.xlsx`
- `Data_Dictionary_v3 (1).xlsx`

Task B was intentionally **not** started.

## Data Processing Outcome
- Total rows: 119
- Unilateral pilot rows used for modeling: 99
- Holdout rows: 20
- Bilateral rows deferred: 6

Files:
- `outputs/pilot_model_input.csv`
- `outputs/pilot_holdout_rows.csv`
- `outputs/manual_side_verification.csv`

## EDA and Quality Checks Completed
- Missing-value summary and heatmap generated
- Target distributions generated (`mhiss_total`, `attha_type`, `mues_score`, `strick_max`)
- Spearman feature-target heatmap generated
- Range anomaly checks generated (4 monitored numeric range rows)

Files:
- `outputs/eda/missing_value_summary.csv`
- `outputs/eda/missing_data_heatmap.png`
- `outputs/eda/target_distributions.png`
- `outputs/eda/spearman_feature_target_heatmap.png`
- `outputs/eda/anomaly_report.csv`

## Quick MHISS Gate (Baseline vs RF)

| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| DummyRegressor | -0.151969 | 35.806962 | 42.025170 |
| RandomForestRegressor | 0.043604 | 30.046745 | 38.291989 |

Result: RF beat baseline with positive R2, so full Task A proceeded.

File:
- `outputs/task_a/quick_mhiss_metrics.csv`

## Task A Model Performance

| Target | Task | Subset Rows | Test Metric 1 | Test Metric 2 | Test Metric 3 |
|---|---|---:|---:|---:|---:|
| `mhiss_total` | Regression | 99 | R2 = 0.145527 | MAE = 27.034693 | RMSE = 36.194147 |
| `attha_type` | Classification | 51 | Weighted F1 = 0.284091 | Accuracy = 0.454545 | AUC (OvR, weighted) = 0.689394 |
| `mues_score` | Regression | 55 | R2 = 0.687250 | MAE = 0.312693 | RMSE = 0.413026 |
| `strick_max` | Regression | 76 | R2 = 0.255670 | MAE = 3.573970 | RMSE = 4.264924 |

File:
- `outputs/task_a/model_performance.csv`

## SHAP Feature-Importance Findings (Top-5 per Target)

### `mhiss_total` (Top-5)
1. `hand_skin_loss_volar_0`
2. `time_injury_to_ed_abx_hrs`
3. `fdp_injured_d3_0`
4. `lig_injury_type_d1_3`
5. `skin_loss_volar_d1_6`

### `attha_type` (Top-5)
1. `age`
2. `fx_type_d4_0`
3. `mechanism_3`
4. `fx_type_d1_0`
5. `nail_bed_damage_d4_0`

### `mues_score` (Top-5)
1. `age`
2. `mechanism_3`
3. `mechanism_1`
4. `hand_skin_loss_volar_3`
5. `hand_skin_loss_volar_0`

### `strick_max` (Top-5)
1. `time_injury_to_ed_abx_hrs`
2. `lig_injury_type_d1_0`
3. `fx_type_d5_0`
4. `time_to_debride_hrs`
5. `bone_d3_0`

Files:
- `outputs/task_a/feature_ranking_long.csv`
- SHAP summaries and dependence plots in `outputs/task_a/`

## Provisional Final Feature List Rule
Rule applied: keep features where `target_count >= 2` from overlap analysis.

Result: 13 provisional cross-target features.

File:
- `outputs/task_a/provisional_final_feature_list.csv`

Top features in this provisional list (ordered by overlap + rank):
1. `time_injury_to_ed_abx_hrs`
2. `age`
3. `time_to_debride_hrs`
4. `nail_bed_damage_digits_None`
5. `hand_skin_loss_volar_0`
6. `lig_injury_type_d1_0`
7. `mechanism_3`
8. `fx_type_d4_0`
9. `fdp_injured_d3_0`
10. `hand_skin_loss_volar_3`
11. `fx_type_d1_0`
12. `open_fracture_1`
13. `skin_loss_dorsum_d1_0`

## Task Status
- Task A: Completed on current sheet + dictionary inputs
- Task B: Not started by design
