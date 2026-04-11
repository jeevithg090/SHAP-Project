# Final Submission Report

## Project Scope
This run used:
- `Version 2- MIT calculation table.xlsx`
- `Data_Dictionary_v3 (1).xlsx`

Work completed:
- Task A: Feature importance (Random Forest + SHAP) for 4 targets
- Task B: AI Composite Score modeling and model comparison on pilot data

## Data Summary
- Total records available: 119
- Unilateral pilot records used: 99
- Holdout records: 20
- Bilateral deferred records: 6

Data outputs:
- `outputs/pilot_model_input.csv`
- `outputs/pilot_holdout_rows.csv`
- `outputs/manual_side_verification.csv`

## Task A (Feature Importance) Results

### Model Performance
| Target | Task | Subset Rows | Test Metrics |
|---|---|---:|---|
| `mhiss_total` | Regression | 99 | R2 = 0.1455, MAE = 27.0347, RMSE = 36.1941 |
| `attha_type` | Classification | 51 | Weighted F1 = 0.2841, Accuracy = 0.4545, AUC = 0.6894 |
| `mues_score` | Regression | 55 | R2 = 0.6873, MAE = 0.3127, RMSE = 0.4130 |
| `strick_max` | Regression | 76 | R2 = 0.2557, MAE = 3.5740, RMSE = 4.2649 |

Primary Task A files:
- `outputs/task_a/model_performance.csv`
- `outputs/task_a/feature_ranking_long.csv`
- `outputs/task_a/overlap_analysis.csv`
- SHAP plots under `outputs/task_a/`

### Provisional Final Feature List (Rule: `target_count >= 2`)
Total selected cross-target features: 13

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

Feature list file:
- `outputs/task_a/provisional_final_feature_list.csv`

## Task B (AI Composite Score) Results

### AI Target Construction
`AI_target` built as weighted normalized combination:
- 0.30 × MHISS
- 0.25 × Atthakomol
- 0.25 × Strickland
- 0.20 × MUES

Weights renormalized per patient when a score is not applicable.

### Feature Set Used for Task B
12 features used:
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

### Model Comparison
| Model | CV R2 | CV MAE | Test R2 | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.1453 | 7.2977 | 0.3285 | 6.3808 | 9.5365 |
| RandomForestRegressor | 0.0941 | 7.5186 | 0.2779 | 7.2493 | 9.8887 |
| XGBRegressor | -0.0582 | 7.8112 | 0.1300 | 8.0826 | 10.8545 |
| LinearRegression | -0.1076 | 8.2214 | 0.0786 | 6.9288 | 11.1707 |

Best model selected: **Ridge**

### Task B Correlation Check (Test Set)
Spearman of AI prediction vs source scores:
- `mhiss_total`: 0.2810
- `attha_score`: 0.2825
- `mues_score`: 0.4589
- `strick_max`: 0.3520

Task B outputs:
- `outputs/task_b/B1_model_comparison.csv`
- `outputs/task_b/B2_best_model.pkl`
- `outputs/task_b/B3_feature_weights.csv`
- `outputs/task_b/B4_predictions_pilot.csv`
- `outputs/task_b/B5_scatter_ai_vs_ips.png`
- `outputs/task_b/B6_bland_altman_ai_vs_ips.png`
- `outputs/task_b/B7_task_b_summary.md`
- `outputs/task_b/task_b_run_summary.json`

## Limitations and Status
- This is a **pilot dataset** run (99 unilateral cases), not final thesis-scale validation.
- Bilateral and other holdout records are not included in modeling.
- Final external/prospective validation remains pending proper full retrospective/prospective datasets.
