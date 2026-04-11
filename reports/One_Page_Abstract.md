# One-Page Project Abstract

## Title
AI-Based Hand Injury Prognostication using Score-as-Target Modeling (Pilot Dataset)

## Objective
To develop a practical machine-learning workflow that:
1. identifies the most influential clinical variables across validated hand-injury scoring systems (Task A), and  
2. builds a single AI composite score that approximates multi-system prognostic information (Task B).

## Data and Setup
- Source files: `Version 2- MIT calculation table.xlsx` + `Data_Dictionary_v3 (1).xlsx`
- Total records: 119
- Unilateral pilot records modeled: 99
- Holdout records: 20 (including 6 bilateral deferred cases)
- Modeling approach: de-identified unilateral pilot extraction, EDA, Random Forest + SHAP for feature importance, then composite-score regression model comparison.

## Task A: Feature Importance (SHAP)
Four targets were modeled:
- `mhiss_total` (regression)
- `attha_type` (classification)
- `mues_score` (regression)
- `strick_max` (regression)

Test performance summary:
- `mhiss_total`: R2 = 0.1455, MAE = 27.03
- `attha_type`: weighted F1 = 0.2841, AUC = 0.6894
- `mues_score`: R2 = 0.6873, MAE = 0.3127
- `strick_max`: R2 = 0.2557, MAE = 3.5740

Cross-target provisional feature shortlist rule:
- keep features with `target_count >= 2` from overlap analysis.
- resulting provisional list size: 13 features.

## Task B: AI Composite Score
AI target was constructed as normalized weighted combination:
- 0.30 × MHISS
- 0.25 × Atthakomol
- 0.25 × Strickland
- 0.20 × MUES  
with per-patient weight renormalization when a score is not applicable.

Models compared:
- Linear Regression
- Ridge
- Random Forest Regressor
- XGBoost Regressor

Best model on pilot test set:
- **Ridge**
- Test R2 = 0.3285
- Test MAE = 6.3808

Spearman correlation of AI prediction vs source scores (test set):
- vs `mhiss_total`: 0.2810
- vs `attha_score`: 0.2825
- vs `mues_score`: 0.4589
- vs `strick_max`: 0.3520

## Interpretation
- The pipeline is operational end-to-end on current data.
- Feature-importance outputs and overlap shortlist are available for clinical interpretation.
- Composite modeling is feasible and currently favors a regularized linear approach (Ridge) on this pilot sample.

## Current Limitation
This is a pilot-scale run (99 unilateral modeled records), not final thesis-grade external validation. Final validation against full retrospective/prospective cohorts is still pending.

## Key Output Files
- Task A final: `reports/Task_A_Final_Report.md`
- Combined final report: `reports/Final_Submission_Report.md`
- Task B summary: `outputs/task_b/B7_task_b_summary.md`
