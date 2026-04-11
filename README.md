# SHAP-Project

This repository contains a complete hand-injury ML workflow centered on a single entrypoint:

- `all_in_one_pipeline.py`

The script runs an end-to-end pipeline that:

1. Builds a unilateral pilot dataset from workbook + data dictionary
2. Performs Task A feature-importance analysis (Random Forest + SHAP)
3. Performs Task B AI composite scoring and model comparison
4. Generates AI-vs-IPS agreement outputs and run summaries

## Main File

`all_in_one_pipeline.py` is designed as a **single-file delivery script**.  
It includes embedded Task A and Task B logic and executes them sequentially.

### What it does

- Reads source workbook:
  - `Version 2- MIT calculation table.xlsx`
- Reads data dictionary:
  - `/Users/jeevithg/Downloads/Data_Dictionary_v3 (1).xlsx` (as currently configured)
- Creates pilot/holdout datasets
- Runs EDA artifacts
- Trains target-specific models for:
  - `mhiss_total`
  - `attha_type`
  - `mues_score`
  - `strick_max`
- Produces SHAP summaries and dependence plots
- Constructs Task B `AI_target` and trains comparison models:
  - Linear Regression
  - Ridge
  - RandomForestRegressor
  - XGBRegressor
- Exports prediction files, model files, and agreement plots vs IPS

## How to Run

From repository root:

```bash
python3 all_in_one_pipeline.py
```

Optional objective summary script:

```bash
python3 script.py
```

or run pipeline first then summarize:

```bash
python3 script.py --run-pipeline
```

## Output Structure

Primary outputs are written under:

- `outputs/eda/`
- `outputs/task_a/`
- `outputs/task_b/`
- `reports/`

Important run summaries:

- `reports/single_file_run_summary.json` (unified run status)
- `reports/run_summary.json` (Task A summary)
- `outputs/task_b/task_b_run_summary.json` (Task B summary)

## Key Files

- `all_in_one_pipeline.py` — unified end-to-end pipeline
- `script.py` — compact objective-level status/metrics summary
- `outputs/task_a/overlap_analysis.csv` — cross-target feature overlap
- `outputs/task_a/provisional_final_feature_list.csv` — shortlist used for downstream modeling
- `outputs/task_b/B4_predictions_pilot.csv` — patient-wise AI composite predictions

## Notes

- Current implementation runs on the available pilot cohort artifacts in this project.
- Prospective final validation requires prospective dataset availability and matching schema.
