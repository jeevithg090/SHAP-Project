#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"

PIPELINE_PATH = ROOT / "all_in_one_pipeline.py"
RUN_SUMMARY_PATH = REPORTS_DIR / "run_summary.json"
SINGLE_RUN_SUMMARY_PATH = REPORTS_DIR / "single_file_run_summary.json"
TASK_B_RUN_SUMMARY_PATH = OUTPUTS_DIR / "task_b" / "task_b_run_summary.json"
PREDICTIONS_PATH = OUTPUTS_DIR / "task_b" / "B4_predictions_pilot.csv"
OBJECTIVE_SUMMARY_JSON = REPORTS_DIR / "script_objective_summary.json"
OBJECTIVE_SUMMARY_MD = REPORTS_DIR / "script_objective_summary.md"


@dataclass(frozen=True)
class Component:
    name: str
    description: str
    required_files: list[Path]


COMPONENTS = [
    Component(
        name="component_1_data_preparation",
        description="Prepare unilateral pilot cohort from workbook + dictionary with holdout handling.",
        required_files=[
            OUTPUTS_DIR / "pilot_model_input.csv",
            OUTPUTS_DIR / "pilot_holdout_rows.csv",
            OUTPUTS_DIR / "manual_side_verification.csv",
            OUTPUTS_DIR / "eda" / "row_counts.csv",
        ],
    ),
    Component(
        name="component_2_feature_importance",
        description="Train RF models and generate feature-importance outputs for four scoring targets.",
        required_files=[
            OUTPUTS_DIR / "task_a" / "model_performance.csv",
            OUTPUTS_DIR / "task_a" / "feature_ranking_long.csv",
            OUTPUTS_DIR / "task_a" / "overlap_analysis.csv",
            OUTPUTS_DIR / "task_a" / "mhiss_total_shap_summary.png",
            OUTPUTS_DIR / "task_a" / "attha_type_shap_summary.png",
            OUTPUTS_DIR / "task_a" / "mues_score_shap_summary.png",
            OUTPUTS_DIR / "task_a" / "strick_max_shap_summary.png",
        ],
    ),
    Component(
        name="component_3_ai_composite_scoring",
        description="Build AI composite score model from cross-target dominant features.",
        required_files=[
            OUTPUTS_DIR / "task_b" / "B1_model_comparison.csv",
            OUTPUTS_DIR / "task_b" / "B2_best_model.pkl",
            OUTPUTS_DIR / "task_b" / "B3_feature_weights.csv",
            OUTPUTS_DIR / "task_b" / "B4_predictions_pilot.csv",
            OUTPUTS_DIR / "task_b" / "B7_task_b_summary.md",
        ],
    ),
    Component(
        name="component_4_ai_vs_ips_agreement",
        description="Quantify AI composite vs IPS relationship using scatter and Bland-Altman artifacts.",
        required_files=[
            OUTPUTS_DIR / "task_b" / "B5_scatter_ai_vs_ips.png",
            OUTPUTS_DIR / "task_b" / "B6_bland_altman_ai_vs_ips.png",
            OUTPUTS_DIR / "task_b" / "B4_predictions_pilot.csv",
        ],
    ),
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline_if_requested(run_pipeline: bool) -> None:
    if not run_pipeline:
        return
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline file not found: {PIPELINE_PATH}")
    subprocess.run([sys.executable, str(PIPELINE_PATH)], check=True, cwd=str(ROOT))


def evaluate_components() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for component in COMPONENTS:
        missing = [str(path) for path in component.required_files if not path.exists()]
        results.append(
            {
                "name": component.name,
                "description": component.description,
                "status": "completed" if not missing else "missing_outputs",
                "required_count": len(component.required_files),
                "missing_outputs": missing,
            }
        )
    return results


def get_task_a_metrics() -> dict[str, object]:
    run_summary = read_json(RUN_SUMMARY_PATH)
    if not run_summary:
        return {}
    return {
        "rows_total": run_summary.get("row_counts", {}).get("total_rows"),
        "rows_pilot": run_summary.get("row_counts", {}).get("pilot_rows"),
        "rows_holdout": run_summary.get("row_counts", {}).get("holdout_rows"),
        "rows_bilateral": run_summary.get("row_counts", {}).get("bilateral_rows"),
        "targets_completed": run_summary.get("task_a_targets_completed", []),
        "quick_model_gate": run_summary.get("quick_metrics", {}),
    }


def get_task_b_metrics() -> dict[str, object]:
    run_summary = read_json(TASK_B_RUN_SUMMARY_PATH)
    if not run_summary:
        return {}
    return {
        "rows_used": run_summary.get("rows_used"),
        "best_model": run_summary.get("best_model"),
        "best_model_test_r2": run_summary.get("best_model_test_r2"),
        "best_model_test_mae": run_summary.get("best_model_test_mae"),
        "selected_feature_count": run_summary.get("selected_feature_count"),
        "selected_features": run_summary.get("selected_features", []),
    }


def compute_ai_vs_ips_metrics() -> dict[str, float | int | None]:
    if not PREDICTIONS_PATH.exists():
        return {}
    df = pd.read_csv(PREDICTIONS_PATH)
    df = df[df["AI_composite"].notna() & df["ips_total"].notna()].copy()
    if df.empty:
        return {}

    diff = df["AI_composite"] - df["ips_total"]
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.mean(diff**2)))
    mean_diff = float(diff.mean())
    median_diff = float(diff.median())
    sd = float(diff.std(ddof=1))
    loa_lower = float(mean_diff - 1.96 * sd)
    loa_upper = float(mean_diff + 1.96 * sd)
    spearman = float(spearmanr(df["AI_composite"], df["ips_total"]).statistic)

    return {
        "n": int(len(df)),
        "mean_ai": float(df["AI_composite"].mean()),
        "mean_ips": float(df["ips_total"].mean()),
        "mean_diff_ai_minus_ips": mean_diff,
        "median_diff_ai_minus_ips": median_diff,
        "mae": mae,
        "rmse": rmse,
        "spearman_ai_vs_ips": spearman,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "within_5_pct": float((np.abs(diff) <= 5).mean()),
        "within_10_pct": float((np.abs(diff) <= 10).mean()),
    }


def build_summary(run_pipeline: bool) -> dict[str, object]:
    run_pipeline_if_requested(run_pipeline)

    component_status = evaluate_components()
    all_done = all(item["status"] == "completed" for item in component_status)

    single_run = read_json(SINGLE_RUN_SUMMARY_PATH)
    summary = {
        "script_purpose": "Compact objective-level project summary and status checker.",
        "pipeline_invoked": bool(run_pipeline),
        "pipeline_summary": single_run,
        "overall_status": "completed" if all_done else "incomplete",
        "components": component_status,
        "task_a_metrics": get_task_a_metrics(),
        "task_b_metrics": get_task_b_metrics(),
        "ai_vs_ips_metrics": compute_ai_vs_ips_metrics(),
    }
    return summary


def write_reports(summary: dict[str, object]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OBJECTIVE_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Objective Components Summary")
    lines.append("")
    lines.append(f"- Overall status: `{summary['overall_status']}`")
    lines.append(f"- Pipeline invoked now: `{summary['pipeline_invoked']}`")
    lines.append("")
    lines.append("## Components")
    for component in summary["components"]:
        lines.append(f"- {component['name']}: `{component['status']}`")
    lines.append("")

    task_a = summary.get("task_a_metrics", {})
    if task_a:
        lines.append("## Task A Snapshot")
        lines.append(f"- Total rows: {task_a.get('rows_total')}")
        lines.append(f"- Pilot rows: {task_a.get('rows_pilot')}")
        lines.append(f"- Holdout rows: {task_a.get('rows_holdout')}")
        lines.append(f"- Bilateral rows: {task_a.get('rows_bilateral')}")
        lines.append(f"- Targets completed: {', '.join(task_a.get('targets_completed', []))}")
        lines.append("")

    task_b = summary.get("task_b_metrics", {})
    if task_b:
        lines.append("## Task B Snapshot")
        lines.append(f"- Rows used: {task_b.get('rows_used')}")
        lines.append(f"- Best model: {task_b.get('best_model')}")
        lines.append(f"- Test R2: {task_b.get('best_model_test_r2')}")
        lines.append(f"- Test MAE: {task_b.get('best_model_test_mae')}")
        lines.append("")

    agreement = summary.get("ai_vs_ips_metrics", {})
    if agreement:
        lines.append("## AI vs IPS")
        lines.append(f"- n: {agreement.get('n')}")
        lines.append(f"- Mean diff (AI - IPS): {agreement.get('mean_diff_ai_minus_ips')}")
        lines.append(f"- MAE: {agreement.get('mae')}")
        lines.append(f"- RMSE: {agreement.get('rmse')}")
        lines.append(f"- Spearman: {agreement.get('spearman_ai_vs_ips')}")
        lines.append("")

    OBJECTIVE_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Main objective-level script for the hand injury AI project.",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run all_in_one_pipeline.py before building objective summary.",
    )
    args = parser.parse_args()

    summary = build_summary(run_pipeline=args.run_pipeline)
    write_reports(summary)

    print(f"Summary JSON: {OBJECTIVE_SUMMARY_JSON}")
    print(f"Summary Markdown: {OBJECTIVE_SUMMARY_MD}")
    print(f"Overall status: {summary['overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
