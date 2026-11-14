"""Objective-level status report built from pipeline outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import Paths


@dataclass(frozen=True)
class Component:
    name: str
    description: str
    required_files: tuple[Path, ...]


def components(paths: Paths) -> list[Component]:
    out, task_a, task_b = paths.output_dir, paths.task_a_dir, paths.task_b_dir
    return [
        Component(
            "component_1_data_preparation",
            "Prepare unilateral pilot cohort from workbook + dictionary with holdout handling.",
            (
                out / "pilot_model_input.csv",
                out / "pilot_holdout_rows.csv",
                out / "manual_side_verification.csv",
                paths.eda_dir / "row_counts.csv",
            ),
        ),
        Component(
            "component_2_feature_importance",
            "Train RF models and generate feature-importance outputs for four scoring targets.",
            (
                task_a / "model_performance.csv",
                task_a / "feature_ranking_long.csv",
                task_a / "overlap_analysis.csv",
                task_a / "mhiss_total_shap_summary.png",
                task_a / "attha_type_shap_summary.png",
                task_a / "mues_score_shap_summary.png",
                task_a / "strick_max_shap_summary.png",
            ),
        ),
        Component(
            "component_3_ai_composite_scoring",
            "Build AI composite score model from cross-target dominant features.",
            (
                task_b / "B1_model_comparison.csv",
                task_b / "B2_best_model.pkl",
                task_b / "B3_feature_weights.csv",
                task_b / "B4_predictions_pilot.csv",
                task_b / "B7_task_b_summary.md",
            ),
        ),
        Component(
            "component_4_ai_vs_ips_agreement",
            "Quantify AI composite vs IPS relationship using scatter and Bland-Altman artifacts.",
            (
                task_b / "B5_scatter_ai_vs_ips.png",
                task_b / "B6_bland_altman_ai_vs_ips.png",
                task_b / "B4_predictions_pilot.csv",
            ),
        ),
    ]


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def evaluate_components(paths: Paths) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for component in components(paths):
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


def get_task_a_metrics(paths: Paths) -> dict[str, object]:
    run_summary = read_json(paths.task_a_summary)
    if not run_summary:
        return {}
    counts = run_summary.get("row_counts", {})
    return {
        "rows_total": counts.get("total_rows"),
        "rows_pilot": counts.get("pilot_rows"),
        "rows_holdout": counts.get("holdout_rows"),
        "rows_bilateral": counts.get("bilateral_rows"),
        "targets_completed": run_summary.get("task_a_targets_completed", []),
        "quick_model_gate": run_summary.get("quick_metrics", {}),
    }


def get_task_b_metrics(paths: Paths) -> dict[str, object]:
    run_summary = read_json(paths.task_b_summary)
    if not run_summary:
        return {}
    keys = (
        "rows_used",
        "best_model",
        "best_model_test_r2",
        "best_model_test_mae",
        "best_model_oof_r2",
        "selected_feature_count",
    )
    metrics = {key: run_summary.get(key) for key in keys}
    metrics["selected_features"] = run_summary.get("selected_features", [])
    return metrics


def agreement_metrics(ai: pd.Series, ips: pd.Series) -> dict[str, float | int]:
    """Agreement statistics between an AI score and IPS (both on the same scale)."""
    frame = pd.DataFrame({"ai": ai, "ips": ips}).dropna()
    if frame.empty:
        return {}
    diff = frame["ai"] - frame["ips"]
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    spearman = float(spearmanr(frame["ai"], frame["ips"])[0]) if len(frame) >= 3 else float("nan")
    return {
        "n": len(frame),
        "mean_ai": float(frame["ai"].mean()),
        "mean_ips": float(frame["ips"].mean()),
        "mean_diff_ai_minus_ips": mean_diff,
        "median_diff_ai_minus_ips": float(diff.median()),
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "spearman_ai_vs_ips": spearman,
        "loa_lower": mean_diff - 1.96 * sd,
        "loa_upper": mean_diff + 1.96 * sd,
        "within_5_pct": float((diff.abs() <= 5).mean()),
        "within_10_pct": float((diff.abs() <= 10).mean()),
    }


def compute_ai_vs_ips_metrics(paths: Paths) -> dict[str, object]:
    if not paths.predictions.is_file():
        return {}
    df = pd.read_csv(paths.predictions)
    if not {"AI_composite", "ips_total"} <= set(df.columns):
        return {}
    metrics: dict[str, object] = dict(agreement_metrics(df["AI_composite"], df["ips_total"]))
    if "AI_composite_oof" in df.columns:
        metrics["out_of_fold"] = agreement_metrics(df["AI_composite_oof"], df["ips_total"])
    return metrics


def build_summary(paths: Paths, pipeline_invoked: bool = False) -> dict[str, object]:
    component_status = evaluate_components(paths)
    all_done = all(item["status"] == "completed" for item in component_status)
    return {
        "script_purpose": "Compact objective-level project summary and status checker.",
        "pipeline_invoked": pipeline_invoked,
        "pipeline_summary": read_json(paths.pipeline_summary),
        "overall_status": "completed" if all_done else "incomplete",
        "components": component_status,
        "task_a_metrics": get_task_a_metrics(paths),
        "task_b_metrics": get_task_b_metrics(paths),
        "ai_vs_ips_metrics": compute_ai_vs_ips_metrics(paths),
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Objective Components Summary",
        "",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Pipeline invoked now: `{summary['pipeline_invoked']}`",
        "",
        "## Components",
        *[f"- {item['name']}: `{item['status']}`" for item in summary["components"]],
        "",
    ]
    task_a = summary.get("task_a_metrics") or {}
    if task_a:
        lines += [
            "## Task A Snapshot",
            f"- Total rows: {task_a.get('rows_total')}",
            f"- Pilot rows: {task_a.get('rows_pilot')}",
            f"- Holdout rows: {task_a.get('rows_holdout')}",
            f"- Bilateral rows: {task_a.get('rows_bilateral')}",
            f"- Targets completed: {', '.join(task_a.get('targets_completed', []))}",
            "",
        ]
    task_b = summary.get("task_b_metrics") or {}
    if task_b:
        lines += [
            "## Task B Snapshot",
            f"- Rows used: {task_b.get('rows_used')}",
            f"- Best model: {task_b.get('best_model')}",
            f"- Test R2: {task_b.get('best_model_test_r2')}",
            f"- Test MAE: {task_b.get('best_model_test_mae')}",
            f"- Out-of-fold R2 (full cohort): {task_b.get('best_model_oof_r2')}",
            "",
        ]
    agreement = summary.get("ai_vs_ips_metrics") or {}
    if agreement:
        lines += [
            "## AI vs IPS",
            f"- n: {agreement.get('n')}",
            f"- Mean diff (AI - IPS): {agreement.get('mean_diff_ai_minus_ips')}",
            f"- MAE: {agreement.get('mae')}",
            f"- RMSE: {agreement.get('rmse')}",
            f"- Spearman: {agreement.get('spearman_ai_vs_ips')}",
            "",
        ]
    return "\n".join(lines)


def write_reports(summary: dict[str, object], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "script_objective_summary.json"
    md_path = reports_dir / "script_objective_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path
