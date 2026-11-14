"""End-to-end orchestration: Task A followed by Task B, with a run summary."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from . import task_a, task_b
from .config import Paths

logger = logging.getLogger(__name__)


def required_outputs_stage_a(paths: Paths) -> list[Path]:
    return [
        paths.pilot_dataset,
        paths.holdout_dataset,
        paths.model_performance,
        paths.overlap_analysis,
        paths.provisional_feature_list,
    ]


def required_outputs_stage_b(paths: Paths) -> list[Path]:
    names = (
        "B1_model_comparison.csv",
        "B2_best_model.pkl",
        "B3_feature_weights.csv",
        "B4_predictions_pilot.csv",
        "B5_scatter_ai_vs_ips.png",
        "B6_bland_altman_ai_vs_ips.png",
        "B7_task_b_summary.md",
    )
    return [paths.task_b_dir / name for name in names]


def check_outputs(paths: list[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{label} completed but required outputs are missing:\n" + "\n".join(missing)
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def run_full_pipeline(paths: Paths) -> int:
    """Run both stages; always write ``single_file_run_summary.json``. Returns an exit code."""
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, dict[str, str]] = {
        "stage_a": {"status": "not_run"},
        "stage_b": {"status": "not_run"},
    }
    summary: dict[str, object] = {
        "started_at_utc": _now(),
        "pipeline_mode": "single_end_to_end",
        "stages": stages,
    }
    try:
        logger.info("Step 1/2 (Task A): cohort, EDA and SHAP feature rankings")
        task_a.run(paths)
        check_outputs(required_outputs_stage_a(paths), "Stage A")
        stages["stage_a"] = {"status": "completed"}

        logger.info("Step 2/2 (Task B): AI composite model comparison")
        task_b.run(paths)
        check_outputs(required_outputs_stage_b(paths), "Stage B")
        stages["stage_b"] = {"status": "completed"}
        summary["status"] = "completed"
        exit_code = 0
        logger.info("Pipeline completed")
    except Exception as exc:  # noqa: BLE001 - recorded in the summary and re-reported
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Pipeline failed: %s", summary["error"])
        exit_code = 1

    summary["finished_at_utc"] = _now()
    paths.pipeline_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Run summary saved: %s", paths.pipeline_summary)
    return exit_code
