from __future__ import annotations

import json

import pandas as pd
import pytest

from hand_injury_ai import pipeline, summary
from hand_injury_ai.config import Paths


@pytest.mark.slow
def test_full_pipeline_on_synthetic_data(synthetic_paths: Paths) -> None:
    assert pipeline.run_full_pipeline(synthetic_paths) == 0

    run = json.loads(synthetic_paths.pipeline_summary.read_text(encoding="utf-8"))
    assert run["status"] == "completed", run.get("error")

    task_a_summary = json.loads(synthetic_paths.task_a_summary.read_text(encoding="utf-8"))
    assert task_a_summary["row_counts"]["total_rows"] == 90
    assert task_a_summary["row_counts"]["bilateral_rows"] == 2
    assert set(task_a_summary["task_a_targets_completed"]) == {
        "mhiss_total",
        "attha_type",
        "mues_score",
        "strick_max",
    }

    pilot = pd.read_csv(synthetic_paths.pilot_dataset)
    assert pilot["patient_id"].astype(str).str.match(r"^P\d{4}$").all()
    assert "patient_name" not in pilot.columns

    predictions = pd.read_csv(synthetic_paths.predictions)
    assert {"AI_composite", "AI_composite_oof", "AI_target", "ips_total"} <= set(
        predictions.columns
    )
    assert predictions["AI_composite"].notna().all()

    report = summary.build_summary(synthetic_paths)
    assert report["overall_status"] == "completed"
    assert report["ai_vs_ips_metrics"]["n"] == len(predictions)
