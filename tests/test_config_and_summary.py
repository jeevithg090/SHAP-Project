from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hand_injury_ai import cli, summary
from hand_injury_ai.config import Paths


def test_paths_from_env_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "HAND_INJURY_WORKBOOK_PATH",
        "HAND_INJURY_DATA_DICTIONARY_PATH",
        "HAND_INJURY_OUTPUT_DIR",
        "HAND_INJURY_REPORTS_DIR",
        "HAND_INJURY_FEATURE_LIST_PATH",
        "HAND_INJURY_PATIENT_ID_SALT",
    ):
        monkeypatch.delenv(key, raising=False)
    paths = Paths.from_env(tmp_path)
    assert paths.workbook == tmp_path / "data" / "Version 2- MIT calculation table.xlsx"
    assert paths.output_dir == tmp_path / "outputs"
    assert paths.task_b_dir == tmp_path / "outputs" / "task_b"
    assert paths.feature_list_override is None
    assert paths.patient_id_salt is None


def test_paths_from_env_overrides_and_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAND_INJURY_OUTPUT_DIR", "/abs/out")
    monkeypatch.delenv("HAND_INJURY_REPORTS_DIR", raising=False)
    monkeypatch.delenv("HAND_INJURY_PATIENT_ID_SALT", raising=False)
    (tmp_path / ".env").write_text(
        "# comment\nHAND_INJURY_REPORTS_DIR=custom_reports\nHAND_INJURY_OUTPUT_DIR=ignored\n"
        "HAND_INJURY_PATIENT_ID_SALT='s3cret'\n",
        encoding="utf-8",
    )
    paths = Paths.from_env(tmp_path)
    assert paths.output_dir == Path("/abs/out")  # real env var wins over .env
    assert paths.reports_dir == tmp_path / "custom_reports"
    assert paths.patient_id_salt == "s3cret"
    assert "s3cret" not in repr(paths)


def test_agreement_metrics() -> None:
    metrics = summary.agreement_metrics(
        pd.Series([10.0, 20.0, 30.0, None]), pd.Series([12.0, 18.0, 30.0, 5.0])
    )
    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["mean_diff_ai_minus_ips"] == pytest.approx(0.0)
    assert metrics["within_5_pct"] == pytest.approx(1.0)
    assert summary.agreement_metrics(pd.Series([None]), pd.Series([1.0])) == {}


def test_summary_without_outputs_reports_incomplete(synthetic_paths: Paths) -> None:
    report = summary.build_summary(synthetic_paths)
    assert report["overall_status"] == "incomplete"
    assert all(item["status"] == "missing_outputs" for item in report["components"])
    json_path, md_path = summary.write_reports(report, synthetic_paths.reports_dir)
    assert json_path.is_file()
    assert "incomplete" in md_path.read_text(encoding="utf-8")


def test_cli_reports_missing_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAND_INJURY_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("HAND_INJURY_WORKBOOK_PATH", raising=False)
    monkeypatch.delenv("HAND_INJURY_DATA_DICTIONARY_PATH", raising=False)
    monkeypatch.delenv("HAND_INJURY_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("HAND_INJURY_REPORTS_DIR", raising=False)
    assert cli.main(["task-a"]) == 1
    assert cli.main(["task-b"]) == 1
    assert cli.main(["run"]) == 1
    assert (
        '"status": "failed"' in (tmp_path / "reports" / "single_file_run_summary.json").read_text()
    )
