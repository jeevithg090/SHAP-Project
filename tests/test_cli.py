"""CLI behaviour: help, version, check, path overrides and the synthetic demo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hand_injury_ai import __version__, cli

ENV_VARS = (
    "HAND_INJURY_WORKBOOK_PATH",
    "HAND_INJURY_DATA_DICTIONARY_PATH",
    "HAND_INJURY_OUTPUT_DIR",
    "HAND_INJURY_REPORTS_DIR",
    "HAND_INJURY_FEATURE_LIST_PATH",
    "HAND_INJURY_PATIENT_ID_SALT",
)


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HAND_INJURY_PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    help_text = capsys.readouterr().out
    for command in ("run", "task-a", "task-b", "summary", "check", "demo"):
        assert command in help_text
    assert "HAND_INJURY_WORKBOOK_PATH" in help_text


def test_demo_rejects_too_few_patients(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["demo", "--patients", "5"])
    assert exc.value.code == 2
    assert "at least" in capsys.readouterr().err


def test_check_reports_missing_inputs(clean_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check"]) == 1
    out = capsys.readouterr().out
    assert out.count("MISSING  ") == 2
    assert "HAND_INJURY_WORKBOOK_PATH" in out
    assert "Not ready" in out


def test_demo_no_run_then_check_with_overrides(
    clean_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--dir", "demo", "--no-run"]) == 0
    data_dir = clean_env / "demo" / "data"
    assert (data_dir / "synthetic_workbook.xlsx").is_file()
    assert (data_dir / "synthetic_dictionary.xlsx").is_file()
    capsys.readouterr()

    args = [
        "check",
        "--workbook",
        "demo/data/synthetic_workbook.xlsx",
        "--dictionary",
        "demo/data/synthetic_dictionary.xlsx",
        "--output-dir",
        "demo/outputs",
    ]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "MISSING" not in out
    assert "Ready" in out
    assert "demo/outputs" in out


def test_check_never_prints_salt(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HAND_INJURY_PATIENT_ID_SALT", "do-not-print-me")
    cli.main(["check"])
    out = capsys.readouterr().out
    assert "do-not-print-me" not in out
    assert "salt set" in out


@pytest.mark.slow
def test_demo_runs_end_to_end(clean_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--dir", "demo", "--patients", "60", "-q"]) == 0
    out = capsys.readouterr().out
    assert "Done in" in out
    assert "best model" in out
    run = json.loads((clean_env / "demo" / "reports" / "single_file_run_summary.json").read_text())
    assert run["status"] == "completed"
    assert (clean_env / "demo" / "outputs" / "task_b" / "B7_task_b_summary.md").is_file()
