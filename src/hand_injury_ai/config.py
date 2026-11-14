"""Runtime configuration: input locations, output layout and environment overrides.

Every path can be overridden with an environment variable (see ``.env.example``).
Relative paths are resolved against the project root (the current working directory
by default, or ``HAND_INJURY_PROJECT_ROOT`` when set).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RANDOM_STATE = 42

DEFAULT_WORKBOOK = Path("data") / "Version 2- MIT calculation table.xlsx"
DEFAULT_DATA_DICTIONARY = Path("data") / "Data_Dictionary_v3.xlsx"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_REPORTS_DIR = Path("reports")

ENV_PROJECT_ROOT = "HAND_INJURY_PROJECT_ROOT"
ENV_WORKBOOK = "HAND_INJURY_WORKBOOK_PATH"
ENV_DATA_DICTIONARY = "HAND_INJURY_DATA_DICTIONARY_PATH"
ENV_OUTPUT_DIR = "HAND_INJURY_OUTPUT_DIR"
ENV_REPORTS_DIR = "HAND_INJURY_REPORTS_DIR"
ENV_FEATURE_LIST = "HAND_INJURY_FEATURE_LIST_PATH"
ENV_PATIENT_ID_SALT = "HAND_INJURY_PATIENT_ID_SALT"


def _load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file without overriding real env vars."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve(root: Path, value: str | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Paths:
    """All input and output locations used by the pipeline."""

    workbook: Path
    data_dictionary: Path
    output_dir: Path
    reports_dir: Path
    feature_list_override: Path | None = None
    patient_id_salt: str | None = field(default=None, repr=False)

    # ----- derived output locations -------------------------------------------------
    @property
    def eda_dir(self) -> Path:
        return self.output_dir / "eda"

    @property
    def task_a_dir(self) -> Path:
        return self.output_dir / "task_a"

    @property
    def task_b_dir(self) -> Path:
        return self.output_dir / "task_b"

    @property
    def pilot_dataset(self) -> Path:
        return self.output_dir / "pilot_model_input.csv"

    @property
    def holdout_dataset(self) -> Path:
        return self.output_dir / "pilot_holdout_rows.csv"

    @property
    def manual_verification(self) -> Path:
        return self.output_dir / "manual_side_verification.csv"

    @property
    def overlap_analysis(self) -> Path:
        return self.task_a_dir / "overlap_analysis.csv"

    @property
    def default_shortlist(self) -> Path:
        return self.task_a_dir / "default_feature_shortlist.csv"

    @property
    def provisional_feature_list(self) -> Path:
        return self.task_a_dir / "provisional_final_feature_list.csv"

    @property
    def model_performance(self) -> Path:
        return self.task_a_dir / "model_performance.csv"

    @property
    def task_a_summary(self) -> Path:
        return self.reports_dir / "run_summary.json"

    @property
    def task_b_summary(self) -> Path:
        return self.task_b_dir / "task_b_run_summary.json"

    @property
    def pipeline_summary(self) -> Path:
        return self.reports_dir / "single_file_run_summary.json"

    @property
    def predictions(self) -> Path:
        return self.task_b_dir / "B4_predictions_pilot.csv"

    @classmethod
    def from_env(cls, root: Path | None = None) -> Paths:
        """Build paths from environment variables (and an optional ``.env`` file)."""
        base = Path(root or os.environ.get(ENV_PROJECT_ROOT) or Path.cwd()).resolve()
        _load_dotenv(base / ".env")
        feature_list = os.environ.get(ENV_FEATURE_LIST)
        return cls(
            workbook=_resolve(base, os.environ.get(ENV_WORKBOOK), DEFAULT_WORKBOOK),
            data_dictionary=_resolve(
                base, os.environ.get(ENV_DATA_DICTIONARY), DEFAULT_DATA_DICTIONARY
            ),
            output_dir=_resolve(base, os.environ.get(ENV_OUTPUT_DIR), DEFAULT_OUTPUT_DIR),
            reports_dir=_resolve(base, os.environ.get(ENV_REPORTS_DIR), DEFAULT_REPORTS_DIR),
            feature_list_override=_resolve(base, feature_list, Path()) if feature_list else None,
            patient_id_salt=os.environ.get(ENV_PATIENT_ID_SALT) or None,
        )

    def ensure_output_dirs(self) -> None:
        for directory in (
            self.output_dir,
            self.eda_dir,
            self.task_a_dir,
            self.task_b_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
