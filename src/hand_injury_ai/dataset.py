"""Load the bilateral workbook and build the unilateral pilot cohort.

The source workbook has a human-readable form block (which contains direct
identifiers such as names and contact details) followed by a machine-readable
block that starts at the ``patient_id`` column. Only the machine-readable block is
used; the three side/dominance answers from the human block are copied across for
cross-checking, and ``patient_id`` is pseudonymised immediately after loading.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import canonicalize_name, is_positive_signal, make_unique, pseudonymize_ids

logger = logging.getLogger(__name__)

FREE_TEXT_COLUMNS = frozenset({"occupation_text"})

# Weighting factors and MHISS modifiers say nothing about which hand is injured.
NON_SIGNAL_PREFIXES = (
    "digit_weighing_factor_",
    "hand_forearm_weighing_factor",
    "integument_mod",
    "skeletal_mod",
    "motor_mod",
    "nv_mod",
)

SIDE_SPECIFIC_TARGET_ROOTS = frozenset(
    {
        "mhiss_total",
        "mhiss_cat",
        "attha_score",
        "attha_type",
        "mues_score",
        "strick_max",
        "ips_total",
        "ips_band",
    }
)

HUMAN_CROSS_CHECK_COLUMNS = (
    ("side_involved", "human_side_involved"),
    ("injured_hand", "human_injured_hand_label"),
    ("dominance", "human_dominance_label"),
)

_HAND_CODES = {0: "Left", 1: "Right", 2: "Bilateral"}
_SIDE_LABELS = {
    "left": "Left",
    "right": "Right",
    "bilateral": "Bilateral",
    "dominant": "Dominant",
    "non dominant": "Non-Dominant",
}


class DataInputError(RuntimeError):
    """Raised when an input file is missing or does not have the expected layout."""


@dataclass
class CohortResult:
    pilot: pd.DataFrame
    holdout: pd.DataFrame
    manual_verification: pd.DataFrame
    counts: dict[str, int]
    common_input_columns: list[str]
    side_specific_input_columns: list[str]


# --------------------------------------------------------------------------- parsing
def decode_hand_code(value: object) -> str:
    """Decode the machine hand code (0=Left, 1=Right, 2=Bilateral)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return ""
    return _HAND_CODES.get(numeric, "")


def normalize_side_label(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return _SIDE_LABELS.get(text, str(value).strip())


def infer_side_from_dominance_label(human_injured_label: str, dominant_side: str) -> str:
    """Translate "Dominant"/"Non-Dominant" into Left/Right using the dominant hand."""
    label = normalize_side_label(human_injured_label)
    if label == "Bilateral":
        return "Bilateral"
    if label not in {"Dominant", "Non-Dominant"} or dominant_side not in {"Left", "Right"}:
        return ""
    if label == "Dominant":
        return dominant_side
    return "Right" if dominant_side == "Left" else "Left"


def parse_numeric_range(value: object) -> tuple[float, float] | None:
    """Parse ``"0-412"`` style allowed-value ranges from the data dictionary."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip().replace("–", "-").replace("—", "-")
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def base_name_without_side(column: str) -> str:
    return re.sub(r"_(l|r)$", "", column)


def is_side_specific(column: str) -> bool:
    return column.endswith(("_l", "_r"))


# --------------------------------------------------------------------------- loading
def _require_file(path: Path, description: str, env_var: str) -> None:
    if not path.is_file():
        raise DataInputError(
            f"{description} not found at '{path}'. Place the file there or set {env_var}."
        )


def _read_excel(path: Path, description: str, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read an Excel sheet, turning unreadable files and missing sheets into clear errors."""
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as exc:
        if isinstance(sheet_name, str) and sheet_name in str(exc):
            raise DataInputError(
                f"{description} '{path}' has no sheet named '{sheet_name}'."
            ) from exc
        raise DataInputError(f"Could not read {description.lower()} '{path}': {exc}") from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise DataInputError(
            f"Could not read {description.lower()} '{path}' as an Excel workbook: {exc}"
        ) from exc


def load_machine_block(workbook_path: Path, patient_id_salt: str | None = None) -> pd.DataFrame:
    """Return the machine-readable block of the workbook with pseudonymised IDs."""
    _require_file(workbook_path, "Source workbook", "HAND_INJURY_WORKBOOK_PATH")
    full_df = _read_excel(workbook_path, "Source workbook")
    canonical_columns = [canonicalize_name(column) for column in full_df.columns]
    if "patient_id" not in canonical_columns:
        raise DataInputError(
            "Could not find the machine-readable block: no 'patient_id' column in the workbook."
        )
    machine_start = canonical_columns.index("patient_id")

    machine_df = full_df.iloc[:, machine_start:].copy()
    machine_df.columns = make_unique(machine_df.columns.tolist())

    human_lookup: dict[str, object] = {}
    for original_column, canonical in zip(
        full_df.columns[:machine_start], canonical_columns[:machine_start], strict=True
    ):
        if canonical and canonical not in human_lookup:
            human_lookup[canonical] = original_column

    for human_column, output_column in HUMAN_CROSS_CHECK_COLUMNS:
        if human_column in human_lookup:
            machine_df[output_column] = full_df[human_lookup[human_column]].to_numpy()
        else:
            machine_df[output_column] = np.nan

    machine_df["patient_id"] = pseudonymize_ids(machine_df["patient_id"], patient_id_salt)
    return machine_df


def load_dictionary(dictionary_path: Path) -> pd.DataFrame:
    """Load the "Data Dictionary" sheet with canonical column names."""
    _require_file(dictionary_path, "Data dictionary", "HAND_INJURY_DATA_DICTIONARY_PATH")
    dictionary_df = _read_excel(dictionary_path, "Data dictionary", sheet_name="Data Dictionary")
    dictionary_df.columns = [canonicalize_name(column) for column in dictionary_df.columns]
    missing = {"column_name", "layer"} - set(dictionary_df.columns)
    if missing:
        raise DataInputError(f"Data dictionary is missing required columns: {sorted(missing)}")
    for optional in ("allowed_values_range", "data_type"):
        if optional not in dictionary_df.columns:
            dictionary_df[optional] = ""

    dictionary_df = dictionary_df[dictionary_df["column_name"].notna()].copy()
    dictionary_df["column_name"] = dictionary_df["column_name"].map(canonicalize_name)
    dictionary_df = dictionary_df[dictionary_df["column_name"] != ""].copy()
    dictionary_df["layer"] = dictionary_df["layer"].fillna("").astype(str).str.strip().str.lower()
    dictionary_df["allowed_values_range"] = dictionary_df["allowed_values_range"].fillna("")
    dictionary_df = dictionary_df.drop_duplicates(subset="column_name", keep="first")
    dictionary_df = dictionary_df.reset_index(drop=True)

    # The workbook uses a side-less name for this one field.
    alias_source = "inadequate_soft_tissue_coverage_l_r"
    if alias_source in set(dictionary_df["column_name"]):
        alias_row = dictionary_df.loc[dictionary_df["column_name"] == alias_source].iloc[0].copy()
        alias_row["column_name"] = "inadequate_soft_tissue_coverage"
        dictionary_df = pd.concat([dictionary_df, alias_row.to_frame().T], ignore_index=True)

    return dictionary_df


def expand_grouped_dictionary_columns(
    dictionary_df: pd.DataFrame, machine_columns: set[str]
) -> pd.DataFrame:
    """Expand grouped entries such as ``skin_d2_d5_l`` into per-digit rows."""
    expanded_rows: list[pd.Series] = []
    for _, row in dictionary_df.iterrows():
        column_name = row["column_name"]
        candidates: list[str] = []
        if match := re.fullmatch(r"(.+)_d2_d5_(l|r)", column_name):
            prefix, side = match.groups()
            candidates = [f"{prefix}_d{digit}_{side}" for digit in range(2, 6)]
        elif (match := re.fullmatch(r"strick_d1_(l|r)_d5_(l|r)", column_name)) and (
            match.group(1) == match.group(2)
        ):
            side = match.group(1)
            candidates = [f"strick_d{digit}_{side}" for digit in range(1, 6)]
        for candidate in candidates:
            if candidate in machine_columns:
                expanded = row.copy()
                expanded["column_name"] = candidate
                expanded_rows.append(expanded)

    if not expanded_rows:
        return dictionary_df
    expanded_df = pd.concat([dictionary_df, pd.DataFrame(expanded_rows)], ignore_index=True)
    return expanded_df.drop_duplicates(subset="column_name", keep="first").reset_index(drop=True)


def split_dictionary_columns(
    dictionary_df: pd.DataFrame, machine_columns: set[str]
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Return (input columns, target columns, dictionary rows present in the workbook)."""
    dictionary_df = expand_grouped_dictionary_columns(dictionary_df, machine_columns)
    present = dictionary_df[dictionary_df["column_name"].isin(machine_columns)].copy()
    inputs = present.loc[present["layer"] == "input", "column_name"].tolist()
    targets = present.loc[present["layer"] == "target", "column_name"].tolist()
    return inputs, targets, present


def build_feature_data_type_map(dictionary_present: pd.DataFrame) -> dict[str, str]:
    """Map side-less feature names to their dictionary data type."""
    type_map: dict[str, str] = {}
    for _, row in dictionary_present.iterrows():
        column_name = row["column_name"]
        base = base_name_without_side(column_name) if is_side_specific(column_name) else column_name
        data_type = canonicalize_name(row.get("data_type"))
        if base and data_type and base not in type_map:
            type_map[base] = data_type
    return type_map


def build_signal_columns(side_specific_input_columns: list[str]) -> dict[str, list[str]]:
    signal_columns: dict[str, list[str]] = {"Left": [], "Right": []}
    for column in side_specific_input_columns:
        if base_name_without_side(column).startswith(NON_SIGNAL_PREFIXES):
            continue
        signal_columns["Left" if column.endswith("_l") else "Right"].append(column)
    return signal_columns


# --------------------------------------------------------------------------- cohort
def _side_review_reasons(
    selected_side: str,
    machine_side: str,
    inferred_human_side: str,
    is_bilateral: bool,
) -> list[str]:
    reasons: list[str] = []
    if selected_side not in {"Left", "Right", "Bilateral"}:
        reasons.append("invalid_or_missing_side_involved")
    if is_bilateral:
        reasons.append("bilateral_case")
    if selected_side in {"Left", "Right"} and machine_side in {"Left", "Right"}:
        if selected_side != machine_side:
            reasons.append("side_involved_machine_injured_hand_mismatch")
    elif machine_side == "":
        reasons.append("missing_machine_injured_hand")
    if (
        selected_side in {"Left", "Right"}
        and inferred_human_side in {"Left", "Right"}
        and selected_side != inferred_human_side
    ):
        reasons.append("side_involved_human_injured_hand_mismatch")
    return reasons


def build_pilot_and_holdout_datasets(
    machine_df: pd.DataFrame,
    input_columns: list[str],
    target_columns: list[str],
) -> CohortResult:
    """Collapse bilateral columns to the injured side; route doubtful rows to holdout.

    A row joins the pilot cohort only when the side recorded in the form, the machine
    hand code and the dominance-based answer all agree, the injured side has at least
    one positive finding and the opposite side has none. Everything else (including
    bilateral injuries) goes to the holdout table with the reasons attached.
    """
    side_specific_inputs = [column for column in input_columns if is_side_specific(column)]
    common_inputs = [column for column in input_columns if not is_side_specific(column)]
    side_targets = [
        column
        for column in target_columns
        if is_side_specific(column) and base_name_without_side(column) in SIDE_SPECIFIC_TARGET_ROOTS
    ]
    signal_columns = build_signal_columns(side_specific_inputs)

    kept_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    verification_rows: list[dict[str, object]] = []

    for _, row in machine_df.iterrows():
        selected_side = normalize_side_label(row.get("human_side_involved"))
        machine_side = decode_hand_code(row.get("injured_hand"))
        dominant_side = decode_hand_code(row.get("dominant_hand"))
        human_injured_label = normalize_side_label(row.get("human_injured_hand_label"))
        inferred_human_side = infer_side_from_dominance_label(human_injured_label, dominant_side)

        is_bilateral = "Bilateral" in {selected_side, machine_side, human_injured_label}
        reasons = _side_review_reasons(
            selected_side, machine_side, inferred_human_side, is_bilateral
        )

        if selected_side in {"Left", "Right"}:
            own_suffix = "_l" if selected_side == "Left" else "_r"
            own_signals = signal_columns[selected_side]
            other_signals = signal_columns["Right" if selected_side == "Left" else "Left"]
            side_code = 0 if selected_side == "Left" else 1
        else:
            own_suffix = ""
            own_signals, other_signals = [], []
            side_code = 2

        own_count = sum(is_positive_signal(row.get(column)) for column in own_signals)
        other_count = sum(is_positive_signal(row.get(column)) for column in other_signals)
        if selected_side in {"Left", "Right"}:
            if own_count == 0:
                reasons.append("no_positive_signal_on_selected_side")
            if other_count > 0:
                reasons.append("positive_signal_present_on_opposite_side")

        record: dict[str, object] = {
            "patient_id": row.get("patient_id"),
            "selected_side": selected_side,
            "selected_side_code": side_code,
            "dominant_hand_label": dominant_side,
            "machine_injured_hand_label": machine_side,
            "human_side_involved": selected_side,
            "human_injured_hand_label": human_injured_label,
            "human_dominance_label": normalize_side_label(row.get("human_dominance_label")),
            "is_bilateral": int(is_bilateral),
            "needs_review": 0,
            "selected_side_positive_signal_count": int(own_count),
            "opposite_side_positive_signal_count": int(other_count),
        }
        for column in common_inputs:
            if column in {"patient_id", "injured_hand"} or column in FREE_TEXT_COLUMNS:
                continue
            record[column] = row.get(column)
        if own_suffix:
            for column in side_specific_inputs + side_targets:
                if column.endswith(own_suffix):
                    record[base_name_without_side(column)] = row.get(column)

        if reasons:
            holdout_record = dict(record)
            holdout_record["holdout_reason"] = ";".join(sorted(set(reasons)))
            holdout_record["machine_injured_hand_code"] = row.get("injured_hand")
            holdout_record["dominant_hand"] = row.get("dominant_hand")
            holdout_rows.append(holdout_record)
            continue

        kept_rows.append(record)
        if len(verification_rows) < 10:
            verification_rows.append(
                {
                    key: record.get(key)
                    for key in (
                        "patient_id",
                        "selected_side",
                        "dominant_hand_label",
                        "machine_injured_hand_label",
                        "human_injured_hand_label",
                        "selected_side_positive_signal_count",
                        "opposite_side_positive_signal_count",
                        "open_fracture",
                        "mhiss_total",
                        "attha_type",
                        "mues_score",
                        "strick_max",
                    )
                }
            )

    pilot_df = pd.DataFrame(kept_rows)
    holdout_df = pd.DataFrame(holdout_rows)
    if pilot_df.empty:
        raise DataInputError("Pilot dataset is empty after unilateral filtering.")

    pilot_df["is_bilateral"] = pilot_df["is_bilateral"].fillna(0).astype(int)
    pilot_df["needs_review"] = pilot_df["needs_review"].fillna(0).astype(int)
    pilot_df = pilot_df.drop(
        columns=[
            "selected_side_code",
            "dominant_hand_label",
            "machine_injured_hand_label",
            "human_side_involved",
            "human_injured_hand_label",
            "human_dominance_label",
        ],
        errors="ignore",
    )

    holdout_reasons = holdout_df.get("holdout_reason", pd.Series(dtype=str)).astype(str)
    counts = {
        "total_rows": len(machine_df),
        "pilot_rows": len(pilot_df),
        "holdout_rows": len(holdout_df),
        "bilateral_rows": int(holdout_reasons.str.contains("bilateral_case").sum()),
    }
    logger.info(
        "Cohort built: %d rows -> %d unilateral pilot, %d holdout (%d bilateral)",
        counts["total_rows"],
        counts["pilot_rows"],
        counts["holdout_rows"],
        counts["bilateral_rows"],
    )
    return CohortResult(
        pilot=pilot_df,
        holdout=holdout_df,
        manual_verification=pd.DataFrame(verification_rows),
        counts=counts,
        common_input_columns=common_inputs,
        side_specific_input_columns=side_specific_inputs,
    )
