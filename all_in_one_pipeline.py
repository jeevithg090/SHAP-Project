#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "outputs"
TASK_A_DIR = OUTPUT_DIR / "task_a"
TASK_B_DIR = OUTPUT_DIR / "task_b"
SINGLE_FILE_SUMMARY_PATH = REPORTS_DIR / "single_file_run_summary.json"
OVERLAP_PATH = TASK_A_DIR / "overlap_analysis.csv"
PROVISIONAL_FEATURE_PATH = TASK_A_DIR / "provisional_final_feature_list.csv"


FEATURE_SHAP_FAST_START_CODE = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_PATH = ROOT / "Version 2- MIT calculation table.xlsx"
DATA_DICTIONARY_PATH = Path("/Users/jeevithg/Downloads/Data_Dictionary_v3 (1).xlsx")
OUTPUT_DIR = ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
TASK_A_DIR = OUTPUT_DIR / "task_a"
REPORTS_DIR = ROOT / "reports"

PILOT_DATASET_PATH = OUTPUT_DIR / "pilot_model_input.csv"
HOLDOUT_DATASET_PATH = OUTPUT_DIR / "pilot_holdout_rows.csv"
MANUAL_VERIFICATION_PATH = OUTPUT_DIR / "manual_side_verification.csv"
ROW_COUNTS_PATH = EDA_DIR / "row_counts.csv"
MISSING_SUMMARY_PATH = EDA_DIR / "missing_value_summary.csv"
ANOMALY_REPORT_PATH = EDA_DIR / "anomaly_report.csv"
QUICK_MODEL_METRICS_PATH = TASK_A_DIR / "quick_mhiss_metrics.csv"
MODEL_PERFORMANCE_PATH = TASK_A_DIR / "model_performance.csv"
FEATURE_RANKING_LONG_PATH = TASK_A_DIR / "feature_ranking_long.csv"
FEATURE_RANKING_WIDE_PATH = TASK_A_DIR / "feature_ranking_wide.csv"
OVERLAP_ANALYSIS_PATH = TASK_A_DIR / "overlap_analysis.csv"
DEFAULT_SHORTLIST_PATH = TASK_A_DIR / "default_feature_shortlist.csv"
SUMMARY_REPORT_PATH = REPORTS_DIR / "run_summary.json"

TARGET_FAMILY_SUMMARY_COLUMNS = {
    "ips_worse",
    "ips_better",
    "ips_patient",
    "ips_bilateral",
    "ips_bilateral_band",
    "bilateral_penalty",
    "dominant_worse",
    "worse_hand",
}

CORE_TARGET_COLUMNS = {
    "mhiss_total",
    "mhiss_cat",
    "attha_score",
    "attha_type",
    "mues_score",
    "strick_max",
    "ips_total",
    "ips_band",
}

TARGET_COLUMNS_FOR_MODELING = {
    "mhiss_total",
    "attha_type",
    "mues_score",
    "strick_max",
    "ips_total",
    "attha_score",
    "mhiss_cat",
    "ips_band",
}

FREE_TEXT_COLUMNS = {"occupation_text"}

METADATA_COLUMNS = {
    "patient_id",
    "selected_side",
    "is_bilateral",
    "needs_review",
    "holdout_reason",
    "selected_side_positive_signal_count",
    "opposite_side_positive_signal_count",
    "is_open_fracture_case",
    "is_mangled_ue_case",
    "has_digit_injury_case",
    "mangled_domain_count",
}

CATEGORICAL_DATA_TYPES = {"binary", "categorical", "ordinal", "string"}
NUMERIC_DATA_TYPES = {"integer", "continuous"}

NON_SIGNAL_PREFIXES = (
    "digit_weighing_factor_",
    "hand_forearm_weighing_factor",
    "integument_mod",
    "skeletal_mod",
    "motor_mod",
    "nv_mod",
)

MANGLED_DOMAIN_RULES = {
    "skin": (
        "skin_d",
        "skin_loss_",
        "pulp_loss_",
        "skin_lace_",
        "nail_bed_damage",
        "hand_skin_loss_",
        "soft_tissue_defect_cm2",
    ),
    "bone": ("bone_d", "fx_type_", "forearm_fx", "bony_loss_defect", "open_fracture"),
    "joint": ("joint_d", "fx_dislocation_", "lig_injury_type_"),
    "tendon": (
        "tendon_d",
        "fdp_injured_",
        "fds_injured_",
        "fds_zone_2_",
        "extensor_",
        "intrinsic_",
        "wrist_flexor_injured",
        "wrist_extensor_injured",
    ),
    "nerve": ("nerve_d", "digital_nerve_", "major_nerve"),
    "vessel": ("vessel_d", "digital_artery_", "major_artery", "revasc_required"),
}


@dataclass
class TargetConfig:
    name: str
    task_type: str
    subset_filter: Callable[[pd.DataFrame], pd.Series]
    estimator: object


def canonicalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for raw_name in names:
        base = canonicalize_name(raw_name) or "unnamed"
        count = seen.get(base, 0) + 1
        seen[base] = count
        unique_names.append(base if count == 1 else f"{base}__dup{count}")
    return unique_names


def decode_hand_code(value: object) -> str:
    mapping = {0: "Left", 1: "Right", 2: "Bilateral"}
    if pd.isna(value):
        return ""
    try:
        if isinstance(value, str) and value.strip() == "":
            return ""
        numeric = int(float(value))
    except (TypeError, ValueError):
        return ""
    return mapping.get(numeric, "")


def normalize_side_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    mapping = {
        "left": "Left",
        "right": "Right",
        "bilateral": "Bilateral",
        "dominant": "Dominant",
        "non_dominant": "Non-Dominant",
        "non dominant": "Non-Dominant",
    }
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return mapping.get(text, str(value).strip())


def infer_side_from_dominance_label(human_injured_label: str, dominant_side: str) -> str:
    label = normalize_side_label(human_injured_label)
    if label == "Bilateral":
        return "Bilateral"
    if label not in {"Dominant", "Non-Dominant"} or dominant_side not in {"Left", "Right"}:
        return ""
    if label == "Dominant":
        return dominant_side
    return "Right" if dominant_side == "Left" else "Left"


def is_positive_signal(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) > 0
    text = str(value).strip().lower()
    return text not in {"", "0", "0.0", "no", "none", "na", "n/a", "nan"}


def parse_numeric_range(value: object) -> tuple[float, float] | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    text = text.replace("–", "-").replace("—", "-")
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    lower = float(match.group(1))
    upper = float(match.group(2))
    return (lower, upper)


def ensure_output_dirs() -> None:
    for directory in [OUTPUT_DIR, EDA_DIR, TASK_A_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_machine_block() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_df = pd.read_excel(WORKBOOK_PATH)
    full_columns_canonical = [canonicalize_name(column) for column in full_df.columns]
    if "patient_id" not in full_columns_canonical:
        raise RuntimeError("Could not find machine-readable block starting at patient_id.")
    machine_start_index = full_columns_canonical.index("patient_id")

    machine_df = full_df.iloc[:, machine_start_index:].copy()
    machine_df.columns = make_unique(machine_df.columns.tolist())

    raw_cross_check_map: dict[str, str] = {}
    for index, original_column in enumerate(full_df.columns[:machine_start_index]):
        canonical = canonicalize_name(original_column)
        if canonical and canonical not in raw_cross_check_map:
            raw_cross_check_map[canonical] = full_df.columns[index]

    for human_column, output_column in [
        ("side_involved", "human_side_involved"),
        ("injured_hand", "human_injured_hand_label"),
        ("dominance", "human_dominance_label"),
    ]:
        if human_column in raw_cross_check_map:
            machine_df[output_column] = full_df[raw_cross_check_map[human_column]]
        else:
            machine_df[output_column] = np.nan

    return machine_df, full_df


def load_dictionary() -> pd.DataFrame:
    dictionary_df = pd.read_excel(DATA_DICTIONARY_PATH, sheet_name="Data Dictionary")
    dictionary_df.columns = [canonicalize_name(column) for column in dictionary_df.columns]
    dictionary_df = dictionary_df[dictionary_df["column_name"].notna()].copy()
    dictionary_df["column_name"] = dictionary_df["column_name"].map(canonicalize_name)
    dictionary_df = dictionary_df[dictionary_df["column_name"] != ""].copy()
    dictionary_df["layer"] = dictionary_df["layer"].fillna("").astype(str).str.strip().str.lower()
    dictionary_df["allowed_values_range"] = dictionary_df["allowed_values_range"].fillna("")
    dictionary_df = dictionary_df.drop_duplicates(subset="column_name", keep="first").reset_index(drop=True)

    alias_rows = []
    if "inadequate_soft_tissue_coverage_l_r" in set(dictionary_df["column_name"]):
        alias_row = dictionary_df.loc[
            dictionary_df["column_name"] == "inadequate_soft_tissue_coverage_l_r"
        ].iloc[0].copy()
        alias_row["column_name"] = "inadequate_soft_tissue_coverage"
        alias_rows.append(alias_row)
    if alias_rows:
        dictionary_df = pd.concat([dictionary_df, pd.DataFrame(alias_rows)], ignore_index=True)

    return dictionary_df


def expand_grouped_dictionary_columns(
    dictionary_df: pd.DataFrame,
    machine_columns: set[str],
) -> pd.DataFrame:
    expanded_rows: list[pd.Series] = []
    for _, row in dictionary_df.iterrows():
        column_name = row["column_name"]
        d2_d5_match = re.fullmatch(r"(.+)_d2_d5_(l|r)", column_name)
        if d2_d5_match:
            prefix = d2_d5_match.group(1)
            side = d2_d5_match.group(2)
            for digit in range(2, 6):
                expanded_name = f"{prefix}_d{digit}_{side}"
                if expanded_name in machine_columns:
                    expanded_row = row.copy()
                    expanded_row["column_name"] = expanded_name
                    expanded_rows.append(expanded_row)
            continue

        strick_match = re.fullmatch(r"strick_d1_(l|r)_d5_(l|r)", column_name)
        if strick_match and strick_match.group(1) == strick_match.group(2):
            side = strick_match.group(1)
            for digit in range(1, 6):
                expanded_name = f"strick_d{digit}_{side}"
                if expanded_name in machine_columns:
                    expanded_row = row.copy()
                    expanded_row["column_name"] = expanded_name
                    expanded_rows.append(expanded_row)

    if not expanded_rows:
        return dictionary_df

    expanded_df = pd.concat([dictionary_df, pd.DataFrame(expanded_rows)], ignore_index=True)
    return expanded_df.drop_duplicates(subset="column_name", keep="first").reset_index(drop=True)


def split_dictionary_columns(dictionary_df: pd.DataFrame, machine_columns: set[str]) -> tuple[list[str], list[str], pd.DataFrame]:
    dictionary_df = expand_grouped_dictionary_columns(dictionary_df, machine_columns)
    dictionary_present = dictionary_df[dictionary_df["column_name"].isin(machine_columns)].copy()
    input_columns = dictionary_present.loc[dictionary_present["layer"] == "input", "column_name"].tolist()
    target_columns = dictionary_present.loc[dictionary_present["layer"] == "target", "column_name"].tolist()
    return input_columns, target_columns, dictionary_present


def build_feature_data_type_map(dictionary_present: pd.DataFrame) -> dict[str, str]:
    feature_data_type_map: dict[str, str] = {}
    for _, row in dictionary_present.iterrows():
        column_name = row["column_name"]
        base_name = base_name_without_side(column_name) if column_name.endswith(("_l", "_r")) else column_name
        data_type = canonicalize_name(row.get("data_type"))
        if base_name and data_type and base_name not in feature_data_type_map:
            feature_data_type_map[base_name] = data_type
    return feature_data_type_map


def get_side_specific_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.endswith("_l") or column.endswith("_r")]


def get_common_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if not (column.endswith("_l") or column.endswith("_r"))]


def base_name_without_side(column: str) -> str:
    return re.sub(r"_(l|r)$", "", column)


def build_signal_columns(side_specific_input_columns: list[str]) -> dict[str, list[str]]:
    signal_columns = {"Left": [], "Right": []}
    for column in side_specific_input_columns:
        base_name = base_name_without_side(column)
        if any(base_name.startswith(prefix) for prefix in NON_SIGNAL_PREFIXES):
            continue
        if column.endswith("_l"):
            signal_columns["Left"].append(column)
        elif column.endswith("_r"):
            signal_columns["Right"].append(column)
    return signal_columns


def build_pilot_and_holdout_datasets(
    machine_df: pd.DataFrame,
    input_columns: list[str],
    target_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], list[str], list[str]]:
    side_specific_input_columns = get_side_specific_columns(input_columns)
    common_input_columns = get_common_columns(input_columns)

    selected_side_target_roots = {
        "mhiss_total",
        "mhiss_cat",
        "attha_score",
        "attha_type",
        "mues_score",
        "strick_max",
        "ips_total",
        "ips_band",
    }
    selected_side_target_columns = [
        column
        for column in target_columns
        if (column.endswith("_l") or column.endswith("_r"))
        and base_name_without_side(column) in selected_side_target_roots
    ]

    signal_columns = build_signal_columns(side_specific_input_columns)

    kept_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    manual_verification_rows: list[dict[str, object]] = []

    for _, row in machine_df.iterrows():
        patient_id = row.get("patient_id")
        selected_side = normalize_side_label(row.get("human_side_involved"))
        machine_injured_hand_label = decode_hand_code(row.get("injured_hand"))
        dominant_hand_label = decode_hand_code(row.get("dominant_hand"))
        human_injured_hand_label = normalize_side_label(row.get("human_injured_hand_label"))
        inferred_side_from_human_label = infer_side_from_dominance_label(
            human_injured_hand_label,
            dominant_hand_label,
        )

        reasons: list[str] = []
        is_bilateral = selected_side == "Bilateral" or machine_injured_hand_label == "Bilateral"
        if human_injured_hand_label == "Bilateral":
            is_bilateral = True

        if selected_side not in {"Left", "Right", "Bilateral"}:
            reasons.append("invalid_or_missing_side_involved")
        if is_bilateral:
            reasons.append("bilateral_case")
        if selected_side in {"Left", "Right"} and machine_injured_hand_label in {"Left", "Right"}:
            if selected_side != machine_injured_hand_label:
                reasons.append("side_involved_machine_injured_hand_mismatch")
        elif machine_injured_hand_label == "":
            reasons.append("missing_machine_injured_hand")

        if selected_side in {"Left", "Right"} and inferred_side_from_human_label in {"Left", "Right"}:
            if selected_side != inferred_side_from_human_label:
                reasons.append("side_involved_human_injured_hand_mismatch")

        if selected_side == "Left":
            selected_signal_columns = signal_columns["Left"]
            opposite_signal_columns = signal_columns["Right"]
            selected_suffix = "_l"
            opposite_suffix = "_r"
            selected_side_code = 0
        elif selected_side == "Right":
            selected_signal_columns = signal_columns["Right"]
            opposite_signal_columns = signal_columns["Left"]
            selected_suffix = "_r"
            opposite_suffix = "_l"
            selected_side_code = 1
        else:
            selected_signal_columns = []
            opposite_signal_columns = []
            selected_suffix = ""
            opposite_suffix = ""
            selected_side_code = 2

        selected_signal_count = int(sum(is_positive_signal(row.get(column)) for column in selected_signal_columns))
        opposite_signal_count = int(sum(is_positive_signal(row.get(column)) for column in opposite_signal_columns))

        if selected_side in {"Left", "Right"} and selected_signal_count == 0:
            reasons.append("no_positive_signal_on_selected_side")
        if selected_side in {"Left", "Right"} and opposite_signal_count > 0:
            reasons.append("positive_signal_present_on_opposite_side")

        selected_record: dict[str, object] = {
            "patient_id": patient_id,
            "selected_side": selected_side,
            "selected_side_code": selected_side_code,
            "dominant_hand_label": dominant_hand_label,
            "machine_injured_hand_label": machine_injured_hand_label,
            "human_side_involved": selected_side,
            "human_injured_hand_label": human_injured_hand_label,
            "human_dominance_label": normalize_side_label(row.get("human_dominance_label")),
            "is_bilateral": int(is_bilateral),
            "needs_review": 0,
            "selected_side_positive_signal_count": selected_signal_count,
            "opposite_side_positive_signal_count": opposite_signal_count,
        }

        for common_column in common_input_columns:
            if common_column in {"patient_id", "injured_hand"} or common_column in FREE_TEXT_COLUMNS:
                continue
            selected_record[common_column] = row.get(common_column)

        if selected_side in {"Left", "Right"}:
            for side_specific_input in side_specific_input_columns:
                if side_specific_input.endswith(selected_suffix):
                    selected_record[base_name_without_side(side_specific_input)] = row.get(side_specific_input)

            for target_column in selected_side_target_columns:
                if target_column.endswith(selected_suffix):
                    selected_record[base_name_without_side(target_column)] = row.get(target_column)

        if reasons:
            holdout_record = selected_record.copy()
            holdout_record["holdout_reason"] = ";".join(sorted(set(reasons)))
            holdout_record["machine_injured_hand_code"] = row.get("injured_hand")
            holdout_record["dominant_hand"] = row.get("dominant_hand")
            holdout_rows.append(holdout_record)
        else:
            kept_rows.append(selected_record)
            if len(manual_verification_rows) < 10:
                verification_record = {
                    "patient_id": patient_id,
                    "selected_side": selected_side,
                    "dominant_hand_label": dominant_hand_label,
                    "machine_injured_hand_label": machine_injured_hand_label,
                    "human_injured_hand_label": human_injured_hand_label,
                    "selected_side_positive_signal_count": selected_signal_count,
                    "opposite_side_positive_signal_count": opposite_signal_count,
                    "open_fracture": selected_record.get("open_fracture"),
                    "mhiss_total": selected_record.get("mhiss_total"),
                    "attha_type": selected_record.get("attha_type"),
                    "mues_score": selected_record.get("mues_score"),
                    "strick_max": selected_record.get("strick_max"),
                }
                manual_verification_rows.append(verification_record)

    pilot_df = pd.DataFrame(kept_rows)
    holdout_df = pd.DataFrame(holdout_rows)
    manual_verification_df = pd.DataFrame(manual_verification_rows)

    if pilot_df.empty:
        raise RuntimeError("Pilot dataset ended up empty after unilateral filtering.")

    pilot_df["is_bilateral"] = pilot_df["is_bilateral"].fillna(0).astype(int)
    pilot_df["needs_review"] = pilot_df["needs_review"].fillna(0).astype(int)
    pilot_df = pilot_df.drop(
        columns=[
            column
            for column in [
                "selected_side_code",
                "dominant_hand_label",
                "machine_injured_hand_label",
                "human_side_involved",
                "human_injured_hand_label",
                "human_dominance_label",
                "holdout_reason",
            ]
            if column in pilot_df.columns
        ]
    )
    manual_verification_df.to_csv(MANUAL_VERIFICATION_PATH, index=False)

    counts = {
        "total_rows": int(len(machine_df)),
        "pilot_rows": int(len(pilot_df)),
        "holdout_rows": int(len(holdout_df)),
        "bilateral_rows": int((holdout_df.get("holdout_reason", pd.Series(dtype=str)).astype(str).str.contains("bilateral_case")).sum()),
    }

    return pilot_df, holdout_df, counts, common_input_columns, side_specific_input_columns


def write_row_counts(counts: dict[str, int]) -> None:
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in counts.items()]
    ).to_csv(ROW_COUNTS_PATH, index=False)


def build_missing_value_outputs(pilot_df: pd.DataFrame) -> None:
    missing_summary = (
        pilot_df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda frame: (frame["missing_count"] / len(pilot_df) * 100).round(2))
        .reset_index(names="column")
        .sort_values(["missing_count", "column"], ascending=[False, True])
    )
    missing_summary.to_csv(MISSING_SUMMARY_PATH, index=False)

    plot_df = pilot_df.drop(columns=[column for column in ["patient_id"] if column in pilot_df.columns])
    plt.figure(figsize=(28, 10))
    sns.heatmap(plot_df.isna(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Data Heatmap - Unilateral Pilot Dataset")
    plt.xlabel("Columns")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "missing_data_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_target_distribution_plot(pilot_df: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))

    pilot_df["mhiss_total"].dropna().plot(kind="hist", bins=30, ax=axes[0, 0], edgecolor="black")
    axes[0, 0].set_title("MHISS Total")
    axes[0, 0].set_xlabel("Score")

    pilot_df["attha_type"].dropna().astype(str).value_counts().sort_index().plot(
        kind="bar", ax=axes[0, 1], edgecolor="black"
    )
    axes[0, 1].set_title("Atthakomol Type")
    axes[0, 1].set_xlabel("Class")

    pilot_df["mues_score"].dropna().plot(kind="hist", bins=10, ax=axes[1, 0], edgecolor="black")
    axes[1, 0].set_title("MUES Score")
    axes[1, 0].set_xlabel("Score")

    pilot_df["strick_max"].dropna().plot(kind="hist", bins=18, ax=axes[1, 1], edgecolor="black")
    axes[1, 1].set_title("Strick Max")
    axes[1, 1].set_xlabel("Score")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "target_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_correlation_heatmap(pilot_df: pd.DataFrame, feature_columns: list[str]) -> None:
    target_columns = [column for column in ["mhiss_total", "attha_type", "mues_score", "strick_max"] if column in pilot_df.columns]
    numeric_feature_columns = [
        column
        for column in feature_columns
        if column in pilot_df.columns and pd.api.types.is_numeric_dtype(pilot_df[column])
    ]
    correlation_frame = pilot_df[numeric_feature_columns + target_columns].copy()
    correlation_matrix = correlation_frame.corr(method="spearman")
    cross_correlation = correlation_matrix.loc[numeric_feature_columns, target_columns]

    plt.figure(figsize=(12, max(8, len(numeric_feature_columns) * 0.22)))
    sns.heatmap(cross_correlation, cmap="coolwarm", center=0)
    plt.title("Spearman Correlation: Numeric Features vs Targets")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "spearman_feature_target_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_anomaly_report(pilot_df: pd.DataFrame, dictionary_present: pd.DataFrame) -> pd.DataFrame:
    anomaly_rows: list[dict[str, object]] = []
    dictionary_lookup = dictionary_present.set_index("column_name")

    for column in pilot_df.columns:
        if column not in dictionary_lookup.index:
            continue
        numeric_range = parse_numeric_range(dictionary_lookup.at[column, "allowed_values_range"])
        if numeric_range is None:
            continue
        if not pd.api.types.is_numeric_dtype(pilot_df[column]):
            continue
        lower, upper = numeric_range
        series = pilot_df[column].dropna()
        if series.empty:
            min_value = np.nan
            max_value = np.nan
        else:
            min_value = float(series.min())
            max_value = float(series.max())
        anomaly_rows.append(
            {
                "column": column,
                "allowed_min": lower,
                "allowed_max": upper,
                "observed_min": min_value,
                "observed_max": max_value,
                "below_min_count": int((pilot_df[column] < lower).sum()),
                "above_max_count": int((pilot_df[column] > upper).sum()),
            }
        )

    anomaly_report = pd.DataFrame(anomaly_rows).sort_values(
        ["above_max_count", "below_min_count", "column"],
        ascending=[False, False, True],
    )
    anomaly_report.to_csv(ANOMALY_REPORT_PATH, index=False)
    return anomaly_report


def quick_preprocess_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    feature_data_type_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = train_df[feature_columns].copy()
    test_features = test_df[feature_columns].copy()

    numeric_columns = [
        column
        for column in feature_columns
        if feature_data_type_map.get(column) in NUMERIC_DATA_TYPES
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    for column in numeric_columns:
        train_features[column] = pd.to_numeric(train_features[column], errors="coerce")
        test_features[column] = pd.to_numeric(test_features[column], errors="coerce")

    for column in numeric_columns:
        median_value = train_features[column].median()
        if pd.isna(median_value):
            median_value = 0
        train_features[column] = train_features[column].fillna(median_value)
        test_features[column] = test_features[column].fillna(median_value)

    for column in categorical_columns:
        mode_series = train_features[column].mode(dropna=True)
        mode_value = mode_series.iloc[0] if not mode_series.empty else "missing"
        train_features[column] = train_features[column].fillna(mode_value).astype(str)
        test_features[column] = test_features[column].fillna(mode_value).astype(str)

    train_encoded = pd.get_dummies(train_features, columns=categorical_columns, dummy_na=False)
    test_encoded = pd.get_dummies(test_features, columns=categorical_columns, dummy_na=False)
    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)
    return train_encoded, test_encoded


def build_feature_columns(pilot_df: pd.DataFrame, target_column: str | None = None) -> list[str]:
    feature_columns = [
        column
        for column in pilot_df.columns
        if column not in METADATA_COLUMNS
        and column not in TARGET_COLUMNS_FOR_MODELING
        and not column.startswith("domain_")
    ]
    if target_column is not None and target_column in pilot_df.columns:
        feature_columns = [column for column in feature_columns if column != target_column]
    return feature_columns


def run_quick_mhiss_model(
    pilot_df: pd.DataFrame,
    feature_data_type_map: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    model_df = pilot_df.dropna(subset=["mhiss_total"]).copy()
    feature_columns = build_feature_columns(model_df, target_column="mhiss_total")
    stratify_labels = build_regression_stratify_labels(model_df["mhiss_total"])
    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )
    x_train, x_test = quick_preprocess_train_test(train_df, test_df, feature_columns, feature_data_type_map)
    y_train = train_df["mhiss_total"]
    y_test = test_df["mhiss_total"]

    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(x_train, y_train)
    baseline_predictions = baseline_model.predict(x_test)

    random_forest = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_forest.fit(x_train, y_train)
    rf_predictions = random_forest.predict(x_test)

    metrics = {
        "baseline_r2": r2_score(y_test, baseline_predictions),
        "baseline_mae": mean_absolute_error(y_test, baseline_predictions),
        "baseline_rmse": math.sqrt(mean_squared_error(y_test, baseline_predictions)),
        "rf_r2": r2_score(y_test, rf_predictions),
        "rf_mae": mean_absolute_error(y_test, rf_predictions),
        "rf_rmse": math.sqrt(mean_squared_error(y_test, rf_predictions)),
    }

    metrics_frame = pd.DataFrame(
        [
            {"model": "DummyRegressor", "r2": metrics["baseline_r2"], "mae": metrics["baseline_mae"], "rmse": metrics["baseline_rmse"]},
            {"model": "RandomForestRegressor", "r2": metrics["rf_r2"], "mae": metrics["rf_mae"], "rmse": metrics["rf_rmse"]},
        ]
    )
    metrics_frame.to_csv(QUICK_MODEL_METRICS_PATH, index=False)

    if metrics["rf_r2"] > metrics["baseline_r2"] and metrics["rf_r2"] > 0:
        explainer = shap.TreeExplainer(random_forest)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_test, max_display=20, show=False)
        plt.title("Quick MHISS SHAP Summary")
        plt.tight_layout()
        plt.savefig(TASK_A_DIR / "quick_mhiss_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

    return model_df, metrics


def to_dense_frame(matrix: object, feature_names: list[str]) -> pd.DataFrame:
    dense_matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    cleaned_feature_names = [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names]
    return pd.DataFrame(dense_matrix, columns=cleaned_feature_names)


def get_cv_splitter(task_type: str, y_train: pd.Series):
    if task_type == "classification":
        minimum_class_count = int(y_train.value_counts().min())
        n_splits = min(5, minimum_class_count)
        if n_splits < 2:
            return None
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    n_splits = min(5, len(y_train))
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def build_regression_stratify_labels(y: pd.Series, max_bins: int = 5) -> pd.Series | None:
    if y.nunique(dropna=True) < 2:
        return None
    bin_count = min(max_bins, y.nunique(dropna=True))
    if bin_count < 2:
        return None
    try:
        return pd.qcut(y.rank(method="first"), q=bin_count, labels=False, duplicates="drop")
    except ValueError:
        return None


def compute_multiclass_auc(y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) < 2:
        return np.nan
    try:
        encoded = label_binarize(y_true, classes=classes)
        if encoded.shape[1] == 1:
            return np.nan
        return roc_auc_score(encoded, probabilities, multi_class="ovr", average="weighted")
    except ValueError:
        return np.nan


def build_preprocessor(
    feature_frame: pd.DataFrame,
    feature_data_type_map: dict[str, str],
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = [
        column
        for column in feature_frame.columns
        if feature_data_type_map.get(column) in NUMERIC_DATA_TYPES
    ]
    categorical_columns = [column for column in feature_frame.columns if column not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns


def coerce_numeric_feature_columns(
    frame: pd.DataFrame,
    feature_data_type_map: dict[str, str],
) -> pd.DataFrame:
    coerced = frame.copy()
    numeric_columns = [
        column
        for column in coerced.columns
        if feature_data_type_map.get(column) in NUMERIC_DATA_TYPES
    ]
    if numeric_columns:
        coerced[numeric_columns] = coerced[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return coerced


def get_shap_matrix(shap_values: object, predicted_probabilities: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(shap_values)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        if predicted_probabilities is None:
            return values[:, :, 0]
        predicted_indices = predicted_probabilities.argmax(axis=1)
        return np.stack([values[index, :, class_index] for index, class_index in enumerate(predicted_indices)], axis=0)
    raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")


def generate_dependence_plots(
    shap_matrix: np.ndarray,
    transformed_test_frame: pd.DataFrame,
    target_name: str,
    top_feature_names: list[str],
) -> None:
    for rank, feature_name in enumerate(top_feature_names, start=1):
        shap.dependence_plot(feature_name, shap_matrix, transformed_test_frame, show=False)
        plt.title(f"{target_name} - SHAP Dependence - {feature_name}")
        plt.tight_layout()
        plt.savefig(
            TASK_A_DIR / f"{target_name}_shap_dependence_{rank}_{feature_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def run_task_a_models(
    pilot_df: pd.DataFrame,
    feature_data_type_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pilot_df = pilot_df.copy()
    pilot_df["is_open_fracture_case"] = pilot_df["open_fracture"].fillna(0).astype(float).gt(0).astype(int)

    for domain_name, prefixes in MANGLED_DOMAIN_RULES.items():
        domain_columns = [
            column
            for column in pilot_df.columns
            if any(column.startswith(prefix) for prefix in prefixes)
        ]
        pilot_df[f"domain_{domain_name}_present"] = pilot_df[domain_columns].apply(
            lambda row: int(any(is_positive_signal(value) for value in row)),
            axis=1,
        )
    domain_presence_columns = [column for column in pilot_df.columns if column.startswith("domain_") and column.endswith("_present")]
    pilot_df["mangled_domain_count"] = pilot_df[domain_presence_columns].sum(axis=1)
    pilot_df["is_mangled_ue_case"] = pilot_df["mangled_domain_count"].ge(3).astype(int)

    digit_injury_columns = [
        column
        for column in pilot_df.columns
        if re.search(r"_d[1-5](?:_|$)", column) is not None
        and column not in TARGET_COLUMNS_FOR_MODELING
        and column not in METADATA_COLUMNS
    ]
    pilot_df["has_digit_injury_case"] = pilot_df[digit_injury_columns].apply(
        lambda row: int(any(is_positive_signal(value) for value in row)),
        axis=1,
    )

    target_configs = [
        TargetConfig(
            name="mhiss_total",
            task_type="regression",
            subset_filter=lambda frame: frame["mhiss_total"].notna(),
            estimator=RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        TargetConfig(
            name="attha_type",
            task_type="classification",
            subset_filter=lambda frame: frame["is_open_fracture_case"].eq(1) & frame["attha_type"].isin([1, 2, 3]),
            estimator=RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        TargetConfig(
            name="mues_score",
            task_type="regression",
            subset_filter=lambda frame: frame["is_mangled_ue_case"].eq(1) & frame["mues_score"].notna(),
            estimator=RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        TargetConfig(
            name="strick_max",
            task_type="regression",
            subset_filter=lambda frame: frame["has_digit_injury_case"].eq(1) & frame["strick_max"].notna(),
            estimator=RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]

    performance_rows: list[dict[str, object]] = []
    feature_ranking_frames: list[pd.DataFrame] = []

    for config in target_configs:
        target_frame = pilot_df.loc[config.subset_filter(pilot_df)].copy()
        if target_frame.empty:
            continue

        y = target_frame[config.name]
        feature_columns = build_feature_columns(target_frame, target_column=config.name)
        x = coerce_numeric_feature_columns(target_frame[feature_columns].copy(), feature_data_type_map)
        if config.task_type == "classification" and y.nunique() > 1:
            stratify = y
        elif config.task_type == "regression":
            stratify = build_regression_stratify_labels(y)
        else:
            stratify = None

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

        preprocessor, _, _ = build_preprocessor(x_train, feature_data_type_map)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clone(config.estimator)),
            ]
        )

        cv_splitter = get_cv_splitter(config.task_type, y_train)
        if config.task_type == "classification":
            scoring = {
                "weighted_f1": "f1_weighted",
                "accuracy": "accuracy",
            }
        else:
            scoring = {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            }

        if cv_splitter is not None:
            cv_results = cross_validate(
                pipeline,
                x_train,
                y_train,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=None,
                return_train_score=False,
            )
        else:
            cv_results = {}

        pipeline.fit(x_train, y_train)

        transformed_test = pipeline.named_steps["preprocessor"].transform(x_test)
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        transformed_test_frame = to_dense_frame(transformed_test, feature_names)

        if config.task_type == "classification":
            predictions = pipeline.predict(x_test)
            probabilities = pipeline.predict_proba(x_test)
            auc_value = compute_multiclass_auc(y_test, probabilities, pipeline.named_steps["model"].classes_)

            performance_rows.append(
                {
                    "target": config.name,
                    "task_type": config.task_type,
                    "subset_rows": len(target_frame),
                    "train_rows": len(x_train),
                    "test_rows": len(x_test),
                    "cv_weighted_f1_mean": float(np.mean(cv_results["test_weighted_f1"])) if cv_results else np.nan,
                    "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])) if cv_results else np.nan,
                    "test_weighted_f1": f1_score(y_test, predictions, average="weighted"),
                    "test_accuracy": accuracy_score(y_test, predictions),
                    "test_auc_ovr_weighted": auc_value,
                }
            )
            explainer = shap.TreeExplainer(pipeline.named_steps["model"])
            raw_shap_values = explainer.shap_values(transformed_test_frame)
            shap_matrix = get_shap_matrix(raw_shap_values, predicted_probabilities=probabilities)
        else:
            predictions = pipeline.predict(x_test)
            performance_rows.append(
                {
                    "target": config.name,
                    "task_type": config.task_type,
                    "subset_rows": len(target_frame),
                    "train_rows": len(x_train),
                    "test_rows": len(x_test),
                    "cv_r2_mean": float(np.mean(cv_results["test_r2"])) if cv_results else np.nan,
                    "cv_mae_mean": float(-np.mean(cv_results["test_mae"])) if cv_results else np.nan,
                    "cv_rmse_mean": float(-np.mean(cv_results["test_rmse"])) if cv_results else np.nan,
                    "test_r2": r2_score(y_test, predictions),
                    "test_mae": mean_absolute_error(y_test, predictions),
                    "test_rmse": math.sqrt(mean_squared_error(y_test, predictions)),
                }
            )
            explainer = shap.TreeExplainer(pipeline.named_steps["model"])
            raw_shap_values = explainer.shap_values(transformed_test_frame)
            shap_matrix = get_shap_matrix(raw_shap_values)

        shap.summary_plot(shap_matrix, transformed_test_frame, max_display=20, show=False)
        plt.title(f"{config.name} - SHAP Summary")
        plt.tight_layout()
        plt.savefig(TASK_A_DIR / f"{config.name}_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        mean_absolute_shap = np.abs(shap_matrix).mean(axis=0)
        feature_ranking_frame = pd.DataFrame(
            {
                "target": config.name,
                "feature_name": transformed_test_frame.columns,
                "mean_abs_shap": mean_absolute_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        feature_ranking_frame["rank"] = np.arange(1, len(feature_ranking_frame) + 1)
        feature_ranking_frames.append(feature_ranking_frame)

        top_feature_names = feature_ranking_frame.head(5)["feature_name"].tolist()
        generate_dependence_plots(shap_matrix, transformed_test_frame, config.name, top_feature_names)

    performance_df = pd.DataFrame(performance_rows)
    feature_ranking_long_df = pd.concat(feature_ranking_frames, ignore_index=True)

    if feature_ranking_long_df.empty:
        raise RuntimeError("Task A feature ranking table is empty.")

    feature_ranking_wide_df = (
        feature_ranking_long_df.pivot_table(
            index="feature_name",
            columns="target",
            values="mean_abs_shap",
            fill_value=0,
        )
        .reset_index()
        .sort_values("feature_name")
    )

    top_15_sets: list[pd.DataFrame] = []
    for target_name in feature_ranking_long_df["target"].unique():
        subset = feature_ranking_long_df.loc[feature_ranking_long_df["target"] == target_name].head(15).copy()
        subset["target"] = target_name
        top_15_sets.append(subset)
    top_15_df = pd.concat(top_15_sets, ignore_index=True)
    overlap_df = (
        top_15_df.groupby("feature_name")
        .agg(
            target_count=("target", "nunique"),
            targets=("target", lambda values: ",".join(sorted(set(values)))),
            best_rank=("rank", "min"),
            mean_abs_shap_mean=("mean_abs_shap", "mean"),
        )
        .reset_index()
        .sort_values(["target_count", "best_rank", "mean_abs_shap_mean"], ascending=[False, True, False])
    )
    overlap_df["default_shortlist"] = overlap_df["target_count"].ge(2).astype(int)

    performance_df.to_csv(MODEL_PERFORMANCE_PATH, index=False)
    feature_ranking_long_df.to_csv(FEATURE_RANKING_LONG_PATH, index=False)
    feature_ranking_wide_df.to_csv(FEATURE_RANKING_WIDE_PATH, index=False)
    overlap_df.to_csv(OVERLAP_ANALYSIS_PATH, index=False)
    overlap_df.loc[overlap_df["default_shortlist"] == 1].to_csv(DEFAULT_SHORTLIST_PATH, index=False)

    return performance_df, overlap_df


def write_summary_report(summary: dict[str, object]) -> None:
    with SUMMARY_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    ensure_output_dirs()

    machine_df, _ = load_machine_block()
    dictionary_df = load_dictionary()
    input_columns, target_columns, dictionary_present = split_dictionary_columns(
        dictionary_df,
        set(machine_df.columns),
    )

    pilot_df, holdout_df, row_counts, _, _ = build_pilot_and_holdout_datasets(
        machine_df,
        input_columns,
        target_columns,
    )
    feature_data_type_map = build_feature_data_type_map(dictionary_present)

    pilot_df.to_csv(PILOT_DATASET_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_DATASET_PATH, index=False)
    write_row_counts(row_counts)

    common_feature_columns = build_feature_columns(pilot_df, target_column=None)
    build_missing_value_outputs(pilot_df)
    build_target_distribution_plot(pilot_df)
    build_correlation_heatmap(pilot_df, common_feature_columns)
    anomaly_report = build_anomaly_report(pilot_df, dictionary_present)

    _, quick_metrics = run_quick_mhiss_model(pilot_df, feature_data_type_map)
    if not (quick_metrics["rf_r2"] > quick_metrics["baseline_r2"] and quick_metrics["rf_r2"] > 0):
        write_summary_report(
            {
                "status": "stopped_after_quick_model",
                "row_counts": row_counts,
                "quick_metrics": quick_metrics,
                "message": "Quick MHISS model did not beat baseline with positive R2. Debug side logic or feature filtering before expanding.",
            }
        )
        raise RuntimeError("Quick MHISS model failed the gate for full Task A execution.")

    performance_df, overlap_df = run_task_a_models(pilot_df, feature_data_type_map)
    write_summary_report(
        {
            "status": "completed",
            "row_counts": row_counts,
            "quick_metrics": quick_metrics,
            "anomaly_rows": int(len(anomaly_report)),
            "task_a_targets_completed": performance_df["target"].tolist(),
            "default_shortlist_features": overlap_df.loc[
                overlap_df["default_shortlist"] == 1, "feature_name"
            ].tolist(),
        }
    )


if __name__ == "__main__":
    main()
'''


TASK_B_PIPELINE_CODE = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]

PILOT_PATH = ROOT / "outputs" / "pilot_model_input.csv"
TASK_A_DIR = ROOT / "outputs" / "task_a"
TASK_B_DIR = ROOT / "outputs" / "task_b"

PROVISIONAL_FEATURE_PATH = TASK_A_DIR / "provisional_final_feature_list.csv"
OVERLAP_PATH = TASK_A_DIR / "overlap_analysis.csv"

MODEL_COMPARISON_PATH = TASK_B_DIR / "B1_model_comparison.csv"
BEST_MODEL_PATH = TASK_B_DIR / "B2_best_model.pkl"
FEATURE_WEIGHTS_PATH = TASK_B_DIR / "B3_feature_weights.csv"
PREDICTIONS_PATH = TASK_B_DIR / "B4_predictions_pilot.csv"
SCATTER_PATH = TASK_B_DIR / "B5_scatter_ai_vs_ips.png"
BLAND_ALTMAN_PATH = TASK_B_DIR / "B6_bland_altman_ai_vs_ips.png"
SUMMARY_MD_PATH = TASK_B_DIR / "B7_task_b_summary.md"
TASK_B_FEATURE_LIST_PATH = TASK_B_DIR / "task_b_selected_features.csv"
RUN_SUMMARY_PATH = TASK_B_DIR / "task_b_run_summary.json"


def ensure_dirs() -> None:
    TASK_B_DIR.mkdir(parents=True, exist_ok=True)


def is_positive_signal(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) > 0
    text = str(value).strip().lower()
    return text not in {"", "0", "0.0", "no", "none", "na", "n/a", "nan"}


def build_regression_stratify_labels(y: pd.Series, max_bins: int = 5) -> pd.Series | None:
    if y.nunique(dropna=True) < 2:
        return None
    bin_count = min(max_bins, y.nunique(dropna=True))
    if bin_count < 2:
        return None
    try:
        return pd.qcut(y.rank(method="first"), q=bin_count, labels=False, duplicates="drop")
    except ValueError:
        return None


def derive_task_b_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mangled_domain_rules = {
        "skin": (
            "skin_d",
            "skin_loss_",
            "pulp_loss_",
            "skin_lace_",
            "nail_bed_damage",
            "hand_skin_loss_",
            "soft_tissue_defect_cm2",
        ),
        "bone": ("bone_d", "fx_type_", "forearm_fx", "bony_loss_defect", "open_fracture"),
        "joint": ("joint_d", "fx_dislocation_", "lig_injury_type_"),
        "tendon": (
            "tendon_d",
            "fdp_injured_",
            "fds_injured_",
            "extensor_",
            "intrinsic_",
            "wrist_flexor_injured",
            "wrist_extensor_injured",
        ),
        "nerve": ("nerve_d", "major_nerve"),
        "vessel": ("vessel_d", "major_artery", "revasc_required"),
    }

    for domain, prefixes in mangled_domain_rules.items():
        cols = [c for c in out.columns if any(c.startswith(p) for p in prefixes)]
        if not cols:
            out[f"domain_{domain}_present"] = 0
            continue
        out[f"domain_{domain}_present"] = out[cols].apply(
            lambda row: int(any(is_positive_signal(v) for v in row)),
            axis=1,
        )

    domain_cols = [c for c in out.columns if c.startswith("domain_") and c.endswith("_present")]
    out["mangled_domain_count"] = out[domain_cols].sum(axis=1)
    out["is_mangled_ue_case"] = out["mangled_domain_count"].ge(3).astype(int)

    digit_cols = [
        c
        for c in out.columns
        if re.search(r"_d[1-5](?:_|$)", c) is not None
        and c not in {"mhiss_total", "attha_score", "attha_type", "mues_score", "strick_max", "ips_total"}
    ]
    out["has_digit_injury_case"] = out[digit_cols].apply(
        lambda row: int(any(is_positive_signal(v) for v in row)),
        axis=1,
    )

    out["is_open_fracture_case"] = pd.to_numeric(out.get("open_fracture"), errors="coerce").fillna(0).gt(0).astype(int)
    return out


def construct_ai_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mhiss_norm"] = (np.minimum(pd.to_numeric(out["mhiss_total"], errors="coerce"), 500) / 500.0) * 100.0
    out["attha_norm"] = (pd.to_numeric(out["attha_score"], errors="coerce") / 9.0) * 100.0
    out["strick_norm"] = (pd.to_numeric(out["strick_max"], errors="coerce") / 18.0) * 100.0
    out["mues_norm"] = (pd.to_numeric(out["mues_score"], errors="coerce") / 8.0) * 100.0

    weight_base = {
        "mhiss": 0.30,
        "attha": 0.25,
        "strick": 0.25,
        "mues": 0.20,
    }

    attha_app = out["is_open_fracture_case"].eq(1) & out["attha_score"].notna()
    strick_app = out["has_digit_injury_case"].eq(1) & out["strick_max"].notna()
    mues_app = out["is_mangled_ue_case"].eq(1) & out["mues_score"].notna()
    mhiss_app = out["mhiss_total"].notna()

    ai_values = []
    for idx, row in out.iterrows():
        active = {
            "mhiss": bool(mhiss_app.loc[idx]),
            "attha": bool(attha_app.loc[idx]),
            "strick": bool(strick_app.loc[idx]),
            "mues": bool(mues_app.loc[idx]),
        }
        total_w = sum(weight_base[name] for name, on in active.items() if on)
        if total_w <= 0:
            ai_values.append(np.nan)
            continue

        value = 0.0
        if active["mhiss"]:
            value += (weight_base["mhiss"] / total_w) * row["mhiss_norm"]
        if active["attha"]:
            value += (weight_base["attha"] / total_w) * row["attha_norm"]
        if active["strick"]:
            value += (weight_base["strick"] / total_w) * row["strick_norm"]
        if active["mues"]:
            value += (weight_base["mues"] / total_w) * row["mues_norm"]
        ai_values.append(value)

    out["AI_target"] = ai_values
    return out


def map_encoded_to_raw_feature(encoded_name: str, raw_columns: list[str]) -> str | None:
    if encoded_name in raw_columns:
        return encoded_name
    candidates = [col for col in raw_columns if encoded_name.startswith(f"{col}_")]
    if not candidates:
        return None
    return sorted(candidates, key=len, reverse=True)[0]


def select_task_b_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    target_and_meta = {
        "patient_id",
        "selected_side",
        "is_bilateral",
        "needs_review",
        "selected_side_positive_signal_count",
        "opposite_side_positive_signal_count",
        "mhiss_total",
        "mhiss_cat",
        "attha_score",
        "attha_type",
        "mues_score",
        "strick_max",
        "ips_total",
        "ips_band",
        "mhiss_norm",
        "attha_norm",
        "strick_norm",
        "mues_norm",
        "AI_target",
        "is_open_fracture_case",
        "is_mangled_ue_case",
        "has_digit_injury_case",
        "mangled_domain_count",
    }
    candidate_raw_features = [c for c in df.columns if c not in target_and_meta and not c.startswith("domain_")]

    source = pd.read_csv(PROVISIONAL_FEATURE_PATH if PROVISIONAL_FEATURE_PATH.exists() else OVERLAP_PATH)
    selected_raw = []
    mapping_rows = []

    for _, row in source.iterrows():
        enc = row["feature_name"]
        raw = map_encoded_to_raw_feature(enc, candidate_raw_features)
        mapping_rows.append({"encoded_feature": enc, "mapped_raw_feature": raw})
        if raw and raw not in selected_raw:
            selected_raw.append(raw)

    # Ensure at least 10 features for Task B by adding top numeric correlations with AI_target.
    if len(selected_raw) < 10:
        numeric_candidates = [
            c for c in candidate_raw_features if pd.api.types.is_numeric_dtype(df[c])
        ]
        corr_rows = []
        for c in numeric_candidates:
            series = pd.to_numeric(df[c], errors="coerce")
            valid = df["AI_target"].notna() & series.notna()
            if valid.sum() < 8:
                continue
            corr = spearmanr(series[valid], df.loc[valid, "AI_target"])[0]
            if pd.notna(corr):
                corr_rows.append((c, abs(corr)))
        corr_rows.sort(key=lambda x: x[1], reverse=True)
        for c, _ in corr_rows:
            if c not in selected_raw:
                selected_raw.append(c)
            if len(selected_raw) >= 10:
                break

    selected_raw = selected_raw[:15]
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(TASK_B_FEATURE_LIST_PATH, index=False)
    return selected_raw, mapping_df


def build_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def fit_ridge_with_cv(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train, scale_numeric=True)),
            ("model", Ridge(random_state=RANDOM_STATE)),
        ]
    )
    cv = KFold(n_splits=min(5, len(y_train)), shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipe,
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=None,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def evaluate_pipeline_model(
    model_name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, Pipeline]:
    cv = KFold(n_splits=min(5, len(y_train)), shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        return_train_score=False,
    )
    model.fit(x_train, y_train)
    pred_test = model.predict(x_test)
    metrics = {
        "model": model_name,
        "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
        "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
        "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_rmse": float(math.sqrt(mean_squared_error(y_test, pred_test))),
    }
    return metrics, pred_test, model


def build_task_b_summary_md(
    comparison_df: pd.DataFrame,
    selected_features: list[str],
    best_model_name: str,
    spearman_test: dict[str, float],
) -> str:
    lines = []
    lines.append("# Task B Summary (Pilot Dataset)")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Built AI composite target from MHISS/Atthakomol/Strickland/MUES with per-patient weight renormalization.")
    lines.append("- Trained and compared multiple regressors on the current pilot dataset.")
    lines.append("- This is a pilot Task B run; final thesis-grade validation still requires full retrospective/prospective datasets.")
    lines.append("")
    lines.append("## Selected Features")
    for f in selected_features:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Model Comparison")
    lines.append(comparison_df.to_markdown(index=False))
    lines.append("")
    lines.append(f"## Best Model")
    lines.append(f"- `{best_model_name}`")
    lines.append("")
    lines.append("## Spearman (Test Set) : AI Prediction vs Score Systems")
    for k, v in spearman_test.items():
        lines.append(f"- `{k}`: {v:.4f}" if pd.notna(v) else f"- `{k}`: NaN")
    lines.append("")
    lines.append("## Output Files")
    lines.append("- `outputs/task_b/B1_model_comparison.csv`")
    lines.append("- `outputs/task_b/B2_best_model.pkl`")
    lines.append("- `outputs/task_b/B3_feature_weights.csv`")
    lines.append("- `outputs/task_b/B4_predictions_pilot.csv`")
    lines.append("- `outputs/task_b/B5_scatter_ai_vs_ips.png`")
    lines.append("- `outputs/task_b/B6_bland_altman_ai_vs_ips.png`")
    lines.append("- `outputs/task_b/B7_task_b_summary.md`")
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    if not PILOT_PATH.exists():
        raise FileNotFoundError(f"Pilot dataset missing: {PILOT_PATH}")
    if not OVERLAP_PATH.exists():
        raise FileNotFoundError(f"Task A overlap analysis missing: {OVERLAP_PATH}")

    df = pd.read_csv(PILOT_PATH)
    df = derive_task_b_flags(df)
    df = construct_ai_target(df)
    df = df[df["AI_target"].notna()].copy()

    selected_features, mapping_df = select_task_b_features(df)
    if len(selected_features) < 5:
        raise RuntimeError("Too few usable features for Task B after mapping from Task A shortlist.")

    x = df[selected_features].copy()
    y = df["AI_target"].copy()
    stratify = build_regression_stratify_labels(y)
    x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
        x,
        y,
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    models: list[tuple[str, Pipeline | None]] = [
        (
            "LinearRegression",
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(x_train, scale_numeric=True)),
                    ("model", LinearRegression()),
                ]
            ),
        ),
        ("Ridge", fit_ridge_with_cv(x_train, y_train)),
        (
            "RandomForestRegressor",
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(x_train, scale_numeric=False)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=500,
                            min_samples_leaf=5,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "XGBRegressor",
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(x_train, scale_numeric=False)),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=300,
                            learning_rate=0.06,
                            max_depth=6,
                            reg_lambda=1.0,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            objective="reg:squarederror",
                        ),
                    ),
                ]
            ),
        ),
    ]

    comparison_rows = []
    fitted_models: dict[str, Pipeline] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for model_name, model in models:
        metrics, pred_test, fitted = evaluate_pipeline_model(
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        comparison_rows.append(metrics)
        fitted_models[model_name] = fitted
        test_predictions[model_name] = pred_test

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["test_r2", "test_mae"],
        ascending=[False, True],
    )
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    best_model_name = comparison_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    best_model.fit(x, y)

    import joblib

    joblib.dump(best_model, BEST_MODEL_PATH)
    ai_pred_all = best_model.predict(x)

    # Feature weight export
    transformed_feature_names = best_model.named_steps["preprocessor"].get_feature_names_out().tolist()
    transformed_feature_names = [n.replace("numeric__", "").replace("categorical__", "") for n in transformed_feature_names]
    model_obj = best_model.named_steps["model"]
    if hasattr(model_obj, "coef_"):
        weights = np.asarray(model_obj.coef_).reshape(-1)
        weight_df = pd.DataFrame({"feature_name": transformed_feature_names, "weight": weights})
        weight_df["abs_weight"] = np.abs(weight_df["weight"])
        weight_df = weight_df.sort_values("abs_weight", ascending=False)
    elif hasattr(model_obj, "feature_importances_"):
        weights = np.asarray(model_obj.feature_importances_).reshape(-1)
        weight_df = pd.DataFrame({"feature_name": transformed_feature_names, "importance": weights}).sort_values(
            "importance",
            ascending=False,
        )
    else:
        weight_df = pd.DataFrame({"feature_name": transformed_feature_names})
    weight_df.to_csv(FEATURE_WEIGHTS_PATH, index=False)

    prediction_df = df[
        ["patient_id", "AI_target", "mhiss_total", "attha_score", "mues_score", "strick_max", "ips_total"]
    ].copy()
    prediction_df["AI_composite"] = ai_pred_all
    prediction_df["dataset"] = "pilot"
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    # Scatter and Bland-Altman vs IPS
    valid = prediction_df["ips_total"].notna() & prediction_df["AI_composite"].notna()
    scatter_df = prediction_df.loc[valid].copy()
    rho = spearmanr(scatter_df["AI_composite"], scatter_df["ips_total"])[0] if len(scatter_df) >= 3 else np.nan

    plt.figure(figsize=(8, 6))
    plt.scatter(scatter_df["AI_composite"], scatter_df["ips_total"], alpha=0.75)
    plt.xlabel("AI Composite")
    plt.ylabel("IPS Total")
    plt.title(f"AI Composite vs IPS Total (Spearman={rho:.3f})" if pd.notna(rho) else "AI Composite vs IPS Total")
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    mean_vals = (scatter_df["AI_composite"] + scatter_df["ips_total"]) / 2.0
    diff_vals = scatter_df["AI_composite"] - scatter_df["ips_total"]
    mean_diff = diff_vals.mean()
    sd_diff = diff_vals.std(ddof=1)
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.75)
    plt.axhline(mean_diff, linestyle="--")
    plt.axhline(upper, linestyle="--")
    plt.axhline(lower, linestyle="--")
    plt.xlabel("Mean of AI Composite and IPS")
    plt.ylabel("AI Composite - IPS")
    plt.title("Bland-Altman: AI Composite vs IPS")
    plt.tight_layout()
    plt.savefig(BLAND_ALTMAN_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    test_true_idx = df.index[test_idx]
    test_eval = df.loc[test_true_idx].copy()
    test_eval["AI_pred"] = test_predictions[best_model_name]

    spearman_test = {}
    for col in ["mhiss_total", "attha_score", "mues_score", "strick_max"]:
        mask = test_eval[col].notna() & test_eval["AI_pred"].notna()
        if mask.sum() >= 3:
            spearman_test[col] = float(spearmanr(test_eval.loc[mask, "AI_pred"], test_eval.loc[mask, col])[0])
        else:
            spearman_test[col] = np.nan

    summary_md = build_task_b_summary_md(
        comparison_df=comparison_df,
        selected_features=selected_features,
        best_model_name=best_model_name,
        spearman_test=spearman_test,
    )
    SUMMARY_MD_PATH.write_text(summary_md, encoding="utf-8")

    run_summary = {
        "rows_used": int(len(df)),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "best_model": best_model_name,
        "best_model_test_r2": float(comparison_df.iloc[0]["test_r2"]),
        "best_model_test_mae": float(comparison_df.iloc[0]["test_mae"]),
        "spearman_test_vs_scores": spearman_test,
        "files": {
            "B1": str(MODEL_COMPARISON_PATH),
            "B2": str(BEST_MODEL_PATH),
            "B3": str(FEATURE_WEIGHTS_PATH),
            "B4": str(PREDICTIONS_PATH),
            "B5": str(SCATTER_PATH),
            "B6": str(BLAND_ALTMAN_PATH),
            "B7": str(SUMMARY_MD_PATH),
        },
    }
    RUN_SUMMARY_PATH.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


REQUIRED_OUTPUTS_STAGE_A = [
    OUTPUT_DIR / "pilot_model_input.csv",
    OUTPUT_DIR / "pilot_holdout_rows.csv",
    TASK_A_DIR / "model_performance.csv",
    TASK_A_DIR / "overlap_analysis.csv",
    TASK_A_DIR / "provisional_final_feature_list.csv",
]

REQUIRED_OUTPUTS_STAGE_B = [
    TASK_B_DIR / "B1_model_comparison.csv",
    TASK_B_DIR / "B2_best_model.pkl",
    TASK_B_DIR / "B3_feature_weights.csv",
    TASK_B_DIR / "B4_predictions_pilot.csv",
    TASK_B_DIR / "B5_scatter_ai_vs_ips.png",
    TASK_B_DIR / "B6_bland_altman_ai_vs_ips.png",
    TASK_B_DIR / "B7_task_b_summary.md",
]


def execute_embedded_script(script_code: str, fake_script_path: Path, module_name: str) -> None:
    module = types.ModuleType(module_name)
    module.__file__ = str(fake_script_path)
    sys.modules[module_name] = module
    exec(compile(script_code, str(fake_script_path), "exec"), module.__dict__)
    main_fn = module.__dict__.get("main")
    if not callable(main_fn):
        raise RuntimeError(f"Embedded script {module_name} does not define a callable main().")
    main_fn()


def check_outputs(paths: list[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{label} completed but required outputs are missing:\n" + "\n".join(missing))


def ensure_provisional_feature_list() -> None:
    if PROVISIONAL_FEATURE_PATH.exists():
        return
    if not OVERLAP_PATH.exists():
        return
    overlap_df = pd.read_csv(OVERLAP_PATH)
    if "target_count" not in overlap_df.columns:
        return
    shortlist_df = overlap_df.loc[overlap_df["target_count"] >= 2].copy()
    shortlist_df.to_csv(PROVISIONAL_FEATURE_PATH, index=False)


def run_stage_a() -> None:
    execute_embedded_script(
        script_code=FEATURE_SHAP_FAST_START_CODE,
        fake_script_path=ROOT / "scripts" / "feature_shap_fast_start.py",
        module_name="embedded_task_a",
    )
    ensure_provisional_feature_list()


def run_stage_b() -> None:
    ensure_provisional_feature_list()
    execute_embedded_script(
        script_code=TASK_B_PIPELINE_CODE,
        fake_script_path=ROOT / "scripts" / "task_b_pipeline.py",
        module_name="embedded_task_b",
    )


def run_full_pipeline() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_mode": "single_end_to_end",
        "stages": {
            "stage_a": {"status": "not_run"},
            "stage_b": {"status": "not_run"},
        },
    }

    try:
        print("Running unified pipeline...")
        print("Step 1/2: building dataset + feature importance outputs...")
        run_stage_a()
        check_outputs(REQUIRED_OUTPUTS_STAGE_A, "Stage A")
        summary["stages"]["stage_a"] = {"status": "completed"}

        print("Step 2/2: building AI composite outputs...")
        run_stage_b()
        check_outputs(REQUIRED_OUTPUTS_STAGE_B, "Stage B")
        summary["stages"]["stage_b"] = {"status": "completed"}
        print("Unified pipeline completed.")

        summary["status"] = "completed"
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        SINGLE_FILE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Run summary saved: {SINGLE_FILE_SUMMARY_PATH}")
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        SINGLE_FILE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        print(f"Failure summary saved: {SINGLE_FILE_SUMMARY_PATH}", file=sys.stderr)
        return 1


def main() -> int:
    return run_full_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
