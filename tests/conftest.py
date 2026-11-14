"""Synthetic, fully fictional workbook + data dictionary fixtures.

No real patient data is used anywhere in the test suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hand_injury_ai.config import Paths

DOMAINS = ("skin", "bone", "joint", "tendon", "nerve", "vessel")


def _side_row(rng: np.random.Generator, prefix: str, injured: bool) -> dict[str, object]:
    row: dict[str, object] = {}
    for domain in DOMAINS:
        row[f"{domain}_d1{prefix}"] = int(injured and rng.random() < 0.6)
    row[f"fx_type_d1{prefix}"] = (
        int(rng.integers(1, 4)) if injured and row[f"bone_d1{prefix}"] else 0
    )
    return row


def build_synthetic_workbook(n_patients: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for index in range(n_patients):
        side = "Left" if index % 2 == 0 else "Right"
        dominant = "Right" if index % 3 else "Left"
        dominance_label = "Dominant" if side == dominant else "Non-Dominant"
        bilateral = index in (5, 11)
        mismatch = index == 17

        left = _side_row(rng, "_l", injured=side == "Left" or bilateral)
        right = _side_row(rng, "_r", injured=side == "Right" or bilateral)
        open_fracture = int(rng.random() < 0.6)
        age = int(rng.integers(18, 80))
        debride = float(rng.uniform(1, 48))

        record: dict[str, object] = {
            # Human-readable block (fictional values only).
            "Patient Name": f"Synthetic Person {index}",
            "Side Involved": "Bilateral" if bilateral else side,
            "Injured Hand": "Bilateral" if bilateral else dominance_label,
            "Dominance": dominant,
            # Machine-readable block starts at patient_id.
            "patient_id": 900000 + index,
            "injured_hand": 2 if bilateral else (0 if (side == "Left") != mismatch else 1),
            "dominant_hand": 0 if dominant == "Left" else 1,
            "age": age,
            "mechanism": int(rng.integers(1, 4)),
            "open_fracture": open_fracture,
            "time_to_debride_hrs": debride,
            **left,
            **right,
        }
        for suffix, findings in (("_l", left), ("_r", right)):
            injured_count = sum(int(findings[f"{d}_d1{suffix}"]) for d in DOMAINS)
            active = injured_count > 0
            noise = rng.normal(0, 3)
            record[f"mhiss_total{suffix}"] = (
                float(max(0.0, 25 * injured_count + 0.8 * debride + noise)) if active else np.nan
            )
            record[f"attha_score{suffix}"] = (
                float(min(9, injured_count + open_fracture)) if active else np.nan
            )
            record[f"attha_type{suffix}"] = int(1 + (index % 3)) if active else np.nan
            record[f"mues_score{suffix}"] = float(min(8, injured_count)) if active else np.nan
            record[f"strick_max{suffix}"] = float(min(18, 3 * injured_count)) if active else np.nan
            record[f"ips_total{suffix}"] = (
                float(10 + 5 * injured_count + noise) if active else np.nan
            )
        records.append(record)
    return pd.DataFrame(records)


def build_synthetic_dictionary() -> pd.DataFrame:
    rows = [
        ("patient_id", "input", "string", ""),
        ("injured_hand", "input", "categorical", "0-2"),
        ("dominant_hand", "input", "categorical", "0-1"),
        ("age", "input", "integer", "0-120"),
        ("mechanism", "input", "categorical", ""),
        ("open_fracture", "input", "binary", "0-1"),
        ("time_to_debride_hrs", "input", "continuous", "0-500"),
    ]
    for suffix in ("_l", "_r"):
        for domain in DOMAINS:
            rows.append((f"{domain}_d1{suffix}", "input", "binary", "0-1"))
        rows.append((f"fx_type_d1{suffix}", "input", "categorical", ""))
        for score in (
            "mhiss_total",
            "attha_score",
            "attha_type",
            "mues_score",
            "strick_max",
            "ips_total",
        ):
            rows.append((f"{score}{suffix}", "target", "integer", ""))
    return pd.DataFrame(
        rows, columns=["Column Name", "Layer", "Data Type", "Allowed Values / Range"]
    )


@pytest.fixture
def synthetic_paths(tmp_path: Path) -> Paths:
    workbook = tmp_path / "data" / "workbook.xlsx"
    dictionary = tmp_path / "data" / "dictionary.xlsx"
    workbook.parent.mkdir(parents=True)
    build_synthetic_workbook().to_excel(workbook, index=False)
    with pd.ExcelWriter(dictionary) as writer:
        build_synthetic_dictionary().to_excel(writer, sheet_name="Data Dictionary", index=False)
    return Paths(
        workbook=workbook,
        data_dictionary=dictionary,
        output_dir=tmp_path / "outputs",
        reports_dir=tmp_path / "reports",
    )
