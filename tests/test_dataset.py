from __future__ import annotations

import pandas as pd
import pytest

from hand_injury_ai.dataset import (
    DataInputError,
    build_feature_data_type_map,
    build_pilot_and_holdout_datasets,
    decode_hand_code,
    expand_grouped_dictionary_columns,
    infer_side_from_dominance_label,
    load_dictionary,
    load_machine_block,
    normalize_side_label,
    parse_numeric_range,
    split_dictionary_columns,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "Left"), (1.0, "Right"), ("2", "Bilateral"), (5, ""), ("", ""), (None, ""), ("x", "")],
)
def test_decode_hand_code(value: object, expected: str) -> None:
    assert decode_hand_code(value) == expected


def test_normalize_side_label() -> None:
    assert normalize_side_label(" non-dominant ") == "Non-Dominant"
    assert normalize_side_label("LEFT") == "Left"
    assert normalize_side_label("Something else") == "Something else"


@pytest.mark.parametrize(
    ("label", "dominant", "expected"),
    [
        ("Dominant", "Right", "Right"),
        ("Non-Dominant", "Right", "Left"),
        ("Non-Dominant", "Left", "Right"),
        ("Bilateral", "", "Bilateral"),
        ("Dominant", "", ""),
        ("Left", "Right", ""),
    ],
)
def test_infer_side_from_dominance_label(label: str, dominant: str, expected: str) -> None:
    assert infer_side_from_dominance_label(label, dominant) == expected


def test_parse_numeric_range() -> None:
    assert parse_numeric_range("0-412") == (0.0, 412.0)
    assert parse_numeric_range("1.5 – 9") == (1.5, 9.0)
    assert parse_numeric_range("-5 - 5") == (-5.0, 5.0)
    assert parse_numeric_range("yes/no") is None
    assert parse_numeric_range(None) is None


def test_expand_grouped_dictionary_columns() -> None:
    dictionary = pd.DataFrame(
        {
            "column_name": ["skin_d2_d5_l", "strick_d1_r_d5_r", "age"],
            "layer": ["input", "target", "input"],
        }
    )
    machine = {"skin_d2_l", "skin_d3_l", "strick_d1_r", "strick_d5_r", "age"}
    expanded = expand_grouped_dictionary_columns(dictionary, machine)
    assert {"skin_d2_l", "skin_d3_l", "strick_d1_r", "strick_d5_r"} <= set(expanded["column_name"])
    inputs, targets, present = split_dictionary_columns(dictionary, machine)
    assert set(inputs) == {"skin_d2_l", "skin_d3_l", "age"}
    assert set(targets) == {"strick_d1_r", "strick_d5_r"}
    assert set(present["column_name"]) == machine


def test_feature_type_map_strips_side_suffix() -> None:
    present = pd.DataFrame(
        {"column_name": ["age", "skin_d1_l", "skin_d1_r"], "data_type": ["Integer", "Binary", "x"]}
    )
    assert build_feature_data_type_map(present) == {"age": "integer", "skin_d1": "binary"}


def test_load_machine_block_drops_identifiers(synthetic_paths) -> None:
    machine = load_machine_block(synthetic_paths.workbook)
    assert "patient_name" not in machine.columns
    assert machine.columns[0] == "patient_id"
    assert machine["patient_id"].iloc[0] == "P0001"
    assert {"human_side_involved", "human_injured_hand_label", "human_dominance_label"} <= set(
        machine.columns
    )


def test_missing_inputs_raise_helpful_errors(tmp_path) -> None:
    with pytest.raises(DataInputError, match="HAND_INJURY_WORKBOOK_PATH"):
        load_machine_block(tmp_path / "missing.xlsx")
    with pytest.raises(DataInputError, match="HAND_INJURY_DATA_DICTIONARY_PATH"):
        load_dictionary(tmp_path / "missing.xlsx")


def test_workbook_without_patient_id_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    with pytest.raises(DataInputError, match="patient_id"):
        load_machine_block(path)


def _machine_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": "P1",
        "injured_hand": 0,
        "dominant_hand": 1,
        "age": 40,
        "skin_d1_l": 1,
        "skin_d1_r": 0,
        "mhiss_total_l": 30,
        "mhiss_total_r": None,
        "human_side_involved": "Left",
        "human_injured_hand_label": "Non-Dominant",
        "human_dominance_label": "Right",
    }
    row.update(overrides)
    return row


def test_cohort_routing_rules() -> None:
    machine = pd.DataFrame(
        [
            _machine_row(patient_id="ok"),
            _machine_row(patient_id="bilateral", human_side_involved="Bilateral"),
            _machine_row(patient_id="code_mismatch", injured_hand=1),
            _machine_row(patient_id="label_mismatch", human_injured_hand_label="Dominant"),
            _machine_row(patient_id="no_signal", skin_d1_l=0),
            _machine_row(patient_id="opposite", skin_d1_r=1),
        ]
    )
    cohort = build_pilot_and_holdout_datasets(
        machine, ["age", "skin_d1_l", "skin_d1_r"], ["mhiss_total_l", "mhiss_total_r"]
    )
    assert cohort.pilot["patient_id"].tolist() == ["ok"]
    assert cohort.pilot.loc[0, "skin_d1"] == 1
    assert cohort.pilot.loc[0, "mhiss_total"] == 30
    reasons = dict(zip(cohort.holdout["patient_id"], cohort.holdout["holdout_reason"], strict=True))
    assert "bilateral_case" in reasons["bilateral"]
    assert "side_involved_machine_injured_hand_mismatch" in reasons["code_mismatch"]
    assert "side_involved_human_injured_hand_mismatch" in reasons["label_mismatch"]
    assert "no_positive_signal_on_selected_side" in reasons["no_signal"]
    assert "positive_signal_present_on_opposite_side" in reasons["opposite"]
    assert cohort.counts == {
        "total_rows": 6,
        "pilot_rows": 1,
        "holdout_rows": 5,
        "bilateral_rows": 1,
    }


def test_empty_pilot_raises() -> None:
    machine = pd.DataFrame([_machine_row(human_side_involved="Bilateral")])
    with pytest.raises(DataInputError):
        build_pilot_and_holdout_datasets(machine, ["skin_d1_l", "skin_d1_r"], [])


def test_unreadable_inputs_raise_data_input_error(tmp_path) -> None:
    import pandas as pd
    import pytest

    from hand_injury_ai.dataset import DataInputError, load_dictionary, load_machine_block

    not_excel = tmp_path / "not_excel.xlsx"
    not_excel.write_text("this is not a workbook", encoding="utf-8")
    with pytest.raises(DataInputError, match="Could not read source workbook"):
        load_machine_block(not_excel)

    wrong_sheet = tmp_path / "wrong_sheet.xlsx"
    pd.DataFrame({"Column Name": ["age"], "Layer": ["input"]}).to_excel(
        wrong_sheet, sheet_name="Sheet1", index=False
    )
    with pytest.raises(DataInputError, match="no sheet named 'Data Dictionary'"):
        load_dictionary(wrong_sheet)
