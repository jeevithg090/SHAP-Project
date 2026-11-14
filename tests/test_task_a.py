from __future__ import annotations

import numpy as np
import pandas as pd

from hand_injury_ai.task_a import (
    build_feature_columns,
    build_overlap_table,
    coerce_feature_types,
    compute_multiclass_auc,
    get_shap_matrix,
    quick_gate_passed,
    quick_preprocess_train_test,
)


def test_feature_columns_exclude_ids_scores_and_flags() -> None:
    frame = pd.DataFrame(
        columns=[
            "patient_id",
            "selected_side",
            "age",
            "mhiss_total",
            "ips_total",
            "is_mangled_ue_case",
            "domain_skin_present",
            "skin_d1",
        ]
    )
    assert build_feature_columns(frame) == ["age", "skin_d1"]
    assert build_feature_columns(frame, target_column="age") == ["skin_d1"]


def test_quick_preprocess_fits_on_train_only() -> None:
    train = pd.DataFrame({"age": [10.0, np.nan, 30.0], "mech": ["a", "b", None]})
    test = pd.DataFrame({"age": [np.nan], "mech": ["c"]})
    x_train, x_test = quick_preprocess_train_test(train, test, ["age", "mech"], {"age": "integer"})
    assert x_train["age"].tolist() == [10.0, 20.0, 30.0]
    assert x_test["age"].tolist() == [20.0]
    assert list(x_train.columns) == list(x_test.columns)
    assert "mech_c" not in x_test.columns  # unseen category dropped


def test_coerce_feature_types() -> None:
    frame = pd.DataFrame({"age": ["5", "x"], "mech": [1, np.nan]})
    out = coerce_feature_types(frame, {"age": "integer"})
    assert out["age"].tolist()[0] == 5.0 and np.isnan(out["age"].tolist()[1])
    assert out["mech"].tolist()[0] == "1.0" and pd.isna(out["mech"].tolist()[1])


def test_get_shap_matrix_selects_predicted_class() -> None:
    values = np.zeros((2, 3, 2))
    values[0, :, 1] = 1.0
    values[1, :, 0] = 2.0
    probabilities = np.array([[0.2, 0.8], [0.9, 0.1]])
    matrix = get_shap_matrix(values, probabilities)
    assert matrix.tolist() == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert get_shap_matrix(np.ones((4, 2))).shape == (4, 2)
    assert get_shap_matrix([np.ones((4, 2)), np.zeros((4, 2))]).shape == (4, 2)


def test_binary_auc_is_computed() -> None:
    y = pd.Series([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
    assert compute_multiclass_auc(y, probabilities, np.array([0, 1])) == 1.0
    assert np.isnan(compute_multiclass_auc(y, probabilities[:, :1], np.array([0])))


def test_overlap_table_counts_targets_within_top_k() -> None:
    ranking = pd.DataFrame(
        {
            "target": ["a", "a", "b", "b"],
            "feature_name": ["age", "skin", "age", "bone"],
            "mean_abs_shap": [0.5, 0.1, 0.3, 0.2],
            "rank": [1, 2, 1, 2],
        }
    )
    overlap = build_overlap_table(ranking, top_k=2)
    row = overlap.set_index("feature_name").loc["age"]
    assert row["target_count"] == 2 and row["targets"] == "a,b" and row["default_shortlist"] == 1
    assert overlap["feature_name"].iloc[0] == "age"
    assert set(overlap.loc[overlap["default_shortlist"] == 0, "feature_name"]) == {"skin", "bone"}
    assert build_overlap_table(ranking, top_k=1)["feature_name"].tolist() == ["age"]


def test_quick_gate() -> None:
    assert quick_gate_passed({"rf_r2": 0.1, "baseline_r2": -0.2})
    assert not quick_gate_passed({"rf_r2": -0.1, "baseline_r2": -0.2})
    assert not quick_gate_passed({"rf_r2": 0.1, "baseline_r2": 0.2})
