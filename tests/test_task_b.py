from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hand_injury_ai.task_b import (
    bland_altman_stats,
    construct_ai_target,
    map_encoded_to_raw_feature,
    select_task_b_features,
)


def _scores(**overrides: object) -> pd.DataFrame:
    base = {
        "mhiss_total": [100.0],
        "attha_score": [4.5],
        "strick_max": [9.0],
        "mues_score": [4.0],
        "is_open_fracture_case": [1],
        "has_digit_injury_case": [1],
        "is_mangled_ue_case": [1],
    }
    base.update({key: [value] for key, value in overrides.items()})
    return pd.DataFrame(base)


def test_ai_target_all_scores_applicable() -> None:
    out = construct_ai_target(_scores())
    # 0.30*20 + 0.25*50 + 0.25*50 + 0.20*50
    assert out["AI_target"].iloc[0] == pytest.approx(41.0)


def test_ai_target_renormalises_weights_for_inapplicable_scores() -> None:
    out = construct_ai_target(_scores(is_open_fracture_case=0, is_mangled_ue_case=0))
    # Only MHISS (0.30) and Strickland (0.25) apply.
    expected = (0.30 * 20 + 0.25 * 50) / 0.55
    assert out["AI_target"].iloc[0] == pytest.approx(expected)


def test_ai_target_caps_mhiss_and_handles_no_scores() -> None:
    capped = construct_ai_target(
        _scores(
            mhiss_total=900.0,
            is_open_fracture_case=0,
            has_digit_injury_case=0,
            is_mangled_ue_case=0,
        )
    )
    assert capped["AI_target"].iloc[0] == pytest.approx(100.0)
    empty = construct_ai_target(
        _scores(
            mhiss_total=np.nan,
            is_open_fracture_case=0,
            has_digit_injury_case=0,
            is_mangled_ue_case=0,
        )
    )
    assert math.isnan(empty["AI_target"].iloc[0])


def test_map_encoded_to_raw_feature_prefers_longest_prefix() -> None:
    raw = ["fx_type", "fx_type_d1", "age"]
    assert map_encoded_to_raw_feature("age", raw) == "age"
    assert map_encoded_to_raw_feature("fx_type_d1_0", raw) == "fx_type_d1"
    assert map_encoded_to_raw_feature("unknown_1", raw) is None


def test_select_task_b_features_tops_up_with_correlates() -> None:
    rng = np.random.default_rng(0)
    n = 40
    target = rng.normal(size=n)
    df = pd.DataFrame({f"f{i}": target * (i + 1) + rng.normal(size=n) for i in range(12)})
    df["AI_target"] = target
    df["patient_id"] = range(n)
    df["mhiss_total"] = target
    source = pd.DataFrame({"feature_name": ["f0_1", "f0_2", "missing_feature"]})
    selected, mapping = select_task_b_features(df, source)
    assert selected[0] == "f0"
    assert len(selected) == 10
    assert "patient_id" not in selected and "mhiss_total" not in selected
    assert mapping["mapped_raw_feature"].tolist()[:2] == ["f0", "f0"]
    assert pd.isna(mapping["mapped_raw_feature"].iloc[2])


def test_bland_altman_stats() -> None:
    stats = bland_altman_stats(pd.Series([1.0, 2.0, 3.0]), pd.Series([0.0, 2.0, 4.0]))
    assert stats["mean_diff"] == pytest.approx(0.0)
    assert stats["loa_upper"] == pytest.approx(1.96)
    assert stats["loa_lower"] == pytest.approx(-1.96)
