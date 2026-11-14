from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hand_injury_ai.common import (
    build_regression_stratify_labels,
    canonicalize_name,
    derive_case_flags,
    is_positive_signal,
    make_unique,
    pseudonymize_ids,
    safe_classification_stratify,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Age(at time of injury)", "age_at_time_of_injury"),
        ("  Side\nInvolved ", "side_involved"),
        ("MHISS_Total_L", "mhiss_total_l"),
        ("___", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_canonicalize_name(raw: object, expected: str) -> None:
    assert canonicalize_name(raw) == expected


def test_make_unique_suffixes_duplicates_and_blanks() -> None:
    assert make_unique(["Age", "age", "AGE", "", None]) == [
        "age",
        "age__dup2",
        "age__dup3",
        "unnamed",
        "unnamed__dup2",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (0, False),
        (-1, False),
        (2.5, True),
        (np.nan, False),
        (None, False),
        (True, True),
        (False, False),
        ("Yes", True),
        ("no", False),
        (" N/A ", False),
        ("0.0", False),
        ("", False),
    ],
)
def test_is_positive_signal(value: object, expected: bool) -> None:
    assert is_positive_signal(value) is expected


def test_regression_stratify_labels_bins_and_degenerate_cases() -> None:
    labels = build_regression_stratify_labels(pd.Series(range(50)))
    assert labels is not None
    assert labels.nunique() == 5
    assert build_regression_stratify_labels(pd.Series([3.0] * 10)) is None


def test_safe_classification_stratify_requires_two_per_class() -> None:
    assert safe_classification_stratify(pd.Series([1, 1, 2, 2, 3, 3])) is not None
    assert safe_classification_stratify(pd.Series([1, 1, 2, 2, 3])) is None
    assert safe_classification_stratify(pd.Series([1, 1, 1])) is None


def test_derive_case_flags() -> None:
    df = pd.DataFrame(
        {
            "open_fracture": [1, 0, "1", np.nan],
            "skin_d1": [1, 0, 0, 0],
            "bone_d1": [1, 0, 0, 0],
            "tendon_d2": [1, 0, 1, 0],
            "major_nerve": [0, 0, 0, 0],
            "mhiss_total": [10, 0, 5, 0],
        }
    )
    flags = derive_case_flags(df)
    assert flags["is_open_fracture_case"].tolist() == [1, 0, 1, 0]
    # skin + bone (open_fracture counts as bone) + tendon -> three domains.
    assert flags["mangled_domain_count"].tolist() == [3, 0, 2, 0]
    assert flags["is_mangled_ue_case"].tolist() == [1, 0, 0, 0]
    assert flags["has_digit_injury_case"].tolist() == [1, 0, 1, 0]
    assert "is_open_fracture_case" not in df.columns  # input is not mutated


def test_pseudonymize_without_salt_uses_sequential_codes() -> None:
    ids = pd.Series([4003779, 2794659, 4003784])
    assert pseudonymize_ids(ids, None).tolist() == ["P0001", "P0002", "P0003"]


def test_pseudonymize_with_salt_is_stable_and_hides_raw_ids() -> None:
    ids = pd.Series([4003779, "4003779", 4003779.0, 123])
    out = pseudonymize_ids(ids, "secret")
    assert out.iloc[0] == out.iloc[1] == out.iloc[2]
    assert out.iloc[0] != out.iloc[3]
    assert all(str(value).startswith("P-") and "4003779" not in str(value) for value in out)
    assert pseudonymize_ids(ids, "other").iloc[0] != out.iloc[0]
