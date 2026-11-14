"""Helpers shared by Task A (feature importance) and Task B (AI composite score)."""

from __future__ import annotations

import hashlib
import hmac
import re

import numpy as np
import pandas as pd

# Columns that count as evidence that a tissue domain is injured. A patient with three
# or more injured domains is treated as a mangled upper-extremity (UE) case.
MANGLED_DOMAIN_RULES: dict[str, tuple[str, ...]] = {
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

MANGLED_DOMAIN_THRESHOLD = 3

# Scoring-system outputs. They are prediction targets, never model inputs.
SCORE_COLUMNS = frozenset(
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

DERIVED_FLAG_COLUMNS = frozenset(
    {
        "is_open_fracture_case",
        "is_mangled_ue_case",
        "has_digit_injury_case",
        "mangled_domain_count",
    }
)

_NEGATIVE_TOKENS = frozenset({"", "0", "0.0", "no", "none", "na", "n/a", "nan"})
_DIGIT_COLUMN_PATTERN = re.compile(r"_d[1-5](?:_|$)")


def canonicalize_name(value: object) -> str:
    """Normalise a header to ``snake_case`` ASCII (``"Age (yrs)"`` -> ``"age_yrs"``)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def make_unique(names: list[object]) -> list[str]:
    """Canonicalise names and suffix duplicates with ``__dup2``, ``__dup3``..."""
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for raw_name in names:
        base = canonicalize_name(raw_name) or "unnamed"
        count = seen.get(base, 0) + 1
        seen[base] = count
        unique_names.append(base if count == 1 else f"{base}__dup{count}")
    return unique_names


def is_positive_signal(value: object) -> bool:
    """Return True when a cell indicates that something is present / injured."""
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return not pd.isna(value) and float(value) > 0
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in _NEGATIVE_TOKENS


def build_regression_stratify_labels(y: pd.Series, max_bins: int = 5) -> pd.Series | None:
    """Quantile-bin a continuous target so train/test splits keep its distribution."""
    unique_count = y.nunique(dropna=True)
    if unique_count < 2:
        return None
    bin_count = min(max_bins, unique_count)
    try:
        labels = pd.qcut(y.rank(method="first"), q=bin_count, labels=False, duplicates="drop")
    except ValueError:
        return None
    if labels.value_counts().min() < 2:
        return None
    return labels


def safe_classification_stratify(y: pd.Series) -> pd.Series | None:
    """Stratify on class labels only when every class has at least two members."""
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return y


def derive_case_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add the subset flags that decide which scoring systems apply to each patient.

    * ``is_open_fracture_case`` – Atthakomol applies to open fractures.
    * ``domain_<name>_present`` / ``mangled_domain_count`` / ``is_mangled_ue_case`` –
      MUES applies to mangled injuries (three or more injured tissue domains).
    * ``has_digit_injury_case`` – Strickland applies when any digit is injured.
    """
    out = df.copy()
    open_fracture = out["open_fracture"] if "open_fracture" in out.columns else 0
    out["is_open_fracture_case"] = (
        pd.to_numeric(pd.Series(open_fracture, index=out.index), errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    for domain_name, prefixes in MANGLED_DOMAIN_RULES.items():
        domain_columns = [
            column for column in out.columns if any(column.startswith(p) for p in prefixes)
        ]
        out[f"domain_{domain_name}_present"] = _any_positive(out, domain_columns)

    domain_columns = [f"domain_{name}_present" for name in MANGLED_DOMAIN_RULES]
    out["mangled_domain_count"] = out[domain_columns].sum(axis=1).astype(int)
    out["is_mangled_ue_case"] = out["mangled_domain_count"].ge(MANGLED_DOMAIN_THRESHOLD).astype(int)

    digit_columns = [
        column
        for column in out.columns
        if _DIGIT_COLUMN_PATTERN.search(column) and column not in SCORE_COLUMNS
    ]
    out["has_digit_injury_case"] = _any_positive(out, digit_columns)
    return out


def _any_positive(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0, index=frame.index, dtype=int)
    return (
        frame[columns]
        .apply(lambda row: int(any(is_positive_signal(value) for value in row)), axis=1)
        .astype(int)
    )


def pseudonymize_ids(ids: pd.Series, salt: str | None) -> pd.Series:
    """Replace hospital identifiers with stable pseudonyms before anything is written.

    With a secret ``salt`` the pseudonym is a keyed HMAC-SHA256 digest, so the data
    custodian (who holds the salt) can re-link rows to the source record while nobody
    else can. Without a salt, sequential ``P0001``-style codes are used.
    """
    if salt:
        key = salt.encode("utf-8")

        def _digest(value: object) -> object:
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                return pd.NA
            text = str(value).strip()
            if text.endswith(".0"):
                text = text[:-2]
            digest = hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()
            return f"P-{digest[:12]}"

        return ids.map(_digest)

    width = max(4, len(str(len(ids))))
    return pd.Series(
        [f"P{index:0{width}d}" for index in range(1, len(ids) + 1)],
        index=ids.index,
        dtype=object,
    )
