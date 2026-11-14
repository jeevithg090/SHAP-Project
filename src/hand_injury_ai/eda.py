"""Exploratory data analysis artefacts for the pilot cohort."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .dataset import parse_numeric_range

FIGURE_DPI = 200
PLOTTED_TARGETS = ("mhiss_total", "attha_type", "mues_score", "strick_max")


def write_row_counts(counts: dict[str, int], path: Path) -> None:
    pd.DataFrame([{"metric": key, "value": value} for key, value in counts.items()]).to_csv(
        path, index=False
    )


def build_missing_value_outputs(pilot_df: pd.DataFrame, eda_dir: Path) -> pd.DataFrame:
    missing = (
        pilot_df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda frame: (frame["missing_count"] / len(pilot_df) * 100).round(2))
        .reset_index(names="column")
        .sort_values(["missing_count", "column"], ascending=[False, True])
    )
    missing.to_csv(eda_dir / "missing_value_summary.csv", index=False)

    plot_df = pilot_df.drop(columns=["patient_id"], errors="ignore")
    plt.figure(figsize=(28, 10))
    sns.heatmap(plot_df.isna(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Data Heatmap - Unilateral Pilot Dataset")
    plt.xlabel("Columns")
    plt.tight_layout()
    plt.savefig(eda_dir / "missing_data_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return missing


def build_target_distribution_plot(pilot_df: pd.DataFrame, eda_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    panels = [
        ("mhiss_total", "MHISS Total", "hist", 30),
        ("attha_type", "Atthakomol Type", "bar", None),
        ("mues_score", "MUES Score", "hist", 10),
        ("strick_max", "Strickland Max", "hist", 18),
    ]
    for axis, (column, title, kind, bins) in zip(axes.flat, panels, strict=True):
        axis.set_title(title)
        if column not in pilot_df.columns or pilot_df[column].dropna().empty:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
            continue
        series = pilot_df[column].dropna()
        if kind == "bar":
            series.astype(str).value_counts().sort_index().plot(
                kind="bar", ax=axis, edgecolor="black"
            )
            axis.set_xlabel("Class")
        else:
            pd.to_numeric(series, errors="coerce").dropna().plot(
                kind="hist", bins=bins, ax=axis, edgecolor="black"
            )
            axis.set_xlabel("Score")
    figure.tight_layout()
    figure.savefig(eda_dir / "target_distributions.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def build_correlation_heatmap(
    pilot_df: pd.DataFrame, feature_columns: list[str], eda_dir: Path
) -> pd.DataFrame:
    targets = [column for column in PLOTTED_TARGETS if column in pilot_df.columns]
    numeric_features = [
        column
        for column in feature_columns
        if column in pilot_df.columns and pd.api.types.is_numeric_dtype(pilot_df[column])
    ]
    correlation = pilot_df[numeric_features + targets].corr(method="spearman")
    cross = correlation.loc[numeric_features, targets]

    plt.figure(figsize=(12, max(8, len(numeric_features) * 0.22)))
    sns.heatmap(cross, cmap="coolwarm", center=0)
    plt.title("Spearman Correlation: Numeric Features vs Targets")
    plt.tight_layout()
    plt.savefig(
        eda_dir / "spearman_feature_target_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    plt.close()
    return cross


def build_anomaly_report(
    pilot_df: pd.DataFrame, dictionary_present: pd.DataFrame, path: Path
) -> pd.DataFrame:
    """Compare observed numeric values with the dictionary's allowed range."""
    lookup = dictionary_present.drop_duplicates("column_name").set_index("column_name")
    rows: list[dict[str, object]] = []
    for column in pilot_df.columns:
        if column not in lookup.index or not pd.api.types.is_numeric_dtype(pilot_df[column]):
            continue
        allowed = parse_numeric_range(lookup.at[column, "allowed_values_range"])
        if allowed is None:
            continue
        lower, upper = allowed
        series = pilot_df[column].dropna()
        rows.append(
            {
                "column": column,
                "allowed_min": lower,
                "allowed_max": upper,
                "observed_min": float(series.min()) if not series.empty else np.nan,
                "observed_max": float(series.max()) if not series.empty else np.nan,
                "below_min_count": int((pilot_df[column] < lower).sum()),
                "above_max_count": int((pilot_df[column] > upper).sum()),
            }
        )

    columns = [
        "column",
        "allowed_min",
        "allowed_max",
        "observed_min",
        "observed_max",
        "below_min_count",
        "above_max_count",
    ]
    report = pd.DataFrame(rows, columns=columns).sort_values(
        ["above_max_count", "below_min_count", "column"], ascending=[False, False, True]
    )
    report.to_csv(path, index=False)
    return report
