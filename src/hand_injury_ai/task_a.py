"""Task A: per-score Random Forest models explained with SHAP.

For each scoring system (MHISS, Atthakomol, MUES, Strickland) a Random Forest is
trained on the clinically applicable subset of the pilot cohort, SHAP values are
computed on the held-out split, and features are ranked by mean |SHAP|. Features that
rank in the top 15 for two or more scores form the default shortlist used by Task B.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize

from . import eda
from .common import (
    DERIVED_FLAG_COLUMNS,
    SCORE_COLUMNS,
    build_regression_stratify_labels,
    derive_case_flags,
    safe_classification_stratify,
)
from .config import RANDOM_STATE, Paths
from .dataset import (
    build_feature_data_type_map,
    build_pilot_and_holdout_datasets,
    load_dictionary,
    load_machine_block,
    split_dictionary_columns,
)

logger = logging.getLogger(__name__)

FIGURE_DPI = 200
NUMERIC_DATA_TYPES = frozenset({"integer", "continuous"})
METADATA_COLUMNS = frozenset(
    {
        "patient_id",
        "selected_side",
        "is_bilateral",
        "needs_review",
        "holdout_reason",
        "selected_side_positive_signal_count",
        "opposite_side_positive_signal_count",
    }
)
TOP_K_FOR_OVERLAP = 15
TOP_K_DEPENDENCE_PLOTS = 5
MIN_SUBSET_ROWS = 10


class QuickModelGateError(RuntimeError):
    """The quick MHISS sanity model did not beat a mean baseline."""


@dataclass
class TargetConfig:
    name: str
    task_type: str  # "regression" or "classification"
    subset_filter: Callable[[pd.DataFrame], pd.Series]
    estimator: Any


def default_target_configs() -> list[TargetConfig]:
    def forest_regressor() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        )

    return [
        TargetConfig(
            name="mhiss_total",
            task_type="regression",
            subset_filter=lambda frame: frame["mhiss_total"].notna(),
            estimator=forest_regressor(),
        ),
        TargetConfig(
            name="attha_type",
            task_type="classification",
            subset_filter=lambda frame: (
                frame["is_open_fracture_case"].eq(1) & frame["attha_type"].isin([1, 2, 3])
            ),
            estimator=RandomForestClassifier(
                n_estimators=500, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
        TargetConfig(
            name="mues_score",
            task_type="regression",
            subset_filter=lambda frame: (
                frame["is_mangled_ue_case"].eq(1) & frame["mues_score"].notna()
            ),
            estimator=forest_regressor(),
        ),
        TargetConfig(
            name="strick_max",
            task_type="regression",
            subset_filter=lambda frame: (
                frame["has_digit_injury_case"].eq(1) & frame["strick_max"].notna()
            ),
            estimator=forest_regressor(),
        ),
    ]


# --------------------------------------------------------------------- preprocessing
def build_feature_columns(frame: pd.DataFrame, target_column: str | None = None) -> list[str]:
    """Model inputs: everything except identifiers, scores and derived subset flags."""
    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
        and column not in SCORE_COLUMNS
        and column not in DERIVED_FLAG_COLUMNS
        and not column.startswith("domain_")
        and column != target_column
    ]


def split_numeric_categorical(
    columns: list[str], feature_data_type_map: dict[str, str]
) -> tuple[list[str], list[str]]:
    numeric = [c for c in columns if feature_data_type_map.get(c) in NUMERIC_DATA_TYPES]
    categorical = [c for c in columns if c not in numeric]
    return numeric, categorical


def quick_preprocess_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    feature_data_type_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median/mode imputation and one-hot encoding fitted on the training split only."""
    train = train_df[feature_columns].copy()
    test = test_df[feature_columns].copy()
    numeric, categorical = split_numeric_categorical(feature_columns, feature_data_type_map)

    for column in numeric:
        train[column] = pd.to_numeric(train[column], errors="coerce")
        test[column] = pd.to_numeric(test[column], errors="coerce")
        median = train[column].median()
        fill = 0 if pd.isna(median) else median
        train[column] = train[column].fillna(fill)
        test[column] = test[column].fillna(fill)

    for column in categorical:
        mode = train[column].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "missing"
        train[column] = train[column].astype(object).where(train[column].notna(), fill).astype(str)
        test[column] = test[column].astype(object).where(test[column].notna(), fill).astype(str)

    train_encoded = pd.get_dummies(train, columns=categorical, dummy_na=False)
    test_encoded = pd.get_dummies(test, columns=categorical, dummy_na=False)
    return train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)


def build_preprocessor(
    feature_frame: pd.DataFrame, feature_data_type_map: dict[str, str]
) -> ColumnTransformer:
    numeric, categorical = split_numeric_categorical(
        feature_frame.columns.tolist(), feature_data_type_map
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def coerce_feature_types(
    frame: pd.DataFrame, feature_data_type_map: dict[str, str]
) -> pd.DataFrame:
    """Numeric columns -> floats; categorical columns -> strings (NaN preserved)."""
    coerced = frame.copy()
    numeric, categorical = split_numeric_categorical(
        coerced.columns.tolist(), feature_data_type_map
    )
    if numeric:
        coerced[numeric] = coerced[numeric].apply(pd.to_numeric, errors="coerce")
    for column in categorical:
        values = coerced[column].astype(object)
        coerced[column] = values.where(values.isna(), values.astype(str))
    return coerced


def to_dense_frame(matrix: object, feature_names: list[str]) -> pd.DataFrame:
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    names = [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names]
    return pd.DataFrame(dense.astype(float), columns=names)


def get_cv_splitter(task_type: str, y_train: pd.Series) -> KFold | StratifiedKFold | None:
    if task_type == "classification":
        n_splits = min(5, int(y_train.value_counts().min()))
        if n_splits < 2:
            return None
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    n_splits = min(5, len(y_train))
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def compute_multiclass_auc(
    y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray
) -> float:
    if len(classes) < 2:
        return float("nan")
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        encoded = label_binarize(y_true, classes=classes)
        return float(roc_auc_score(encoded, probabilities, multi_class="ovr", average="weighted"))
    except ValueError:
        return float("nan")


def get_shap_matrix(
    shap_values: object, predicted_probabilities: np.ndarray | None = None
) -> np.ndarray:
    """Reduce SHAP output to (rows, features).

    For multi-class models each row takes the SHAP values of its predicted class.
    """
    if isinstance(shap_values, list):
        values = np.stack([np.asarray(v) for v in shap_values], axis=-1)
    else:
        values = np.asarray(shap_values)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        if predicted_probabilities is None:
            return values[:, :, 0]
        predicted = predicted_probabilities.argmax(axis=1)
        return values[np.arange(values.shape[0]), :, predicted]
    raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")


# ----------------------------------------------------------------------------- models
def run_quick_mhiss_model(
    pilot_df: pd.DataFrame, feature_data_type_map: dict[str, str], task_a_dir: Path
) -> dict[str, float]:
    """Sanity gate: a quick RF on MHISS must beat a mean predictor before the full run."""
    model_df = pilot_df.dropna(subset=["mhiss_total"]).copy()
    feature_columns = build_feature_columns(model_df, target_column="mhiss_total")
    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=build_regression_stratify_labels(model_df["mhiss_total"]),
    )
    x_train, x_test = quick_preprocess_train_test(
        train_df, test_df, feature_columns, feature_data_type_map
    )
    y_train, y_test = train_df["mhiss_total"], test_df["mhiss_total"]

    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
    forest = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(x_train, y_train)
    baseline_pred = baseline.predict(x_test)
    forest_pred = forest.predict(x_test)

    metrics = {
        "baseline_r2": float(r2_score(y_test, baseline_pred)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
        "baseline_rmse": math.sqrt(mean_squared_error(y_test, baseline_pred)),
        "rf_r2": float(r2_score(y_test, forest_pred)),
        "rf_mae": float(mean_absolute_error(y_test, forest_pred)),
        "rf_rmse": math.sqrt(mean_squared_error(y_test, forest_pred)),
    }
    pd.DataFrame(
        [
            {
                "model": "DummyRegressor",
                "r2": metrics["baseline_r2"],
                "mae": metrics["baseline_mae"],
                "rmse": metrics["baseline_rmse"],
            },
            {
                "model": "RandomForestRegressor",
                "r2": metrics["rf_r2"],
                "mae": metrics["rf_mae"],
                "rmse": metrics["rf_rmse"],
            },
        ]
    ).to_csv(task_a_dir / "quick_mhiss_metrics.csv", index=False)

    if quick_gate_passed(metrics):
        x_test_float = x_test.astype(float)
        shap_values = shap.TreeExplainer(forest).shap_values(x_test_float)
        shap.summary_plot(shap_values, x_test_float, max_display=20, show=False)
        plt.title("Quick MHISS SHAP Summary")
        plt.tight_layout()
        plt.savefig(
            task_a_dir / "quick_mhiss_shap_summary.png", dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.close()
    return metrics


def quick_gate_passed(metrics: dict[str, float]) -> bool:
    return metrics["rf_r2"] > metrics["baseline_r2"] and metrics["rf_r2"] > 0


def _fit_and_explain_target(
    config: TargetConfig,
    target_frame: pd.DataFrame,
    feature_data_type_map: dict[str, str],
    task_a_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    y = target_frame[config.name]
    if config.task_type == "classification":
        y = y.astype(int)
        stratify = safe_classification_stratify(y)
    else:
        y = pd.to_numeric(y, errors="coerce")
        stratify = build_regression_stratify_labels(y)
    feature_columns = build_feature_columns(target_frame, target_column=config.name)
    x = coerce_feature_types(target_frame[feature_columns], feature_data_type_map)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train, feature_data_type_map)),
            ("model", clone(config.estimator)),
        ]
    )

    if config.task_type == "classification":
        scoring = {"weighted_f1": "f1_weighted", "accuracy": "accuracy"}
    else:
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
    cv_splitter = get_cv_splitter(config.task_type, y_train)
    cv_results = (
        cross_validate(pipeline, x_train, y_train, cv=cv_splitter, scoring=scoring)
        if cv_splitter is not None
        else {}
    )

    def cv_mean(key: str, sign: float = 1.0) -> float:
        return float(sign * np.mean(cv_results[key])) if cv_results else float("nan")

    pipeline.fit(x_train, y_train)
    preprocessor = pipeline.named_steps["preprocessor"]
    test_frame = to_dense_frame(
        preprocessor.transform(x_test), preprocessor.get_feature_names_out().tolist()
    )
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    predictions = pipeline.predict(x_test)
    performance: dict[str, object] = {
        "target": config.name,
        "task_type": config.task_type,
        "subset_rows": len(target_frame),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }

    if config.task_type == "classification":
        probabilities = pipeline.predict_proba(x_test)
        performance |= {
            "cv_weighted_f1_mean": cv_mean("test_weighted_f1"),
            "cv_accuracy_mean": cv_mean("test_accuracy"),
            "test_weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_auc_ovr_weighted": compute_multiclass_auc(
                y_test, probabilities, pipeline.named_steps["model"].classes_
            ),
        }
        shap_matrix = get_shap_matrix(explainer.shap_values(test_frame), probabilities)
    else:
        performance |= {
            "cv_r2_mean": cv_mean("test_r2"),
            "cv_mae_mean": cv_mean("test_mae", -1.0),
            "cv_rmse_mean": cv_mean("test_rmse", -1.0),
            "test_r2": float(r2_score(y_test, predictions)),
            "test_mae": float(mean_absolute_error(y_test, predictions)),
            "test_rmse": math.sqrt(mean_squared_error(y_test, predictions)),
        }
        shap_matrix = get_shap_matrix(explainer.shap_values(test_frame))

    shap.summary_plot(shap_matrix, test_frame, max_display=20, show=False)
    plt.title(f"{config.name} - SHAP Summary")
    plt.tight_layout()
    plt.savefig(task_a_dir / f"{config.name}_shap_summary.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    ranking = pd.DataFrame(
        {
            "target": config.name,
            "feature_name": test_frame.columns,
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, kind="stable")
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    for rank, feature_name in enumerate(
        ranking.head(TOP_K_DEPENDENCE_PLOTS)["feature_name"], start=1
    ):
        shap.dependence_plot(feature_name, shap_matrix, test_frame, show=False)
        plt.title(f"{config.name} - SHAP Dependence - {feature_name}")
        plt.tight_layout()
        plt.savefig(
            task_a_dir / f"{config.name}_shap_dependence_{rank}_{feature_name}.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close()

    return performance, ranking


def build_overlap_table(ranking_long: pd.DataFrame, top_k: int = TOP_K_FOR_OVERLAP) -> pd.DataFrame:
    """Count how many targets each feature reaches the top ``top_k`` for."""
    top = ranking_long.loc[ranking_long["rank"] <= top_k]
    overlap = (
        top.groupby("feature_name")
        .agg(
            target_count=("target", "nunique"),
            targets=("target", lambda values: ",".join(sorted(set(values)))),
            best_rank=("rank", "min"),
            mean_abs_shap_mean=("mean_abs_shap", "mean"),
        )
        .reset_index()
        .sort_values(
            ["target_count", "best_rank", "mean_abs_shap_mean"], ascending=[False, True, False]
        )
    )
    overlap["default_shortlist"] = overlap["target_count"].ge(2).astype(int)
    return overlap


def run_task_a_models(
    pilot_df: pd.DataFrame,
    feature_data_type_map: dict[str, str],
    task_a_dir: Path,
    target_configs: list[TargetConfig] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flagged = derive_case_flags(pilot_df)
    performance_rows: list[dict[str, object]] = []
    rankings: list[pd.DataFrame] = []

    for config in target_configs or default_target_configs():
        if config.name not in flagged.columns:
            logger.warning("Skipping %s: column not present in pilot dataset.", config.name)
            continue
        subset = flagged.loc[config.subset_filter(flagged)].copy()
        if len(subset) < MIN_SUBSET_ROWS:
            logger.warning(
                "Skipping %s: only %d applicable rows (need %d).",
                config.name,
                len(subset),
                MIN_SUBSET_ROWS,
            )
            continue
        logger.info("Task A: fitting %s on %d rows", config.name, len(subset))
        performance, ranking = _fit_and_explain_target(
            config, subset, feature_data_type_map, task_a_dir
        )
        performance_rows.append(performance)
        rankings.append(ranking)

    if not rankings:
        raise RuntimeError("Task A produced no feature rankings (no target had enough rows).")

    performance_df = pd.DataFrame(performance_rows)
    ranking_long = pd.concat(rankings, ignore_index=True)
    ranking_wide = (
        ranking_long.pivot_table(
            index="feature_name", columns="target", values="mean_abs_shap", fill_value=0
        )
        .reset_index()
        .sort_values("feature_name")
    )
    overlap = build_overlap_table(ranking_long)
    shortlist = overlap.loc[overlap["default_shortlist"] == 1]

    performance_df.to_csv(task_a_dir / "model_performance.csv", index=False)
    ranking_long.to_csv(task_a_dir / "feature_ranking_long.csv", index=False)
    ranking_wide.to_csv(task_a_dir / "feature_ranking_wide.csv", index=False)
    overlap.to_csv(task_a_dir / "overlap_analysis.csv", index=False)
    shortlist.to_csv(task_a_dir / "default_feature_shortlist.csv", index=False)
    shortlist.to_csv(task_a_dir / "provisional_final_feature_list.csv", index=False)
    return performance_df, overlap


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run(paths: Paths) -> dict[str, object]:
    """Build the cohort, write EDA artefacts and run the SHAP models."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    paths.ensure_output_dirs()

    machine_df = load_machine_block(paths.workbook, paths.patient_id_salt)
    dictionary_df = load_dictionary(paths.data_dictionary)
    input_columns, target_columns, dictionary_present = split_dictionary_columns(
        dictionary_df, set(machine_df.columns)
    )
    cohort = build_pilot_and_holdout_datasets(machine_df, input_columns, target_columns)
    pilot_df = cohort.pilot
    type_map = build_feature_data_type_map(dictionary_present)

    pilot_df.to_csv(paths.pilot_dataset, index=False)
    cohort.holdout.to_csv(paths.holdout_dataset, index=False)
    cohort.manual_verification.to_csv(paths.manual_verification, index=False)
    eda.write_row_counts(cohort.counts, paths.eda_dir / "row_counts.csv")

    eda.build_missing_value_outputs(pilot_df, paths.eda_dir)
    eda.build_target_distribution_plot(pilot_df, paths.eda_dir)
    eda.build_correlation_heatmap(pilot_df, build_feature_columns(pilot_df), paths.eda_dir)
    anomaly_report = eda.build_anomaly_report(
        pilot_df, dictionary_present, paths.eda_dir / "anomaly_report.csv"
    )

    quick_metrics = run_quick_mhiss_model(pilot_df, type_map, paths.task_a_dir)
    if not quick_gate_passed(quick_metrics):
        _write_json(
            paths.task_a_summary,
            {
                "status": "stopped_after_quick_model",
                "row_counts": cohort.counts,
                "quick_metrics": quick_metrics,
                "message": (
                    "Quick MHISS model did not beat the mean baseline with a positive R2. "
                    "Check side-selection logic or feature filtering before expanding."
                ),
            },
        )
        raise QuickModelGateError(
            "Quick MHISS sanity check failed: Random Forest R² "
            f"{quick_metrics['rf_r2']:.3f} vs mean-baseline R² {quick_metrics['baseline_r2']:.3f} "
            "(needs to be positive and above the baseline). Full Task A was not run. "
            "Check the injured-side columns and the data dictionary's input/target layers; "
            f"details in {paths.task_a_summary}."
        )

    performance_df, overlap = run_task_a_models(pilot_df, type_map, paths.task_a_dir)
    summary: dict[str, object] = {
        "status": "completed",
        "row_counts": cohort.counts,
        "quick_metrics": quick_metrics,
        "anomaly_rows": len(anomaly_report),
        "task_a_targets_completed": performance_df["target"].tolist(),
        "default_shortlist_features": overlap.loc[
            overlap["default_shortlist"] == 1, "feature_name"
        ].tolist(),
    }
    _write_json(paths.task_a_summary, summary)
    return summary
