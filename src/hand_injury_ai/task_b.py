"""Task B: an AI composite injury score learned from the Task A shortlist.

1. Each scoring system is normalised to 0-100 and combined into ``AI_target`` with
   base weights (MHISS 0.30, Atthakomol 0.25, Strickland 0.25, MUES 0.20). Scores
   that do not apply to a patient are dropped and the remaining weights renormalised.
2. Linear, Ridge, Random Forest and XGBoost regressors are compared on the shortlisted
   features; the best model by test R2 is refit on the full pilot cohort.
3. The composite is compared with the IPS score (scatter + Bland-Altman).
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .common import (
    DERIVED_FLAG_COLUMNS,
    SCORE_COLUMNS,
    build_regression_stratify_labels,
    derive_case_flags,
)
from .config import RANDOM_STATE, Paths

logger = logging.getLogger(__name__)

FIGURE_DPI = 200
MIN_FEATURES = 5
TARGET_FEATURE_COUNT = 10
MAX_FEATURES = 15

# score -> (column, maximum used for 0-100 normalisation, base weight)
SCORE_COMPONENTS: dict[str, tuple[str, float, float]] = {
    "mhiss": ("mhiss_total", 500.0, 0.30),
    "attha": ("attha_score", 9.0, 0.25),
    "strick": ("strick_max", 18.0, 0.25),
    "mues": ("mues_score", 8.0, 0.20),
}

NON_FEATURE_COLUMNS = (
    frozenset(
        {
            "patient_id",
            "selected_side",
            "is_bilateral",
            "needs_review",
            "selected_side_positive_signal_count",
            "opposite_side_positive_signal_count",
            "mhiss_norm",
            "attha_norm",
            "strick_norm",
            "mues_norm",
            "AI_target",
        }
    )
    | SCORE_COLUMNS
    | DERIVED_FLAG_COLUMNS
)


class TaskBInputError(RuntimeError):
    """Task B inputs (pilot dataset / Task A outputs) are missing or unusable."""


# ----------------------------------------------------------------------- AI target
def construct_ai_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<score>_norm`` columns and the renormalised weighted ``AI_target``.

    Expects the flags from :func:`hand_injury_ai.common.derive_case_flags`.
    """
    out = df.copy()
    applicable: dict[str, pd.Series] = {}
    for name, (column, maximum, _) in SCORE_COMPONENTS.items():
        raw = pd.to_numeric(out[column], errors="coerce")
        if name == "mhiss":
            raw = np.minimum(raw, maximum)
        out[f"{name}_norm"] = raw / maximum * 100.0
        applicable[name] = raw.notna()

    applicable["attha"] &= out["is_open_fracture_case"].eq(1)
    applicable["strick"] &= out["has_digit_injury_case"].eq(1)
    applicable["mues"] &= out["is_mangled_ue_case"].eq(1)

    weighted_sum = pd.Series(0.0, index=out.index)
    total_weight = pd.Series(0.0, index=out.index)
    for name, (_, _, weight) in SCORE_COMPONENTS.items():
        mask = applicable[name]
        weighted_sum = weighted_sum + out[f"{name}_norm"].where(mask, 0.0) * weight
        total_weight = total_weight + mask.astype(float) * weight

    out["AI_target"] = (weighted_sum / total_weight).where(total_weight > 0)
    return out


# ------------------------------------------------------------------ feature choice
def map_encoded_to_raw_feature(encoded_name: str, raw_columns: list[str]) -> str | None:
    """Map a one-hot name such as ``mechanism_3`` back to its raw column ``mechanism``."""
    if encoded_name in raw_columns:
        return encoded_name
    candidates = [column for column in raw_columns if encoded_name.startswith(f"{column}_")]
    return max(candidates, key=len) if candidates else None


def read_feature_source(paths: Paths) -> pd.DataFrame:
    for candidate in (
        paths.feature_list_override,
        paths.provisional_feature_list,
        paths.overlap_analysis,
    ):
        if candidate is not None and candidate.is_file():
            logger.info("Task B feature source: %s", candidate)
            source = pd.read_csv(candidate)
            if "feature_name" not in source.columns:
                raise TaskBInputError(f"{candidate} has no 'feature_name' column.")
            return source
    raise TaskBInputError(
        f"No Task A feature list found (looked for {paths.provisional_feature_list} and "
        f"{paths.overlap_analysis}). Run Task A first."
    )


def select_task_b_features(
    df: pd.DataFrame, feature_source: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    """Map shortlisted encoded features to raw columns; top up with Spearman correlates."""
    candidates = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and not column.startswith("domain_")
    ]
    selected: list[str] = []
    mapping_rows: list[dict[str, object]] = []
    for encoded in feature_source["feature_name"].astype(str):
        raw = map_encoded_to_raw_feature(encoded, candidates)
        mapping_rows.append({"encoded_feature": encoded, "mapped_raw_feature": raw})
        if raw and raw not in selected:
            selected.append(raw)

    if len(selected) < TARGET_FEATURE_COUNT:
        correlations: list[tuple[str, float]] = []
        for column in candidates:
            if column in selected or not pd.api.types.is_numeric_dtype(df[column]):
                continue
            series = pd.to_numeric(df[column], errors="coerce")
            valid = df["AI_target"].notna() & series.notna()
            if valid.sum() < 8 or series[valid].nunique() < 2:
                continue
            rho = spearmanr(series[valid], df.loc[valid, "AI_target"])[0]
            if pd.notna(rho):
                correlations.append((column, abs(float(rho))))
        correlations.sort(key=lambda item: item[1], reverse=True)
        for column, _ in correlations:
            if len(selected) >= TARGET_FEATURE_COUNT:
                break
            selected.append(column)

    return selected[:MAX_FEATURES], pd.DataFrame(mapping_rows)


# --------------------------------------------------------------------------- models
def build_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [column for column in x.columns if column not in numeric]
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def _kfold(n_rows: int) -> KFold:
    return KFold(n_splits=max(2, min(5, n_rows)), shuffle=True, random_state=RANDOM_STATE)


def fit_ridge_with_cv(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train, scale_numeric=True)),
            ("model", Ridge(random_state=RANDOM_STATE)),
        ]
    )
    search = GridSearchCV(
        estimator=pipe,
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="neg_mean_absolute_error",
        cv=_kfold(len(y_train)),
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def candidate_models(x_train: pd.DataFrame, y_train: pd.Series) -> list[tuple[str, Pipeline]]:
    return [
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


def evaluate_pipeline_model(
    model_name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float | str], np.ndarray, Pipeline]:
    cv_scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=_kfold(len(y_train)),
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics: dict[str, float | str] = {
        "model": model_name,
        "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
        "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
        "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
        "test_r2": float(r2_score(y_test, predictions)),
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": math.sqrt(mean_squared_error(y_test, predictions)),
    }
    return metrics, predictions, model


def export_feature_weights(model: Pipeline) -> pd.DataFrame:
    names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in model.named_steps["preprocessor"].get_feature_names_out()
    ]
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        weights = np.asarray(estimator.coef_).reshape(-1)
        frame = pd.DataFrame({"feature_name": names, "weight": weights})
        frame["abs_weight"] = frame["weight"].abs()
        return frame.sort_values("abs_weight", ascending=False)
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_).reshape(-1)
        return pd.DataFrame({"feature_name": names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
    return pd.DataFrame({"feature_name": names})


# ---------------------------------------------------------------------- agreement
def bland_altman_stats(ai: pd.Series, reference: pd.Series) -> dict[str, float]:
    """Mean difference and 95% limits of agreement (AI - reference)."""
    diff = ai - reference
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    return {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "loa_lower": mean_diff - 1.96 * sd_diff,
        "loa_upper": mean_diff + 1.96 * sd_diff,
    }


def plot_agreement(predictions: pd.DataFrame, scatter_path: Path, bland_altman_path: Path) -> float:
    valid = predictions["ips_total"].notna() & predictions["AI_composite"].notna()
    frame = predictions.loc[valid]
    rho = (
        float(spearmanr(frame["AI_composite"], frame["ips_total"])[0])
        if len(frame) >= 3
        else float("nan")
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(frame["AI_composite"], frame["ips_total"], alpha=0.75)
    plt.xlabel("AI Composite")
    plt.ylabel("IPS Total")
    title = "AI Composite vs IPS Total"
    plt.title(f"{title} (Spearman={rho:.3f})" if pd.notna(rho) else title)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    stats = bland_altman_stats(frame["AI_composite"], frame["ips_total"])
    plt.figure(figsize=(8, 6))
    plt.scatter(
        (frame["AI_composite"] + frame["ips_total"]) / 2.0,
        frame["AI_composite"] - frame["ips_total"],
        alpha=0.75,
    )
    for level in (stats["mean_diff"], stats["loa_upper"], stats["loa_lower"]):
        if pd.notna(level):
            plt.axhline(level, linestyle="--")
    plt.xlabel("Mean of AI Composite and IPS")
    plt.ylabel("AI Composite - IPS")
    plt.title("Bland-Altman: AI Composite vs IPS")
    plt.tight_layout()
    plt.savefig(bland_altman_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return rho


def build_task_b_summary_md(
    comparison_df: pd.DataFrame,
    selected_features: list[str],
    best_model_name: str,
    spearman_test: dict[str, float],
) -> str:
    lines = [
        "# Task B Summary (Pilot Dataset)",
        "",
        "## Scope",
        "- Built AI composite target from MHISS/Atthakomol/Strickland/MUES with per-patient "
        "weight renormalization.",
        "- Trained and compared multiple regressors on the current pilot dataset.",
        "- This is a pilot Task B run; final thesis-grade validation still requires full "
        "retrospective/prospective datasets.",
        "",
        "## Selected Features",
        *[f"- `{feature}`" for feature in selected_features],
        "",
        "## Model Comparison",
        comparison_df.to_markdown(index=False),
        "",
        "## Best Model",
        f"- `{best_model_name}`",
        "",
        "## Spearman (Test Set) : AI Prediction vs Score Systems",
        *[
            f"- `{name}`: {value:.4f}" if pd.notna(value) else f"- `{name}`: NaN"
            for name, value in spearman_test.items()
        ],
        "",
        "## Output Files",
        *[
            f"- `outputs/task_b/{name}`"
            for name in (
                "B1_model_comparison.csv",
                "B2_best_model.pkl",
                "B3_feature_weights.csv",
                "B4_predictions_pilot.csv",
                "B5_scatter_ai_vs_ips.png",
                "B6_bland_altman_ai_vs_ips.png",
                "B7_task_b_summary.md",
            )
        ],
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------ run
def run(paths: Paths) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    out_dir = paths.task_b_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not paths.pilot_dataset.is_file():
        raise TaskBInputError(f"Pilot dataset missing: {paths.pilot_dataset}. Run Task A first.")

    df = pd.read_csv(paths.pilot_dataset)
    df = construct_ai_target(derive_case_flags(df))
    df = df[df["AI_target"].notna()].reset_index(drop=True)

    selected, mapping_df = select_task_b_features(df, read_feature_source(paths))
    mapping_df.to_csv(out_dir / "task_b_selected_features.csv", index=False)
    if len(selected) < MIN_FEATURES:
        raise TaskBInputError(
            f"Only {len(selected)} usable features for Task B (need {MIN_FEATURES})."
        )
    if len(df) < 10:
        raise TaskBInputError(f"Only {len(df)} rows with an AI target; need at least 10.")

    x = df[selected].copy()
    y = df["AI_target"].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=build_regression_stratify_labels(y),
    )

    comparison_rows: list[dict[str, float | str]] = []
    fitted: dict[str, Pipeline] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, model in candidate_models(x_train, y_train):
        logger.info("Task B: evaluating %s", name)
        metrics, predictions, fitted_model = evaluate_pipeline_model(
            name, model, x_train, y_train, x_test, y_test
        )
        comparison_rows.append(metrics)
        fitted[name] = fitted_model
        test_predictions[name] = predictions

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["test_r2", "test_mae"], ascending=[False, True]
    )
    comparison_df.to_csv(out_dir / "B1_model_comparison.csv", index=False)

    best_name = str(comparison_df.iloc[0]["model"])
    best_template = clone(fitted[best_name])
    oof_predictions = cross_val_predict(best_template, x, y, cv=_kfold(len(y)))
    best_model = fitted[best_name].fit(x, y)
    joblib.dump(best_model, out_dir / "B2_best_model.pkl")
    export_feature_weights(best_model).to_csv(out_dir / "B3_feature_weights.csv", index=False)

    prediction_df = df[
        [
            "patient_id",
            "AI_target",
            "mhiss_total",
            "attha_score",
            "mues_score",
            "strick_max",
            "ips_total",
        ]
    ].copy()
    # In-sample fit of the final model (as in the original pipeline) plus honest
    # out-of-fold predictions for agreement analysis.
    prediction_df["AI_composite"] = best_model.predict(x)
    prediction_df["AI_composite_oof"] = oof_predictions
    prediction_df["dataset"] = "pilot"
    prediction_df.to_csv(paths.predictions, index=False)

    plot_agreement(
        prediction_df,
        out_dir / "B5_scatter_ai_vs_ips.png",
        out_dir / "B6_bland_altman_ai_vs_ips.png",
    )

    test_eval = df.loc[x_test.index].copy()
    test_eval["AI_pred"] = test_predictions[best_name]
    spearman_test: dict[str, float] = {}
    for column in ("mhiss_total", "attha_score", "mues_score", "strick_max"):
        mask = test_eval[column].notna() & test_eval["AI_pred"].notna()
        spearman_test[column] = (
            float(spearmanr(test_eval.loc[mask, "AI_pred"], test_eval.loc[mask, column])[0])
            if mask.sum() >= 3
            else float("nan")
        )

    (out_dir / "B7_task_b_summary.md").write_text(
        build_task_b_summary_md(comparison_df, selected, best_name, spearman_test),
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "rows_used": len(df),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "best_model": best_name,
        "best_model_test_r2": float(comparison_df.iloc[0]["test_r2"]),
        "best_model_test_mae": float(comparison_df.iloc[0]["test_mae"]),
        "best_model_oof_r2": float(r2_score(y, oof_predictions)),
        "spearman_test_vs_scores": spearman_test,
        "files": {
            key: str(out_dir / name)
            for key, name in {
                "B1": "B1_model_comparison.csv",
                "B2": "B2_best_model.pkl",
                "B3": "B3_feature_weights.csv",
                "B4": "B4_predictions_pilot.csv",
                "B5": "B5_scatter_ai_vs_ips.png",
                "B6": "B6_bland_altman_ai_vs_ips.png",
                "B7": "B7_task_b_summary.md",
            }.items()
        },
    }
    paths.task_b_summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
