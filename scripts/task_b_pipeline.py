#!/usr/bin/env python3
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
