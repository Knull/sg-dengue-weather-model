"""
Gradient Boosting Model (LightGBM) for high-precision dengue forecasting.
Includes isotonic calibration to ensure probabilities are realistic.
"""
import sys
from pathlib import Path
from typing import Dict, Optional, List
import json

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from ..logging_setup import setup_logging

LABEL_COL = "y_cluster_present"
EXCLUDE_COLS = {
    LABEL_COL,
    "cluster_key",
    "peak_cases",
    "left_censored",
    "right_censored",
    "spatial_block",
    "rain_mm",
    "temp_c",
    "rh_pct",
    "wind_kmh",
    "abs_humidity_gm3",
    "iso_year",
    "self_lag_1",
}
# Note: We keep 'h3' (as category) and 'iso_year'/'iso_week' in exclusion usually, 
# but LightGBM can handle 'week_of_year' explicitly if passed.
# For now, we exclude identifiers.

def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    # Select numeric features + the new spatial neighbor features
    cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "h3"]
    # Filter out non-numeric just in case, unless we want categorical support
    numeric_cols = df[cols].select_dtypes(include=["number"]).columns.tolist()
    return numeric_cols

def fit_gbm(
    features_path: str,
    out: str,
    *,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    calibration_cv: int = 3,
) -> None:
    """Train a Calibrated LightGBM model."""
    setup_logging()
    print(f"Loading data from {features_path}...")
    df = pd.read_parquet(features_path)
    
    feature_cols = _select_feature_cols(df)
    print(f"Training on {len(feature_cols)} features: {feature_cols}")
    
    # Handle NaNs (LightGBM handles them natively, but good to be explicit or fill)
    # Spatial lags might have NaNs for the very first week -> fill 0
    df = df.fillna(0)
    
    X = df[feature_cols].values
    y = df[LABEL_COL].astype(int).values

    # Base Model: LightGBM
    # We use 'is_unbalance=True' or 'scale_pos_weight' for dengue (rare events)
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    
    base_model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        objective="binary",
        class_weight="balanced", # or scale_pos_weight=ratio
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )

    # Wrap in Calibration
    # This is crucial for "Top-K" accuracy
    print(f"fitting CalibratedClassifierCV (cv={calibration_cv})...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=calibration_cv
    )
    
    calibrated_model.fit(X, y)
    
    # Save
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": calibrated_model,
        "feature_cols": feature_cols,
        "model_type": "lightgbm_calibrated"
    }
    joblib.dump(payload, out)
    print(f"Saved GBM model to {out}")

def eval_gbm(model_path: str, features_path: str, out: str) -> None:
    """Evaluate GBM model."""
    setup_logging()
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    df = pd.read_parquet(features_path).fillna(0)
    X = df[feature_cols].values
    y = df[LABEL_COL].astype(int).values

    # Predict
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y, y_pred_proba)
    ap = average_precision_score(y, y_pred_proba)
    
    # Top-K Precision (Tactical Metric)
    # Just a quick global check, though weekly is better
    df["pred"] = y_pred_proba
    top_k_precisions = []
    for _, grp in df.groupby(["iso_year", "iso_week"]):
        if grp[LABEL_COL].sum() > 0:
            top_20 = grp.sort_values("pred", ascending=False).head(20)
            top_k_precisions.append(top_20[LABEL_COL].mean())
    
    mean_p20 = np.mean(top_k_precisions) if top_k_precisions else 0.0

    metrics = {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "mean_precision_at_20": float(mean_p20)
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"GBM Evaluation: {metrics} -> {out}")

def run_cv(
    features_path: str,
    out: str,
    n_splits: int = 3,  # kept for CLI compatibility; ignored in walk-forward
) -> None:
    """Run walk-forward temporal cross-validation.

    For each test year (from the 2nd valid year onward), train ONLY on years
    strictly before it. No future leakage.
    """
    setup_logging()
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path).fillna(0)

    # 1. Filter for years that actually have positive labels.
    valid_years = []
    all_years = sorted(df["iso_year"].unique())

    for year in all_years:
        n_pos = df[df["iso_year"] == year][LABEL_COL].sum()
        if n_pos > 0:
            valid_years.append(year)
        else:
            print(f"[Warn] Excluding year {year} from CV (0 positive labels found).")

    if len(valid_years) < 2:
        print(f"Error: Need at least 2 valid labeled years for walk-forward CV (got {len(valid_years)}).")
        return

    df = df[df["iso_year"].isin(valid_years)].copy()
    print(f"Running walk-forward CV on valid years: {valid_years}")

    # 2. Walk-forward folds: skip earliest year (nothing to train on).
    feature_cols = _select_feature_cols(df)
    print(f"Using {len(feature_cols)} features.")
    results = []

    test_years_list = valid_years[1:]
    print(f"\nStarting walk-forward CV across {len(test_years_list)} folds...")

    for i, test_year in enumerate(test_years_list):
        train_years = [y for y in valid_years if y < test_year]
        print(f"Fold {i+1}/{len(test_years_list)}: train={train_years}, test={test_year}")

        train_mask = df["iso_year"].isin(train_years)
        test_mask = df["iso_year"] == test_year

        X_tr = df.loc[train_mask, feature_cols].values
        y_tr = df.loc[train_mask, LABEL_COL].astype(int).values
        X_te = df.loc[test_mask, feature_cols].values
        y_te = df.loc[test_mask, LABEL_COL].astype(int).values

        if len(np.unique(y_tr)) < 2:
            print("  Skipping fold (only 1 class in training).")
            continue
        if len(np.unique(y_te)) < 2:
            print(f"  Skipping fold (only 1 class in test for year {test_year}).")
            continue

        base = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calibrated.fit(X_tr, y_tr)
        probs = calibrated.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, probs)
        ap = average_precision_score(y_te, probs)

        fold_df = df.loc[test_mask, ["iso_year", "iso_week", LABEL_COL]].copy()
        fold_df["pred"] = probs
        precisions = []
        for _, wk_df in fold_df.groupby(["iso_year", "iso_week"]):
            if wk_df[LABEL_COL].sum() > 0:
                top_20 = wk_df.sort_values("pred", ascending=False).head(20)
                precisions.append(top_20[LABEL_COL].mean())
        mean_p20 = float(np.mean(precisions)) if precisions else 0.0

        print(f"  --> AUC: {auc:.4f}, AP: {ap:.4f}, P@20: {mean_p20:.4f}")
        results.append({
            "fold": i + 1,
            "train_years": str(train_years),
            "test_year": test_year,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_pos_test": int(y_te.sum()),
            "roc_auc": float(auc),
            "average_precision": float(ap),
            "precision_at_20": mean_p20,
        })

    # Save summary
    res_df = pd.DataFrame(results)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out, index=False)

    if "tabulate" in sys.modules:
        print("\nCV Results Summary:")
        print(res_df.to_markdown(index=False))
    else:
        print("\nCV Results Summary:")
        print(res_df)