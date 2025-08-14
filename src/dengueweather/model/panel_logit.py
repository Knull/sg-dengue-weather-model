"""Panel logistic regression utilities."""

from pathlib import Path
from typing import Dict, Optional
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from ..logging_setup import setup_logging

LABEL_COL = "y_cluster_present"
EXCLUDE_COLS = {
    LABEL_COL,
    "h3",
    "cluster_key",
    "peak_cases",
    "left_censored",
    "right_censored",
    "iso_year",
    "iso_week",
    "week_of_year",
    "woy_sin",
    "woy_cos",
}

def _select_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=["number"]).columns
    cols = [c for c in numeric if c not in EXCLUDE_COLS]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols

def fit_model(
    features_path: str,
    out: str,
    *,
    class_weight: Optional[str | Dict[int, float]] = "balanced",
    max_iter: int = 2000,
    **model_kwargs: object,
) -> None:
    """Fit a scaled logistic regression on numeric features and save it."""
    setup_logging()
    df = pd.read_parquet(features_path)
    if LABEL_COL not in df:
        raise ValueError(f"Missing label column '{LABEL_COL}' in {features_path}")

    feature_cols = _select_numeric_feature_cols(df)
    if not feature_cols:
        raise ValueError("No eligible numeric feature columns found for model training.")

    X = df[feature_cols].fillna(0.0).astype("float64").values
    y = df[LABEL_COL].astype(int).values

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    cw = "balanced" if class_weight == "balanced" else (class_weight if isinstance(class_weight, dict) else None)
    model = LogisticRegression(max_iter=max_iter, class_weight=cw, **model_kwargs)
    model.fit(Xs, y)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols, "scaler": scaler}, out)
    print(f"model: trained on {len(X):,} rows, {len(feature_cols)} features → {out}")

def eval_model(model_path: str, features_path: str, out: str) -> None:
    """Compute evaluation metrics using saved scaler + features."""
    setup_logging()
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]
    scaler: StandardScaler = payload["scaler"]

    df = pd.read_parquet(features_path)
    X = df[feature_cols].fillna(0.0).astype("float64").values
    y = df[LABEL_COL].astype(int).values

    Xs = scaler.transform(X)
    y_pred_proba = model.predict_proba(Xs)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    ap = average_precision_score(y, y_pred_proba)
    metrics = {"roc_auc": float(auc), "average_precision": float(ap)} # eval: {'roc_auc': 0.9621794871794871, 'average_precision': 0.9972520005321366} → data/processed/metrics.json (outputs)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"eval: {metrics} → {out}")
