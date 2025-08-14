"""Evaluation utilities for dengue-weather models (panel logistic)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

LABEL_COL = "y_cluster_present"
EXCLUDE_COLS = {
    "h3",
    "cluster_key",
    LABEL_COL,
    "peak_cases",
    "left_censored",
    "right_censored",
}

def compute_basic_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute common evaluation metrics"""
    out: dict = {}
    # Brier score is defined regardless of class distribution
    out["brier_score"] = float(brier_score_loss(y_true, y_pred_proba))
    # If only one class is present, ROC AUC and average precision are undefined.
    # so, explicitly check for this condition and assign NaN to the metrics.
    unique_classes = set(int(v) for v in np.unique(y_true))
    if len(unique_classes) < 2:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
        return out
    # Compute AUC; fallback to NaN on error
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        out["roc_auc"] = float("nan")
    # Compute average precision; fallback to NaN on error
    try:
        out["average_precision"] = float(average_precision_score(y_true, y_pred_proba))
    except Exception:
        out["average_precision"] = float("nan")
    return out

def _select_numeric_features(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=["number"]).columns
    cols = [c for c in numeric if c not in EXCLUDE_COLS]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols

def _load_df(df_or_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(df_or_path, (str, Path)):
        return pd.read_parquet(df_or_path)
    return df_or_path

def _contiguous_year_blocks(years: List[int], n_splits: int) -> List[List[int]]:
    years = sorted(years)
    k, n = n_splits, len(years)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    blocks, idx = [], 0
    for s in sizes:
        blocks.append(years[idx: idx + s])
        idx += s
    return blocks

def year_block_cv(
    df_or_path: Union[str, Path, pd.DataFrame],
    n_splits: int = 3,
    *,
    model_params: Optional[dict] = None,
    random_state: int = 42,  # kept for API compatibility
) -> pd.DataFrame:
    df = _load_df(df_or_path)
    if "iso_year" not in df.columns:
        raise ValueError("DataFrame must contain 'iso_year' for year-block CV.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    feature_cols = _select_numeric_features(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for CV after exclusions.")

    uniq_years = sorted(df["iso_year"].dropna().astype(int).unique().tolist())
    if len(uniq_years) < n_splits:
        n_splits = len(uniq_years)
    blocks = _contiguous_year_blocks(uniq_years, n_splits)

    rows = []
    for i, test_years in enumerate(blocks, start=1):
        train_years = [y for y in uniq_years if y not in test_years]
        train_df = df[df["iso_year"].isin(train_years)].copy()
        test_df = df[df["iso_year"].isin(test_years)].copy()

        X_tr = train_df[feature_cols].fillna(0.0).astype("float64").values
        y_tr = train_df[LABEL_COL].astype(int).values
        X_te = test_df[feature_cols].fillna(0.0).astype("float64").values
        y_te = test_df[LABEL_COL].astype(int).values

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        params = {"max_iter": 2000, "class_weight": "balanced"}
        if model_params:
            params.update({k: v for k, v in model_params.items() if v is not None})

        # If the training set contains only a single class, a logistic regression model cannot be fitted. Instead, fall back to a constant predictor equal to the sole class value (0 or 1) and compute metrics directly. This avoids raising an exception from scikit-learn.
        uniq_train = np.unique(y_tr)
        if len(uniq_train) < 2:
            # constant predicted probability: 0.0 if all labels are 0, 1.0 if all are 1
            const_pred = float(uniq_train[0]) if len(uniq_train) == 1 else 0.0
            p_te = np.full(len(y_te), const_pred, dtype=float)
            metrics = compute_basic_metrics(y_te, p_te)
        else:
            clf = LogisticRegression(**params)
            clf.fit(X_tr, y_tr)
            p_te = clf.predict_proba(X_te)[:, 1]
            metrics = compute_basic_metrics(y_te, p_te)

        rows.append(
            {
                "fold": i,
                "train_years": ",".join(map(str, train_years)),
                "test_years": ",".join(map(str, test_years)),
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)
