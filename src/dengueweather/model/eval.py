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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

LABEL_COL = "y_cluster_present"
EXCLUDE_COLS = {
    "h3",
    "cluster_key",
    LABEL_COL,
    "peak_cases",
    "left_censored",
    "right_censored",
    "spatial_block", # Exclude the block identifier from features
}

def compute_basic_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute common evaluation metrics"""
    out: dict = {}
    out["brier_score"] = float(brier_score_loss(y_true, y_pred_proba))
    
    unique_classes = set(int(v) for v in np.unique(y_true))
    if len(unique_classes) < 2:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
        return out
    
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        out["roc_auc"] = float("nan")
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
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Temporal cross-validation splitting by contiguous blocks of years.
    """
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

        metrics = _train_eval_fold(train_df, test_df, feature_cols, model_params)
        
        rows.append(
            {
                "fold": i,
                "type": "temporal",
                "train_years": ",".join(map(str, train_years)),
                "test_years": ",".join(map(str, test_years)),
                "n_train": len(train_df),
                "n_test": len(test_df),
                **metrics,
            }
        )

    return pd.DataFrame(rows)

def spatial_block_cv(
    df_or_path: Union[str, Path, pd.DataFrame],
    block_col: str = "subzone_id", # Or 'cluster_id', 'region', etc.
    n_splits: int = 5,
    *,
    model_params: Optional[dict] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Spatial cross-validation splitting by geographic blocks (e.g. subzones).
    
    This ensures that when we test on a specific region, the model has not 
    seen data from that specific region during training, reducing spatial leakage.
    """
    df = _load_df(df_or_path)
    
    # If the block column isn't present, we might need to derive it or fail
    if block_col not in df.columns:
         # Fallback: if 'h3' exists, we can perhaps use the parent resolution as a crude block
         if "h3" in df.columns:
             import h3
             # Use a coarser resolution (e.g., res 6) as the spatial block
             df["spatial_block"] = df["h3"].apply(lambda x: h3.h3_to_parent(x, 6) if h3.h3_is_valid(x) else "unknown")
             block_col = "spatial_block"
         else:
            raise ValueError(f"Column '{block_col}' not found for spatial blocking.")

    feature_cols = _select_numeric_features(df)
    
    # Filter out blocks with too few samples if necessary, or just proceed
    # We use GroupKFold to ensure blocks are not split across train/test
    gkf = GroupKFold(n_splits=n_splits)
    
    # We need arrays for the indices
    # We can just use a dummy X and y for the splitter, provided 'groups' is correct
    groups = df[block_col].astype(str)
    
    rows = []
    fold_idx = 1
    
    # GroupKFold.split yields train/test INDICES
    for train_idx, test_idx in gkf.split(df, groups=groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Identify which blocks are in test
        test_blocks = sorted(test_df[block_col].unique())
        test_blocks_str = ",".join(map(str, test_blocks[:5])) 
        if len(test_blocks) > 5:
            test_blocks_str += "..."

        metrics = _train_eval_fold(train_df, test_df, feature_cols, model_params)

        rows.append({
            "fold": fold_idx,
            "type": "spatial",
            "test_blocks": test_blocks_str,
            "n_train": len(train_df),
            "n_test": len(test_df),
            **metrics
        })
        fold_idx += 1

    return pd.DataFrame(rows)


def _train_eval_fold(train_df, test_df, feature_cols, model_params):
    """Helper to train and evaluate a single fold."""
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

    uniq_train = np.unique(y_tr)
    if len(uniq_train) < 2:
        const_pred = float(uniq_train[0]) if len(uniq_train) == 1 else 0.0
        p_te = np.full(len(y_te), const_pred, dtype=float)
        metrics = compute_basic_metrics(y_te, p_te)
    else:
        clf = LogisticRegression(**params)
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        metrics = compute_basic_metrics(y_te, p_te)
    
    return metrics