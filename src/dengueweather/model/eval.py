"""
Evaluation utilities for dengue-weather models.

This module contains helper functions to compute common classification
metrics, perform year-block cross-validation and generate calibration
curves.  You can import these helpers in notebooks or scripts to
evaluate models beyond a simple train/test split.  The functions are
written to be independent of the specific model implementation- as
long as a model exposes a `predict_proba` method, it can be used
with these evaluators.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score,
    roc_auc_score,
)

from .panel_logit import fit_model  # reuse logistic trainer if needed
from ..logging_setup import setup_logging


def compute_basic_metrics(
    y_true: Sequence[int], y_pred_proba: Sequence[float]
) -> Dict[str, float]:
    """Compute fundamental classification metrics.

    Parameters
    ----------
    y_true : sequence of int
        Ground truth binary labels (0/1).
    y_pred_proba : sequence of float
        Model-predicted probabilities of the positive class.

    Returns
    -------
    dict
        A dictionary containing ROC AUC, average precision and Brier score.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred_proba)
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_pred_arr)),
        "average_precision": float(average_precision_score(y_true_arr, y_pred_arr)),
        "brier_score": float(brier_score_loss(y_true_arr, y_pred_arr)),
    }


def calibration_curve(
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Compute calibration (reliability) curve data.

    Parameters
    ----------
    y_true : sequence of int
        True binary outcomes.
    y_pred_proba : sequence of float
        Predicted probabilities.
    n_bins : int, optional
        Number of bins to compute.  Defaults to 10.
    strategy : {"uniform", "quantile"}, optional
        Binning strategy.  ``"quantile"`` bins contain equal
        numbers of samples; ``"uniform"`` bins span equal
        probability ranges.

    Returns
    -------
    DataFrame
        Columns ``bin_center``, ``mean_predicted``, ``fraction_positive``, ``count``.
    """
    # Sort by predicted probabilities
    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_proba})
    if strategy == "quantile":
        data["bin"] = pd.qcut(data["y_pred"], q=n_bins, duplicates="drop")
    else:
        data["bin"] = pd.cut(data["y_pred"], bins=n_bins, include_lowest=True)
    grouped = data.groupby("bin")
    summary = grouped.agg(
        count=("y_true", "count"),
        mean_predicted=("y_pred", "mean"),
        fraction_positive=("y_true", "mean"),
    ).reset_index(drop=True)
    summary["bin_center"] = summary["mean_predicted"]
    return summary


def year_block_cv(
    df: pd.DataFrame,
    model_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 3,
    target_col: str = "y_cluster_present",
    random_state: int = 42,
) -> pd.DataFrame:
    """Perform block cross-validation by ISO year.

    The dataset is partitioned by unique ISO years into ``n_splits` nearly
    equal folds.  Each fold acts as the test set exactly once.  A
    :class:`~sklearn.linear_model.LogisticRegression` is trained on the
    remaining folds and evaluated on the holdout. The resulting metrics
    for each fold are returned as a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing features, labels and an `iso_year` column.
    model_params : dict, optional
        Parameters passed to the logistic regression constructor.  If
        `None`, sensible defaults are used (`max_iter=1000`,
        `class_weight='balanced'`).
    n_splits : int, optional
        Number of folds (must be >= 2).  Defaults to 3.
    target_col : str, optional
        Name of the target column.  Defaults to ``"y_cluster_present"``.
    random_state : int, optional
        Random seed used only for reproducibility when splitting years
        into folds.

    Returns
    -------
    DataFrame
        One row per fold with columns ``fold``, ``years`` (list of
        holdout years) and metric columns from :func:`compute_basic_metrics`.
    """
    setup_logging()
    if "iso_year" not in df.columns:
        raise ValueError("DataFrame must contain 'iso_year' for year-block CV.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    model_params = model_params or {}
    # Determine feature columns (exclude identifiers and target)
    exclude = {target_col, "h3", "cluster_key"}
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and not pd.api.types.is_object_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric features found for cross-validation.")
    # Unique ISO years sorted for reproducibility
    years = sorted(df["iso_year"].dropna().astype(int).unique())
    if len(years) < n_splits:
        raise ValueError(
            f"Not enough unique iso_year values ({len(years)}) for {n_splits} splits."
        )
    # Partition years into n_splits contiguous groups
    # Use numpy array_split to balance sizes
    year_folds = np.array_split(years, n_splits)
    results = []
    for fold_idx, hold_years in enumerate(year_folds):
        hold_years = list(hold_years)
        train_df = df[~df["iso_year"].isin(hold_years)]
        test_df = df[df["iso_year"].isin(hold_years)]
        X_train = train_df[feature_cols].fillna(0.0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0.0)
        y_test = test_df[target_col]
        # Fit logistic regression on training years
        # Use balanced class weights by default unless specified
        default_params = {"max_iter": 1000, "class_weight": "balanced"}
        params = {**default_params, **model_params}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_basic_metrics(y_test, y_pred_prob)
        metrics.update({"fold": fold_idx, "years": hold_years})
        results.append(metrics)
    return pd.DataFrame(results)