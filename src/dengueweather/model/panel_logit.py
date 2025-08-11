"""
Panel logistic regression utilities.

This module defines helpers for fitting and saving a logistic regression
model on the unit-week feature set. The resulting model is continued with 
joblib and with the list of feature columns used for training.

Use the companion functions in mod`dengueweather.model.eval` to
perform cross-validation, compute metrics and generate calibration
curves.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from ..logging_setup import setup_logging


def fit_model(
    features_path: str,
    out: str,
    *,
    class_weight: Optional[str | Dict[int, float]] = "balanced",
    max_iter: int = 1000,
    **model_kwargs: object,
) -> None:
    """Fit a logistic regression on the unit-week features and save it.

    Parameters
    ----------
    features_path : str
        Path to the features Parquet produced by func:`build_features`.
    out : str
        Path to save the trained model (joblib format).
    class_weight : "balanced" or dict, optional
        How to balance classes.  Defaults to "balanced" to compensate
        for class imbalance.  Pass a dictionary `{0: w0, 1: w1}` or
        `None` for unweighted training.
    max_iter : int, optional
        Maximum number of solver iterations.  Defaults to 1000.
    **model_kwargs
        Additional keyword arguments passed to
        class:`~sklearn.linear_model.LogisticRegression` (e.g.,
        `penalty`, `C`, `solver`).

    Notes
    -----
    The function automatically selects all numeric feature columns except
    non-feature keys (`h3`, `cluster_key`) and the target column
    `y_cluster_present`. Missing values are given zeros.  The
    fitted model and feature column names are continued via joblib so
    evaluation functions can reconstruct the model later.
    """
    setup_logging()
    df = pd.read_parquet(features_path)
    # Select numeric features excluding identifiers and target
    exclude = {"y_cluster_present", "h3", "cluster_key"}
    feature_cols: list[str] = [
        c
        for c in df.columns
        if c not in exclude and not pd.api.types.is_object_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for model training.")
    X = df[feature_cols].fillna(0.0)
    y = df["y_cluster_present"]
    cw = class_weight # computes class weights
    if class_weight == "balanced":
        # let scikit learn balance internally
        cw = "balanced"
    elif isinstance(class_weight, dict):
        cw = class_weight
    else:
        # IF None or unsupported type, leave as is (unweighted)
        cw = None
    model = LogisticRegression(max_iter=max_iter, class_weight=cw, **model_kwargs)
    model.fit(X, y)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, out)


def eval_model(model_path: str, features_path: str, out: str) -> None:
    """Compute evaluation metrics for a trained logistic model.

    Parameters
    ----------
    model_path : str
        Path to the trained model file produced by func:`fit_model`.
    features_path : str
        Path to the Parquet with features and labels.
    out : str
        Path to save a JSON file with evaluation metrics.

    Notes
    -----
    This function evaluates the model on the entire feature set. For
    cross-validated performance, use the functions in
    mod:`dengueweather.model.eval`.
    """
    setup_logging()
    import json
    model_dict = joblib.load(model_path)
    model = model_dict["model"]
    feature_cols = model_dict["feature_cols"]
    df = pd.read_parquet(features_path)
    X = df[feature_cols].fillna(0.0)
    y = df["y_cluster_present"]
    # Evaluate on the full dataset
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    ap = average_precision_score(y, y_pred_proba)
    metrics = {"roc_auc": float(auc), "average_precision": float(ap)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)