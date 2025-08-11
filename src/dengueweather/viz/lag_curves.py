"""Plot lag-response curves for lagged weather features.

This module provides a function to visualise the coefficients of a fitted
logistic regression model across temporal lags for each weather variable. 
Feature list is stored via joblib.

When called, the function loads the trained model, parses the feature
names to identify base variables and their lag indices, extracts the
corresponding coefficients, and produces a line chart for each base
variable showing how the coefficient (effect size) changes with lag.  The
plots are saved into the specified output directory with filenames of
the form `<variable>_lag_response.png`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import joblib  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from dengueweather.logging_setup import setup_logging


def plot_lag_curves(model_path: str, out_dir: str) -> None:
    """Generate and save lag-response curve figures."""
    setup_logging()
    # Load the trained model and related feature names
    model_dict = joblib.load(model_path)
    model = model_dict.get("model")
    feature_cols = model_dict.get("feature_cols", [])
    if model is None or not feature_cols:
        print(f"Could not load model or feature list from {model_path}")
        return
    # Extract coefficients for the positive class
    try:
        coefs = model.coef_[0]
    except Exception:
        print("Model does not show coefficients in the expected format.")
        return
    if len(coefs) != len(feature_cols):
        print(
            f"Number of coefficients ({len(coefs)}) does not match number of"
            f" features ({len(feature_cols)}); aborting lag curve plot."
        )
        return
    # Parse feature names to identify base variables and lag indices
    pattern = re.compile(r"(.+)_lag_(\d+)$")
    records = []
    for feat, coef in zip(feature_cols, coefs):
        m = pattern.match(feat)
        if m:
            base = m.group(1)
            lag = int(m.group(2))
            records.append((base, lag, coef))
    if not records:
        print("No lagged features found; skipping lag curve plotting.")
        return
    df = pd.DataFrame(records, columns=["variable", "lag", "coef"])
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # Plot each base variable separately and also build a combined figure
    variables = sorted(df["variable"].unique())
    if variables:
        fig_all, ax_all = plt.subplots()
        for variable in variables:
            sub = df[df["variable"] == variable].sort_values("lag")
            ax_all.plot(
                sub["lag"],
                sub["coef"],
                marker="o",
                label=variable,
            )
        ax_all.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax_all.set_xlabel("Lag (weeks)")
        ax_all.set_ylabel("Coefficient (log‑odds)")
        ax_all.set_title("Lag-response curves (all variables)")
        ax_all.grid(True, linewidth=0.3, linestyle=":")
        ax_all.legend(loc="best")
        fig_all_filename = out_path / "lag_curves_combined.png"
        fig_all.savefig(fig_all_filename, dpi=300, bbox_inches="tight")
        plt.close(fig_all)
    # Individual plots for each base variable
    for variable in variables:
        sub = df[df["variable"] == variable].sort_values("lag")
        fig, ax = plt.subplots()
        ax.plot(sub["lag"], sub["coef"], marker="o")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lag (weeks)")
        ax.set_ylabel("Coefficient (log‑odds)")
        ax.set_title(f"Lag-response curve for {variable}")
        ax.grid(True, linewidth=0.3, linestyle=":")

        fig_filename = out_path / f"{variable}_lag_response.png"
        fig.savefig(fig_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)