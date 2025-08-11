"""Plot rainfall-temperature interaction surfaces for logistic models.

This module is used to visualise how predicted dengue cluster
probabilities vary with two weather variables (by default rainfall and
temperature) while holding all other covariates at zero. It evaluates the model over
a grid of values for the chosen variables and produces a heatmap showing
the predicted probabilities.

The function does not attempt to include spatial covariates or seasonal
variables; instead, it fixes all other features at zero. So, it is usedful for exploring interaction patterns rather than generating absolute risk predictions.
"""

from __future__ import annotations

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import joblib  # type: ignore[import]
from pathlib import Path
from typing import Optional

from dengueweather.config import load_config
from dengueweather.logging_setup import setup_logging


def plot_interaction(
    model_path: str,
    out_dir: str,
    features_path: Optional[str] = None,
    var_x: str = "rain_mm",
    var_y: str = "temp_c",
) -> None:
    """Generate a heatmap of predicted probabilities over a two-variable grid.

    Parameters
    ----------
    model_path : str
        Path to the joblib file containing the trained logistic model and feature list.
    out_dir : str
        Directory in which to save the output figure
    features_path : str, optional
        Path to the Parquet file containing the feature matrix.  If not provided,
        the function looks for `unit_week_features.parquet` in the processed
        directory defined in the project configuration.
    var_x : str, optional
        Base name of the first variable (e.g. `'rain_mm'`).  The function
        searches for a lag 0 column named `<var_x>_lag_0` to set the range.
    var_y : str, optional
        Base name of the second variable (e.g.'temp_c').  Similar to `var_x`.

    Notes
    -----
    The ranges for the variables are derived from the 5th to 95th percentiles
    of the corresponding lag 0 columns in the feature table. This will avoids
    extreme outliers influencing the grid.
    Only lag 0 variables are varied. All other features are set to zero
    for prediction. If the model uses higher lags or derived variables,
    interaction patterns may differ.
    The resulting heatmap is saved as `interaction_surface.png` in the specified output directory.
    """
    setup_logging()
    cfg = load_config()
    if features_path is None:
        processed_dir = cfg.data.get("processed_dir", "data/processed")
        features_path = str(Path(processed_dir) / "unit_week_features.parquet")
    # Load features table
    try:
        df = pd.read_parquet(features_path)
    except Exception as exc:
        print(f"Could not load features from {features_path}: {exc}")
        return
    # Identify lag_0 columns for the chosen variables
    def find_col(df: pd.DataFrame, base: str) -> Optional[str]:
        candidates = [c for c in df.columns if c.startswith(base) and c.endswith("_lag_0")]
        return candidates[0] if candidates else None
    x_col = find_col(df, var_x) or next((c for c in df.columns if var_x in c), None)
    y_col = find_col(df, var_y) or next((c for c in df.columns if var_y in c), None)
    if x_col is None or y_col is None:
        print(f"Unable to locate columns for {var_x} and/or {var_y} in features table.")
        return
    # Determine ranges using 5th and 95th percentiles
    x_vals = df[x_col].dropna()
    y_vals = df[y_col].dropna()
    if x_vals.empty or y_vals.empty:
        print(f"No data found in columns {x_col} or {y_col}; cannot plot interaction.")
        return
    x_min, x_max = x_vals.quantile(0.05), x_vals.quantile(0.95)
    y_min, y_max = y_vals.quantile(0.05), y_vals.quantile(0.95)

    x_grid = np.linspace(x_min, x_max, 30)
    y_grid = np.linspace(y_min, y_max, 30)
    Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="xy")

    model_dict = joblib.load(model_path)
    model = model_dict.get("model")
    feature_cols = model_dict.get("feature_cols", [])
    if model is None or not feature_cols:
        print(f"Model or feature list missing in {model_path}")
        return
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    idx = {f: i for i, f in enumerate(feature_cols)}
    base = np.zeros(len(feature_cols), dtype=float)
    xi = idx.get(x_col)
    yi = idx.get(y_col)
    if xi is None or yi is None:
        print(f"Columns {x_col} and/or {y_col} not found in model feature list.")
        return
    # Compute predictions over grid
    Z = np.zeros_like(Xg, dtype=float)
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            vec = base.copy()
            vec[xi] = Xg[i, j]
            vec[yi] = Yg[i, j]
            logit = intercept + np.dot(vec, coefs)
            Z[i, j] = 1.0 / (1.0 + np.exp(-logit))
    # Plot heatmap
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(Xg, Yg, Z, levels=20, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Predicted probability")
    ax.set_xlabel(f"{var_x} (lag 0)")
    ax.set_ylabel(f"{var_y} (lag 0)")
    ax.set_title(f"Interaction surface: {var_x} vs {var_y}")
    ax.grid(False)
    fig.savefig(out_dir_path / "interaction_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)