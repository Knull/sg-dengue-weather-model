import h3
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from pathlib import Path


PAGE_TITLE = "Dengue Risk Ranking"
FEATURE_PATH = Path("data/processed/unit_week_features.parquet")
MODEL_PATH = Path("data/processed/model_gbm.joblib")

st.set_page_config(page_title=PAGE_TITLE, layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Load the processed feature table."""
    if not FEATURE_PATH.exists():
        return None

    df = pd.read_parquet(FEATURE_PATH)
    if {"iso_year", "iso_week"}.issubset(df.columns):
        df = df.sort_values(["iso_year", "iso_week"])

    return df


@st.cache_resource
def load_model() -> dict | None:
    """Load the trained model payload."""
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def predict_risk(df_week: pd.DataFrame, payload: dict) -> np.ndarray:
    """Run the model on a selected week."""
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    missing = [col for col in feature_cols if col not in df_week.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")

    X = df_week[feature_cols].fillna(0).astype("float64")
    return model.predict_proba(X)[:, 1]


def colour_from_risk(risk: float) -> list[int]:
    risk = float(np.clip(risk, 0.0, 1.0))
    red = int(risk * 255)
    green = int((1.0 - risk) * 255)
    return [red, green, 0, 160]


def display_table_columns(df: pd.DataFrame) -> list[str]:
    cols = ["h3", "risk_score"]

    if "y_cluster_present" in df.columns and df["y_cluster_present"].notna().any():
        cols.append("y_cluster_present")

    for optional_col in [
        "rain_mm_lag_1",
        "temp_c_lag_1",
        "rh_pct_lag_1",
        "neighbor_pressure_lag_1",
    ]:
        if optional_col in df.columns:
            cols.append(optional_col)

    return cols


st.title(PAGE_TITLE)
st.caption(
    "Research dashboard for inspecting weekly H3 risk rankings. "
    "Scores are model outputs for active cluster-presence risk and should not be treated as operational decisions."
)

df = load_data()
model_payload = load_model()

if df is None:
    st.error("Feature table not found. Run the feature-building pipeline first.")
    st.stop()

if model_payload is None:
    st.error("Model file not found. Train the GBM model first.")
    st.stop()

required_time_cols = {"iso_year", "iso_week"}
if not required_time_cols.issubset(df.columns):
    st.error("Feature table must contain iso_year and iso_week columns.")
    st.stop()

st.sidebar.header("Time Selection")

years = sorted(df["iso_year"].dropna().unique())
selected_year = st.sidebar.select_slider("Year", options=years, value=years[-1])

weeks = sorted(df.loc[df["iso_year"] == selected_year, "iso_week"].dropna().unique())
selected_week = st.sidebar.select_slider("Week", options=weeks, value=weeks[-1])

subset = df[(df["iso_year"] == selected_year) & (df["iso_week"] == selected_week)].copy()

if subset.empty:
    st.warning("No data found for this week.")
    st.stop()

try:
    with st.spinner("Scoring selected week..."):
        subset["risk_score"] = predict_risk(subset, model_payload)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

subset["risk_percent"] = subset["risk_score"] * 100.0
subset["color"] = subset["risk_score"].apply(colour_from_risk)
subset["elevation"] = subset["risk_score"] * 2000.0

col1, col2, col3 = st.columns(3)

col1.metric("Selected Week", f"{int(selected_year)}-W{int(selected_week):02d}")
col2.metric("Zones Scored", f"{len(subset):,}")
col3.metric("Max Score", f"{subset['risk_score'].max():.1%}")

if "y_cluster_present" in subset.columns and subset["y_cluster_present"].notna().any():
    known_labels = subset["y_cluster_present"].notna().sum()
    active_clusters = int(subset["y_cluster_present"].fillna(0).sum())
    st.caption(
        f"Historical labels available for {known_labels:,} zones in this week. "
        f"Active labelled zones: {active_clusters:,}."
    )
else:
    st.caption("No target labels are available for this selected week. Rankings are model outputs only.")

st.subheader("Risk Ranking Map")

map_style_options = {
    "Dark": pdk.map_styles.CARTO_DARK,
    "Light": pdk.map_styles.CARTO_LIGHT,
    "Roads": pdk.map_styles.CARTO_ROAD,
}

selected_style_name = st.selectbox("Map style", options=list(map_style_options.keys()))
selected_style_url = map_style_options[selected_style_name]

tooltip_parts = [
    "<b>H3:</b> {h3}",
    "<b>Risk score:</b> {risk_percent}%",
]
if "rain_mm_lag_1" in subset.columns:
    tooltip_parts.append("<b>Rain lag 1:</b> {rain_mm_lag_1}")
if "neighbor_pressure_lag_1" in subset.columns:
    tooltip_parts.append("<b>Neighbour pressure lag 1:</b> {neighbor_pressure_lag_1}")

tooltip_html = "<br/>".join(tooltip_parts)

layer = pdk.Layer(
    "H3HexagonLayer",
    subset,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=True,
    get_hexagon="h3",
    get_fill_color="color",
    get_elevation="elevation",
    elevation_scale=1,
    elevation_range=[0, 1000],
    opacity=0.8,
)

view_state = pdk.ViewState(
    latitude=1.3521,
    longitude=103.8198,
    zoom=11,
    pitch=50,
    bearing=0,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": tooltip_html,
        "style": {"backgroundColor": "white", "color": "black"},
    },
    map_style=selected_style_url,
)

st.pydeck_chart(deck)

st.subheader("Top Ranked Zones")

top_k = st.slider("Number of zones to show", min_value=5, max_value=100, value=20, step=5)
top_zones = subset.sort_values("risk_score", ascending=False).head(top_k).copy()

table_cols = display_table_columns(top_zones)
st.dataframe(top_zones[table_cols], use_container_width=True)

export_df = top_zones[table_cols]
st.download_button(
    "Download ranked zones as CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"dengue_risk_ranking_{int(selected_year)}_W{int(selected_week):02d}.csv",
    mime="text/csv",
)
