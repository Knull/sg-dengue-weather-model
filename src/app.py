import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib
import h3
import numpy as np
from pathlib import Path

# --- CONFIG ---
PAGE_TITLE = "🦟 Dengue Response"
feat_path = "data/processed/unit_week_features.parquet"
model_path = "data/processed/model_gbm.joblib"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- LOADERS ---
@st.cache_data
def load_data():
    if not Path(feat_path).exists():
        return None
    df = pd.read_parquet(feat_path)
    # Ensure sorted by time
    df = df.sort_values(["iso_year", "iso_week"])
    return df

@st.cache_resource
def load_model():
    if not Path(model_path).exists():
        return None
    payload = joblib.load(model_path)
    return payload

def predict_risk(df_week, payload):
    """Run the model on a specific slice of data."""
    model = payload["model"]
    features = payload["feature_cols"]
    
    # Prepare X (fill NaNs with 0 like in training)
    X = df_week[features].fillna(0).values
    
    # Predict Probability (Risk)
    probs = model.predict_proba(X)[:, 1]
    return probs

# --- MAIN APP ---
st.title(f"{PAGE_TITLE}")

df = load_data()
model_payload = load_model()

if df is None or model_payload is None:
    st.error("Data or Model not found! Run `make features` and `make model` first.")
    st.stop()

# SIDEBAR: Time Selection
st.sidebar.header("Temporal Controls")
years = sorted(df["iso_year"].unique())
selected_year = st.sidebar.select_slider("Year", options=years, value=years[-1])

# Filter weeks for that year
weeks = sorted(df[df["iso_year"] == selected_year]["iso_week"].unique())
selected_week = st.sidebar.select_slider("Week", options=weeks, value=weeks[-1])

# Filter Data
subset = df[(df["iso_year"] == selected_year) & (df["iso_week"] == selected_week)].copy()

# PREDICT
if not subset.empty:
    with st.spinner("Calculating Risk Profiles..."):
        risk_scores = predict_risk(subset, model_payload)
        subset["risk_score"] = risk_scores
        
        # Color Scaling logic (Green -> Red)
        # R, G, B, A
        def get_color(risk):
            # Simple gradient: Low risk = Green, High risk = Red
            # 0.0 -> [0, 255, 0]
            # 1.0 -> [255, 0, 0]
            r = int(risk * 255)
            g = int((1 - risk) * 255)
            return [r, g, 0, 160] # 160 is transparency

        subset["color"] = subset["risk_score"].apply(get_color)
        
        # Elevation logic (Higher risk = Taller hexagon)
        subset["elevation"] = subset["risk_score"] * 2000 

    # METRICS ROW
    col1, col2, col3 = st.columns(3)
    n_high_risk = len(subset[subset["risk_score"] > 0.5])
    top_risk = subset["risk_score"].max()
    
    col1.metric("Selected Time", f"{selected_year}-W{selected_week:02d}")
    col2.metric("High Risk Zones (>50%)", n_high_risk)
    col3.metric("Max Risk Detected", f"{top_risk:.1%}")

    # MAP VISUALIZATION (PyDeck)
    st.subheader("Tactical Map")
    
    # 1. Add Map Style Picker (Updated with No-Key styles)
    map_style_options = {
        "Dark (Best for Contrast)": pdk.map_styles.CARTO_DARK,
        "Light (Clean)": pdk.map_styles.CARTO_LIGHT,
        "Roads (Detailed)": pdk.map_styles.CARTO_ROAD,
    }
    
    selected_style_name = st.select_slider("Map Style", options=list(map_style_options.keys()))
    # Get the actual style string based on the name selected
    selected_style_url = map_style_options[selected_style_name]
    
    # H3 Layer (Keep this the same)
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
        render_sub_layers=True,
        opacity=0.8
    )

    # View State (Keep this the same)
    view_state = pdk.ViewState(
        latitude=1.3521,
        longitude=103.8198,
        zoom=11,
        pitch=50,
        bearing=0
    )

    # Tooltip (Keep this the same)
    tooltip = {
        "html": "<b>H3:</b> {h3} <br/> <b>Risk:</b> {risk_score} <br/> <b>Rain:</b> {rain_mm_lag_1}mm",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=selected_style_url  # <--- Pass the Carto URL here
    )

    st.pydeck_chart(r)
    
    # DATA TABLE
    st.subheader("Priority List (Top 20)")
    top_zones = subset.sort_values("risk_score", ascending=False).head(20)
    st.dataframe(
        top_zones[["h3", "risk_score", "y_cluster_present", "rain_mm_lag_1", "neighbor_pressure_lag_1"]],
        use_container_width=True
    )

else:
    st.warning("No data found for this week.")