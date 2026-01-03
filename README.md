# 🦟 Singapore Dengue Tactical Response Model

A high-precision, spatiotemporal forecasting system designed to identify active dengue clusters in Singapore before they expand.

Unlike traditional weather-only models, this system calculates **Spatial Infection Pressure** (the "spark") alongside environmental suitability (the "fuel") to generate tactical intervention lists for NEA/Town Councils.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Model](https://img.shields.io/badge/Model-LightGBM-green)
![Spatial](https://img.shields.io/badge/Spatial-H3_Hexagons-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## 🎯 The Goal: Save as many people.
To prevent outbreaks, broad "risk maps" are insufficient. Resources are finite. This model answers one specific question:
> **"Which 20 specific neighborhoods (H3 Hexagons) require boots-on-the-ground intervention *today*?"**

### Key Features
* **Spatial Intelligence:** Uses H3 Hierarchical Geospatial Indexing to measure "Infection Pressure" from neighboring zones.
* **Non-Linear Modeling:** Powered by a Calibrated **LightGBM** (Gradient Boosting Machine) to capture complex weather-lag interactions.
* **Live "Patching" Engine:** Combines live NEA cluster data (via API) with persistence weather forecasts to predict risk for the *current* week.
* **3D Command Dashboard:** Interactive Streamlit visualization for identifying risk towers on a street map.

---

## 🚀 Quick Start

### 1. Installation
Clone the repo and install dependencies (including `lightgbm`, `h3`, `pydeck`).

```bash
# Windows
pip install -e .[dev]

```

### 2. The "Weekly Tactical" Workflow

To generate the **Kill List** for the current week (e.g., Monday morning routine):

```bash
# 1. Fetch the latest live cluster map from NEA
python -m src.cli ingest-nea-live

# 2. Patch the data (Merges Live Clusters + Latest Weather Forecast)
# Replace the filename below with the one just downloaded in step 1
python -m src.cli patch-live-week --live-geojson data/raw/nea_live/2026-01-03.geojson

# 3. Generate the Priority List (Top 20 Riskiest Zones)
# Use the Forecast Week printed by the patch command (e.g., 2026-01)
python -m src.cli rank-riskiest --model-path data/processed/model_gbm.joblib --iso-year 2026 --iso-week 1

```

### 3. Launch the Dashboard

Visualize the active Red Zones on a 3D map.

```bash
streamlit run src/app.py

```

---

## 📊 Model Performance

The model was evaluated using **Time-Block Cross-Validation** (strict temporal separation) to simulate real-world deployment.

| Metric | Result (Outbreak Years) | Meaning |
| --- | --- | --- |
| **ROC AUC** | **0.9375** | Excellent ability to distinguish safe vs. dangerous zones globally. |
| **Average Precision** | **0.9950** | Extremely high reliability in risk scoring. |
| **Precision @ 20** | **1.0000** | **100% Hit Rate.** In historical validation, every single one of the Top 20 zones flagged by the model contained an active cluster. |

*Evaluation performed on 2017–2020 data.*

---

## 🛠️ Engineering Pipeline

The system uses a modular ETL pipeline managed by `src.cli`:

1. **Ingest:**
* `download-weather`: Scrapes MSS daily weather data.
* `ingest-nea-live`: Fetches active clusters from Data.gov.sg.


2. **Process (`build-features`):**
* Calculates **Spatiotemporal Lags** (e.g., `neighbor_pressure_lag_1`).
* Aggregates weather (Rain, Temp, Humidity) to weekly H3 resolutions.


3. **Model (`fit-gbm`):**
* Trains a `LGBMClassifier` with `CalibratedClassifierCV` (Isotonic) to ensure risk scores are realistic probabilities (0–100%).



---

## 📂 Project Structure

```text
.
├── data/
│   ├── raw/                # MSS Weather & NEA GeoJSONs
│   ├── interim/            # Parquet checkpoints
│   └── processed/          # Final feature tables & trained models
├── src/
│   ├── app.py              # Streamlit Command Center
│   ├── cli.py              # The "Controller" (CLI commands)
│   └── dengueweather/
│       ├── build/          # Feature Engineering logic
│       └── model/          # LightGBM training & inference code
└── pyproject.toml          # Dependencies

```

---

## 🔮 Future Upgrades

* **Automated Cron Job:** GitHub Action to run the pipeline every Monday at 0800H.
* **Explainable AI:** Integrate SHAP values into the dashboard to explain *why* a specific block is high risk (e.g., "High Rain 2 weeks ago + Neighbor Infection").
