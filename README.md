# Singapore Dengue Tactical Response Model

A high-precision, spatiotemporal forecasting system designed to identify active dengue clusters in Singapore before they expand.

Unlike traditional weather-only models, this system calculates **Spatial Infection Pressure** (the "spark") alongside environmental suitability (the "fuel") to generate tactical intervention lists for NEA/Town Councils.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Model](https://img.shields.io/badge/Model-LightGBM-green)
![Spatial](https://img.shields.io/badge/Spatial-H3_Hexagons-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## The Goal: Save as many people.
To prevent outbreaks, broad "risk maps" are insufficient. Resources are finite. This model answers one specific question:
> **"Which 20 specific neighborhoods (H3 Hexagons) require boots-on-the-ground intervention *today*?"**

### Key Features
* **Spatial Intelligence:** Uses H3 Hierarchical Geospatial Indexing to measure "Infection Pressure" from neighboring zones.
* **Non-Linear Modeling:** Powered by a Calibrated **LightGBM** (Gradient Boosting Machine) to capture complex weather-lag interactions.
* **Live "Patching" Engine:** Combines live NEA cluster data (via API) with persistence weather forecasts to predict risk for the *current* week.
* **3D Command Dashboard:** Interactive Streamlit visualization for identifying risk towers on a street map.

---

## Quick Start

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

## Model Performance

7-fold walk-forward CV on 2013–2020 NEA archive (train strictly precedes test):

| Test Year | ROC AUC | AP | P@20 |
| --- | --- | --- | --- |
| 2014 | 0.66 | 0.22 | 0.37 |
| 2015 | 0.72 | 0.28 | 0.49 |
| 2016 | 0.74 | 0.45 | 0.59 |
| 2017 | 0.68 | 0.13 | 0.16 |
| 2018 | 0.75 | 0.19 | 0.23 |
| 2019 | 0.81 | 0.52 | 0.69 |
| 2020 | 0.72 | 0.33 | 0.71 |
| **Mean** | **0.73** | **0.31** | **0.46** |
*Evaluation performed on 2017–2020 data.*

---

## Limitations
- **Target is cluster presence, not new-cluster onset.** Active clusters
  persist for multiple weeks, so part of the signal reflects last-week
  carryover rather than true forecasting of new outbreaks.
- **Archive coverage varies.** Positive-rate by year ranges from 2% (2017)
  to 17% (2019); low-positive years may reflect transmission lulls or
  gaps in the SGCharts/NEA archive ingest, and per-fold metrics should
  be read in that light.
- **No data 2021–present.** The current archive has no positive labels
  after 2020; the live patching engine has not been validated against
  out-of-sample 2022 outbreak data.
---
## Engineering Pipeline

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

## Project Structure

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

## Future Upgrades

* **Automated Cron Job:** GitHub Action to run the pipeline every Monday at 0800H.
* **Explainable AI:** Integrate SHAP values into the dashboard to explain *why* a specific block is high risk (e.g., "High Rain 2 weeks ago + Neighbor Infection").
