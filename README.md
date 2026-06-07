# Singapore Dengue Risk Ranking Model

A research prototype for ranking geographic areas in Singapore by dengue cluster-presence risk using weather, spatial, and temporal features.

The project studies whether recent cluster history, neighbouring cluster pressure, and lagged weather variables can improve weekly risk ranking over weather-only baselines. It should be read as an experimental modelling pipeline rather than an operational public-health decision tool.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Model](https://img.shields.io/badge/Model-LightGBM-green)
![Spatial](https://img.shields.io/badge/Spatial-H3_Hexagons-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## Goal

The goal is to generate weekly ranked lists of H3 zones for retrospective evaluation and prospective monitoring. The model ranks areas by estimated active-cluster risk using spatial cluster-pressure features, weather lags, and seasonal information.

The current target is active cluster presence. Active dengue clusters can persist across weeks, so some of the signal comes from recent cluster carryover rather than the first appearance of new clusters. The results should therefore be interpreted as risk-ranking evidence for this specific target.

### Key Features

* **Spatial features:** Uses H3 geospatial indexing to measure cluster pressure from nearby zones.
* **Nonlinear modelling:** Uses LightGBM with isotonic calibration for weather, temporal, and spatial features.
* **Walk-forward validation:** Evaluates each test year using only earlier years for training.
* **Live feature patching:** Supports a prospective monitoring workflow using the latest observed NEA cluster map as lagged spatial context.
* **Dashboard:** Provides an interactive Streamlit map for inspecting ranked zones and model inputs.

---

## Quick Start

### 1. Installation

Clone the repository and install dependencies.

```bash
pip install -e .[dev]
```

### 2. Historical evaluation

Run walk-forward cross-validation on the processed historical feature table.

```bash
python -m src.cli cv-gbm
```

### 3. Prospective monitoring workflow

Fetch the latest live cluster map and use it to patch the feature table for a prospective ranking week.

```bash
# 1. Fetch the latest live cluster map from NEA
python -m src.cli ingest-nea-live

# 2. Patch the feature table using the latest observed live clusters
# Replace the filename below with the file just downloaded in step 1
python -m src.cli patch-live-week --live-geojson data/raw/nea_live/2026-01-03.geojson

# 3. Generate the top-ranked zones
# Use the year and week printed by the patch command
python -m src.cli rank-riskiest --model-path data/processed/model_gbm.joblib --iso-year 2026 --iso-week 1
```

### 4. Launch the dashboard

```bash
streamlit run src/app.py
```

---

## Model Performance

Current walk-forward CV on valid labelled years in the 2013 to 2020 archive, with each test year trained only on previous years:

| Test Year | ROC AUC | AP | P@20 |
| --- | ---: | ---: | ---: |
| 2014 | 0.6600 | 0.2159 | 0.3714 |
| 2015 | 0.7217 | 0.2793 | 0.4875 |
| 2016 | 0.7424 | 0.4525 | 0.5902 |
| 2017 | 0.6754 | 0.1318 | 0.1580 |
| 2018 | 0.7507 | 0.1899 | 0.2317 |
| 2019 | 0.8099 | 0.5177 | 0.6942 |
| 2020 | 0.7179 | 0.3280 | 0.7111 |
| **Mean** | **0.7254** | **0.3027** | **0.4634** |

These results are for active cluster presence. Because active clusters persist over time, Precision@20 should be read as a ranking result for a persistence-influenced target rather than as a clean new-outbreak forecasting metric.

---

## Limitations

- **Target definition:** The target is active cluster presence. It does not attempt to model new-cluster onset directly.
- **Cluster persistence:** Part of the signal comes from recent self and neighbouring cluster activity.
- **Archive coverage:** Positive-rate by year varies substantially. Low-positive years may reflect transmission lulls or gaps in the SGCharts and NEA archive ingest.
- **Recent years:** The current archive has no positive labels after 2020, so the live monitoring workflow has not yet been validated on labelled post-2020 outcomes.
- **Operational use:** The project is a research prototype. It should not be used for public-health intervention decisions without prospective validation, uncertainty analysis, and domain review.

---

## Engineering Pipeline

The system uses a modular ETL pipeline managed by `src.cli`:

1. **Ingest**
   * `download-weather`: Downloads historical MSS daily weather data.
   * `ingest-archive`: Merges historical SGCharts or NEA archive snapshots.
   * `ingest-nea-live`: Fetches the latest active cluster map from Data.gov.sg.

2. **Process**
   * `build-history`: Constructs stable cluster histories from archived snapshots.
   * `build-cluster-week`: Converts cluster histories into weekly H3 labels.
   * `build-features`: Builds weekly H3 features, including weather lags and spatial cluster-pressure features.

3. **Model**
   * `fit-gbm`: Trains a LightGBM classifier with isotonic calibration.
   * `cv-gbm`: Runs walk-forward validation by test year.
   * `rank-riskiest`: Produces a ranked list of H3 zones for a selected year and week.

4. **Visualise**
   * `streamlit run src/app.py`: Opens the dashboard for inspecting ranked zones and input features.

---

## Project Structure

```text
.
├── data/
│   ├── raw/                # MSS weather files and NEA GeoJSONs
│   ├── interim/            # Parquet checkpoints
│   └── processed/          # Feature tables and trained models
├── src/
│   ├── app.py              # Streamlit dashboard
│   ├── cli.py              # Command-line interface
│   └── dengueweather/
│       ├── build/          # Feature engineering logic
│       ├── ingest/         # Data ingestion utilities
│       ├── model/          # Model training and evaluation code
│       └── viz/            # Mapping and plotting utilities
└── pyproject.toml          # Dependencies
```

---

## Future Work

* Add prospective validation once labelled outcomes are available for recent years.
* Add SHAP or permutation-based explanations for top-ranked zones.
* Separate active-cluster persistence from new-cluster onset in a future target definition.
* Add uncertainty summaries for top-k rankings.
