# Singapore Dengue-Weather Model (`dengueweather`)

**Weather–dengue spatiotemporal analysis for Singapore**

This project provides a data pipeline and modelling framework to analyze the relationship between weather variables (temperature, rainfall, wind, humidity) and dengue cluster formation in Singapore. It includes tools for data ingestion, feature engineering, panel logistic regression modeling, and risk visualization.

## Features

* **Data Ingestion**: Automated tools to fetch and archive data from:
    * **SGCharts**: Historical dengue cluster snapshots.
    * **MSS**: Meteorological Service Singapore daily weather logs.
    * **MOH**: Weekly infectious disease bulletins (PDF/CSV).
    * **NEA**: Live dengue cluster data (GeoJSON).
* **Spatio-temporal Analysis**:
    * Builds stable cluster histories and "cluster-week" presence panels.
    * Generates distributed lag models (DLMs) to capture delayed weather effects.
    * Captures interactions (e.g., rainfall × temperature).
* **Modelling**: Panel logistic regression to predict outbreak risk.
* **Visualization**:
    * Lag-response curves.
    * Interaction surface plots.
    * Hindcast risk maps.
* **Dashboard**: A Streamlit app for interactive exploration.

## Installation

**Prerequisites**: Python ≥ 3.11

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sg-dengue-weather-model.git
    cd sg-dengue-weather-model
    ```

2.  **Install dependencies:**
    This project uses `uv` for dependency management, but falls back to `pip` if unavailable.
    ```bash
    make init
    ```
    *Or manually:* `pip install -e .[dev]`

## Usage / Data Pipeline

The project uses a `Makefile` to orchestrate the main stages of the pipeline.

### 1. Data Ingestion
Download and standardize raw data sources.
```bash
make ingest
```

* **Archives**: Merges SGCharts CSV snapshots into a single Parquet file.
* **Weather**: Aggregates daily MSS station data into weekly summaries (including Absolute Humidity calculations).

### 2. Panel Construction & Feature Engineering

Build the spatiotemporal panel and generate features for modeling.

```bash
make features
```

* **Panel**: Constructs a "cluster-week" table indicating where and when clusters existed.
* **Features**: Merges weather data and computes distributed lags (e.g., weather conditions 1–12 weeks prior).

### 3. Modelling

Train the panel logistic regression model.

```bash
make model
```

* Fits the model on the `unit_week_features.parquet` dataset.
* Saves the trained model to `data/processed/model.joblib`.

### 4. Evaluation

Evaluate model performance (precision, recall, AUC, etc.).

```bash
make eval
```

### 5. Visualization

Generate static plots and maps.

```bash
make figs
```

* **Lag Curves**: Visualizes how risk responds to weather variables over time (`docs/figures_lag/`).
* **Interactions**: Visualizes 3D surfaces for variable interactions (`docs/figures_interaction/`).
* **Maps**: Generates hindcast risk maps for specific weeks (`docs/maps/`).

### 6. Dashboard

Run the interactive Streamlit dashboard.

```bash
make app
```

## CLI Reference

The project exposes a rich Command Line Interface (CLI) via `src/cli.py`. You can access detailed help for any command:

```bash
python -m src.cli --help
```

**Key Commands:**

* `ingest-archive`: Merge SGCharts snapshots.
* `ingest-weather`: Process daily weather files.
* `ingest-moh`: Parse MOH weekly bulletin PDFs/CSVs.
* `ingest-nea-live`: Archive today's live NEA clusters.
* `fit-panel-logit`: Train the main model.
* `cv-panel-logit`: Run year-block cross-validation.
* `map-hindcast`: Generate a risk map for a specific ISO week.

## Development

* **Linting**: Check code quality.
```bash
make lint
```

* **Formatting**: Auto-format code.
```bash
make fmt
```

* **Testing**: Run unit tests.
```bash
make test
```

## Project Structure

```text
├── config/                 # Configuration files
├── dashboard/              # Streamlit application
├── data/                   # Data directory (raw, interim, processed)
├── docs/                   # Documentation and generated figures
├── scripts/                # Standalone utility scripts
├── src/
│   └── dengueweather/      # Main package source code
│       ├── build/          # Feature engineering logic
│       ├── ingest/         # Data fetching and parsing
│       ├── model/          # Model training and evaluation
│       └── viz/            # Visualization modules
└── pyproject.toml          # Project metadata and dependencies
```
