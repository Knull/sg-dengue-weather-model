PYTHON := python

init:
	uv sync || pip install -e .[dev]

ingest: ingest-archive ingest-weather

ingest-archive:
	$(PYTHON) -m src.cli ingest-archive --input data/raw/sgcharts_csv --out data/interim/archive_merged.parquet

ingest-weather:
	$(PYTHON) -m src.cli ingest-weather --out data/interim/weather_weekly.parquet

panel:
	$(PYTHON) -m src.cli build-history --in data/interim/archive_merged.parquet --out data/interim/cluster_history.parquet
	$(PYTHON) -m src.cli build-cluster-week --in data/interim/cluster_history.parquet --out data/processed/cluster_week.parquet

features: panel
	$(PYTHON) -m src.cli build-features --cluster data/processed/cluster_week.parquet --weather data/interim/weather_weekly.parquet --out data/processed/unit_week_features.parquet

model:
	$(PYTHON) -m src.cli fit-panel-logit --features data/processed/unit_week_features.parquet --out data/processed/model.joblib

eval:
	$(PYTHON) -m src.cli eval-model --model data/processed/model.joblib --features data/processed/unit_week_features.parquet --out data/processed/metrics.json

figs:
	$(PYTHON) -m src.cli plot-lag-curves --model data/processed/model.joblib --out docs/figures_lag/
	$(PYTHON) -m src.cli plot-interaction --model data/processed/model.joblib --out docs/figures_interaction/
	$(PYTHON) -m src.cli map-hindcast --model data/processed/model.joblib --out docs/maps/

app:
	streamlit run dashboard/app.py

lint:
	ruff check .

fmt:
	ruff format .

test:
	pytest -q

.PHONY: init ingest ingest-archive ingest-weather panel features model eval figs app lint fmt test