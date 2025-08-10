# Dengue‐Weather Singapore Project

This repository provides a starting point for analysing the relationship between
dengue clusters and local weather patterns in Singapore.  It contains a well
structured folder hierarchy, configuration files, stubbed scripts and notes to
help you focus on the analytical work rather than boilerplate setup.

## Getting Started

Follow these steps to get up and running quickly:

1. **Install dependencies.** Use the package manager of your choice to
   initialise the environment.  For example:
   ```bash
   make init
   ```

2. **Add raw data.** Place the archived SGCharts CSV files under
   `data/raw/sgcharts_csv/`.  Other raw datasets (weather, population
   shapefiles, etc.) should live under appropriate folders in `data/raw/`.

3. **Ingest and build panels.** Run the ingestion and processing pipeline to
   merge the archived snapshots, build cluster histories, and prepare the
   unit–week panel.  Use:
   ```bash
   make ingest
   make panel
   make features
   ```

4. **Train and evaluate models.** Fit the baseline panel logistic model and
   evaluate its performance:
   ```bash
   make model
   make eval
   ```

5. **Visualise results.** Generate lag–response curves, interaction surfaces
   and risk maps using:
   ```bash
   make figs
   ```

6. **Run the dashboard.** Launch the interactive Streamlit application to
   explore predictions and risk maps:
   ```bash
   make app
   ```

Consult the `docs/` directory for details on data schemas, planned figures
and methodological notes.
