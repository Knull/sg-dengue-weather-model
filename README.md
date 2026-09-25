# Singapore Dengue Cluster-Presence Ranking

A research project for ranking H3 cells in Singapore by **active dengue-cluster presence** using lagged weather, seasonality, and recent spatial cluster activity.

The project began as a tactical dengue-risk prototype, but the leakage-fixed experiments support a narrower conclusion: the current LightGBM pipeline is a useful spatiotemporal modelling exercise, **not evidence that the learned model improves on a simple persistence baseline**. On the historical archive used here, carrying last week's active cells forward is already a very strong predictor.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Model](https://img.shields.io/badge/Model-LightGBM-green)
![Spatial](https://img.shields.io/badge/Spatial-H3_res_8-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## What the project predicts

The unit of prediction is an **H3 resolution-8 cell × ISO week**. The target is:

```text
y_cluster_present = 1 if an archived cluster snapshot places an active dengue cluster
                    in that H3 cell during that ISO week; otherwise 0
```

This is **cluster presence**, not new-cluster onset and not dengue case counts. Active clusters often persist across adjacent weeks, so persistence is an important part of the problem.

Historical cluster labels are reconstructed from SGCharts archive snapshots. Since the May 2026 leakage fix, the pipeline uses only weeks in which a cluster was actually observed in a raw snapshot; it no longer fills every week between a cluster's first and last appearance as active.

---

## Current model

The main historical model is a `LightGBM` binary classifier wrapped with isotonic `CalibratedClassifierCV`.

The recorded walk-forward run used 26 numeric features. The current feature pipeline includes:

- lagged weather features at **1, 2, 3, 4, 6, 8 and 12 weeks**;
- seasonal week-of-year features;
- `neighbor_pressure_lag_1`, the number of active cells in the previous week's H3 `k_ring(1)`.

Important detail: the explicit same-cell feature `self_lag_1` is excluded from the GBM. However, `neighbor_pressure_lag_1` is currently **inclusive of the centre cell**, so the model still receives a local persistence signal containing both the same H3 cell and its immediate neighbours.

Same-week raw weather variables and `iso_year` are also excluded from model fitting.

---

## Historical evaluation

### Walk-forward LightGBM

The recorded leakage-fixed cross-validation uses valid labelled years from 2013–2020. Each test year is trained only on earlier labelled years.

| Test year | ROC AUC | Average precision | P@20 |
| --- | ---: | ---: | ---: |
| 2014 | 0.6600 | 0.2159 | 0.3714 |
| 2015 | 0.7217 | 0.2793 | 0.4875 |
| 2016 | 0.7424 | 0.4525 | 0.5902 |
| 2017 | 0.6754 | 0.1318 | 0.1580 |
| 2018 | 0.7507 | 0.1899 | 0.2317 |
| 2019 | 0.8099 | 0.5177 | 0.6942 |
| 2020 | 0.7179 | 0.3280 | 0.7111 |
| **Macro mean** | **0.7254** | **0.3027** | **0.4634** |

`P@20` is computed week by week on weeks containing at least one positive cell, then averaged within each test year. The table above is the historical run saved in the project logs; the processed feature parquet itself is not committed to Git.

### Last-week-clusters baseline

A naive persistence baseline was rerun in September 2026 from a public mirror of the same SGCharts historical archive. It marks a cell as high risk when that **same H3 cell was active in the previous week**.

Because this baseline produces many tied scores, its P@20 below is the expected P@20 under uniform random ordering inside score ties. This avoids making the result depend on arbitrary H3-string ordering.

| Test year | ROC AUC | Average precision | Expected P@20 |
| --- | ---: | ---: | ---: |
| 2014 | 0.7074 | 0.2611 | 0.4509 |
| 2015 | 0.7940 | 0.4378 | 0.6192 |
| 2016 | 0.8056 | 0.4913 | 0.5902 |
| 2017 | 0.7271 | 0.2262 | 0.2024 |
| 2018 | 0.7670 | 0.3250 | 0.3140 |
| 2019 | 0.8135 | 0.5337 | 0.7095 |
| 2020 | 0.7372 | 0.3781 | 0.7315 |
| **Macro mean** | **0.7645** | **0.3790** | **0.5168** |

The same audit also evaluated the model's one-week **local-pressure primitive** by ranking cells on the previous week's inclusive H3 `k_ring(1)` active-cell count:

| Baseline | ROC AUC | Average precision | Expected P@20 |
| --- | ---: | ---: | ---: |
| Same-cell last week | 0.7645 | 0.3790 | 0.5168 |
| Local pressure last week | 0.7832 | 0.3430 | 0.4946 |
| LightGBM | 0.7254 | 0.3027 | 0.4634 |

**Current conclusion:** on these historical active-cluster labels, the learned LightGBM does not beat the simple persistence baselines on the macro headline metrics. The strong 2019/2020 P@20 values therefore should not be presented as evidence of incremental forecasting skill over “last week's clusters”.

The reproducible baseline script is:

```bash
python scripts/evaluate_last_week_baseline.py
```

It downloads a public GitHub mirror of `dengue_clusters_archive.zip`, reconstructs H3-resolution-8 weekly labels, and writes `results/last_week_baseline.csv`.

The original processed `unit_week_features.parquet` used for the LightGBM run is gitignored, so the baseline rerun is a reconstruction from the source archive rather than a byte-for-byte replay of that local parquet.

---

## Data and leakage fixes

The historical pipeline combines:

- **Dengue cluster snapshots:** SGCharts historical archive, derived from publicly displayed Singapore dengue-cluster information.
- **Weather:** Meteorological Service Singapore historical weather ingestion.
- **Spatial grid:** H3 resolution 8.

The May 2026 leakage repair made several material changes:

1. cluster labels are built from **observed raw snapshot weeks**, rather than interpolating all weeks from `first_seen` to `last_seen`;
2. pre-existing 0/1 labels are preserved when building the feature panel instead of turning all joined rows into positives;
3. `self_lag_1`, `iso_year`, and same-week raw weather fields are excluded from GBM fitting;
4. weather lag 0 was removed;
5. temporal CV became strict walk-forward training on years earlier than the test year.

Years with zero positive labels in the local feature table are excluded from the reported CV rather than interpreted as dengue-free years. In the recorded run, 2011–2012 and 2021–2025 were excluded; the labelled evaluation period is therefore 2013–2020.

---

## Reproduction

### Installation

```bash
pip install -e .[dev]
```

The Streamlit dashboard has an additional requirement:

```bash
pip install -r dashboard/requirements.txt
```

### Persistence baseline

This is the most self-contained evaluation because the script downloads its public archive mirror automatically:

```bash
python scripts/evaluate_last_week_baseline.py
```

### Historical LightGBM

The raw and processed datasets are not committed, so a fresh clone cannot run `cv-gbm` immediately. After the historical archive and weather data have been ingested and the processed feature table has been built:

```bash
python -m src.cli cv-gbm
```

The main ETL commands are:

```bash
python -m src.cli ingest-archive
python -m src.cli ingest-weather
python -m src.cli build-history
python -m src.cli build-cluster-week
python -m src.cli build-features
python -m src.cli cv-gbm
```

See `python -m src.cli --help` for paths and options.

---

## Live-week prototype

The repository contains a live-cluster patching path:

```bash
python -m src.cli ingest-nea-live
python -m src.cli patch-live-week --live-geojson <downloaded-file.geojson>
python -m src.cli rank-riskiest --model-path data/processed/model_gbm.joblib --iso-year <year> --iso-week <week>
```

This should be treated as **prototype scaffolding**, not as a validated prospective forecast.

`patch-live-week` currently:

- maps the latest live NEA cluster polygons into H3 cells;
- uses them to populate the one-week spatial cluster features;
- targets approximately one week ahead;
- **copies the latest available non-cluster feature values forward**.

It does not currently ingest an actual future weather forecast, and the live path has not been prospectively validated against labelled post-2020 outcomes.

---

## Dashboard

```bash
streamlit run src/app.py
```

The dashboard loads the processed feature table and trained GBM payload, allows a year/week selection, and visualises model scores over H3 cells. It is an exploratory interface, not an operational public-health tool.

---

## Project structure

```text
.
├── config/
│   └── config.default.yaml
├── dashboard/
│   └── requirements.txt
├── results/
│   └── last_week_baseline.csv
├── scripts/
│   └── evaluate_last_week_baseline.py
├── src/
│   ├── app.py
│   ├── cli.py
│   └── dengueweather/
│       ├── build/
│       ├── ingest/
│       ├── model/
│       └── viz/
└── pyproject.toml
```

---

## Limitations

- **Target mismatch:** active-cluster presence is not new-cluster onset, incidence, or outbreak expansion.
- **Persistence dominates:** a one-week persistence baseline currently outperforms the learned model on macro AUC, AP and tie-aware P@20.
- **Archive completeness:** historical labels depend on archived snapshots; low-positive periods may represent real transmission changes, incomplete archival coverage, or both.
- **Reproducibility gap:** the exact processed feature parquet from the recorded GBM run is not committed.
- **Live path is unvalidated:** post-2020 prospective performance has not been measured, and future-weather inputs are not implemented.
- **No causal interpretation:** feature importance or risk score should not be interpreted as a causal effect of weather or neighbouring clusters.
- **Not for intervention decisions:** this repository is a research/engineering prototype.

---

## Next useful experiments

The current results point to a clearer next research question: can a model add signal **beyond persistence**?

Useful follow-ups would be:

1. predict **new-cluster onset** among cells that were inactive in the prior week;
2. compare every learned model directly against same-cell and local-pressure persistence;
3. make the local-pressure feature exclude the centre cell to separate persistence from neighbourhood spread;
4. add real prospective weather inputs before evaluating the live path;
5. retain a frozen, versioned feature table or data manifest so historical results are fully replayable.
