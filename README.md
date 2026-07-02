# Industrial Predictive Maintenance Platform

This repository is a portfolio-grade predictive maintenance platform. The goal is not only to train a model on a notebook dataset, but to build the project the way an industrial machine learning system would grow in practice: data pipelines first, then model training, then model serving, backend persistence, and finally a dashboard.

Current progress: milestones 0.1 through 1.2 are implemented. That means the repository has the project structure, dataset download scripts and documentation, C-MAPSS EDA notes, and a reusable feature engineering pipeline that turns raw turbofan sensor logs into model-ready training data.

## What Problem This Project Solves

Predictive maintenance tries to answer a practical question:

> Given the recent sensor history of a machine, how much useful life is left before maintenance or failure?

In this project, that target is called Remaining Useful Life, or RUL. Instead of predicting a simple class like `healthy` or `failed`, the model learns to predict a number of cycles remaining. This is more useful in industrial settings because maintenance teams need to plan when to inspect, repair, or replace equipment.

The first dataset is NASA C-MAPSS, a simulated turbofan engine degradation dataset. Each engine starts healthy, runs for many cycles, gradually degrades, and eventually reaches failure in the training data. Every row contains:

- `engine_id`: which engine the row belongs to
- `cycle`: the time step for that engine
- `setting_1` to `setting_3`: operating conditions
- `s_1` to `s_21`: sensor readings

The machine learning task is: use the operating settings and sensor readings to predict RUL.

## Architecture Direction

The long-term architecture is intentionally split into separate services:

```text
Raw public datasets
        |
        v
ML pipeline in /ml
  - download data
  - audit data
  - run EDA
  - label RUL
  - engineer features
  - train models
        |
        v
Model registry in /ml/models/registry        (planned)
        |
        v
Serving API in /serving                       (planned)
        |
        v
FastAPI backend in /backend + MongoDB         (planned)
        |
        v
Frontend dashboard                            (planned)
```

Why split it this way?

- The ML code can train and evaluate models without being tangled with API code.
- The serving API can load trained model artifacts and expose predictions over HTTP.
- The backend can own business concepts such as machines, sensor readings, prediction history, and maintenance records.
- The frontend can focus on operators and maintenance users without knowing model internals.

This mirrors how production ML systems are usually separated: training, serving, business logic, storage, and user interface evolve at different speeds.

## Repository Layout

```text
backend/    FastAPI business backend, planned for Phase 3
docs/       Dataset notes, EDA findings, roadmap, and future design docs
infra/      Docker Compose and local infrastructure
ml/         Data acquisition, EDA, feature engineering, and future model training
serving/    Standalone model-serving API, planned for Phase 2
```

Important files through Milestone 1.2:

```text
ROADMAP.md                         Technical milestone plan
docs/DATASETS.md                   Dataset sources, licenses, and citations
docs/EDA_FINDINGS.md               C-MAPSS exploratory analysis summary
ml/data/download_cmapss.py          NASA C-MAPSS download script
ml/data/download_mimii.py           MIMII download script
ml/data/download_cwru.py            CWRU download script
ml/notebooks/01_eda_cmapss.ipynb    Exploratory notebook
ml/src/data/cmapss_loader.py        Raw C-MAPSS parser and training-data entry point
ml/src/data/rul_labeling.py         RUL target computation
ml/src/data/feature_engineering.py  Sensor filtering, rolling features, normalization
ml/src/data/windowing.py            Sliding-window sequence generation
ml/tests/test_data_pipeline.py      Unit tests for the data pipeline
```

## Milestone's Details and Findings

### Milestone 0.1: Project Foundation

The repository was organized as a monorepo with separate areas for backend, ML, serving, infrastructure, and documentation. MongoDB local infrastructure is defined in `infra/docker-compose.yml`.

At this stage the project became more than a notebook folder. It got a shape that can support a real system.

### Milestone 0.2: Dataset Acquisition and Audit

The project uses public datasets only. The raw data is intentionally not committed to git because these datasets can become large and should be reproducible from scripts.

Dataset scripts live in `ml/data/`:

- C-MAPSS for turbofan RUL prediction
- MIMII for future industrial audio anomaly detection
- CWRU bearing data for future vibration/fault diagnosis work

Dataset documentation lives in `docs/DATASETS.md`, including source, license, structure, project use, and citation notes.

### Milestone 1.1: C-MAPSS EDA

The EDA milestone answers a basic but important question: what is inside the data before modeling?

Key findings documented in `docs/EDA_FINDINGS.md`:

- Some FD001 sensors are nearly constant and do not help prediction: `s_1`, `s_5`, `s_6`, `s_10`, `s_16`, `s_18`, and `s_19`.
- Several sensors show meaningful correlation with RUL and visible degradation trends.
- FD002 and FD004 contain multiple operating regimes, so normalization should respect operating condition clusters instead of treating all rows as one distribution.

These findings directly inform the feature engineering pipeline in Milestone 1.2.

### Milestone 1.2: Feature Engineering and RUL Labeling

This milestone creates the reusable data pipeline for all future models.

The main entry point is:

```python
from src.data import get_training_data

data = get_training_data(dataset="FD001")

X_train = data.X_train
y_train = data.y_train
X_test = data.X_test
y_test = data.y_test

train_windows = data.train_windows
y_train_windows = data.y_train_windows
```

The same function returns:

- tabular features for Random Forest and XGBoost baselines
- sequence windows for LSTM, GRU, and transformer models later
- labeled train/test data
- fitted feature engineering metadata

## The Data Pipeline Explained Simply

The pipeline turns raw C-MAPSS text files into clean model inputs.

Step 1: Parse raw text files.

The NASA files are whitespace-separated text files without headers. `cmapss_loader.py` assigns readable column names such as `engine_id`, `cycle`, `setting_1`, and `s_1`.

Step 2: Compute RUL labels.

For training engines, the final cycle is treated as failure. If an engine fails at cycle 200, then cycle 199 has RUL 1, cycle 198 has RUL 2, and so on.

Very early cycles can have extremely large RUL values, which are less useful for learning degradation. The pipeline applies a standard C-MAPSS piecewise cap, usually 125 cycles. This means early-life rows are labeled as `125` instead of very large values, and the target becomes more realistic.

Step 3: Drop uninformative sensors.

The EDA showed that several FD001 sensors are flat or nearly flat. The pipeline drops those by default so models do not waste capacity learning from constant columns.

Step 4: Add rolling features.

A single sensor reading can be noisy. Rolling statistics summarize recent behavior:

- rolling mean: recent average value
- rolling standard deviation: recent variability

These features help classic ML models understand short-term trends without needing a full sequence model.

Step 5: Normalize features.

Sensor scales differ. Normalization makes features easier for models to learn from. For FD002 and FD004, the pipeline can normalize per operating regime because those subsets contain multiple operating conditions.

Step 6: Create sequence windows.

Future deep learning models should look at sensor history, not only one row. `windowing.py` creates fixed-length sliding windows per engine and prevents leakage across engine boundaries.

That last part matters: a window must never contain the end of engine 1 and the beginning of engine 2. The tests check this.

## How To Run Locally

Create or activate the ML environment, then install dependencies:

```bash
python3 -m venv ml/.venv
source ml/.venv/bin/activate
pip install -r ml/requirements.txt
```

Download the datasets:

```bash
make download-data
```

Run the Milestone 1.2 tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/ipmp_pycache ml/.venv/bin/pytest ml/tests/ -q
```

Expected result:

```text
8 passed
```

Start local MongoDB infrastructure when working on later backend milestones:

```bash
cd infra
docker-compose up -d
```

MongoDB is included now because the final platform will persist machines, sensor readings, predictions, and maintenance logs. It is not required for the Milestone 1.2 ML pipeline.

## How To Explain This Project

A concise explanation:

> This is an industrial predictive maintenance platform. I started with NASA C-MAPSS turbofan data and built the foundation that production RUL modeling needs: dataset documentation, EDA, RUL target generation, feature engineering, normalization, and sequence windowing. The pipeline now produces both tabular training data for Random Forest/XGBoost and sequence windows for future LSTM or transformer models. Later milestones add model training, a standalone serving API, a FastAPI backend, MongoDB persistence, and a dashboard.

An interviewer-friendly explanation:

> The important design choice is that I did not jump straight to a model. I first built a reusable pipeline. It parses the raw sensor logs, computes capped Remaining Useful Life labels, removes sensors shown by EDA to be uninformative, adds rolling statistics, handles operating-regime normalization for harder C-MAPSS subsets, and generates sliding windows without leaking across engines. That makes the next modeling milestones much cleaner because every model family can consume the same trusted data layer.


See `ROADMAP.md` for the full technical plan.
