# Industrial Predictive Maintenance Platform

This repository is a portfolio-grade predictive maintenance platform. The goal is not only to train a model on a notebook dataset, but to build the project the way an industrial machine learning system would grow in practice: data pipelines first, then model training, then model serving, backend persistence, and finally a dashboard.

Current progress: milestones 0.1 through 1.6 are implemented through code. That means the repository has the project structure, dataset download scripts and documentation, C-MAPSS EDA notes, a reusable feature engineering pipeline, trained Random Forest and XGBoost baselines, a running results table, saved baseline artifacts with metadata, a PyTorch LSTM/GRU sequence-model trainer, a PatchTST-style transformer trainer ready to run on Colab/Kaggle, and a SHAP-based explainability module with demo notebooks and serving schemas. PatchTST training metrics are intentionally pending until the GPU run.

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
Model registry in /ml/models/registry
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

Important files through Milestone 1.6:

```text
ROADMAP.md                         Technical milestone plan
docs/DATASETS.md                   Dataset sources, licenses, and citations
docs/EDA_FINDINGS.md               C-MAPSS exploratory analysis summary
docs/SCHEMAS.md                    API data contract schemas for prediction & explanations
ml/data/download_cmapss.py          NASA C-MAPSS download script
ml/data/download_mimii.py           MIMII download script
ml/data/download_cwru.py            CWRU download script
ml/notebooks/01_eda_cmapss.ipynb    Exploratory notebook
ml/src/data/cmapss_loader.py        Raw C-MAPSS parser and training-data entry point
ml/src/data/rul_labeling.py         RUL target computation
ml/src/data/feature_engineering.py  Sensor filtering, rolling features, normalization
ml/src/data/windowing.py            Sliding-window sequence generation
ml/src/evaluation/metrics.py        RMSE, MAE, R2, and NASA score
ml/src/models/baseline_rf.py         Random Forest baseline training
ml/src/models/baseline_xgb.py        XGBoost baseline training
ml/src/models/lstm_rul.py            LSTM/GRU sequence-model training
ml/src/models/patchtst_rul.py        PatchTST-style transformer training
ml/src/models/registry.py            Lightweight model artifact registry
ml/src/models/train_baselines.py     Combined baseline training script
ml/src/explain/shap_explainer.py     SHAP interpretability and explanation generator
ml/notebooks/02_train_lstm_colab.ipynb Colab-oriented LSTM training notebook
ml/notebooks/03_train_patchtst_colab.ipynb Colab-oriented PatchTST training notebook
ml/notebooks/04_shap_explainability.ipynb Colab-oriented SHAP explainability notebook
ml/tests/test_data_pipeline.py      Unit tests for the data pipeline
ml/tests/test_metrics_and_registry.py Unit tests for metrics and registry behavior
ml/tests/test_lstm_rul.py           Unit tests for sequence model behavior
docs/RESULTS.md                     Running model leaderboard
docs/MODEL_COMPARISON.md            Cross-family model comparison notes
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

### Milestone 1.3: Classic ML Baselines

This milestone establishes the first real model leaderboard. The goal is not to claim the final best model yet; it is to create a baseline that every later LSTM, GRU, or transformer model must beat.

Implemented components:

- `baseline_rf.py` trains a Random Forest regressor.
- `baseline_xgb.py` trains an XGBoost regressor.
- `metrics.py` evaluates RMSE, MAE, R2, and NASA score.
- `search.py` performs a small holdout grid search before final training.
- `registry.py` saves each trained model as `model.joblib` with a matching `metadata.json`.
- `docs/RESULTS.md` stores the running leaderboard.

Current FD001 results:

| Model | RMSE | MAE | R2 | NASA Score |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 18.1382 | 12.6108 | 0.7951 | 1084.3293 |
| XGBoost | 18.9330 | 13.2661 | 0.7768 | 1208.9589 |

Findings:

- Random Forest is the current FD001 leader, with lower RMSE, lower MAE, higher R2, and lower NASA score than XGBoost.
- Both models use the same engineered cycle-level features from Milestone 1.2: operating settings, selected sensors, rolling means, and rolling standard deviations.
- The Random Forest grid search selected `n_estimators=400`, `max_depth=None`, and `min_samples_leaf=1`.
- The XGBoost grid search selected `n_estimators=600`, `max_depth=4`, and `learning_rate=0.05`.
- NASA score is especially important because over-estimating RUL is riskier than under-estimating it. On this run, Random Forest is not only more accurate by RMSE/MAE, but also safer by the asymmetric NASA metric.

Saved artifact versions:

- Random Forest: `ml/models/registry/random_forest/FD001/20260702T071720Z/`
- XGBoost: `ml/models/registry/xgboost/FD001/20260702T071739Z/`

The main lesson from this milestone is that the project now has a measurable baseline. Future sequence models should be compared against these exact numbers, not only against intuition.

### Milestone 1.4: LSTM/GRU Sequence Model

This milestone moves from tabular snapshots to sequence learning. Instead of asking a model to predict RUL from engineered features at one cycle, the PyTorch model consumes a sliding window of recent sensor history.

Implemented components:

- `lstm_rul.py` defines an LSTM/GRU regressor over C-MAPSS windows.
- The trainer reuses the Milestone 1.2 `get_training_data()` window outputs.
- Evaluation uses the final available test window per engine, so the score lines up with the official C-MAPSS test RUL target.
- Training includes early stopping, gradient clipping, AdamW, and `ReduceLROnPlateau` scheduling.
- Training displays a single `tqdm` progress bar over epochs with train loss, validation loss, best validation loss, and learning rate.
- Checkpoints are saved as `model.pt` with JSON metadata under `ml/models/registry/lstm_rul/` or `ml/models/registry/gru_rul/`.
- `02_train_lstm_colab.ipynb` provides a GPU-friendly notebook workflow.

Run locally:

```bash
make train-lstm
```

Colab FD001 results:

| Model | Run | Max Epochs | RMSE | MAE | R2 | NASA Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | `20260702T102020Z` | 30 | 16.3312 | 11.9987 | 0.8339 | 835.6180 |
| LSTM | `20260702T103659Z` | 100 | 16.5961 | 12.6642 | 0.8285 | 815.3898 |
| GRU | `20260702T111034Z` | 100 | 15.5333 | 11.4046 | 0.8497 | 545.8152 |

Findings:

- Both sequence models beat the Random Forest baseline on FD001. Random Forest had RMSE `18.1382`, MAE `12.6108`, R2 `0.7951`, and NASA score `1084.3293`.
- GRU is the current overall FD001 leader, with the best RMSE, MAE, R2, and NASA score in the project so far.
- Among LSTM runs, `20260702T102020Z` is best by RMSE, MAE, and R2: RMSE improved from the Random Forest baseline `18.1382` to `16.3312`.
- Among LSTM runs, `20260702T103659Z` is best by NASA score: NASA score improved from the Random Forest baseline `1084.3293` to `815.3898`.
- The GRU run improved RMSE from the best LSTM value `16.3312` to `15.5333`, and improved NASA score from the best LSTM value `815.3898` to `545.8152`.
- The LSTM runs showed a useful industrial trade-off: the 30-epoch run was better on average accuracy, while the 100-epoch run was better under the asymmetric NASA penalty. The GRU removed that trade-off in this experiment by winning on both accuracy and NASA score.
- All sequence runs used `window_size=30`, `hidden_size=64`, `num_layers=2`, `dropout=0.2`, `batch_size=128`, and the same 73 engineered sequence features from Milestone 1.2.

Saved sequence metadata versions:

- Best RMSE/MAE/R2 run: `ml/models/registry/lstm_rul/FD001/20260702T102020Z/`
- Best LSTM NASA score run: `ml/models/registry/lstm_rul/FD001/20260702T103659Z/`
- Best overall sequence run: `ml/models/registry/gru_rul/FD001/20260702T111034Z/`

The main lesson from this milestone is that sequence learning is clearly adding value over the classic tabular baseline, and GRU is currently the strongest Phase 1 model. The next milestone should test whether a transformer-style time-series model can beat the GRU result or offer a better accuracy/cost trade-off.

### Milestone 1.5: PatchTST Transformer Model

This milestone adds a transformer-style time-series model for RUL prediction. The implementation uses the central PatchTST idea: split each sensor channel into temporal patches, encode those patch tokens with a shared Transformer encoder, then predict one RUL value for the full window.

Implemented components:

- `patchtst_rul.py` defines a PatchTST-style PyTorch regressor over C-MAPSS windows.
- The trainer reuses the same Milestone 1.2 `get_training_data()` sequence outputs as the LSTM/GRU trainer.
- Evaluation again keeps only the final available test window per engine, matching the official C-MAPSS test target.
- Training includes RevIN-style input normalization, early stopping, gradient clipping, AdamW, and `ReduceLROnPlateau` scheduling.
- Checkpoints are saved as `model.pt` with JSON metadata under `ml/models/registry/patchtst_rul/`.
- `03_train_patchtst_colab.ipynb` provides a GPU-friendly notebook workflow.
- `docs/MODEL_COMPARISON.md` is the comparison write-up scaffold for RF/XGBoost, LSTM/GRU, and PatchTST across FD001-FD004.

Run on Colab:

```bash
PYTHONPATH=ml python -m src.models.patchtst_rul --dataset FD001 --max-epochs 100 --patience 12
```

No PatchTST metrics are committed yet because this milestone's training is intended to run on Colab/Kaggle GPU.

### Milestone 1.6: Explainability (SHAP)

This milestone adds SHAP-based feature attribution to provide interpretability for predictions. This helps explain which sensors and timesteps contributed most to each Remaining Useful Life (RUL) estimate.

Implemented components:

- `shap_explainer.py` implements the core explainability module. It uses `shap.TreeExplainer` for XGBoost models, and `shap.GradientExplainer` for PyTorch sequence models (LSTM, GRU, and PatchTST).
- `docs/SCHEMAS.md` defines the API data contract schemas that the serving API will use for `/predict/rul` and `/predict/rul/explain` response models.
- `04_shap_explainability.ipynb` demonstrates computing global feature importances, per-engine waterfall/beeswarm plots, and attribution heatmaps over sliding window sequences.

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

Train the Milestone 1.3 Random Forest and XGBoost baselines:

```bash
make train-baselines
```

This writes model artifacts to `ml/models/registry/` and appends metrics to `docs/RESULTS.md`.

Train the Milestone 1.4 LSTM sequence model:

```bash
make train-lstm
```

For GPU training, open `ml/notebooks/02_train_lstm_colab.ipynb` in Colab or Kaggle.

Train the Milestone 1.5 PatchTST transformer model:

```bash
make train-patchtst
```

For GPU training, open `ml/notebooks/03_train_patchtst_colab.ipynb` in Colab or Kaggle.

## How To Explain This Project

A concise explanation:

> This is an industrial predictive maintenance platform. I started with NASA C-MAPSS turbofan data and built the foundation that production RUL modeling needs: dataset documentation, EDA, RUL target generation, feature engineering, normalization, and sequence windowing. The pipeline now produces tabular training data for Random Forest/XGBoost, saves trained baseline artifacts with metadata, and also prepares sequence windows for future LSTM or transformer models. Later milestones add a standalone serving API, a FastAPI backend, MongoDB persistence, and a dashboard.

An interviewer-friendly explanation:

> The important design choice is that I did not jump straight to a model. I first built a reusable pipeline. It parses the raw sensor logs, computes capped Remaining Useful Life labels, removes sensors shown by EDA to be uninformative, adds rolling statistics, handles operating-regime normalization for harder C-MAPSS subsets, and generates sliding windows without leaking across engines. That makes the next modeling milestones much cleaner because every model family can consume the same trusted data layer.

Milestone 1.3 builds on that layer by training Random Forest and XGBoost baselines, evaluating RMSE, MAE, R2, and NASA score, then saving model artifacts with metadata in a lightweight registry. Milestone 1.4 adds the sequence-model path: LSTM/GRU models that consume sliding windows of sensor history and can be trained on a GPU notebook. Milestone 1.5 adds a PatchTST-style transformer so the project can compare classic ML, recurrent sequence models, and transformer sequence models on the same benchmark. Milestone 1.6 incorporates model interpretability using SHAP explainers (TreeExplainer and GradientExplainer), allowing operators to inspect feature attributions and understand individual RUL predictions.

See `ROADMAP.md` for the full technical plan.
