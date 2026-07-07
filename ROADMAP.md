# Industrial Predictive Maintenance Platform — Technical Roadmap

**Purpose:** Resume-grade portfolio project. A multi-modal predictive maintenance platform that goes beyond "predict RUL on C-MAPSS in a notebook" — it's a real architecture with a separate ML training pipeline, a separate model-serving API, a FastAPI backend, MongoDB persistence, and (later) a frontend dashboard.

**Repo layout (monorepo, single repo):**
```
/backend     -> FastAPI app: business logic, MongoDB, orchestrates calls to the serving API
/ml          -> training pipelines, notebooks/scripts, model registry, exported model artifacts
/serving     -> standalone model-serving API (loads ml/ artifacts, exposes prediction endpoints)
/frontend    -> (later milestone block, not started yet)
/infra       -> docker-compose, Dockerfiles, env templates
/docs        -> this roadmap + architecture decision records (ADRs)
```

**Core stack decisions locked in:**
- Backend: FastAPI (MongoDB via PyMongo)
- Database: MongoDB
- Model family path: Classic ML (XGBoost/Random Forest) → Deep Learning (LSTM/GRU) → Transformer-based time series (PatchTST) — each becomes a registered, comparable model
- Modality order: tabular (sensor time series) first → audio (MIMII) → image (defect/corrosion datasets) later
- Model serving: a **separate service** from the backend (backend calls it over HTTP) — more realistic, mirrors real MLOps separation
- Data: public datasets only for now; every milestone is designed so a "swap in proprietary data" step is possible later without re-architecting
- Frontend: deliberately deferred — all milestones below are backend + ML only, frontend gets its own roadmap block once this is stable

**Git workflow convention:** each milestone = one feature branch = one PR/commit (or a small stack of commits) merged to `main` once its deliverable checklist is met and it runs end-to-end locally. Tag releases as `v0.1`, `v0.2`, etc. at the end of each Phase.

---

## Phase 0 — Foundations

### Milestone 0.1: Repo Scaffolding & Tooling
**What we're doing:** Set up the monorepo skeleton, dependency management per subproject, linting/formatting, pre-commit hooks, and a docker-compose that brings up MongoDB locally.

**Tasks:**
- Create `backend/`, `ml/`, `serving/`, `infra/`, `docs/` folders, each with its own `pyproject.toml`/`requirements.txt` (keep dependency trees independent — `ml` will need torch/xgboost, `backend` shouldn't).
- Root-level `.gitignore`, `.editorconfig`, `README.md` (project pitch + architecture diagram).
- `infra/docker-compose.yml` with a MongoDB service + persistent volume.

**Deliverables:**
- Empty-but-structured repo that builds/runs `docker-compose up` and gets a live MongoDB instance.
- `README.md` with project description, architecture diagram (placeholder ok), and "how to run locally" section.

**Commit message:** `chore: scaffold monorepo structure, docker-compose, CI skeleton`

---

### Milestone 0.2: Data Acquisition & Audit
**What we're doing:** Download and audit all public datasets we'll use, document their licenses, and store raw data outside git (with a fetch script, not committed binary data).

**Tasks:**
- Write `ml/data/download_cmapss.py` — fetches NASA C-MAPSS (FD001–FD004 subsets) from NASA's repository, unzips into `ml/data/raw/cmapss/`.
- Write `ml/data/download_mimii.py` — fetches MIMII (start with one machine type, e.g. valve or fan, to keep size manageable).
- Write `ml/data/download_cwru.py` — fetches CWRU bearing vibration dataset.
- Add `.gitignore` entries so `ml/data/raw/` and `ml/data/processed/` are never committed (use Git LFS or just document re-download steps — for a resume project, NOT committing multi-GB data is correct).
- Create `docs/DATASETS.md` documenting: source, license, size, structure, what we'll use each for, and citation.

**Deliverables:**
- Reproducible `make download-data` (or equivalent script) that pulls all raw datasets fresh on any machine.
- `docs/DATASETS.md` fully filled in.

**Commit message:** `feat(data): add dataset download scripts and dataset documentation`

---

## Phase 1 — Tabular Pipeline: RUL Prediction (Classic ML Baseline)

This phase is the spine of the project. Everything else (DL, transformers, audio, image, RAG) hangs off this working pipeline.

### Milestone 1.1: Exploratory Data Analysis (C-MAPSS)
**What we're doing:** Understand the C-MAPSS structure deeply before modeling — sensor distributions, operating regimes, degradation trends, which sensors are flat/uninformative.

**Tasks:**
- `ml/notebooks/01_eda_cmapss.ipynb`: load FD001, plot sensor trajectories per engine, identify constant/non-informative sensors (commonly sensors 1, 5, 6, 10, 16, 18, 19 are flat in FD001), visualize operating condition clusters for FD002/FD004.
- Compute and document per-sensor correlation with cycle-to-failure.
- Write findings to `docs/EDA_FINDINGS.md`.

**Deliverables:**
- Notebook with visualizations (sensor trends, degradation curves, operating regime clustering).
- `docs/EDA_FINDINGS.md` summarizing which sensors/conditions matter, used to justify feature selection in 1.2.

**Commit message:** `docs(ml): EDA on C-MAPSS, document sensor relevance findings`

---

### Milestone 1.2: Feature Engineering & RUL Labeling Pipeline
**What we're doing:** Build the reusable data pipeline that turns raw C-MAPSS sensor logs into model-ready (X, y) — this pipeline gets reused by every model family.

**Tasks:**
- `ml/src/data/cmapss_loader.py`: parses raw txt files into structured DataFrames (engine_id, cycle, op_settings, sensor readings).
- `ml/src/data/rul_labeling.py`: implements RUL target computation, including the standard **piecewise-linear RUL cap** (e.g. cap at 125 cycles) used in C-MAPSS literature so labels aren't unrealistically large early in an engine's life.
- `ml/src/data/feature_engineering.py`: rolling statistics (mean/std over window), normalization per operating condition (important for FD002/FD004 which have 6 operating regimes), drop uninformative sensors found in 1.1.
- `ml/src/data/windowing.py`: sliding-window sequence generation for sequence models (used later by LSTM/PatchTST, but build it now so it's shared infra).
- Unit tests in `ml/tests/test_data_pipeline.py` (e.g. assert RUL is monotonically decreasing per engine, assert windowing doesn't leak across engines).

**Deliverables:**
- A clean `get_training_data(dataset="FD001")` function returning train/test splits ready for both tabular and sequence models.
- Passing unit tests (`pytest ml/tests/`).

**Commit message:** `feat(ml): RUL labeling + feature engineering pipeline with tests`

---

### Milestone 1.3: Classic ML Baseline (Random Forest + XGBoost)
**What we're doing:** Establish the baseline everything else has to beat. This is also the fastest milestone to a "working end-to-end result."

**Tasks:**
- `ml/src/models/baseline_rf.py`, `ml/src/models/baseline_xgb.py`: train on engineered tabular features (last-cycle snapshot + rolling stats) to predict RUL.
- Evaluation: RMSE, MAE, and the **NASA scoring function** (asymmetric penalty — late predictions penalized harder than early ones; this is the standard C-MAPSS metric and using it signals domain knowledge to anyone reviewing the repo).
- Hyperparameter search (Optuna or simple grid search) logged via MLflow or a lightweight custom run-logger.
- `ml/src/models/registry.py`: simple model registry pattern — saves model artifact + metadata (metrics, params, dataset version, timestamp) to `ml/models/registry/`.

**Deliverables:**
- Trained RF and XGBoost models on FD001, with metrics logged.
- `docs/RESULTS.md` table: model, dataset, RMSE, NASA score (this table gets a new row every milestone from here on — it becomes the project's running leaderboard).

**Commit message:** `feat(ml): classic ML baselines (RF, XGBoost) with NASA scoring + model registry`

---

### Milestone 1.4: Deep Learning — LSTM/GRU Sequence Model
**What we're doing:** Move to sequence models that use the full sensor trajectory window instead of hand-engineered snapshots, trainable on Colab/Kaggle GPU.

**Tasks:**
- `ml/src/models/lstm_rul.py`: PyTorch LSTM (or GRU) regressor over the windowed sequences from 1.2.
- `ml/notebooks/02_train_lstm_colab.ipynb`: self-contained Colab-runnable notebook (clones repo or pulls data, trains, saves artifact back) — this is what "trainable on Colab/Kaggle" means in practice.
- Compare against baseline on the same NASA scoring function.
- Add early stopping, learning rate scheduling, basic experiment tracking (W&B free tier or MLflow).

**Deliverables:**
- Trained LSTM model checkpoint + metrics added to `docs/RESULTS.md`.
- Working Colab notebook (linked from README) — proof the project doesn't require a GPU machine to reproduce.

**Commit message:** `feat(ml): LSTM/GRU sequence model for RUL, Colab training notebook`

---

### Milestone 1.5: Transformer Time-Series Model (PatchTST)
**What we're doing:** Add a modern transformer-based time-series model — this is the piece that differentiates the project from the sea of LSTM-only predictive maintenance repos.

**Tasks:**
- `ml/src/models/patchtst_rul.py`: implement or adapt PatchTST (patch-based transformer for time series) for the RUL regression task. Use an existing reference implementation as a base rather than reinventing from the paper, but document the adaptation.
- Train on Colab/Kaggle (same notebook pattern as 1.4).
- Run all 3 model families (RF/XGB, LSTM, PatchTST) on FD001 **and** the harder FD002/FD004 subsets (multiple operating conditions) to show robustness, not just a cherry-picked easy subset.

**Deliverables:**
- Trained PatchTST model, full comparison table across FD001–FD004 for all model families in `docs/RESULTS.md`.
- Short write-up in `docs/MODEL_COMPARISON.md`: trade-offs observed (accuracy vs. inference latency vs. training cost) — this write-up itself is a strong resume artifact.

**Commit message:** `feat(ml): PatchTST transformer model for RUL, full cross-subset benchmark`

---

### Milestone 1.6: Explainability (SHAP)
**What we're doing:** Add interpretability — critical for an "industrial AI" pitch since black-box predictions aren't trusted on factory floors.

**Tasks:**
- `ml/src/explain/shap_explainer.py`: SHAP TreeExplainer for the XGBoost model (fast, exact) and SHAP DeepExplainer/KernelExplainer (or integrated gradients) for the LSTM/PatchTST models.
- Generate per-prediction explanation: which sensors/cycles contributed most to a given RUL estimate.
- Store explanation alongside prediction output schema (this matters for Milestone 2.x when serving has to return it too).

**Deliverables:**
- `ml/notebooks/03_shap_explainability.ipynb` with SHAP summary plots and example per-engine explanations.
- A documented explanation JSON schema (`docs/SCHEMAS.md`) that the serving API will implement in Phase 2.

**Commit message:** `feat(ml): SHAP-based explainability for tree and DL models`

---

## Phase 2 — Model Serving API

### Milestone 2.1: Serving Service Skeleton
**Status:** Implemented.

**What we're doing:** Stand up the separate model-serving service (its own FastAPI app, separate from the business-logic backend) that loads artifacts from `ml/models/registry/` and exposes prediction endpoints.

**Tasks:**
- `serving/app/main.py`: FastAPI app with `/health`, `/models` (list registered models + metadata).
- `serving/app/model_loader.py`: loads a model artifact by name/version from the registry (start with the XGBoost baseline, simplest to serve).
- `serving/app/schemas.py`: Pydantic request/response schemas matching `docs/SCHEMAS.md` from 1.6.
- `POST /predict/rul`: accepts a sensor window, returns RUL estimate + confidence/uncertainty if available.
- Dockerfile for the serving service.

**Deliverables:**
- Serving API runs standalone (`docker run` or `uvicorn`), `/predict/rul` returns a real prediction from the trained XGBoost model.
- Postman/curl examples in `serving/README.md`.

**Commit message:** `feat(serving): standalone model-serving API with XGBoost RUL endpoint`

---

### Milestone 2.2: Multi-Model Serving + Explainability Endpoint
**What we're doing:** Extend serving to support all three model families (selectable by request) and return SHAP explanations.

**Tasks:**
- `POST /predict/rul?model=xgboost|lstm|patchtst` — model selection.
- `POST /predict/rul/explain` — returns prediction + SHAP-based feature attribution.
- Add a simple in-memory or Redis-based cache for repeated identical requests (nice-to-have, shows systems thinking).
- Load testing with `locust` or simple async load script — document latency per model family (this is genuinely useful: e.g. PatchTST will be slower than XGBoost, which is an interesting real finding to write up).

**Deliverables:**
- All 3 models servable through one API.
- `docs/SERVING_BENCHMARKS.md`: latency/throughput per model.

**Commit message:** `feat(serving): multi-model selection, explainability endpoint, latency benchmarks`

---

## Phase 3 — Backend & Persistence

### Milestone 3.1: Backend Skeleton + MongoDB Models
**What we're doing:** Build the FastAPI backend that owns business logic — machines, sensor ingestion, prediction history — and talks to MongoDB and to the serving API.

**Tasks:**
- `backend/app/main.py`, `backend/app/db.py` (Motor async MongoDB client).
- MongoDB collections design: `machines`, `sensor_readings`, `predictions`, `maintenance_logs` (placeholder for Phase 5).
- `backend/app/models/` — Pydantic schemas for each collection.
- `backend/app/routers/machines.py`: CRUD for registering a "machine" (maps to a C-MAPSS engine_id for now, but modeled like a real asset).
- Dockerfile + add backend service to `infra/docker-compose.yml`.

**Deliverables:**
- Backend runs, connects to MongoDB, CRUD on `/machines` works end-to-end.
- `docs/DB_SCHEMA.md` documenting collection structures.

**Commit message:** `feat(backend): FastAPI skeleton, MongoDB models, machine CRUD`

---

### Milestone 3.2: Sensor Ingestion + Prediction Orchestration
**What we're doing:** Wire the backend to ingest sensor windows and call the serving API, persisting predictions — this is the first end-to-end flow (data in → prediction out → stored).

**Tasks:**
- `POST /machines/{id}/sensor-readings`: ingest a sensor window, store in `sensor_readings`.
- `POST /machines/{id}/predict`: backend pulls the latest window, calls `serving` API, stores result in `predictions` with timestamp + model version used.
- `GET /machines/{id}/predictions`: history endpoint.
- Background task (FastAPI `BackgroundTasks` or a simple scheduler) to simulate periodic prediction runs.

**Deliverables:**
- Full flow demonstrable via curl/Postman: ingest reading → trigger prediction → backend calls serving → result stored and retrievable.
- Integration test (`backend/tests/test_prediction_flow.py`) covering this flow with a test MongoDB instance (mongomock or test container).

**Commit message:** `feat(backend): sensor ingestion + prediction orchestration calling serving API`

---

### Milestone 3.3: Sensor Stream Simulator
**What we're doing:** Build a script that replays C-MAPSS test engines as a live stream, hitting the ingestion endpoint over time — this is what makes the eventual frontend dashboard feel like a "live" industrial system instead of a static demo.

**Tasks:**
- `ml/scripts/sensor_simulator.py`: replays a test-set engine's cycles at configurable speed (e.g. 1 cycle/sec), POSTing to the backend.
- Support multiple simulated machines concurrently (async).
- Add a `/machines/{id}/alerts` endpoint: simple rule (e.g. predicted RUL < threshold) that flags a machine — first version of the "alerting system" feature.

**Deliverables:**
- Running the simulator against a live backend produces a growing prediction history and occasional alerts, observable via API.

**Commit message:** `feat: sensor stream simulator + threshold-based alerting`

---

## Phase 4 — Audio Modality (MIMII)

### Milestone 4.1: Audio Anomaly Detection Model
**What we're doing:** Add the second modality — audio-based anomaly detection on MIMII, following the same baseline→better-model pattern as Phase 1 but scoped smaller since it's a secondary modality.

**Tasks:**
- `ml/src/data/mimii_loader.py`: loads normal/abnormal WAV clips for one machine type, extracts log-mel spectrograms.
- Baseline: autoencoder (reconstruction-error anomaly detection — the standard MIMII baseline approach) trained only on normal sounds.
- Stretch: a small CNN classifier if time allows, compared against the autoencoder.
- Evaluation: AUC, per the MIMII benchmark convention.

**Deliverables:**
- Trained audio anomaly model + metrics in `docs/RESULTS.md` (new audio section).
- `ml/notebooks/04_audio_anomaly_mimii.ipynb`.

**Commit message:** `feat(ml): audio anomaly detection on MIMII (autoencoder baseline)`

---

### Milestone 4.2: Audio Serving + Backend Integration
**What we're doing:** Extend serving and backend to accept audio uploads and return anomaly scores, reusing the same architecture pattern from Phase 2–3.

**Tasks:**
- `serving`: `POST /predict/audio-anomaly` accepting a WAV file or feature vector.
- `backend`: `POST /machines/{id}/audio-readings`, store anomaly score in a new `audio_predictions` collection (or extend `predictions` with a `modality` field — decide and document in `docs/DB_SCHEMA.md`).

**Deliverables:**
- End-to-end audio flow working, mirroring the sensor flow from 3.2.

**Commit message:** `feat: audio anomaly detection serving + backend integration`

---

## Phase 5 — Maintenance Logs & RAG

### Milestone 5.1: Synthetic Maintenance Log Generation
**What we're doing:** Generate realistic technician maintenance logs since public datasets don't have them (as noted in the source material) — needed for the RAG/document-search feature.

**Tasks:**
- `ml/scripts/generate_maintenance_logs.py`: uses an LLM (Claude API or similar) to generate a few hundred structured logs (problem/inspection/action format), conditioned on plausible C-MAPSS failure modes so they're not generic.
- Store as JSON, seed into MongoDB `maintenance_logs` collection.

**Deliverables:**
- A few hundred realistic, varied maintenance log records seeded into the DB.
- `docs/SYNTHETIC_DATA.md` documenting generation methodology (honest about it being synthetic — this transparency is actually a plus on a resume, shows good judgment).

**Commit message:** `feat(data): synthetic maintenance log generation + seeding`

---

### Milestone 5.2: Vector Store + RAG Pipeline
**What we're doing:** Build retrieval over maintenance logs and (later) equipment manuals.

**Tasks:**
- Choose embedding model (sentence-transformers, local, no API dependency needed for embeddings).
- Choose vector store: start simple — MongoDB Atlas Vector Search (keeps the stack consolidated, avoids adding e.g. Pinecone/Chroma as a new infra dependency) or a lightweight local FAISS index if Atlas isn't available locally.
- `backend/app/routers/search.py`: `POST /search/logs` semantic search over maintenance logs.
- `POST /assistant/query`: combines retrieved logs + sensor/prediction context into an LLM prompt, returns a grounded answer (e.g. "why might Engine 12 be flagged?").

**Deliverables:**
- Working semantic search over logs.
- Working "ask a question, get a grounded answer" endpoint.

**Commit message:** `feat(backend): vector search + RAG query endpoint over maintenance logs`

---

### Milestone 5.3: Equipment Manual Ingestion (PDF RAG)
**What we're doing:** Extend RAG to public equipment manual PDFs, the second document source.

**Tasks:**
- `ml/scripts/ingest_manuals.py`: chunk PDFs (by section/page), embed, store with metadata (manual name, page).
- Extend `/assistant/query` to retrieve from both logs and manuals, citing source (manual + page, or log ID) in the response.

**Deliverables:**
- A query like "how should bearing replacement be performed" returns a grounded answer citing the relevant manual section.

**Commit message:** `feat: PDF manual ingestion + multi-source RAG`

---

## Phase 6 — Image Modality (Optional/Stretch, if time allows)

### Milestone 6.1: Visual Defect Classification
**What we're doing:** Add the third modality — image-based defect detection (bearings, gears, corrosion) using a pretrained CNN fine-tuned on public defect datasets.

**Tasks:**
- `ml/src/models/defect_classifier.py`: fine-tune a small pretrained model (ResNet18/EfficientNet-B0) on a public surface-defect dataset.
- `serving`: `POST /predict/defect-image`.
- `backend`: `POST /machines/{id}/image-readings`.

**Deliverables:**
- Working image classification endpoint, integrated the same way as sensor/audio.

**Commit message:** `feat(ml): visual defect classification model + serving/backend integration`

*(This phase is explicitly marked optional/stretch — sequence it last and only after Phases 1–5 are solid, since tabular+audio+RAG already cover the "multi-modal" bar from the source brief.)*

---

## Phase 7 — Root Cause Analysis & Reporting

### Milestone 7.1: Root-Cause Analysis Agent
**What we're doing:** Tie everything together — given a flagged machine, combine sensor SHAP explanation + audio anomaly (if available) + retrieved similar past maintenance logs into a single root-cause hypothesis.

**Tasks:**
- `backend/app/services/root_cause.py`: orchestrates calls to (a) serving's explain endpoint, (b) RAG search for similar historical cases, (c) an LLM call that synthesizes these into a structured hypothesis.
- `POST /machines/{id}/root-cause-analysis`.

**Deliverables:**
- Endpoint returns a structured root-cause report: likely cause, supporting evidence (SHAP features, similar past logs), confidence.

**Commit message:** `feat(backend): root-cause analysis agent combining explainability + RAG`

---

### Milestone 7.2: Automatic Maintenance Report Generation
**What we're doing:** Generate a human-readable maintenance report (PDF or markdown) from a root-cause analysis — the final "product" feel.

**Tasks:**
- `backend/app/services/report_generator.py`: template-based report (machine info, prediction history chart data, root cause, recommended action) rendered to PDF.
- `GET /machines/{id}/report` — downloadable PDF.

**Deliverables:**
- A real downloadable PDF report for a given machine, generated from live data.

**Commit message:** `feat(backend): automatic PDF maintenance report generation`

---

## Phase 8 — Productionization

### Milestone 8.1: Full Dockerization + Compose Orchestration
**What we're doing:** Bring backend, serving, MongoDB (and later frontend) under one `docker-compose up`.

**Tasks:**
- Finalize Dockerfiles for backend and serving (multi-stage builds, slim images).
- `infra/docker-compose.yml` wires all services with proper networking, env vars, health checks.
- `.env.example` documenting all required environment variables.

**Deliverables:**
- `docker-compose up` from a clean clone brings up the entire backend system (no frontend yet) and it's usable via curl/Postman.

**Commit message:** `chore(infra): full docker-compose orchestration for backend + serving + db`

---

### Milestone 8.2: Testing, CI, and Documentation Pass
**What we're doing:** Harden the project — this is what makes it look maintained rather than abandoned after the fun parts were built.

**Tasks:**
- Expand test coverage: unit tests for `ml/`, integration tests for `backend/` and `serving/`.
- GitHub Actions: lint + test on every PR, across all three subprojects.
- Final documentation pass: top-level `README.md` with architecture diagram, setup instructions, demo GIF/screenshots (once frontend exists), and a "Design Decisions" section explaining the classic ML → DL → transformer progression and the separate-serving-service choice.

**Deliverables:**
- CI green on a fresh PR.
- A README that could stand alone as a portfolio piece even before someone opens the code.

**Commit message:** `chore: CI hardening, expanded test coverage, final documentation pass`

---

## Phase 9 — Frontend (Deferred — Separate Roadmap)

Not detailed here per your request — to be scoped once Phases 0–8 are stable. At minimum it will need: live dashboard (sensor simulation visualization), prediction/RUL charts, alert feed, root-cause report viewer, maintenance log/manual search UI. Will produce its own milestone breakdown when you're ready.

---

## Path to Real Industrial Data (Future, Post-Resume-Project)

Each phase above is built so that swapping public data for proprietary/real industrial data later requires changes in only two places:
1. **`ml/data/*_loader.py`** — write a new loader matching the same internal schema (engine_id/asset_id, cycle/timestamp, sensor columns) that feature engineering and model code already expect.
2. **`backend` ingestion endpoints** — already accept arbitrary sensor windows; a real PLC/SCADA/MQTT ingestion adapter would sit in front of `POST /machines/{id}/sensor-readings`, not replace it.

No retraining-from-scratch architecture change is needed — this is intentional and worth calling out explicitly in interviews.

---

## Suggested Sequencing Summary

| Phase | Focus | Est. Time |
|---|---|---|
| 0 | Scaffolding + data acquisition | 2-3 days |
| 1 | Tabular RUL pipeline (baseline → DL → transformer → SHAP) | 2-3 weeks |
| 2 | Model serving API | 3-5 days |
| 3 | Backend + MongoDB + orchestration + simulator | 1-1.5 weeks |
| 4 | Audio modality | 1 week |
| 5 | Maintenance logs + RAG | 1-1.5 weeks |
| 6 | Image modality (optional) | 1 week |
| 7 | Root cause + reporting | 1 week |
| 8 | Productionization | 3-5 days |
| 9 | Frontend | separate roadmap |

This totals roughly 8-10 weeks of focused work for Phases 0-8 (backend/ML only, matching your "deal with frontend later" plan), which fits comfortably in your stated 2-3 month window with room left for the frontend phase.
