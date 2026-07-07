# Model Serving API

Standalone FastAPI service for registered RUL prediction models.

## Run Locally

```bash
PYTHONPATH=serving uvicorn app.main:app --reload --port 8001
```

## Docker

```bash
docker build -f serving/Dockerfile -t ipmp-serving .
docker run --rm -p 8001:8001 ipmp-serving
```

The Docker image copies `ml/models/registry/` so committed metadata is available. Real model artifacts are git-ignored; build the image from a workspace where the desired `model.joblib` artifact exists.

## Health

```bash
curl http://localhost:8001/health
```

## List Registered Models

```bash
curl "http://localhost:8001/models?model=xgboost&dataset=FD001"
```

## Predict RUL

Create a payload using the registered feature contract:

```bash
python - <<'PY'
import json
from pathlib import Path

metadata = json.loads(Path("ml/models/registry/xgboost/FD001/latest.json").read_text())
metadata_path = Path(metadata["metadata_path"])
if not metadata_path.exists():
    metadata_path = Path("ml/models/registry/xgboost/FD001") / metadata["version"] / "metadata.json"
metadata = json.loads(metadata_path.read_text())
payload = {
    "dataset": "FD001",
    "model": "xgboost",
    "version": metadata["version"],
    "features": {name: 0.0 for name in metadata["feature_columns"]},
}
Path("/tmp/rul-payload.json").write_text(json.dumps(payload, indent=2))
PY

curl -X POST http://localhost:8001/predict/rul \
  -H "Content-Type: application/json" \
  --data @/tmp/rul-payload.json
```

For production-like calls, replace the zero values with engineered feature values from the Phase 1 feature pipeline.

## Tests

```bash
pip install -r serving/requirements-dev.txt
PYTHONPATH=serving pytest serving/tests
```
