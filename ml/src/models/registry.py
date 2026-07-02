"""Lightweight file-based model registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_ROOT = ML_ROOT / "models" / "registry"


@dataclass(frozen=True)
class ModelVersion:
    model_name: str
    version: str
    dataset: str
    artifact_path: str
    metadata_path: str
    metrics: dict[str, float]
    params: dict[str, Any]
    feature_columns: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Persist model artifacts and JSON metadata under ml/models/registry."""

    def __init__(self, root: Path | str = DEFAULT_REGISTRY_ROOT) -> None:
        self.root = Path(root)

    def save(
        self,
        model: Any,
        model_name: str,
        dataset: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        feature_columns: list[str],
        description: str = "",
    ) -> ModelVersion:
        created_at = datetime.now(timezone.utc)
        version = created_at.strftime("%Y%m%dT%H%M%SZ")
        model_dir = self.root / model_name / dataset / version
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = model_dir / "model.joblib"
        metadata_path = model_dir / "metadata.json"
        joblib.dump(model, artifact_path)

        record = ModelVersion(
            model_name=model_name,
            version=version,
            dataset=dataset,
            artifact_path=str(artifact_path),
            metadata_path=str(metadata_path),
            metrics=metrics,
            params=params,
            feature_columns=feature_columns,
            created_at=created_at.isoformat(),
            description=description,
        )
        metadata_path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        self._write_latest_pointer(model_name, dataset, record)
        return record

    def load_metadata(self, model_name: str, dataset: str, version: str = "latest") -> dict[str, Any]:
        metadata_path = self._metadata_path(model_name, dataset, version)
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def load_model(self, model_name: str, dataset: str, version: str = "latest") -> Any:
        metadata = self.load_metadata(model_name, dataset, version)
        return joblib.load(metadata["artifact_path"])

    def list_versions(self, model_name: str | None = None, dataset: str | None = None) -> list[dict[str, Any]]:
        roots = [self.root]
        if model_name is not None:
            roots = [self.root / model_name]
        records: list[dict[str, Any]] = []
        for metadata_path in roots[0].glob("**/metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if dataset is not None and metadata["dataset"] != dataset:
                continue
            records.append(metadata)
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def _metadata_path(self, model_name: str, dataset: str, version: str) -> Path:
        if version == "latest":
            pointer = self.root / model_name / dataset / "latest.json"
            if not pointer.exists():
                raise FileNotFoundError(f"No latest model for {model_name}/{dataset}")
            latest = json.loads(pointer.read_text(encoding="utf-8"))
            return Path(latest["metadata_path"])
        return self.root / model_name / dataset / version / "metadata.json"

    def _write_latest_pointer(self, model_name: str, dataset: str, record: ModelVersion) -> None:
        pointer = self.root / model_name / dataset / "latest.json"
        pointer.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "dataset": dataset,
                    "version": record.version,
                    "metadata_path": record.metadata_path,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
