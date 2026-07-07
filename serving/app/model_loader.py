"""Registry-aware model loading for the serving API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np


SERVING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVING_ROOT.parent
DEFAULT_REGISTRY_ROOT = REPO_ROOT / "ml" / "models" / "registry"
TABULAR_MODELS = {"xgboost", "random_forest"}


class ModelNotFoundError(RuntimeError):
    """Raised when registry metadata or an artifact cannot be found."""


class UnsupportedModelError(RuntimeError):
    """Raised when a model family is not implemented by this serving milestone."""


class FeatureValidationError(ValueError):
    """Raised when a prediction payload does not match the registered feature contract."""


class RegisteredModel:
    """Loaded model plus its registry metadata."""

    def __init__(self, metadata: dict[str, Any], artifact: Any) -> None:
        self.metadata = metadata
        self.artifact = artifact

    @property
    def model_name(self) -> str:
        return str(self.metadata["model_name"])

    @property
    def dataset(self) -> str:
        return str(self.metadata["dataset"])

    @property
    def version(self) -> str:
        return str(self.metadata["version"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata.get("feature_columns", []))

    def predict_rul(self, features: dict[str, float] | None) -> float:
        if self.model_name not in TABULAR_MODELS:
            raise UnsupportedModelError(f"Prediction is not implemented for model '{self.model_name}' in milestone 2.1.")
        if features is None:
            raise FeatureValidationError("Tabular models require a 'features' object.")

        missing = [name for name in self.feature_columns if name not in features]
        extra = sorted(set(features) - set(self.feature_columns))
        if missing:
            preview = ", ".join(missing[:8])
            raise FeatureValidationError(f"Missing {len(missing)} required feature(s): {preview}")
        if extra:
            preview = ", ".join(extra[:8])
            raise FeatureValidationError(f"Unexpected feature(s): {preview}")

        row = np.array([[float(features[name]) for name in self.feature_columns]], dtype=float)
        prediction = self.artifact.predict(row)
        return float(np.asarray(prediction).reshape(-1)[0])


class ModelLoader:
    """List and load models from the file-based ML registry."""

    def __init__(self, registry_root: Path | str = DEFAULT_REGISTRY_ROOT) -> None:
        self.registry_root = Path(registry_root)

    def list_models(self, model_name: str | None = None, dataset: str | None = None) -> list[dict[str, Any]]:
        root = self.registry_root / model_name if model_name else self.registry_root
        if not root.exists():
            return []

        records: list[dict[str, Any]] = []
        for metadata_path in root.glob("**/metadata.json"):
            metadata = self._read_json(metadata_path)
            if dataset is not None and metadata.get("dataset") != dataset:
                continue
            metadata = dict(metadata)
            metadata["artifact_available"] = self._artifact_path(metadata, metadata_path).exists()
            records.append(metadata)
        return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)

    def load(self, model_name: str, dataset: str, version: str = "latest") -> RegisteredModel:
        metadata_path = self._metadata_path(model_name, dataset, version)
        metadata = self._read_json(metadata_path)
        artifact_path = self._artifact_path(metadata, metadata_path)
        if not artifact_path.exists():
            raise ModelNotFoundError(f"Artifact not found for {model_name}/{dataset}/{version}: {artifact_path}")
        artifact = self._load_artifact(str(artifact_path))
        return RegisteredModel(metadata=metadata, artifact=artifact)

    def _metadata_path(self, model_name: str, dataset: str, version: str) -> Path:
        if version == "latest":
            pointer_path = self.registry_root / model_name / dataset / "latest.json"
            if not pointer_path.exists():
                raise ModelNotFoundError(f"No latest pointer found for {model_name}/{dataset}.")
            pointer = self._read_json(pointer_path)
            pointed_path = Path(str(pointer["metadata_path"]))
            if pointed_path.exists():
                return pointed_path
            return self.registry_root / model_name / dataset / str(pointer["version"]) / "metadata.json"

        metadata_path = self.registry_root / model_name / dataset / version / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"No metadata found for {model_name}/{dataset}/{version}.")
        return metadata_path

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _artifact_path(metadata: dict[str, Any], metadata_path: Path) -> Path:
        recorded_path = Path(str(metadata.get("artifact_path", "")))
        if recorded_path.exists():
            return recorded_path
        return metadata_path.parent / recorded_path.name

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_artifact(path: str) -> Any:
        return joblib.load(path)
