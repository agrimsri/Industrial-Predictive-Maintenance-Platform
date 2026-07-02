import json

import joblib
import numpy as np

from src.evaluation import evaluate_rul, nasa_score
from src.models.registry import ModelRegistry
from src.models.search import expand_grid


class TinyModel:
    def predict(self, values):
        return np.zeros(len(values))


def test_nasa_score_penalizes_late_predictions_more_than_early_predictions():
    y_true = np.array([100.0])

    early_score = nasa_score(y_true, np.array([90.0]))
    late_score = nasa_score(y_true, np.array([110.0]))

    assert late_score > early_score


def test_evaluate_rul_returns_expected_keys():
    metrics = evaluate_rul(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 2.0])).to_dict()

    assert set(metrics) == {"rmse", "mae", "r2", "nasa_score"}
    assert metrics["rmse"] > 0


def test_model_registry_saves_artifact_metadata_and_latest_pointer(tmp_path):
    registry = ModelRegistry(root=tmp_path)

    record = registry.save(
        model=TinyModel(),
        model_name="tiny",
        dataset="FD001",
        metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9, "nasa_score": 2.0},
        params={"alpha": 1},
        feature_columns=["s_2"],
        description="test model",
    )

    metadata = json.loads((tmp_path / "tiny" / "FD001" / record.version / "metadata.json").read_text())
    latest = json.loads((tmp_path / "tiny" / "FD001" / "latest.json").read_text())
    loaded_model = joblib.load(record.artifact_path)

    assert metadata["model_name"] == "tiny"
    assert latest["version"] == record.version
    assert loaded_model.predict([[1], [2]]).tolist() == [0.0, 0.0]


def test_expand_grid_returns_cartesian_product():
    candidates = expand_grid({"a": [1, 2], "b": ["x", "y"]})

    assert candidates == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
