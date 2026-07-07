from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.main import health, list_models, predict_rul
from app.model_loader import ModelLoader
from app.schemas import RULPredictionRequest


def test_health() -> None:
    response = health()

    assert response.status == "ok"


def test_models_lists_registry_metadata() -> None:
    response = list_models(model="xgboost", dataset="FD001")

    models = response.models
    assert models
    assert models[0].model_name == "xgboost"
    assert models[0].dataset == "FD001"
    assert "rmse" in models[0].metrics


def test_xgboost_prediction_with_registered_feature_contract() -> None:
    loader = ModelLoader()
    metadata = loader.list_models(model_name="xgboost", dataset="FD001")[0]
    request = RULPredictionRequest(
        dataset="FD001",
        model="xgboost",
        version=metadata["version"],
        features={feature: 0.0 for feature in metadata["feature_columns"]},
    )

    response = predict_rul(request)

    assert response.model_name == "xgboost"
    assert response.dataset == "FD001"
    assert isinstance(response.rul_prediction, float)
    assert response.explained is False


def test_prediction_rejects_missing_features() -> None:
    request = RULPredictionRequest(
        dataset="FD001",
        model="xgboost",
        features={"setting_1": 0.0},
    )

    with pytest.raises(HTTPException) as exc_info:
        predict_rul(request)

    assert exc_info.value.status_code == 422
    assert "Missing" in exc_info.value.detail
