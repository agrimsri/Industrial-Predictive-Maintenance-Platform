"""Pydantic schemas for the model-serving API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "ipmp-model-serving"


class ModelMetadata(BaseModel):
    model_name: str
    dataset: str
    version: str
    created_at: str
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    feature_columns: list[str] = Field(default_factory=list)
    description: str = ""
    artifact_available: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelMetadata]


class RULPredictionRequest(BaseModel):
    dataset: str = "FD001"
    model: str = "xgboost"
    version: str = "latest"
    features: dict[str, float] | None = None
    window: list[list[float]] | None = None

    @model_validator(mode="after")
    def require_features_or_window(self) -> "RULPredictionRequest":
        if self.features is None and self.window is None:
            raise ValueError("Provide either 'features' for tabular models or 'window' for sequence models.")
        return self


class RULPredictionResponse(BaseModel):
    model_name: str
    dataset: str
    model_version: str
    rul_prediction: float
    explained: bool = False
    uncertainty: float | None = None


class ExplanationMetadata(BaseModel):
    explainer_type: str
    background_n: int
    window_shape: list[int] | None = None
    shap_values_1d: list[float] | None = None
    explained_at: str


class ExplanationResponse(RULPredictionResponse):
    feature_importances: dict[str, float]
    shap_values_full: list[list[float]] | None = None
    explanation_metadata: ExplanationMetadata
