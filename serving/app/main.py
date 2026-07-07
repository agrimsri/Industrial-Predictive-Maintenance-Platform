"""FastAPI application for standalone model serving."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from .model_loader import FeatureValidationError, ModelLoader, ModelNotFoundError, UnsupportedModelError
from .schemas import HealthResponse, ModelMetadata, ModelsResponse, RULPredictionRequest, RULPredictionResponse


app = FastAPI(
    title="Industrial Predictive Maintenance Model Serving",
    version="0.2.1",
    description="Standalone serving API for registered RUL prediction models.",
)
loader = ModelLoader()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/models", response_model=ModelsResponse)
def list_models(
    model: str | None = Query(default=None, description="Optional registry model name filter."),
    dataset: str | None = Query(default=None, description="Optional C-MAPSS dataset filter, for example FD001."),
) -> ModelsResponse:
    models = [ModelMetadata(**metadata) for metadata in loader.list_models(model_name=model, dataset=dataset)]
    return ModelsResponse(models=models)


@app.post("/predict/rul", response_model=RULPredictionResponse)
def predict_rul(request: RULPredictionRequest) -> RULPredictionResponse:
    try:
        registered_model = loader.load(request.model, request.dataset, request.version)
        prediction = registered_model.predict_rul(request.features)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnsupportedModelError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RULPredictionResponse(
        model_name=registered_model.model_name,
        dataset=registered_model.dataset,
        model_version=registered_model.version,
        rul_prediction=prediction,
        explained=False,
        uncertainty=None,
    )
