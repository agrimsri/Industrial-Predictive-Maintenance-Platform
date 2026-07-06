"""SHAP-based explainability for tree and deep learning RUL models."""

from src.explain.shap_explainer import (
    ExplanationResult,
    SHAPExplainer,
    explain_sequence_prediction,
    explain_xgboost_prediction,
    load_sequence_model_from_registry,
    load_xgboost_from_registry,
)

__all__ = [
    "ExplanationResult",
    "SHAPExplainer",
    "explain_sequence_prediction",
    "explain_xgboost_prediction",
    "load_sequence_model_from_registry",
    "load_xgboost_from_registry",
]
