from __future__ import annotations

import numpy as np
import torch

from src.explain import SHAPExplainer


class _FakeGradientExplainer:
    def shap_values(self, window: torch.Tensor) -> np.ndarray:
        return np.ones(tuple(window.shape), dtype=np.float32)


class _SequenceModel(torch.nn.Module):
    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return window.mean(dim=(1, 2), keepdim=True)


def test_explain_sequence_uses_retained_model_for_prediction() -> None:
    explainer = SHAPExplainer(
        shap_explainer=_FakeGradientExplainer(),
        feature_columns=["sensor_1", "sensor_2"],
        model_name="gru_rul",
        explainer_type="GradientExplainer",
        background_n=2,
        is_sequence=True,
    )
    explainer._model = _SequenceModel()
    explainer._device = torch.device("cpu")
    explainer._dataset = "FD001"

    result = explainer.explain_sequence(np.full((1, 3, 2), 6.0, dtype=np.float32))

    assert result.rul_prediction == 6.0
    assert result.shap_values_full == [[1.0, 1.0]] * 3
