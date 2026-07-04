import pytest
import torch

from src.models.patchtst_rul import PatchTSTRegressor


def test_patchtst_regressor_outputs_one_prediction_per_window():
    model = PatchTSTRegressor(
        input_size=6,
        window_size=30,
        patch_length=10,
        patch_stride=5,
        d_model=16,
        num_layers=1,
        num_heads=4,
        dim_feedforward=32,
        dropout=0.0,
        head_dropout=0.0,
    )
    inputs = torch.zeros((4, 30, 6), dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (4, 1)


def test_patchtst_rejects_patch_longer_than_window():
    with pytest.raises(ValueError, match="patch_length cannot exceed window_size"):
        PatchTSTRegressor(input_size=6, window_size=8, patch_length=10)


def test_patchtst_requires_attention_heads_to_divide_model_width():
    with pytest.raises(ValueError, match="d_model must be divisible by num_heads"):
        PatchTSTRegressor(input_size=6, d_model=30, num_heads=8)
