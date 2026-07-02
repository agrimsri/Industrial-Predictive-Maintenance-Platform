import numpy as np
import torch

from src.data.windowing import WindowMetadata
from src.models.lstm_rul import RulSequenceRegressor, select_final_test_windows


def test_rul_sequence_regressor_outputs_one_prediction_per_window():
    model = RulSequenceRegressor(input_size=6, hidden_size=8, num_layers=1, dropout=0.0, model_type="lstm")
    inputs = torch.zeros((4, 12, 6), dtype=torch.float32)

    outputs = model(inputs)

    assert outputs.shape == (4, 1)


def test_select_final_test_windows_keeps_latest_window_per_engine():
    windows = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
    targets = np.array([50, 40, 30, 20, 10], dtype=np.float32)
    metadata = [
        WindowMetadata(engine_id=1, start_cycle=1, end_cycle=3),
        WindowMetadata(engine_id=1, start_cycle=2, end_cycle=4),
        WindowMetadata(engine_id=2, start_cycle=1, end_cycle=3),
        WindowMetadata(engine_id=2, start_cycle=2, end_cycle=4),
        WindowMetadata(engine_id=2, start_cycle=3, end_cycle=5),
    ]

    final_windows, final_targets = select_final_test_windows(windows, targets, metadata)

    assert final_windows.shape == (2, 3, 2)
    assert final_targets.tolist() == [40.0, 10.0]
    assert np.array_equal(final_windows[0], windows[1])
    assert np.array_equal(final_windows[1], windows[4])
