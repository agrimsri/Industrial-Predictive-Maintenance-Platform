from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cmapss_loader import SENSOR_COLUMNS, get_training_data, load_cmapss
from src.data.feature_engineering import FeatureEngineeringConfig, build_features
from src.data.rul_labeling import add_piecewise_rul, add_test_rul
from src.data.windowing import make_engine_windows


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cmapss"


def _tiny_frame() -> pd.DataFrame:
    rows = []
    for engine_id in [1, 2]:
        for cycle in range(1, 6):
            values = {
                "engine_id": engine_id,
                "cycle": cycle,
                "setting_1": float(engine_id),
                "setting_2": float(cycle % 2),
                "setting_3": 100.0,
            }
            values.update({sensor: float(engine_id * 100 + cycle + idx) for idx, sensor in enumerate(SENSOR_COLUMNS)})
            rows.append(values)
    return pd.DataFrame(rows)


def test_load_cmapss_parses_raw_files():
    data = load_cmapss("FD001", data_root=DATA_ROOT)

    assert {"engine_id", "cycle", "setting_1", "s_21"}.issubset(data.train.columns)
    assert data.train["engine_id"].dtype == int
    assert data.test_rul.index.name == "engine_id"
    assert len(data.test_rul) == data.test["engine_id"].nunique()


def test_load_cmapss_uses_ml_data_root_by_default():
    data = load_cmapss("FD001")

    assert not data.train.empty
    assert not data.test.empty


def test_train_rul_is_monotonically_decreasing_per_engine():
    labeled = add_piecewise_rul(_tiny_frame(), cap=3)

    for _, engine_frame in labeled.groupby("engine_id"):
        assert engine_frame["rul"].is_monotonic_decreasing
        assert engine_frame["rul"].iloc[-1] == 0
        assert engine_frame["rul"].max() <= 3


def test_test_rul_uses_official_final_cycle_rul():
    frame = _tiny_frame()
    final_rul = pd.Series({1: 10, 2: 20})

    labeled = add_test_rul(frame, final_rul, cap=None)

    engine_1 = labeled[labeled["engine_id"] == 1]
    assert engine_1.loc[engine_1["cycle"].eq(5), "rul"].item() == 10
    assert engine_1.loc[engine_1["cycle"].eq(1), "rul"].item() == 14


def test_feature_engineering_drops_flat_sensors_and_adds_rolling_features():
    labeled = add_piecewise_rul(_tiny_frame(), cap=5)
    result = build_features(
        labeled,
        labeled,
        config=FeatureEngineeringConfig(rolling_windows=(3,), normalize_by_regime=False),
    )

    assert "s_1" not in result.train.columns
    assert "s_2_roll3_mean" in result.train.columns
    assert "s_2_roll3_std" in result.train.columns
    assert not result.train.isna().any().any()


def test_regime_normalization_centers_train_groups():
    frame = pd.concat([_tiny_frame(), _tiny_frame().assign(engine_id=lambda df: df["engine_id"] + 2)])
    frame.loc[frame["engine_id"].isin([3, 4]), "setting_1"] = 50.0
    labeled = add_piecewise_rul(frame, cap=5)
    result = build_features(
        labeled,
        labeled,
        config=FeatureEngineeringConfig(rolling_windows=(), normalize_by_regime=True, n_regimes=2),
    )

    grouped_means = result.train.groupby("operating_regime")["s_2"].mean().abs()
    assert (grouped_means < 1e-6).all()


def test_windowing_never_crosses_engine_boundaries():
    labeled = add_piecewise_rul(_tiny_frame(), cap=5)
    windows, targets, metadata = make_engine_windows(
        labeled,
        feature_columns=["s_2", "s_3"],
        target_column="rul",
        window_size=3,
        stride=2,
    )

    assert windows.shape == (4, 3, 2)
    assert targets.shape == (4,)
    assert all(item.end_cycle - item.start_cycle == 2 for item in metadata)
    assert [item.engine_id for item in metadata] == [1, 1, 2, 2]


def test_get_training_data_returns_tabular_and_sequence_splits():
    data = get_training_data(
        "FD001",
        data_root=DATA_ROOT,
        rolling_windows=(5,),
        window_size=20,
        window_stride=10,
    )

    assert len(data.X_train) == len(data.y_train)
    assert len(data.X_test) == len(data.y_test)
    assert data.y_train.nunique() > 1
    assert data.train_windows.ndim == 3
    assert data.train_windows.shape[1] == 20
    assert len(data.train_window_metadata) == len(data.y_train_windows)
    assert len(data.test_window_metadata) == len(data.y_test_windows)
    assert np.isfinite(data.X_train.to_numpy()).all()
