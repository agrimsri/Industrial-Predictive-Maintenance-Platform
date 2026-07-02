"""Load NASA C-MAPSS turbofan run-to-failure data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .feature_engineering import (
    DEFAULT_DROP_SENSORS,
    FeatureEngineeringConfig,
    FeatureEngineeringResult,
    build_features,
)
from .rul_labeling import add_piecewise_rul, add_test_rul
from .windowing import make_engine_windows

DatasetName = Literal["FD001", "FD002", "FD003", "FD004"]

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CMAPSS_ROOT = ML_ROOT / "data" / "raw" / "cmapss"

INDEX_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = [f"setting_{idx}" for idx in range(1, 4)]
SENSOR_COLUMNS = [f"s_{idx}" for idx in range(1, 22)]
CMAPSS_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


@dataclass(frozen=True)
class CmapssData:
    """Raw train/test splits plus official test RUL values."""

    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series


@dataclass(frozen=True)
class TrainingData:
    """Model-ready C-MAPSS outputs for tabular and sequence model families."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    train_windows: object
    y_train_windows: object
    test_windows: object
    y_test_windows: object
    feature_result: FeatureEngineeringResult


def _read_sensor_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    frame["engine_id"] = frame["engine_id"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    return frame


def _read_rul_file(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS RUL file not found: {path}")

    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
    values.index = values.index + 1
    values.index.name = "engine_id"
    return values.rename("final_rul").astype(int)


def load_cmapss(dataset: DatasetName = "FD001", data_root: Path | str | None = None) -> CmapssData:
    """Parse one C-MAPSS subset into structured DataFrames."""

    root = Path(data_root) if data_root is not None else DEFAULT_CMAPSS_ROOT
    dataset = dataset.upper()  # type: ignore[assignment]
    if dataset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unsupported C-MAPSS dataset: {dataset}")

    train = _read_sensor_file(root / f"train_{dataset}.txt")
    test = _read_sensor_file(root / f"test_{dataset}.txt")
    test_rul = _read_rul_file(root / f"RUL_{dataset}.txt")
    return CmapssData(train=train, test=test, test_rul=test_rul)


def _last_cycle_snapshot(features: pd.DataFrame) -> pd.DataFrame:
    idx = features.groupby("engine_id")["cycle"].idxmax()
    return features.loc[idx].sort_values("engine_id").reset_index(drop=True)


def get_training_data(
    dataset: DatasetName = "FD001",
    data_root: Path | str | None = None,
    rul_cap: int = 125,
    rolling_windows: tuple[int, ...] = (5, 10),
    window_size: int = 30,
    window_stride: int = 1,
    drop_sensors: tuple[str, ...] = DEFAULT_DROP_SENSORS,
    normalize_by_regime: bool | None = None,
) -> TrainingData:
    """Return train/test splits ready for tabular and sequence RUL models.

    Tabular outputs are last-cycle snapshots per engine. Sequence outputs are
    sliding windows over all available cycles.
    """

    raw = load_cmapss(dataset=dataset, data_root=data_root)
    normalize = dataset in {"FD002", "FD004"} if normalize_by_regime is None else normalize_by_regime

    train_labeled = add_piecewise_rul(raw.train, cap=rul_cap)
    test_labeled = add_test_rul(raw.test, raw.test_rul, cap=rul_cap)

    config = FeatureEngineeringConfig(
        drop_sensors=drop_sensors,
        rolling_windows=rolling_windows,
        normalize_by_regime=normalize,
    )
    feature_result = build_features(train_labeled, test_labeled, config=config)

    train_snapshot = _last_cycle_snapshot(feature_result.train)
    test_snapshot = _last_cycle_snapshot(feature_result.test)
    exclude = {"engine_id", "cycle", "rul", "operating_regime"}
    feature_columns = [column for column in train_snapshot.columns if column not in exclude]

    train_windows, y_train_windows, _ = make_engine_windows(
        feature_result.train,
        feature_columns=feature_columns,
        target_column="rul",
        window_size=window_size,
        stride=window_stride,
    )
    test_windows, y_test_windows, _ = make_engine_windows(
        feature_result.test,
        feature_columns=feature_columns,
        target_column="rul",
        window_size=window_size,
        stride=window_stride,
    )

    return TrainingData(
        X_train=train_snapshot[feature_columns].reset_index(drop=True),
        y_train=train_snapshot["rul"].reset_index(drop=True),
        X_test=test_snapshot[feature_columns].reset_index(drop=True),
        y_test=test_snapshot["rul"].reset_index(drop=True),
        train_features=feature_result.train,
        test_features=feature_result.test,
        train_windows=train_windows,
        y_train_windows=y_train_windows,
        test_windows=test_windows,
        y_test_windows=y_test_windows,
        feature_result=feature_result,
    )
