"""Reusable C-MAPSS feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SETTING_COLUMNS = [f"setting_{idx}" for idx in range(1, 4)]
SENSOR_COLUMNS = [f"s_{idx}" for idx in range(1, 22)]
DEFAULT_DROP_SENSORS = ("s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19")


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    drop_sensors: tuple[str, ...] = DEFAULT_DROP_SENSORS
    rolling_windows: tuple[int, ...] = (5, 10)
    normalize_by_regime: bool = False
    n_regimes: int = 6
    random_state: int = 42


@dataclass(frozen=True)
class NormalizationStats:
    columns: tuple[str, ...]
    means: pd.DataFrame
    stds: pd.DataFrame
    global_means: pd.Series
    global_stds: pd.Series


@dataclass(frozen=True)
class FeatureEngineeringResult:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: tuple[str, ...]
    sensor_columns: tuple[str, ...]
    rolling_columns: tuple[str, ...]
    scaler: NormalizationStats
    regime_model: KMeans | None = field(repr=False)


def active_sensor_columns(drop_sensors: tuple[str, ...] = DEFAULT_DROP_SENSORS) -> list[str]:
    return [column for column in SENSOR_COLUMNS if column not in set(drop_sensors)]


def add_operating_regime(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_regimes: int = 6,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """Cluster operating settings into regimes, fitted on train settings only."""

    train_out = train.copy()
    test_out = test.copy()
    model = KMeans(n_clusters=n_regimes, n_init=10, random_state=random_state)
    train_out["operating_regime"] = model.fit_predict(train_out[SETTING_COLUMNS])
    test_out["operating_regime"] = model.predict(test_out[SETTING_COLUMNS])
    return train_out, test_out, model


def add_rolling_features(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    windows: tuple[int, ...] = (5, 10),
    engine_column: str = "engine_id",
) -> tuple[pd.DataFrame, list[str]]:
    """Add per-engine rolling mean/std features without crossing engine boundaries."""

    out = frame.sort_values([engine_column, "cycle"]).copy()
    rolling_columns: list[str] = []

    grouped = out.groupby(engine_column, sort=False)
    for window in windows:
        if window <= 0:
            raise ValueError("rolling windows must be positive")
        rolled = grouped[sensor_columns].rolling(window=window, min_periods=1)

        means = rolled.mean().reset_index(level=0, drop=True)
        mean_columns = [f"{column}_roll{window}_mean" for column in sensor_columns]
        means.columns = mean_columns

        stds = rolled.std().reset_index(level=0, drop=True).fillna(0.0)
        std_columns = [f"{column}_roll{window}_std" for column in sensor_columns]
        stds.columns = std_columns

        out = pd.concat([out, means, stds], axis=1)
        rolling_columns.extend(mean_columns + std_columns)

    return out, rolling_columns


def _fit_normalization(
    train: pd.DataFrame,
    columns: list[str],
    by_regime: bool,
) -> NormalizationStats:
    if by_regime:
        means = train.groupby("operating_regime")[columns].mean()
        stds = train.groupby("operating_regime")[columns].std().replace(0, 1.0).fillna(1.0)
    else:
        means = pd.DataFrame([train[columns].mean()], index=pd.Index(["global"], name="scope"))
        stds = pd.DataFrame([train[columns].std().replace(0, 1.0).fillna(1.0)], index=pd.Index(["global"], name="scope"))

    global_means = train[columns].mean()
    global_stds = train[columns].std().replace(0, 1.0).fillna(1.0)
    return NormalizationStats(
        columns=tuple(columns),
        means=means,
        stds=stds,
        global_means=global_means,
        global_stds=global_stds,
    )


def _normalize(frame: pd.DataFrame, stats: NormalizationStats, by_regime: bool) -> pd.DataFrame:
    out = frame.copy()
    columns = list(stats.columns)
    if by_regime:
        normalized = []
        for regime, chunk in out.groupby("operating_regime", sort=False):
            means = stats.means.loc[regime] if regime in stats.means.index else stats.global_means
            stds = stats.stds.loc[regime] if regime in stats.stds.index else stats.global_stds
            part = chunk.copy()
            part[columns] = (part[columns] - means) / stds
            normalized.append(part)
        return pd.concat(normalized).sort_index()

    out[columns] = (out[columns] - stats.global_means) / stats.global_stds
    return out


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: FeatureEngineeringConfig | None = None,
) -> FeatureEngineeringResult:
    """Drop flat sensors, add rolling statistics, and normalize features."""

    config = config or FeatureEngineeringConfig()
    sensor_columns = active_sensor_columns(config.drop_sensors)
    train_out = train.drop(columns=list(config.drop_sensors), errors="ignore").copy()
    test_out = test.drop(columns=list(config.drop_sensors), errors="ignore").copy()

    regime_model: KMeans | None = None
    if config.normalize_by_regime:
        train_out, test_out, regime_model = add_operating_regime(
            train_out,
            test_out,
            n_regimes=config.n_regimes,
            random_state=config.random_state,
        )

    train_out, rolling_columns = add_rolling_features(train_out, sensor_columns, config.rolling_windows)
    test_out, _ = add_rolling_features(test_out, sensor_columns, config.rolling_windows)

    base_feature_columns = SETTING_COLUMNS + sensor_columns + rolling_columns
    if config.normalize_by_regime:
        base_feature_columns = ["operating_regime"] + base_feature_columns
        columns_to_scale = sensor_columns + rolling_columns
    else:
        columns_to_scale = sensor_columns + rolling_columns

    scaler = _fit_normalization(train_out, columns_to_scale, by_regime=config.normalize_by_regime)
    train_out = _normalize(train_out, scaler, by_regime=config.normalize_by_regime)
    test_out = _normalize(test_out, scaler, by_regime=config.normalize_by_regime)

    train_out = train_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_out = test_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ordered = ["engine_id", "cycle", *base_feature_columns, "rul"]
    return FeatureEngineeringResult(
        train=train_out[ordered],
        test=test_out[ordered],
        feature_columns=tuple(base_feature_columns),
        sensor_columns=tuple(sensor_columns),
        rolling_columns=tuple(rolling_columns),
        scaler=scaler,
        regime_model=regime_model,
    )
