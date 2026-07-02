"""Sliding-window sequence generation for RUL sequence models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowMetadata:
    engine_id: int
    start_cycle: int
    end_cycle: int


def make_engine_windows(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str = "rul",
    window_size: int = 30,
    stride: int = 1,
    engine_column: str = "engine_id",
    cycle_column: str = "cycle",
) -> tuple[np.ndarray, np.ndarray, list[WindowMetadata]]:
    """Create fixed-length windows without leaking cycles across engines."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[WindowMetadata] = []

    sorted_frame = frame.sort_values([engine_column, cycle_column])
    for engine_id, engine_frame in sorted_frame.groupby(engine_column, sort=True):
        values = engine_frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
        targets_for_engine = engine_frame[target_column].to_numpy(dtype=np.float32)
        cycles = engine_frame[cycle_column].to_numpy(dtype=int)
        if len(engine_frame) < window_size:
            continue

        for start in range(0, len(engine_frame) - window_size + 1, stride):
            end = start + window_size
            windows.append(values[start:end])
            targets.append(targets_for_engine[end - 1])
            metadata.append(
                WindowMetadata(
                    engine_id=int(engine_id),
                    start_cycle=int(cycles[start]),
                    end_cycle=int(cycles[end - 1]),
                )
            )

    if not windows:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            metadata,
        )

    return np.stack(windows), np.asarray(targets, dtype=np.float32), metadata
