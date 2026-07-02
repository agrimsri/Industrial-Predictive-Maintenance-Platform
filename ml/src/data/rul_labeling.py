"""Remaining Useful Life label generation for C-MAPSS."""

from __future__ import annotations

import pandas as pd


def apply_rul_cap(rul: pd.Series, cap: int | None = 125) -> pd.Series:
    """Apply the standard piecewise-linear RUL cap used in C-MAPSS work."""

    if cap is None:
        return rul.astype(int)
    if cap <= 0:
        raise ValueError("rul cap must be positive")
    return rul.clip(upper=cap).astype(int)


def add_piecewise_rul(
    frame: pd.DataFrame,
    cap: int | None = 125,
    engine_column: str = "engine_id",
    cycle_column: str = "cycle",
) -> pd.DataFrame:
    """Add capped train RUL labels from each engine's observed failure cycle."""

    labeled = frame.copy()
    max_cycle = labeled.groupby(engine_column)[cycle_column].transform("max")
    raw_rul = max_cycle - labeled[cycle_column]
    labeled["rul"] = apply_rul_cap(raw_rul, cap=cap)
    return labeled


def add_test_rul(
    frame: pd.DataFrame,
    final_rul_by_engine: pd.Series,
    cap: int | None = 125,
    engine_column: str = "engine_id",
    cycle_column: str = "cycle",
) -> pd.DataFrame:
    """Add capped RUL labels to test data using official final-cycle RUL."""

    labeled = frame.copy()
    max_observed_cycle = labeled.groupby(engine_column)[cycle_column].transform("max")
    engine_final_rul = labeled[engine_column].map(final_rul_by_engine)
    if engine_final_rul.isna().any():
        missing = sorted(labeled.loc[engine_final_rul.isna(), engine_column].unique())
        raise ValueError(f"Missing official test RUL for engine ids: {missing}")

    raw_rul = (max_observed_cycle - labeled[cycle_column]) + engine_final_rul.astype(int)
    labeled["rul"] = apply_rul_cap(raw_rul, cap=cap)
    return labeled
