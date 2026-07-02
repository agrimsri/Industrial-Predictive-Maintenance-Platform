"""Small hyperparameter search helpers for baseline models."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import evaluate_rul


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def validation_grid_search(
    estimator_factory: Callable[[dict[str, Any]], Any],
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict[str, list[Any]],
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select params using a small holdout set and RMSE as the objective."""

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
    )

    search_log: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_rmse: float | None = None

    for params in expand_grid(param_grid):
        model = estimator_factory(params)
        model.fit(X_fit, y_fit)
        predictions = model.predict(X_valid)
        metrics = evaluate_rul(y_valid.to_numpy(), predictions).to_dict()
        search_log.append({"params": params, "metrics": metrics})
        if best_rmse is None or metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_params = params

    if best_params is None:
        raise ValueError("Parameter grid produced no candidates")
    return best_params, search_log
