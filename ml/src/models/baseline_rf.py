"""Random Forest baseline for C-MAPSS RUL prediction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestRegressor

from src.data import get_training_data
from src.evaluation import RegressionMetrics, evaluate_rul
from src.models.registry import ModelRegistry, ModelVersion
from src.models.search import validation_grid_search


@dataclass(frozen=True)
class TrainingResult:
    model: RandomForestRegressor
    metrics: RegressionMetrics
    registry_record: ModelVersion | None
    feature_columns: list[str]
    search_log: list[dict[str, Any]]


def default_params(random_state: int = 42) -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": random_state,
    }


def default_param_grid() -> dict[str, list[Any]]:
    return {
        "n_estimators": [200, 400],
        "max_depth": [12, None],
        "min_samples_leaf": [1, 2],
    }


def train_random_forest(
    dataset: str = "FD001",
    data_root: Path | str | None = None,
    registry_root: Path | str | None = None,
    save_model: bool = True,
    params: dict[str, Any] | None = None,
    tune: bool = True,
    param_grid: dict[str, list[Any]] | None = None,
) -> TrainingResult:
    """Train and optionally register a Random Forest RUL baseline."""

    training_data = get_training_data(dataset=dataset, data_root=data_root)
    model_params = default_params()
    if params:
        model_params.update(params)

    search_log: list[dict[str, Any]] = []
    if tune:
        grid = param_grid or default_param_grid()

        def factory(candidate_params: dict[str, Any]) -> RandomForestRegressor:
            merged = {**model_params, **candidate_params}
            return RandomForestRegressor(**merged)

        best_params, search_log = validation_grid_search(
            factory,
            training_data.X_train,
            training_data.y_train,
            grid,
            random_state=model_params["random_state"],
        )
        model_params.update(best_params)

    model = RandomForestRegressor(**model_params)
    model.fit(training_data.X_train, training_data.y_train)

    predictions = model.predict(training_data.X_test)
    metrics = evaluate_rul(training_data.y_test.to_numpy(), predictions)

    registry_record = None
    if save_model:
        registry = ModelRegistry(root=registry_root) if registry_root is not None else ModelRegistry()
        registry_record = registry.save(
            model=model,
            model_name="random_forest",
            dataset=dataset,
            metrics=metrics.to_dict(),
            params={"final": model_params, "search": search_log},
            feature_columns=list(training_data.X_train.columns),
            description="Random Forest baseline trained on engineered C-MAPSS cycle-level features.",
        )

    return TrainingResult(
        model=model,
        metrics=metrics,
        registry_record=registry_record,
        feature_columns=list(training_data.X_train.columns),
        search_log=search_log,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest RUL baseline.")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--registry-root", default=None)
    parser.add_argument("--no-save", action="store_true", help="Train/evaluate without writing a registry artifact.")
    parser.add_argument("--no-tune", action="store_true", help="Skip the validation grid search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_random_forest(
        dataset=args.dataset,
        data_root=args.data_root,
        registry_root=args.registry_root,
        save_model=not args.no_save,
        tune=not args.no_tune,
    )
    print(result.metrics.to_dict())
    if result.registry_record is not None:
        print(f"saved: {result.registry_record.metadata_path}")


if __name__ == "__main__":
    main()
