"""Train all classic C-MAPSS baselines and update the results table."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.baseline_rf import train_random_forest
from src.models.baseline_xgb import train_xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "docs" / "RESULTS.md"


def _format_row(model_name: str, dataset: str, metrics: dict[str, float], artifact_path: str) -> str:
    return (
        f"| {model_name} | {dataset} | "
        f"{metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
        f"{metrics['r2']:.4f} | {metrics['nasa_score']:.4f} | `{artifact_path}` |"
    )


def append_results(rows: list[str], results_path: Path = DEFAULT_RESULTS_PATH) -> None:
    if not results_path.exists():
        results_path.write_text(
            "# Model Results\n\n"
            "This table is the running leaderboard for C-MAPSS RUL experiments.\n\n"
            "| Model | Dataset | RMSE | MAE | R2 | NASA Score | Artifact |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- |\n",
            encoding="utf-8",
        )

    with results_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row}\n")


def train_all(
    dataset: str,
    data_root: str | None,
    registry_root: str | None,
    results_path: Path,
    tune: bool = True,
) -> None:
    rows: list[str] = []

    rf = train_random_forest(
        dataset=dataset,
        data_root=data_root,
        registry_root=registry_root,
        save_model=True,
        tune=tune,
    )
    rf_artifact = rf.registry_record.artifact_path if rf.registry_record else ""
    rows.append(_format_row("Random Forest", dataset, rf.metrics.to_dict(), rf_artifact))

    xgb = train_xgboost(
        dataset=dataset,
        data_root=data_root,
        registry_root=registry_root,
        save_model=True,
        tune=tune,
    )
    xgb_artifact = xgb.registry_record.artifact_path if xgb.registry_record else ""
    rows.append(_format_row("XGBoost", dataset, xgb.metrics.to_dict(), xgb_artifact))

    append_results(rows, results_path=results_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Random Forest and XGBoost C-MAPSS baselines.")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--registry-root", default=None)
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--no-tune", action="store_true", help="Skip validation grid search for faster smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_all(
        dataset=args.dataset,
        data_root=args.data_root,
        registry_root=args.registry_root,
        results_path=Path(args.results_path),
        tune=not args.no_tune,
    )


if __name__ == "__main__":
    main()
