"""PyTorch LSTM/GRU sequence model for C-MAPSS RUL prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.data import get_training_data
from src.evaluation import RegressionMetrics, evaluate_rul
from src.models.registry import DEFAULT_REGISTRY_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "docs" / "RESULTS.md"


@dataclass(frozen=True)
class SequenceModelConfig:
    dataset: str = "FD001"
    model_type: Literal["lstm", "gru"] = "lstm"
    window_size: int = 30
    window_stride: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 10
    validation_fraction: float = 0.2
    random_state: int = 42
    rul_cap: int = 125
    rolling_windows: tuple[int, ...] = (5, 10)


@dataclass(frozen=True)
class SequenceTrainingResult:
    metrics: RegressionMetrics
    artifact_path: str
    metadata_path: str
    history: list[dict[str, float]]
    config: SequenceModelConfig


class RulSequenceDataset(Dataset):
    """Torch dataset wrapping fixed-length sensor windows and RUL targets."""

    def __init__(self, windows: np.ndarray, targets: np.ndarray) -> None:
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.targets[index]


class RulSequenceRegressor(nn.Module):
    """LSTM/GRU regressor that predicts RUL from the final hidden state."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_type: Literal["lstm", "gru"] = "lstm",
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        recurrent_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        self.encoder = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.encoder(inputs)
        final_state = output[:, -1, :]
        return self.regressor(final_state)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_final_test_windows(
    windows: np.ndarray,
    targets: np.ndarray,
    metadata: list[object],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the latest available test window for each engine."""

    latest_by_engine: dict[int, int] = {}
    for index, item in enumerate(metadata):
        engine_id = int(getattr(item, "engine_id"))
        end_cycle = int(getattr(item, "end_cycle"))
        current = latest_by_engine.get(engine_id)
        if current is None or end_cycle > int(getattr(metadata[current], "end_cycle")):
            latest_by_engine[engine_id] = index

    final_indices = [latest_by_engine[engine_id] for engine_id in sorted(latest_by_engine)]
    return windows[final_indices], targets[final_indices]


def _make_loaders(
    dataset: RulSequenceDataset,
    config: SequenceModelConfig,
) -> tuple[DataLoader, DataLoader]:
    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("Not enough windows to create a train/validation split")

    generator = torch.Generator().manual_seed(config.random_state)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, validation_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0

    for windows, targets in loader:
        windows = windows.to(device)
        targets = targets.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            predictions = model(windows)
            loss = criterion(predictions, targets)
            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch_size = len(targets)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def _predict(model: nn.Module, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    dataset = RulSequenceDataset(windows, np.zeros(len(windows), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_windows, _ in loader:
            batch_predictions = model(batch_windows.to(device)).cpu().numpy().reshape(-1)
            predictions.append(batch_predictions)
    return np.concatenate(predictions)


def _save_checkpoint(
    model: nn.Module,
    config: SequenceModelConfig,
    metrics: RegressionMetrics,
    history: list[dict[str, float]],
    feature_columns: list[str],
    input_size: int,
    registry_root: Path,
) -> tuple[str, str]:
    created_at = datetime.now(timezone.utc)
    version = created_at.strftime("%Y%m%dT%H%M%SZ")
    model_name = f"{config.model_type}_rul"
    model_dir = registry_root / model_name / config.dataset / version
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "input_size": input_size,
            "feature_columns": feature_columns,
            "metrics": metrics.to_dict(),
            "history": history,
            "created_at": created_at.isoformat(),
        },
        artifact_path,
    )
    metadata = {
        "model_name": model_name,
        "version": version,
        "dataset": config.dataset,
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics.to_dict(),
        "params": asdict(config),
        "feature_columns": feature_columns,
        "input_size": input_size,
        "created_at": created_at.isoformat(),
        "description": f"{config.model_type.upper()} sequence model trained on C-MAPSS sliding windows.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    latest_path = registry_root / model_name / config.dataset / "latest.json"
    latest_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "dataset": config.dataset,
                "version": version,
                "metadata_path": str(metadata_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(artifact_path), str(metadata_path)


def append_result_row(result: SequenceTrainingResult, results_path: Path = DEFAULT_RESULTS_PATH) -> None:
    if not results_path.exists():
        results_path.write_text(
            "# Model Results\n\n"
            "| Model | Dataset | RMSE | MAE | R2 | NASA Score | Artifact |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- |\n",
            encoding="utf-8",
        )

    metrics = result.metrics.to_dict()
    model_label = result.config.model_type.upper()
    row = (
        f"| {model_label} | {result.config.dataset} | "
        f"{metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
        f"{metrics['r2']:.4f} | {metrics['nasa_score']:.4f} | "
        f"`{result.artifact_path}` |\n"
    )
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def train_sequence_model(
    config: SequenceModelConfig,
    data_root: Path | str | None = None,
    registry_root: Path | str = DEFAULT_REGISTRY_ROOT,
    results_path: Path = DEFAULT_RESULTS_PATH,
    save_model: bool = True,
    append_results: bool = True,
    device_name: str | None = None,
) -> SequenceTrainingResult:
    """Train an LSTM/GRU model on windowed C-MAPSS sequences."""

    set_seed(config.random_state)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    training_data = get_training_data(
        dataset=config.dataset,
        data_root=data_root,
        rul_cap=config.rul_cap,
        rolling_windows=config.rolling_windows,
        window_size=config.window_size,
        window_stride=config.window_stride,
    )

    dataset = RulSequenceDataset(training_data.train_windows, training_data.y_train_windows)
    train_loader, validation_loader = _make_loaders(dataset, config)

    input_size = int(training_data.train_windows.shape[-1])
    model = RulSequenceRegressor(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        model_type=config.model_type,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        validation_loss = _run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    final_test_windows, final_test_targets = select_final_test_windows(
        training_data.test_windows,
        training_data.y_test_windows,
        training_data.test_window_metadata,
    )
    predictions = _predict(model, final_test_windows, config.batch_size, device)
    metrics = evaluate_rul(final_test_targets, predictions)

    artifact_path = ""
    metadata_path = ""
    if save_model:
        artifact_path, metadata_path = _save_checkpoint(
            model=model.cpu(),
            config=config,
            metrics=metrics,
            history=history,
            feature_columns=list(training_data.X_train.columns),
            input_size=input_size,
            registry_root=Path(registry_root),
        )

    result = SequenceTrainingResult(
        metrics=metrics,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        history=history,
        config=config,
    )
    if append_results:
        append_result_row(result, results_path=results_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM/GRU RUL sequence model.")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--registry-root", default=str(DEFAULT_REGISTRY_ROOT))
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SequenceModelConfig(
        dataset=args.dataset,
        model_type=args.model_type,
        window_size=args.window_size,
        window_stride=args.window_stride,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    result = train_sequence_model(
        config=config,
        data_root=args.data_root,
        registry_root=args.registry_root,
        results_path=Path(args.results_path),
        save_model=not args.no_save,
        append_results=not args.no_results,
        device_name=args.device,
    )
    print(result.metrics.to_dict())
    if result.metadata_path:
        print(f"saved: {result.metadata_path}")


if __name__ == "__main__":
    main()
