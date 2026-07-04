"""PatchTST-style transformer model for C-MAPSS RUL prediction.

This implementation follows the core PatchTST adaptation for regression:
split each sensor channel into temporal patches, encode patches with a shared
Transformer encoder, then regress Remaining Useful Life from the flattened
channel embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    class tqdm:  # type: ignore[no-redef]
        """Minimal fallback so tests/imports work before installing tqdm."""

        def __init__(self, iterable, *args, **kwargs) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs) -> None:
            return None

        def write(self, message: str) -> None:
            print(message)

from src.data import get_training_data
from src.evaluation import RegressionMetrics, evaluate_rul
from src.models.lstm_rul import RulSequenceDataset, select_final_test_windows, set_seed
from src.models.registry import DEFAULT_REGISTRY_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "docs" / "RESULTS.md"


@dataclass(frozen=True)
class PatchTSTConfig:
    dataset: str = "FD001"
    window_size: int = 30
    window_stride: int = 1
    patch_length: int = 10
    patch_stride: int = 5
    d_model: int = 64
    num_layers: int = 3
    num_heads: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.2
    head_dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 10
    validation_fraction: float = 0.2
    random_state: int = 42
    rul_cap: int = 125
    rolling_windows: tuple[int, ...] = (5, 10)
    use_revin: bool = True
    show_progress: bool = True


@dataclass(frozen=True)
class PatchTSTTrainingResult:
    metrics: RegressionMetrics
    artifact_path: str
    metadata_path: str
    history: list[dict[str, float]]
    config: PatchTSTConfig


class RevIN(nn.Module):
    """Reversible instance normalization for time-series inputs."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, num_features))
        self.shift = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return ((inputs - mean) / std) * self.scale + self.shift


class PatchTSTRegressor(nn.Module):
    """Patch-based Transformer encoder for RUL regression."""

    def __init__(
        self,
        input_size: int,
        window_size: int = 30,
        patch_length: int = 10,
        patch_stride: int = 5,
        d_model: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        head_dropout: float = 0.2,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        if patch_length <= 0:
            raise ValueError("patch_length must be positive")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if patch_length > window_size:
            raise ValueError("patch_length cannot exceed window_size")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.input_size = input_size
        self.window_size = window_size
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = 1 + (window_size - patch_length) // patch_stride
        self.revin = RevIN(input_size) if use_revin else nn.Identity()

        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(input_size * self.num_patches * d_model),
            nn.Dropout(head_dropout),
            nn.Linear(input_size * self.num_patches * d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.kaiming_uniform_(self.patch_projection.weight, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        normalized = self.revin(inputs)
        patches = normalized.transpose(1, 2).unfold(dimension=-1, size=self.patch_length, step=self.patch_stride)
        patches = patches.contiguous().view(batch_size * self.input_size, self.num_patches, self.patch_length)
        encoded = self.patch_projection(patches) + self.position_embedding
        encoded = self.encoder(encoded)
        encoded = encoded.view(batch_size, self.input_size, self.num_patches, -1)
        return self.head(encoded)


def _make_loaders(
    dataset: RulSequenceDataset,
    config: PatchTSTConfig,
) -> tuple[DataLoader, DataLoader]:
    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("Not enough windows to create a train/validation split")

    generator = torch.Generator().manual_seed(config.random_state)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
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


def _predict(
    model: nn.Module,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
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
    config: PatchTSTConfig,
    metrics: RegressionMetrics,
    history: list[dict[str, float]],
    feature_columns: list[str],
    input_size: int,
    registry_root: Path,
) -> tuple[str, str]:
    created_at = datetime.now(timezone.utc)
    version = created_at.strftime("%Y%m%dT%H%M%SZ")
    model_name = "patchtst_rul"
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
        "description": "PatchTST-style transformer trained on C-MAPSS sliding windows.",
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


def append_result_row(result: PatchTSTTrainingResult, results_path: Path = DEFAULT_RESULTS_PATH) -> None:
    if not results_path.exists():
        results_path.write_text(
            "# Model Results\n\n"
            "| Model | Dataset | RMSE | MAE | R2 | NASA Score | Artifact |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- |\n",
            encoding="utf-8",
        )

    metrics = result.metrics.to_dict()
    row = (
        f"| PatchTST | {result.config.dataset} | "
        f"{metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
        f"{metrics['r2']:.4f} | {metrics['nasa_score']:.4f} | "
        f"`{result.artifact_path}` |\n"
    )
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def train_patchtst_model(
    config: PatchTSTConfig,
    data_root: Path | str | None = None,
    registry_root: Path | str = DEFAULT_REGISTRY_ROOT,
    results_path: Path = DEFAULT_RESULTS_PATH,
    save_model: bool = True,
    append_results: bool = True,
    device_name: str | None = None,
) -> PatchTSTTrainingResult:
    """Train PatchTST on windowed C-MAPSS sequences."""

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
    model = PatchTSTRegressor(
        input_size=input_size,
        window_size=config.window_size,
        patch_length=config.patch_length,
        patch_stride=config.patch_stride,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        head_dropout=config.head_dropout,
        use_revin=config.use_revin,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    epoch_progress = tqdm(range(1, config.max_epochs + 1), desc="PatchTST training", disable=not config.show_progress)
    for epoch in epoch_progress:
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
        epoch_progress.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{validation_loss:.4f}",
            best=f"{best_validation_loss:.4f}",
            lr=f"{learning_rate:.2e}",
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            if config.show_progress:
                epoch_progress.write(f"Early stopping after epoch {epoch}. Best validation loss: {best_validation_loss:.4f}")
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

    result = PatchTSTTrainingResult(
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
    parser = argparse.ArgumentParser(description="Train a PatchTST RUL sequence model.")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--registry-root", default=str(DEFAULT_REGISTRY_ROOT))
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--patch-length", type=int, default=10)
    parser.add_argument("--patch-stride", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda.")
    parser.add_argument("--no-revin", action="store_true", help="Disable RevIN input normalization.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-results", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PatchTSTConfig(
        dataset=args.dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        use_revin=not args.no_revin,
        show_progress=not args.quiet,
    )
    result = train_patchtst_model(
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
