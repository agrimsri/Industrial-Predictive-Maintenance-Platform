"""SHAP-based explainability for C-MAPSS RUL models.

Supports three model families:

- **XGBoost** (tabular): uses ``shap.TreeExplainer`` — exact, fast, and the
  standard choice for gradient-boosted trees.
- **LSTM / GRU** (sequence): uses ``shap.GradientExplainer`` — gradient-based
  attribution that works reliably with any differentiable PyTorch model.
- **PatchTST** (sequence): also uses ``shap.GradientExplainer`` via a thin
  wrapper that flattens the channel-patch reshaping so SHAP can trace through
  the forward pass.

The ``ExplanationResult`` dataclass is the serialisation contract between this
module and the Milestone 2.x serving API.  Every field maps 1-to-1 with the
``/predict/rul/explain`` JSON schema documented in ``docs/SCHEMAS.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.models.registry import DEFAULT_REGISTRY_ROOT


# ---------------------------------------------------------------------------
# Public data contract
# ---------------------------------------------------------------------------


@dataclass
class ExplanationResult:
    """Per-prediction explanation returned by any explainer.

    Attributes
    ----------
    model_name:
        Registry name of the model (e.g. ``"xgboost"``, ``"gru_rul"``).
    dataset:
        C-MAPSS subset the model was trained on (e.g. ``"FD001"``).
    rul_prediction:
        Scalar RUL estimate produced by the model for this sample.
    feature_importances:
        Mean-absolute-SHAP value per feature, sorted descending.
        Keys are feature column names; values are non-negative floats.
    shap_values_full:
        For sequence models: SHAP matrix of shape ``(T, F)`` where *T* is the
        window length and *F* is the number of feature channels.  The value at
        ``[t, f]`` is the SHAP attribution for feature *f* at timestep *t*.
        ``None`` for tabular (XGBoost) models.
    explanation_metadata:
        Provenance information: explainer type, background sample count, and
        ISO-8601 timestamp.
    """

    model_name: str
    dataset: str
    rul_prediction: float
    feature_importances: dict[str, float]
    shap_values_full: list[list[float]] | None = None
    explanation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict matching ``docs/SCHEMAS.md``."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# SHAPExplainer
# ---------------------------------------------------------------------------


class SHAPExplainer:
    """Unified SHAP explainer that wraps different backends per model family.

    Instantiate via the class-method factories rather than ``__init__``.

    Examples
    --------
    **XGBoost**::

        explainer = SHAPExplainer.for_xgboost(model, feature_columns)
        result = explainer.explain_tabular(X_test.iloc[[0]])

    **LSTM / GRU**::

        explainer = SHAPExplainer.for_lstm_gru(
            model, background_windows[:50], feature_columns
        )
        result = explainer.explain_sequence(window[np.newaxis])

    **PatchTST**::

        explainer = SHAPExplainer.for_patchtst(
            model, background_windows[:50], feature_columns
        )
        result = explainer.explain_sequence(window[np.newaxis])
    """

    def __init__(
        self,
        shap_explainer: Any,
        feature_columns: list[str],
        model_name: str,
        explainer_type: str,
        background_n: int,
        is_sequence: bool,
    ) -> None:
        self._explainer = shap_explainer
        self.feature_columns = feature_columns
        self.model_name = model_name
        self.explainer_type = explainer_type
        self.background_n = background_n
        self.is_sequence = is_sequence

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_xgboost(
        cls,
        model: Any,
        feature_columns: list[str],
        dataset: str = "FD001",
    ) -> "SHAPExplainer":
        """Build a TreeExplainer for an XGBoost model.

        Parameters
        ----------
        model:
            A trained ``XGBRegressor``.
        feature_columns:
            Ordered list of feature names matching the model's training columns.
        dataset:
            C-MAPSS subset name (for provenance only).
        """
        import shap

        explainer = shap.TreeExplainer(model)
        instance = cls(
            shap_explainer=explainer,
            feature_columns=feature_columns,
            model_name="xgboost",
            explainer_type="TreeExplainer",
            background_n=0,  # tree explainer doesn't use a background set
            is_sequence=False,
        )
        instance._dataset = dataset
        return instance

    @classmethod
    def for_lstm_gru(
        cls,
        model: Any,
        background_windows: np.ndarray,
        feature_columns: list[str],
        dataset: str = "FD001",
        model_name: str = "lstm_rul",
        device_name: str = "cpu",
    ) -> "SHAPExplainer":
        """Build a GradientExplainer for an LSTM or GRU model.

        Parameters
        ----------
        model:
            A trained ``RulSequenceRegressor`` PyTorch module.
        background_windows:
            Reference dataset for SHAP — array of shape ``(N, T, F)``.
            50–200 random samples from the training windows work well.
        feature_columns:
            Ordered list of feature names.
        dataset:
            C-MAPSS subset name.
        model_name:
            Registry name (``"lstm_rul"`` or ``"gru_rul"``).
        device_name:
            Torch device to run on (``"cpu"`` or ``"cuda"``).
        """
        import shap
        import torch

        device = torch.device(device_name)
        model = model.to(device).eval()
        bg_tensor = torch.as_tensor(background_windows, dtype=torch.float32).to(device)
        explainer = shap.GradientExplainer(model, bg_tensor)
        instance = cls(
            shap_explainer=explainer,
            feature_columns=feature_columns,
            model_name=model_name,
            explainer_type="GradientExplainer",
            background_n=len(background_windows),
            is_sequence=True,
        )
        instance._dataset = dataset
        instance._device = device
        return instance

    @classmethod
    def for_patchtst(
        cls,
        model: Any,
        background_windows: np.ndarray,
        feature_columns: list[str],
        dataset: str = "FD001",
        device_name: str = "cpu",
    ) -> "SHAPExplainer":
        """Build a GradientExplainer for a PatchTST model.

        PatchTST's forward uses an internal channel-patch reshape that is fully
        differentiable, so ``GradientExplainer`` can trace through it unchanged.
        If the gradient path breaks (e.g. due to a custom CUDA extension), fall
        back to ``KernelExplainer`` by wrapping the model in a numpy-callable
        and calling ``shap.KernelExplainer`` directly.

        Parameters
        ----------
        model:
            A trained ``PatchTSTRegressor`` PyTorch module.
        background_windows:
            Reference dataset — array of shape ``(N, T, F)``.
        feature_columns:
            Ordered list of feature names.
        dataset:
            C-MAPSS subset name.
        device_name:
            Torch device to run on.
        """
        import shap
        import torch

        device = torch.device(device_name)
        model = model.to(device).eval()
        bg_tensor = torch.as_tensor(background_windows, dtype=torch.float32).to(device)
        explainer = shap.GradientExplainer(model, bg_tensor)
        instance = cls(
            shap_explainer=explainer,
            feature_columns=feature_columns,
            model_name="patchtst_rul",
            explainer_type="GradientExplainer",
            background_n=len(background_windows),
            is_sequence=True,
        )
        instance._dataset = dataset
        instance._device = device
        return instance

    # ------------------------------------------------------------------
    # Explain methods
    # ------------------------------------------------------------------

    def explain_tabular(
        self,
        X: Any,
        prediction: float | None = None,
    ) -> ExplanationResult:
        """Explain a single tabular prediction (XGBoost).

        Parameters
        ----------
        X:
            A single-row DataFrame or 2-D array of shape ``(1, F)``.
        prediction:
            Pre-computed model prediction.  If ``None``, the XGBoost model's
            ``predict`` method is called internally.

        Returns
        -------
        ExplanationResult
            SHAP feature importances and scalar prediction.
        """
        import pandas as pd

        shap_values = self._explainer.shap_values(X)
        # shap_values shape: (1, F) for regression
        if isinstance(shap_values, list):
            # Some SHAP versions return a list for multi-output
            shap_values = shap_values[0]
        values_1d = np.asarray(shap_values).reshape(-1)

        if prediction is None:
            raw = self._explainer.expected_value
            # Fall back to model predict if available
            if hasattr(self._explainer, "model") and hasattr(self._explainer.model, "predict"):
                if isinstance(X, pd.DataFrame):
                    prediction = float(self._explainer.model.predict(X)[0])
                else:
                    prediction = float(self._explainer.model.predict(np.asarray(X))[0])
            else:
                prediction = float(raw + values_1d.sum())

        abs_importance = np.abs(values_1d)
        order = np.argsort(abs_importance)[::-1]
        feature_importances = {
            self.feature_columns[i]: float(abs_importance[i]) for i in order
        }

        return ExplanationResult(
            model_name=self.model_name,
            dataset=getattr(self, "_dataset", "unknown"),
            rul_prediction=float(prediction),
            feature_importances=feature_importances,
            shap_values_full=None,
            explanation_metadata={
                "explainer_type": self.explainer_type,
                "background_n": self.background_n,
                "shap_values_1d": values_1d.tolist(),
                "explained_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def explain_sequence(
        self,
        window: np.ndarray,
        prediction: float | None = None,
    ) -> ExplanationResult:
        """Explain a single sequence prediction (LSTM/GRU/PatchTST).

        Parameters
        ----------
        window:
            Sensor window of shape ``(1, T, F)``.
        prediction:
            Pre-computed scalar RUL prediction.  If ``None``, the wrapped model
            is called to obtain the prediction.

        Returns
        -------
        ExplanationResult
            Per-feature importance (mean |SHAP| over time) and the full
            ``(T, F)`` SHAP matrix.
        """
        import torch

        window_arr = np.asarray(window, dtype=np.float32)
        if window_arr.ndim == 2:
            window_arr = window_arr[np.newaxis]  # ensure (1, T, F)

        device = getattr(self, "_device", torch.device("cpu"))
        window_tensor = torch.as_tensor(window_arr).to(device)

        shap_values = self._explainer.shap_values(window_tensor)
        # GradientExplainer returns ndarray of shape (1, T, F)
        sv = np.asarray(shap_values).reshape(window_arr.shape)  # (1, T, F)
        sv_2d = sv[0]  # (T, F)

        if prediction is None:
            model = self._explainer.model
            with torch.no_grad():
                prediction = float(model(window_tensor).cpu().numpy().reshape(-1)[0])

        # Per-feature importance: mean absolute SHAP over all timesteps
        abs_mean_per_feature = np.abs(sv_2d).mean(axis=0)  # (F,)
        order = np.argsort(abs_mean_per_feature)[::-1]
        feature_importances = {
            self.feature_columns[i]: float(abs_mean_per_feature[i]) for i in order
        }

        return ExplanationResult(
            model_name=self.model_name,
            dataset=getattr(self, "_dataset", "unknown"),
            rul_prediction=float(prediction),
            feature_importances=feature_importances,
            shap_values_full=sv_2d.tolist(),
            explanation_metadata={
                "explainer_type": self.explainer_type,
                "background_n": self.background_n,
                "window_shape": list(window_arr.shape[1:]),
                "explained_at": datetime.now(timezone.utc).isoformat(),
            },
        )


# ---------------------------------------------------------------------------
# Convenience functional helpers
# ---------------------------------------------------------------------------


def explain_xgboost_prediction(
    model: Any,
    X_sample: Any,
    feature_columns: list[str],
    dataset: str = "FD001",
) -> ExplanationResult:
    """One-shot XGBoost explanation for a single tabular sample.

    Parameters
    ----------
    model:
        Trained ``XGBRegressor``.
    X_sample:
        Single-row DataFrame or 2-D array of shape ``(1, F)``.
    feature_columns:
        Ordered feature names.
    dataset:
        C-MAPSS subset name (provenance).
    """
    explainer = SHAPExplainer.for_xgboost(model, feature_columns, dataset=dataset)
    return explainer.explain_tabular(X_sample)


def explain_sequence_prediction(
    model: Any,
    window: np.ndarray,
    background_windows: np.ndarray,
    feature_columns: list[str],
    model_name: str = "lstm_rul",
    dataset: str = "FD001",
    device_name: str = "cpu",
) -> ExplanationResult:
    """One-shot LSTM/GRU/PatchTST explanation for a single sequence window.

    Parameters
    ----------
    model:
        Trained PyTorch module (``RulSequenceRegressor`` or
        ``PatchTSTRegressor``).
    window:
        Single window of shape ``(1, T, F)`` or ``(T, F)``.
    background_windows:
        Reference dataset array of shape ``(N, T, F)`` — 50–200 samples.
    feature_columns:
        Ordered feature names.
    model_name:
        Registry name for the model.
    dataset:
        C-MAPSS subset name (provenance).
    device_name:
        Torch device (``"cpu"`` or ``"cuda"``).
    """
    if model_name == "patchtst_rul":
        explainer = SHAPExplainer.for_patchtst(
            model, background_windows, feature_columns, dataset=dataset, device_name=device_name
        )
    else:
        explainer = SHAPExplainer.for_lstm_gru(
            model, background_windows, feature_columns,
            dataset=dataset, model_name=model_name, device_name=device_name,
        )
    return explainer.explain_sequence(window)


# ---------------------------------------------------------------------------
# Registry loaders
# ---------------------------------------------------------------------------


def load_xgboost_from_registry(
    dataset: str = "FD001",
    version: str = "latest",
    registry_root: Path | str = DEFAULT_REGISTRY_ROOT,
) -> tuple[Any, list[str]]:
    """Load a trained XGBoost model + its feature columns from the registry.

    Returns
    -------
    (model, feature_columns)
        The ``XGBRegressor`` and the list of feature names used during training.
    """
    import joblib

    registry_root = Path(registry_root)
    if version == "latest":
        pointer = registry_root / "xgboost" / dataset / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError(
                f"No XGBoost registry entry for dataset={dataset}. "
                "Run `make train-baselines` first."
            )
        meta = json.loads(pointer.read_text(encoding="utf-8"))
        meta = json.loads(Path(meta["metadata_path"]).read_text(encoding="utf-8"))
    else:
        meta_path = registry_root / "xgboost" / dataset / version / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model = joblib.load(meta["artifact_path"])
    feature_columns: list[str] = meta["feature_columns"]
    return model, feature_columns


def load_sequence_model_from_registry(
    model_name: str,
    dataset: str = "FD001",
    version: str = "latest",
    registry_root: Path | str = DEFAULT_REGISTRY_ROOT,
    device_name: str = "cpu",
) -> tuple[Any, list[str], dict[str, Any]]:
    """Load a trained LSTM/GRU/PatchTST model from the registry.

    Reconstructs the PyTorch architecture from saved config and loads
    the stored state dict.

    Parameters
    ----------
    model_name:
        Registry name — one of ``"lstm_rul"``, ``"gru_rul"``,
        ``"patchtst_rul"``.
    dataset:
        C-MAPSS subset name.
    version:
        Timestamp version string or ``"latest"``.
    registry_root:
        Root of the model registry (defaults to ``ml/models/registry``).
    device_name:
        Torch device to map the checkpoint onto.

    Returns
    -------
    (model, feature_columns, checkpoint_meta)
        The reconstructed ``nn.Module`` in eval mode, feature column list,
        and the full checkpoint metadata dict.
    """
    import torch

    registry_root = Path(registry_root)
    if version == "latest":
        pointer = registry_root / model_name / dataset / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError(
                f"No registry entry for {model_name}/{dataset}. "
                "Train the model first."
            )
        ptr = json.loads(pointer.read_text(encoding="utf-8"))
        artifact_path = Path(
            json.loads(Path(ptr["metadata_path"]).read_text(encoding="utf-8"))["artifact_path"]
        )
    else:
        artifact_path = registry_root / model_name / dataset / version / "model.pt"

    checkpoint = torch.load(artifact_path, map_location=device_name, weights_only=False)
    config = checkpoint["config"]
    input_size: int = checkpoint["input_size"]
    feature_columns: list[str] = checkpoint["feature_columns"]

    device = torch.device(device_name)

    if model_name == "patchtst_rul":
        from src.models.patchtst_rul import PatchTSTRegressor

        model = PatchTSTRegressor(
            input_size=input_size,
            window_size=config["window_size"],
            patch_length=config["patch_length"],
            patch_stride=config["patch_stride"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            head_dropout=config["head_dropout"],
            use_revin=config.get("use_revin", True),
        )
    else:
        from src.models.lstm_rul import RulSequenceRegressor

        model = RulSequenceRegressor(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            model_type=config["model_type"],
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    return model, feature_columns, checkpoint
