"""RUL regression metrics used by C-MAPSS literature."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    """Common RUL regression metrics plus NASA's asymmetric score."""

    rmse: float
    mae: float
    r2: float
    nasa_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the NASA C-MAPSS asymmetric scoring function.

    Error is prediction minus truth. Over-estimating RUL means predicting that an
    engine has more life than it really has, so late maintenance is penalized
    more strongly than conservative early maintenance.
    """

    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    error = pred - truth

    early = np.exp(-error[error < 0] / 13.0) - 1.0
    late = np.exp(error[error >= 0] / 10.0) - 1.0
    return float(np.sum(early) + np.sum(late))


def evaluate_rul(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Evaluate RUL predictions with symmetric and asymmetric metrics."""

    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(truth, pred)
    return RegressionMetrics(
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(truth, pred)),
        r2=float(r2_score(truth, pred)),
        nasa_score=nasa_score(truth, pred),
    )
