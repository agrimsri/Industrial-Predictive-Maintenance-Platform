# API Schemas

This document is the contract between the `ml/src/explain/` module (Milestone 1.6) and the Milestone 2.x serving API.  Any field added here must be implemented in both `shap_explainer.py` and the corresponding Pydantic model in `serving/app/schemas.py`.

---

## RUL Prediction — `/predict/rul`

### Request

```json
{
  "dataset": "FD001",
  "model": "xgboost",
  "features": {
    "setting_1": 0.0,
    "setting_2": 0.0002,
    "s_2": 641.82,
    "s_3": 1589.7,
    "...": "..."
  }
}
```

For sequence models (`lstm`, `gru`, `patchtst`), the `features` field is replaced by a `window` array:

```json
{
  "dataset": "FD001",
  "model": "gru",
  "window": [
    [0.0, 0.0002, 641.82, 1589.7, "..."],
    "... (30 timesteps × n_features)"
  ]
}
```

### Response

```json
{
  "model_name": "xgboost",
  "dataset": "FD001",
  "rul_prediction": 87.4,
  "explained": false
}
```

---

## RUL Prediction with Explanation — `/predict/rul/explain`

Same request schema as `/predict/rul`.  Response includes the full `ExplanationResult`.

### Response

```json
{
  "model_name": "gru_rul",
  "dataset": "FD001",
  "rul_prediction": 52.3,
  "feature_importances": {
    "s_4_roll_mean_10": 4.812,
    "s_11_roll_mean_10": 3.941,
    "s_2_roll_std_5": 2.107,
    "s_3": 1.883,
    "...": "..."
  },
  "shap_values_full": [
    [-0.12, 0.45, -0.03, "..."],
    "... (T rows × F columns — one row per timestep in the window)"
  ],
  "explanation_metadata": {
    "explainer_type": "GradientExplainer",
    "background_n": 100,
    "window_shape": [30, 14],
    "explained_at": "2026-07-05T14:00:00+00:00"
  }
}
```

For **XGBoost** (tabular model):

- `shap_values_full` is `null` — there is no temporal dimension.
- `explanation_metadata.explainer_type` is `"TreeExplainer"`.
- `explanation_metadata.background_n` is `0` (TreeExplainer does not use a background dataset).
- `explanation_metadata` also includes `shap_values_1d`: the raw signed SHAP values per feature (useful for waterfall/force plots on the client side).

---

## Field Reference

| Field | Type | Present for | Description |
|---|---|---|---|
| `model_name` | `string` | all | Registry model name |
| `dataset` | `string` | all | C-MAPSS subset (e.g. `"FD001"`) |
| `rul_prediction` | `float` | all | Scalar RUL estimate in cycles |
| `feature_importances` | `object` | all | `{feature_name: float}` — mean absolute SHAP value per feature, sorted descending |
| `shap_values_full` | `array\|null` | sequence models | 2-D array of shape `(T, F)` — SHAP attribution at each timestep |
| `explanation_metadata.explainer_type` | `string` | all | `"TreeExplainer"` or `"GradientExplainer"` |
| `explanation_metadata.background_n` | `integer` | all | Number of background samples used (0 for TreeExplainer) |
| `explanation_metadata.window_shape` | `array\|null` | sequence models | `[T, F]` — shape of the explained window |
| `explanation_metadata.shap_values_1d` | `array\|null` | XGBoost | Raw signed SHAP values per feature (length `F`) |
| `explanation_metadata.explained_at` | `string` | all | ISO-8601 UTC timestamp of the explanation |

---

## Notes for the Serving API (Milestone 2.x)

1. **Pydantic model**: create `serving/app/schemas.py` with `ExplanationResponse` mirroring this schema.  Use `Optional[list[list[float]]]` for `shap_values_full`.
2. **Lazy explainer initialisation**: instantiate `SHAPExplainer` once at serving startup (not per-request) and cache it in app state — explainer construction (background tensor loading) is expensive.
3. **Background dataset**: for sequence model explainers, sample 50–200 windows from the training set at startup and store them in the explainer.  More samples increase explanation fidelity but increase memory and latency.
4. **Explainability is opt-in**: the `/predict/rul` endpoint should default to `explained=false` and skip SHAP entirely unless the caller explicitly requests it, since GradientExplainer adds meaningful latency (see `docs/SERVING_BENCHMARKS.md` once populated in Milestone 2.2).
