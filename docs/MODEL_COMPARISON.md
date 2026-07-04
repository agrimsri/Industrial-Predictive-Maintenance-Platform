# Model Comparison

Milestone 1.5 adds a PatchTST-style transformer so the project can compare three model families on the same C-MAPSS RUL task:

- Classic tabular baselines: Random Forest and XGBoost
- Recurrent sequence models: LSTM and GRU
- Transformer sequence model: PatchTST

## Current FD001 Leaderboard

| Model | Dataset | RMSE | MAE | R2 | NASA Score | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Random Forest | FD001 | 18.1382 | 12.6108 | 0.7951 | 1084.3293 | Best classic ML baseline so far |
| XGBoost | FD001 | 18.9330 | 13.2661 | 0.7768 | 1208.9589 | Strong baseline, but behind RF on this run |
| LSTM | FD001 | 16.3312 | 11.9987 | 0.8339 | 835.6180 | Best LSTM run by RMSE/MAE/R2 |
| GRU | FD001 | 15.5333 | 11.4046 | 0.8497 | 545.8152 | Current overall leader before PatchTST training |
| PatchTST | FD001 | TBD | TBD | TBD | TBD | Added in Milestone 1.5; train on Colab |

## Cross-Subset Benchmark Plan

Run each model family on FD001, FD002, FD003, and FD004. FD002 and FD004 are especially important because they contain multiple operating regimes, which tests whether the feature normalization and sequence models remain robust outside the easiest subset.

| Model Family | FD001 | FD002 | FD003 | FD004 |
| --- | --- | --- | --- | --- |
| Random Forest | Done | TODO | TODO | TODO |
| XGBoost | Done | TODO | TODO | TODO |
| LSTM/GRU | Done | TODO | TODO | TODO |
| PatchTST | TODO | TODO | TODO | TODO |

## Expected Trade-Offs To Record

- Accuracy: compare RMSE, MAE, R2, and NASA score. NASA score should be weighted heavily because late maintenance predictions are riskier.
- Training cost: record hardware, epochs, wall-clock time, and whether early stopping triggered.
- Inference cost: compare model size and batch prediction latency once the serving API milestone begins.
- Robustness: compare whether the relative ranking changes on FD002 and FD004.

## PatchTST Adaptation Notes

The implementation in `ml/src/models/patchtst_rul.py` uses the central PatchTST idea from the paper: each variable is split into temporal patches, patches are projected into tokens, and a shared Transformer encoder learns over patch sequences. The adaptation for this project is supervised RUL regression instead of forecasting: the encoded patches from all channels are flattened into a regression head that predicts one RUL value per window.
