# Model Results

This table is the running leaderboard for C-MAPSS RUL experiments. Milestone 1.3 adds the training code for Random Forest and XGBoost baselines; populate the table after running the training script locally.

Run:

```bash
make train-baselines
```

| Model | Dataset | RMSE | MAE | R2 | NASA Score | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Random Forest | FD001 | 18.1382 | 12.6108 | 0.7951 | 1084.3293 | `/home/silvanus/CODES/ipmp-platform/ml/models/registry/random_forest/FD001/20260702T071720Z/model.joblib` |
| XGBoost | FD001 | 18.9330 | 13.2661 | 0.7768 | 1208.9589 | `/home/silvanus/CODES/ipmp-platform/ml/models/registry/xgboost/FD001/20260702T071739Z/model.joblib` |
| LSTM | FD001 | 16.3312 | 11.9987 | 0.8339 | 835.6180 | `/content/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/ml/models/registry/lstm_rul/FD001/20260702T102020Z/model.pt` |
| LSTM | FD001 | 16.5961 | 12.6642 | 0.8285 | 815.3898 | `/content/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/ml/models/registry/lstm_rul/FD001/20260702T103659Z/model.pt` |
| GRU | FD001 | 15.5333 | 11.4046 | 0.8497 | 545.8152 | `/content/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/ml/models/registry/gru_rul/FD001/20260702T111034Z/model.pt` |
| GRU | FD001 | 17.4359 | 12.3735 | 0.8107 | 994.1531 | `/content/Industrial-Predictive-Maintenance-Platform/Industrial-Predictive-Maintenance-Platform/ml/models/registry/gru_rul/FD001/20260702T161238Z/model.pt` |
