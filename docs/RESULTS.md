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
