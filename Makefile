.PHONY: download-data train-baselines train-lstm train-patchtst

download-data:
	@echo "Fetching datasets..."
	python3 ml/data/download_cmapss.py
	python3 ml/data/download_mimii.py
	python3 ml/data/download_cwru.py
	@echo "All data downloaded!"

train-baselines:
	PYTHONPATH=ml ml/.venv/bin/python -m src.models.train_baselines --dataset FD001

train-lstm:
	PYTHONPATH=ml ml/.venv/bin/python -m src.models.lstm_rul --dataset FD001 --model-type lstm

train-patchtst:
	PYTHONPATH=ml ml/.venv/bin/python -m src.models.patchtst_rul --dataset FD001
