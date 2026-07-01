.PHONY: download-data

download-data:
	@echo "Fetching datasets..."
	python3 ml/data/download_cmapss.py
	python3 ml/data/download_mimii.py
	python3 ml/data/download_cwru.py
	@echo "All data downloaded!"
