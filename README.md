# Titanic Survival Prediction (Machine Learning)

This repository contains a Python script to train a Logistic Regression model to predict survival on the Titanic.

## Files added
- `titanic_survival.py` — main script (CLI) to train & evaluate model
- `requirements.txt` — Python dependencies
- `.gitignore`
- `LICENSE` (MIT)
- `.github/workflows/python-app.yml` — optional CI
- `models/` — output directory for saved model

## Usage

1. Put your dataset CSV (train.csv) in the project root or pass its path to the script:
```bash
python titanic_survival.py --data path/to/train.csv
