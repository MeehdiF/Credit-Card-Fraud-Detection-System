# Credit Card Fraud Detection

Explore a classification workflow for identifying potentially fraudulent credit-card transactions.

## Why this project

This repository is part of my practical machine-learning portfolio. It focuses on a complete, understandable workflow rather than claiming production readiness.

## Dataset

The README references the Kaggle Credit Card Fraud Detection dataset. Review the dataset terms and provide a download step rather than committing third-party data.

## Approach

Data cleaning, standardization, an RBF-kernel Support Vector Classifier, and grid search for model selection.

### Features

Transaction features from the referenced credit-card dataset, with preprocessing performed in the notebook.

## Evaluation and current result

F1 score, Jaccard score, classification report, and confusion matrix. Because fraud detection is imbalanced, F1 and class-level recall should be prioritized over accuracy.

## Run locally

```bash
git clone https://github.com/MeehdiF/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python untitled.py
```

For notebook exploration, open the `.ipynb` file with Jupyter after installing the same dependencies.

## Limitations and next steps

The README should record the class distribution, split strategy, selected hyperparameters, and final minority-class recall. A production version should address leakage, calibration, threshold selection, and computational cost.

## Repository structure

- `README.md` — project context and reproducibility notes
- `requirements.txt` — Python dependencies used by the scripts
- `.ipynb` / `.py` files — analysis and model experiments

## License

See [`LICENSE`](LICENSE). Check the dataset's own terms separately; repository code licensing does not automatically license bundled data.
