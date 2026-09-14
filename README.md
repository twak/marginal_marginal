# Warping Earth Observations for better ice labelling in the Marginal Marginal Ice Zone [[PDF](https://arxiv.org/html/2608.11883v1)]

This folder contains the dataset and eval code for Sea Ice feature importance and model comparison.

## Contents
- `dataset/`: Dataset files (`.npy` with warped and unwarped pixel featuers, `.json` with classifications, `mean_std.json`).
- `dataset.py`: Standalone dataset loader that normalizes channels and provides `load_data`.
- `evaluate.py`: Demo evalution script.

## Requirements
You need `numpy`, `pandas`, and `scikit-learn`:
`pip install numpy pandas scikit-learn`

## Running
To run the evalution and repoduce the tables:
`python evaluate.py`

This outputs the Base Feature Importance, MODIS Feature Importance, and Model Comparsion tables.
