# Generic Churn Prediction System

A small end-to-end churn analysis project that can:

- auto-detect a churn target column in a CSV file
- train a churn prediction model when labels are available
- generate customer-level churn risk predictions
- expose the workflow in a Streamlit dashboard

## Features

- Works with a CSV dataset and tries to infer the target, identifier, and date columns
- Trains and compares two classifiers: Random Forest and Logistic Regression
- Produces accuracy, weighted F1, classification report, confusion matrix, and feature importance
- Generates prediction tables with churn probability and risk levels
- Provides a Streamlit dashboard with charts, summary metrics, and a downloadable report

## Project Files

- `app.py` - Streamlit dashboard
- `churn_engine.py` - core detection, training, prediction, and reporting logic
- `train.py` - command-line script to train and save a model
- `predict.py` - command-line script to generate predictions from a saved model
- `requirements.txt` - Python dependencies
- `dataset.csv` / `a.csv` - example datasets

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

Upload a CSV file in the browser. The app will try to detect the churn target automatically and show:

- KPI cards
- target distribution charts
- feature importance
- risk distribution
- a ranked customer table
- a downloadable analysis report

## Train a Model From the Command Line

```bash
python train.py --data dataset.csv --output model.pkl
```

If a churn target column is detected, this script trains a model and saves it to `model.pkl`.

## Generate Predictions

If you already have a saved model:

```bash
python predict.py --data dataset.csv --model model.pkl --output predictions.csv
```

If the saved model cannot be loaded or no model exists, the script falls back to analyzing the dataset directly.

## Expected Input Data

The dataset should ideally include:

- a churn or target column, such as `churn`, `exit`, `attrition`, `target`, or `status`
- an identifier column, such as `customer_id` or `user_id`
- a mix of numeric and categorical feature columns

The project will still run if some of these are missing, but the output will be less complete.

## Outputs

- `model.pkl` - saved trained model
- `predictions.csv` - generated predictions
- `churn_analysis_report.csv` - downloadable report from the Streamlit app

## Notes

- The engine automatically handles missing values and categorical encoding.
- Feature importance is grouped by base feature name for readability.
- High churn risk is shown as `High`, `Medium`, or `Low` based on predicted probability.
