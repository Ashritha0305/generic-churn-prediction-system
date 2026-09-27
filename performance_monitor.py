from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Configuration
# ============================================================

TRACKING_URI = (
    "sqlite:///C:/Users/Ashritha/"
    "OneDrive/Desktop/churn_project/mlflow.db"
)

EXPERIMENT_NAME = (
    "Adaptive Predictive Analytics - Monitoring"
)

DATASET = "production_data.csv"

TARGET_COLUMN = "working_status"


# ============================================================
# MLflow setup
# ============================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# Load production data
# ============================================================

df = pd.read_csv(DATASET)


print("=" * 60)
print("MODEL PERFORMANCE MONITORING")
print("=" * 60)

print(
    f"Dataset rows : {len(df)}"
)

print(
    f"Target       : {TARGET_COLUMN}"
)


# ============================================================
# For this exercise, create model predictions
# ============================================================
#
# We will use the currently saved model.
#
# The important concept here is:
#
#     Prediction
#          +
#     Actual outcome
#          ↓
#     Performance metrics
#
# ============================================================

import joblib

model_data = joblib.load(
    "model.pkl"
)


# The saved artifact contains the trained
# ChurnAnalysisEngine model information.
#
# We use the engine's prediction functionality
# instead of manually rebuilding preprocessing.

from churn_engine import ChurnAnalysisEngine

engine = ChurnAnalysisEngine()

engine.trained_model = model_data


# ============================================================
# Generate predictions
# ============================================================

predictions_df = engine.predict_frame(
    df
)


# ============================================================
# Extract predictions and actual outcomes
# ============================================================

actual = df[TARGET_COLUMN]

predicted = predictions_df[
    "Churn Prediction"
]


# ============================================================
# Calculate performance metrics
# ============================================================

accuracy = accuracy_score(
    actual,
    predicted
)

precision = precision_score(
    actual,
    predicted,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    actual,
    predicted,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    actual,
    predicted,
    average="weighted",
    zero_division=0,
)

report = classification_report(
    actual,
    predicted,
    zero_division=0,
)


# ============================================================
# Print results
# ============================================================

print("\nPERFORMANCE")
print("-" * 60)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print("\nClassification Report")
print("-" * 60)

print(report)


# ============================================================
# Save performance report
# ============================================================

performance_summary = {
    "dataset": DATASET,
    "target_column": TARGET_COLUMN,
    "rows": len(df),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
}


summary_path = Path(
    "performance_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        performance_summary,
        file,
        indent=4,
    )


# ============================================================
# Log to MLflow
# ============================================================

with mlflow.start_run(
    run_name="model-performance-monitoring"
):

    # ------------------------------
    # Tags
    # ------------------------------

    mlflow.set_tag(
        "run_type",
        "monitoring",
    )

    mlflow.set_tag(
        "monitoring_type",
        "model_performance",
    )

    # ------------------------------
    # Parameters
    # ------------------------------

    mlflow.log_param(
        "dataset",
        DATASET,
    )

    mlflow.log_param(
        "target_column",
        TARGET_COLUMN,
    )

    # ------------------------------
    # Metrics
    # ------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy,
    )

    mlflow.log_metric(
        "precision",
        precision,
    )

    mlflow.log_metric(
        "recall",
        recall,
    )

    mlflow.log_metric(
        "f1_score",
        f1,
    )

    # ------------------------------
    # Artifact
    # ------------------------------

    mlflow.log_artifact(
        str(summary_path),
        artifact_path="performance",
    )


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 60)
print("PERFORMANCE MONITORING COMPLETED")
print("=" * 60)

print(
    "Performance metrics logged to MLflow."
)