from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd

from churn_engine import ChurnAnalysisEngine
from monitoring import DataQualityMonitor
from drift_detector import DriftDetector


# ============================================================
# Configuration
# ============================================================

TRACKING_URI = (
    "sqlite:///C:/Users/Ashritha/"
    "OneDrive/Desktop/churn_project/mlflow.db"
)

EXPERIMENT_NAME = "Adaptive Predictive Analytics - Monitoring"

REFERENCE_DATASET = "dataset.csv"

# Simulated current/production data
#CURRENT_DATASET = "dataset.csv"
CURRENT_DATASET = "production_data.csv"


# ============================================================
# MLflow setup
# ============================================================

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Load datasets
# ============================================================

reference_df = pd.read_csv(
    REFERENCE_DATASET
)

current_df = pd.read_csv(
    CURRENT_DATASET
)


# ============================================================
# Discover identifier and target dynamically
# ============================================================

engine = ChurnAnalysisEngine()

profile = engine.profile_dataset(
    reference_df
)

identifier_column = profile.identifier_column
target_column = profile.detected_target


print("=" * 60)
print("MLFLOW MONITORING")
print("=" * 60)

print(f"Reference rows       : {len(reference_df)}")
print(f"Current rows         : {len(current_df)}")
print(f"Identifier column    : {identifier_column}")
print(f"Target column        : {target_column}")


# ============================================================
# Data Quality Monitoring
# ============================================================

quality_monitor = DataQualityMonitor()

quality_report = quality_monitor.compare(
    reference_df,
    current_df,
)


print("\nDATA QUALITY")
print("-" * 60)

print(
    f"Missing cells        : "
    f"{quality_report.missing_cells}"
)

print(
    f"Missing percentage   : "
    f"{quality_report.missing_percentage:.2f}%"
)

print(
    f"Duplicate rows       : "
    f"{quality_report.duplicate_rows}"
)


# ============================================================
# Data Drift Monitoring
# ============================================================

drift_detector = DriftDetector()

drift_results = drift_detector.detect(
    reference_df,
    current_df,
    identifier_column=identifier_column,
    target_column=target_column,
)


print("\nDATA DRIFT")
print("-" * 60)

for result in drift_results:

    print(
        f"{result.feature:25}"
        f"{result.drift_score:<10.4f}"
        f"{result.status}"
    )


# ============================================================
# Drift summary
# ============================================================

high_drift_features = [
    result
    for result in drift_results
    if result.status == "HIGH"
]

moderate_drift_features = [
    result
    for result in drift_results
    if result.status == "MODERATE"
]

low_drift_features = [
    result
    for result in drift_results
    if result.status == "LOW"
]


print("\nDRIFT SUMMARY")
print("-" * 60)

print(
    f"HIGH drift features     : "
    f"{len(high_drift_features)}"
)

print(
    f"MODERATE drift features : "
    f"{len(moderate_drift_features)}"
)

print(
    f"LOW drift features      : "
    f"{len(low_drift_features)}"
)


# ============================================================
# Create monitoring report
# ============================================================

drift_report_df = pd.DataFrame(
    [
        {
            "feature": result.feature,
            "feature_type": result.feature_type,
            "drift_score": result.drift_score,
            "status": result.status,
        }
        for result in drift_results
    ]
)


# ============================================================
# Save temporary monitoring artifacts
# ============================================================

drift_report_path = Path(
    "drift_report.csv"
)

drift_report_df.to_csv(
    drift_report_path,
    index=False,
)


monitoring_summary = {
    "reference_rows": len(reference_df),
    "current_rows": len(current_df),
    "identifier_column": identifier_column,
    "target_column": target_column,
    "missing_cells": quality_report.missing_cells,
    "missing_percentage": quality_report.missing_percentage,
    "duplicate_rows": quality_report.duplicate_rows,
    "high_drift_features": len(
        high_drift_features
    ),
    "moderate_drift_features": len(
        moderate_drift_features
    ),
    "low_drift_features": len(
        low_drift_features
    ),
}


summary_path = Path(
    "monitoring_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        monitoring_summary,
        file,
        indent=4,
    )


# ============================================================
# Log monitoring results to MLflow
# ============================================================

with mlflow.start_run(
    run_name="data-quality-and-drift-monitoring"
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
        "data_quality_and_drift",
    )

    # ------------------------------
    # Parameters
    # ------------------------------

    mlflow.log_param(
        "reference_dataset",
        REFERENCE_DATASET,
    )

    mlflow.log_param(
        "current_dataset",
        CURRENT_DATASET,
    )

    if identifier_column:
        mlflow.log_param(
            "identifier_column",
            identifier_column,
        )

    if target_column:
        mlflow.log_param(
            "target_column",
            target_column,
        )

    # ------------------------------
    # Data quality metrics
    # ------------------------------

    mlflow.log_metric(
        "current_rows",
        len(current_df),
    )

    mlflow.log_metric(
        "missing_cells",
        quality_report.missing_cells,
    )

    mlflow.log_metric(
        "missing_percentage",
        quality_report.missing_percentage,
    )

    mlflow.log_metric(
        "duplicate_rows",
        quality_report.duplicate_rows,
    )

    # ------------------------------
    # Drift metrics
    # ------------------------------

    mlflow.log_metric(
        "high_drift_features",
        len(high_drift_features),
    )

    mlflow.log_metric(
        "moderate_drift_features",
        len(moderate_drift_features),
    )

    mlflow.log_metric(
        "low_drift_features",
        len(low_drift_features),
    )

    # ------------------------------
    # Artifacts
    # ------------------------------

    mlflow.log_artifact(
        str(drift_report_path),
        artifact_path="monitoring",
    )

    mlflow.log_artifact(
        str(summary_path),
        artifact_path="monitoring",
    )


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 60)
print("MONITORING RUN COMPLETED")
print("=" * 60)

print(
    "Monitoring results logged to MLflow."
)

print(
    f"Experiment: {EXPERIMENT_NAME}"
)