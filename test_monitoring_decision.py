from monitoring_decision import (
    MonitoringDecisionEngine,
    print_monitoring_decision,
)

from drift_detector import (
    DriftDetector,
)

import pandas as pd


# ============================================================
# Load data
# ============================================================

reference_df = pd.read_csv(
    "dataset.csv"
)

current_df = pd.read_csv(
    "production_data.csv"
)


# ============================================================
# Detect drift
# ============================================================

detector = DriftDetector()

drift_results = detector.detect(
    reference_df,
    current_df,
    identifier_column="studentID",
    target_column="working_status",
)


# ============================================================
# Create decision engine
# ============================================================

decision_engine = (
    MonitoringDecisionEngine(
        max_missing_percentage=5.0,
        min_f1_score=0.70,
    )
)


# ============================================================
# Evaluate monitoring state
# ============================================================

decision = decision_engine.evaluate(
    missing_percentage=0.0,
    duplicate_rows=0,
    drift_results=drift_results,
    f1_score=0.9501,
)


# ============================================================
# Print result
# ============================================================

print_monitoring_decision(
    decision
)