import pandas as pd

from drift_detector import (
    DriftDetector,
    print_drift_report,
)


# --------------------------------------------------
# Load reference dataset
# --------------------------------------------------

reference_df = pd.read_csv("dataset.csv")


# --------------------------------------------------
# Create simulated current dataset
# --------------------------------------------------

current_df = reference_df.copy()


# Simulate numeric drift
current_df["study_hours"] = (
    current_df["study_hours"] + 5
)


# Simulate categorical drift
current_df["device_type"] = "DifferentDevice"


# --------------------------------------------------
# Create drift detector
# --------------------------------------------------

detector = DriftDetector()


# --------------------------------------------------
# Detect drift
# --------------------------------------------------

results = detector.detect(
    reference_df,
    current_df,
    identifier_column="studentID",
    target_column="working_status",
)


# --------------------------------------------------
# Print report
# --------------------------------------------------

print_drift_report(results)