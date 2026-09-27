import pandas as pd

from monitoring import (
    DataQualityMonitor,
    print_quality_report,
)


REFERENCE_DATASET = "dataset.csv"


def main():

    reference_df = pd.read_csv(
        REFERENCE_DATASET
    )

    # --------------------------------------------------
    # Simulate incoming production data
    # --------------------------------------------------

    current_df = reference_df.copy()

    # Introduce a missing value
    if len(current_df) > 0:
        current_df.iloc[0, 1] = None

    # Introduce a duplicate row
    if len(current_df) > 0:
        current_df = pd.concat(
            [
                current_df,
                current_df.iloc[[0]],
            ],
            ignore_index=True,
        )

    monitor = DataQualityMonitor()

    report = monitor.compare(
        reference_df,
        current_df,
    )

    print_quality_report(report)


if __name__ == "__main__":
    main()
    