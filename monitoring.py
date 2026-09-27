from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class DataQualityReport:
    rows: int
    columns: int
    missing_cells: int
    missing_percentage: float
    duplicate_rows: int
    missing_columns: List[str]
    unexpected_columns: List[str]
    missing_expected_columns: List[str]


class DataQualityMonitor:

    def compare(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> DataQualityReport:

        reference_columns = set(reference_df.columns)
        current_columns = set(current_df.columns)

        unexpected_columns = sorted(
            current_columns - reference_columns
        )

        missing_expected_columns = sorted(
            reference_columns - current_columns
        )

        missing_cells = int(
            current_df.isna().sum().sum()
        )

        total_cells = max(
            current_df.shape[0]
            * max(current_df.shape[1], 1),
            1,
        )

        missing_percentage = (
            missing_cells / total_cells
        ) * 100

        duplicate_rows = int(
            current_df.duplicated().sum()
        )

        missing_columns = [
            column
            for column in current_df.columns
            if current_df[column].isna().any()
        ]

        return DataQualityReport(
            rows=int(current_df.shape[0]),
            columns=int(current_df.shape[1]),
            missing_cells=missing_cells,
            missing_percentage=round(
                missing_percentage,
                2,
            ),
            duplicate_rows=duplicate_rows,
            missing_columns=missing_columns,
            unexpected_columns=unexpected_columns,
            missing_expected_columns=missing_expected_columns,
        )


def print_quality_report(
    report: DataQualityReport,
) -> None:

    print("=" * 60)
    print("DATA QUALITY MONITORING")
    print("=" * 60)

    print(f"Rows                 : {report.rows}")
    print(f"Columns              : {report.columns}")
    print(f"Missing cells        : {report.missing_cells}")
    print(
        f"Missing percentage   : "
        f"{report.missing_percentage:.2f}%"
    )
    print(f"Duplicate rows       : {report.duplicate_rows}")

    print()
    print(
        "Columns containing missing values:"
    )

    if report.missing_columns:
        for column in report.missing_columns:
            print(f"  - {column}")
    else:
        print("  None")

    print()
    print("Unexpected columns:")

    if report.unexpected_columns:
        for column in report.unexpected_columns:
            print(f"  - {column}")
    else:
        print("  None")

    print()
    print("Missing expected columns:")

    if report.missing_expected_columns:
        for column in report.missing_expected_columns:
            print(f"  - {column}")
    else:
        print("  None")