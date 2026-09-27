from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


@dataclass
class DriftResult:
    feature: str
    feature_type: str
    drift_score: float
    status: str


class DriftDetector:

    def __init__(
        self,
        low_threshold: float = 0.10,
        high_threshold: float = 0.25,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    # --------------------------------------------------
    # Numeric drift
    # --------------------------------------------------

    def _numeric_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> float:

        reference = pd.to_numeric(
            reference,
            errors="coerce",
        ).dropna()

        current = pd.to_numeric(
            current,
            errors="coerce",
        ).dropna()

        if reference.empty or current.empty:
            return 0.0

        raw_distance = wasserstein_distance(
            reference,
            current,
        )

        # Normalize by the scale of the reference
        # distribution so different numeric features
        # can be compared more meaningfully.
        reference_std = float(
            reference.std()
        )

        if (
            np.isnan(reference_std)
            or reference_std <= 1e-12
        ):
            reference_range = float(
                reference.max() - reference.min()
            )

            if reference_range <= 1e-12:
                return 0.0

            return float(
                raw_distance / reference_range
            )

        return float(
            raw_distance / reference_std
        )

    # --------------------------------------------------
    # Categorical drift
    # --------------------------------------------------

    def _categorical_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> float:

        reference = reference.astype(
            "string"
        ).fillna("__MISSING__")

        current = current.astype(
            "string"
        ).fillna("__MISSING__")

        categories = sorted(
            set(reference.unique())
            | set(current.unique())
        )

        if not categories:
            return 0.0

        reference_distribution = (
            reference
            .value_counts(normalize=True)
            .reindex(
                categories,
                fill_value=0.0,
            )
            .to_numpy()
        )

        current_distribution = (
            current
            .value_counts(normalize=True)
            .reindex(
                categories,
                fill_value=0.0,
            )
            .to_numpy()
        )

        score = jensenshannon(
            reference_distribution,
            current_distribution,
        )

        if np.isnan(score):
            return 0.0

        return float(score)

    # --------------------------------------------------
    # Status
    # --------------------------------------------------

    def _status(
        self,
        score: float,
    ) -> str:

        if score >= self.high_threshold:
            return "HIGH"

        if score >= self.low_threshold:
            return "MODERATE"

        return "LOW"

    # --------------------------------------------------
    # Detect drift
    # --------------------------------------------------

    def detect(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        identifier_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> List[DriftResult]:

        common_columns = [
            column
            for column in reference_df.columns
            if column in current_df.columns
        ]

        # Do not monitor identifiers or the known target
        # as model-input features.
        excluded_columns = set()

        if identifier_column:
            excluded_columns.add(
                identifier_column
            )

        if target_column:
            excluded_columns.add(
                target_column
            )

        common_columns = [
            column
            for column in common_columns
            if column not in excluded_columns
        ]

        results: List[DriftResult] = []

        for column in common_columns:

            reference_series = (
                reference_df[column]
            )

            current_series = (
                current_df[column]
            )

            # ------------------------------------------
            # Numeric feature
            # ------------------------------------------

            if pd.api.types.is_numeric_dtype(
                reference_series
            ):

                score = self._numeric_drift(
                    reference_series,
                    current_series,
                )

                feature_type = "Numeric"

            # ------------------------------------------
            # Categorical feature
            # ------------------------------------------

            else:

                score = self._categorical_drift(
                    reference_series,
                    current_series,
                )

                feature_type = "Categorical"

            results.append(
                DriftResult(
                    feature=column,
                    feature_type=feature_type,
                    drift_score=round(
                        score,
                        4,
                    ),
                    status=self._status(score),
                )
            )

        return results


def print_drift_report(
    results: List[DriftResult],
) -> None:

    print("=" * 60)
    print("DATA DRIFT MONITORING")
    print("=" * 60)

    if not results:
        print(
            "No common model-input features "
            "available for drift analysis."
        )
        return

    print(
        f"{'Feature':25}"
        f"{'Type':15}"
        f"{'Score':10}"
        f"{'Status'}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result.feature[:24]:25}"
            f"{result.feature_type:15}"
            f"{result.drift_score:<10.4f}"
            f"{result.status}"
        )