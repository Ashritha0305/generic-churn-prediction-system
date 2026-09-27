from __future__ import annotations

from dataclasses import dataclass
from typing import List


# ============================================================
# Decision result
# ============================================================

@dataclass
class MonitoringDecision:

    data_quality_status: str
    drift_status: str
    performance_status: str

    overall_status: str
    recommended_action: str

    high_drift_features: List[str]


# ============================================================
# Monitoring decision engine
# ============================================================

class MonitoringDecisionEngine:

    def __init__(
        self,
        max_missing_percentage: float = 5.0,
        min_f1_score: float = 0.70,
    ):
        self.max_missing_percentage = (
            max_missing_percentage
        )

        self.min_f1_score = min_f1_score

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    def evaluate_data_quality(
        self,
        missing_percentage: float,
        duplicate_rows: int,
    ) -> str:

        if (
            missing_percentage
            > self.max_missing_percentage
        ):
            return "ALERT"

        if duplicate_rows > 0:
            return "WARNING"

        return "HEALTHY"

    # --------------------------------------------------------
    # Drift
    # --------------------------------------------------------

    def evaluate_drift(
        self,
        drift_results,
    ) -> str:

        high_drift = [
            result
            for result in drift_results
            if result.status == "HIGH"
        ]

        moderate_drift = [
            result
            for result in drift_results
            if result.status == "MODERATE"
        ]

        if high_drift:
            return "ALERT"

        if moderate_drift:
            return "WARNING"

        return "HEALTHY"

    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    def evaluate_performance(
        self,
        f1_score: float,
    ) -> str:

        if f1_score < self.min_f1_score:
            return "ALERT"

        return "HEALTHY"

    # --------------------------------------------------------
    # Overall decision
    # --------------------------------------------------------

    def evaluate(
        self,
        missing_percentage: float,
        duplicate_rows: int,
        drift_results,
        f1_score: float,
    ) -> MonitoringDecision:

        data_quality_status = (
            self.evaluate_data_quality(
                missing_percentage,
                duplicate_rows,
            )
        )

        drift_status = (
            self.evaluate_drift(
                drift_results
            )
        )

        performance_status = (
            self.evaluate_performance(
                f1_score
            )
        )

        high_drift_features = [
            result.feature
            for result in drift_results
            if result.status == "HIGH"
        ]

        # ----------------------------------------------------
        # Overall status
        # ----------------------------------------------------

        statuses = [
            data_quality_status,
            drift_status,
            performance_status,
        ]

        if "ALERT" in statuses:
            overall_status = "ALERT"

        elif "WARNING" in statuses:
            overall_status = "WARNING"

        else:
            overall_status = "HEALTHY"

        # ----------------------------------------------------
        # Recommended action
        # ----------------------------------------------------

        if performance_status == "ALERT":

            recommended_action = (
                "Model performance has degraded. "
                "Investigate the cause and consider "
                "retraining after validation."
            )

        elif data_quality_status == "ALERT":

            recommended_action = (
                "Data quality problems detected. "
                "Investigate the incoming data before "
                "using it for predictions."
            )

        elif drift_status == "ALERT":

            recommended_action = (
                "Significant feature drift detected. "
                "Investigate the changed features and "
                "evaluate model performance before "
                "considering retraining."
            )

        elif drift_status == "WARNING":

            recommended_action = (
                "Moderate feature drift detected. "
                "Continue monitoring and investigate "
                "if the drift persists."
            )

        elif data_quality_status == "WARNING":

            recommended_action = (
                "Duplicate records detected. "
                "Investigate the incoming dataset."
            )

        else:

            recommended_action = (
                "Monitoring checks are healthy. "
                "Continue normal operation."
            )

        return MonitoringDecision(
            data_quality_status=data_quality_status,
            drift_status=drift_status,
            performance_status=performance_status,
            overall_status=overall_status,
            recommended_action=recommended_action,
            high_drift_features=high_drift_features,
        )


# ============================================================
# Print decision
# ============================================================

def print_monitoring_decision(
    decision: MonitoringDecision,
) -> None:

    print("\n" + "=" * 60)
    print("MONITORING DECISION")
    print("=" * 60)

    print(
        f"Data Quality : "
        f"{decision.data_quality_status}"
    )

    print(
        f"Data Drift   : "
        f"{decision.drift_status}"
    )

    print(
        f"Performance  : "
        f"{decision.performance_status}"
    )

    print(
        f"Overall      : "
        f"{decision.overall_status}"
    )

    if decision.high_drift_features:

        print(
            "\nHigh drift features:"
        )

        for feature in (
            decision.high_drift_features
        ):
            print(
                f"  - {feature}"
            )

    print(
        "\nRecommended Action:"
    )

    print(
        decision.recommended_action
    )