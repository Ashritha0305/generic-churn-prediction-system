from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


# These terms describe outcomes that can represent leaving, cancellation,
# closure, termination, dropout, or similar exit events.
OUTCOME_NAME_KEYWORDS = (
    "churn",
    "attrition",
    "dropout",
    "drop_out",
    "exit",
    "left",
    "leave",
    "cancel",
    "cancellation",
    "cancelled",
    "canceled",
    "closed",
    "closure",
    "terminated",
    "termination",
    "resigned",
    "resignation",
    "withdrawn",
    "withdrawal",
    "inactive",
    "status",
    "target",
    "label",
)

# Values that often represent an exit/negative outcome.
POSITIVE_OUTCOME_VALUES = {
    "yes",
    "true",
    "1",
    "churn",
    "churned",
    "left",
    "exit",
    "exited",
    "attrition",
    "cancel",
    "cancelled",
    "canceled",
    "dropout",
    "dropped",
    "closed",
    "terminated",
    "resigned",
    "withdrawn",
    "inactive",
}

# Column-name signals that can participate in a meaningful state/event
# interpretation when there is no explicit target.
STATE_KEYWORDS = (
    "status",
    "state",
    "stage",
    "subscription",
    "account",
    "membership",
    "employment",
    "enrollment",
    "activity",
    "login",
    "last_login",
    "last_activity",
    "cancellation",
    "termination",
    "closure",
    "end_date",
    "cancel_date",
)


@dataclass
class OutcomeCandidate:
    column: str
    score: float
    reason: str
    unique_values: int
    sample_values: List[str]


@dataclass
class OutcomeDiscoveryResult:
    status: str
    target_column: Optional[str]
    positive_class: Optional[str]
    confidence: float
    reason: str
    explicit_candidates: List[OutcomeCandidate]
    state_columns: List[str]
    date_columns: List[str]

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OutcomeDiscovery:
    """
    Discover whether a dataset contains a defensible outcome for supervised
    churn/exit analysis.

    This module does NOT invent a churn label from arbitrary mathematical
    relationships. It distinguishes:
      - explicit outcome
      - meaningful state/event evidence
      - behavioral-only data
      - insufficient evidence
    """

    @staticmethod
    def _normalize(value: object) -> str:
        return str(value).strip().lower()

    @staticmethod
    def _sample_values(series: pd.Series, limit: int = 5) -> List[str]:
        return [
            str(value)
            for value in series.dropna().astype(str).drop_duplicates().head(limit)
        ]

    @staticmethod
    def _positive_class(series: pd.Series) -> Optional[str]:
        values = series.dropna().astype(str).str.strip()
        unique = list(values.str.lower().unique())

        matches = [
            value
            for value in unique
            if value in POSITIVE_OUTCOME_VALUES
        ]

        if len(matches) == 1:
            match = matches[0]

            for original in values.unique():
                if str(original).strip().lower() == match:
                    return str(original).strip()

        return None

    def _score_explicit_candidate(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> Optional[OutcomeCandidate]:

        series = df[column].dropna()

        if series.empty:
            return None

        unique_count = series.nunique()

        # A churn/exit target should normally have relatively few classes.
        if unique_count < 2 or unique_count > 10:
            return None

        name = self._normalize(column)

        score = 0.0
        reasons: List[str] = []

        if any(keyword in name for keyword in OUTCOME_NAME_KEYWORDS):
            score += 0.55
            reasons.append("outcome-related column name")

        positive = self._positive_class(series)

        if positive is not None:
            score += 0.30
            reasons.append(f"exit-like value '{positive}' detected")

        if unique_count == 2:
            score += 0.15
            reasons.append("binary outcome")

        # Keep confidence bounded.
        score = min(score, 1.0)

        if score < 0.55:
            return None

        return OutcomeCandidate(
            column=column,
            score=score,
            reason="; ".join(reasons),
            unique_values=unique_count,
            sample_values=self._sample_values(series),
        )

    def _detect_state_columns(
        self,
        df: pd.DataFrame,
    ) -> List[str]:

        state_columns: List[str] = []

        for column in df.columns:
            name = self._normalize(column)

            if any(keyword in name for keyword in STATE_KEYWORDS):
                state_columns.append(column)

        return state_columns

    @staticmethod
    def _detect_date_columns(
        df: pd.DataFrame,
    ) -> List[str]:

        date_columns: List[str] = []

        date_name_keywords = (
            "date",
            "datetime",
            "timestamp",
            "signup",
            "sign_up",
            "joined",
            "joining",
            "registration",
            "registered",
            "last_login",
            "last_activity",
            "cancel_date",
            "cancellation_date",
            "end_date",
            "start_date",
        )

        for column in df.columns:
            name = str(column).strip().lower()

            if any(keyword in name for keyword in date_name_keywords):
                date_columns.append(column)
                continue

            series = df[column]

            if pd.api.types.is_datetime64_any_dtype(series):
                date_columns.append(column)

        return date_columns

    def discover(
        self,
        df: pd.DataFrame,
    ) -> OutcomeDiscoveryResult:

        candidates: List[OutcomeCandidate] = []

        for column in df.columns:
            candidate = self._score_explicit_candidate(
                df,
                column,
            )

            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        state_columns = self._detect_state_columns(df)
        date_columns = self._detect_date_columns(df)

        # ---------------------------------------------------------
        # 1. Explicit outcome
        # ---------------------------------------------------------
        if candidates:
            best = candidates[0]

            positive = self._positive_class(
                df[best.column]
            )

            # Exclude detected target from state/event and date columns
            additional_state_cols = [
                col for col in state_columns if col != best.column
            ]
            additional_date_cols = [
                col for col in date_columns if col != best.column
            ]

            return OutcomeDiscoveryResult(
                status="explicit",
                target_column=best.column,
                positive_class=positive,
                confidence=best.score,
                reason=(
                    f"An explicit binary/low-cardinality outcome was "
                    f"detected in '{best.column}'. {best.reason}."
                ),
                explicit_candidates=candidates,
                state_columns=additional_state_cols,
                date_columns=additional_date_cols,
            )

        # ---------------------------------------------------------
        # 2. Meaningful state/event evidence
        # ---------------------------------------------------------
        meaningful_state_columns = []

        for column in state_columns:
            series = df[column].dropna()

            if series.empty:
                continue

            unique_count = series.nunique()

            if 2 <= unique_count <= 20:
                meaningful_state_columns.append(column)

        if meaningful_state_columns:
            return OutcomeDiscoveryResult(
                status="derivable",
                target_column=None,
                positive_class=None,
                confidence=0.60,
                reason=(
                    "No explicit churn/exit target was detected, but "
                    "state/event-related columns exist and may contain "
                    "meaningful outcome information. Further semantic "
                    "inspection is required before deriving an outcome."
                ),
                explicit_candidates=candidates,
                state_columns=meaningful_state_columns,
                date_columns=date_columns,
            )

        # ---------------------------------------------------------
        # 3. Behavioral-only data
        # ---------------------------------------------------------
        behavioral_columns = []

        numeric_count = len(
            df.select_dtypes(include="number").columns
        )

        categorical_count = len(
            df.select_dtypes(
                include=["object", "string", "category", "bool"]
            ).columns
        )

        if numeric_count >= 2 or categorical_count >= 2:
            behavioral_columns = list(df.columns)

        if behavioral_columns:
            return OutcomeDiscoveryResult(
                status="behavioral_only",
                target_column=None,
                positive_class=None,
                confidence=0.0,
                reason=(
                    "No defensible churn/exit outcome was detected. "
                    "The dataset can still be analyzed for behavioral "
                    "relationships, but supervised churn prediction "
                    "should not be claimed."
                ),
                explicit_candidates=candidates,
                state_columns=state_columns,
                date_columns=date_columns,
            )

        # ---------------------------------------------------------
        # 4. Insufficient evidence
        # ---------------------------------------------------------
        return OutcomeDiscoveryResult(
            status="insufficient",
            target_column=None,
            positive_class=None,
            confidence=0.0,
            reason=(
                "The dataset does not contain enough outcome or behavioral "
                "information for a defensible churn analysis."
            ),
            explicit_candidates=candidates,
            state_columns=state_columns,
            date_columns=date_columns,
        )


def print_report(result: OutcomeDiscoveryResult) -> None:

    print("\n" + "=" * 70)
    print("OUTCOME DISCOVERY")
    print("=" * 70)

    print(f"Outcome status : {result.status.upper()}")
    print(f"Target column  : {result.target_column or 'None'}")
    print(f"Positive class : {result.positive_class or 'Not established'}")
    print(f"Confidence     : {result.confidence:.2f}")

    print("\nReason:")
    print(result.reason)

    if result.explicit_candidates:
        print("\nExplicit outcome candidates:")
        for candidate in result.explicit_candidates:
            print(
                f"  - {candidate.column}: "
                f"score={candidate.score:.2f}; "
                f"{candidate.reason}; "
                f"values={candidate.sample_values}"
            )

    print("\nState/event columns:")
    if result.state_columns:
        print("  " + ", ".join(result.state_columns))
    else:
        print("  No additional state or event columns were detected.")

    if result.date_columns:
        print("\nDate-related columns:")
        print("  " + ", ".join(result.date_columns))
    else:
        print("\nDate-related columns:")
        print("  No date-related columns were detected.")

    print("\nInterpretation:")
    if result.status == "explicit":
        print(
            "Supervised ML can be considered because an explicit outcome "
            "was detected."
        )
    elif result.status == "derivable":
        print(
            "The dataset may contain an outcome represented through "
            "state/event columns. Do not create a target automatically; "
            "inspect the meaning of those columns first."
        )
    elif result.status == "behavioral_only":
        print(
            "Behavioral/relationship analysis is appropriate, but a true "
            "churn probability cannot be claimed without an outcome."
        )
    else:
        print(
            "There is not enough evidence to make a defensible churn "
            "prediction."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Discover whether a dataset contains a defensible churn/exit outcome."
    )

    parser.add_argument(
        "--data",
        default="dataset.csv",
        help="Path to CSV dataset",
    )

    args = parser.parse_args()

    dataframe = pd.read_csv(args.data)

    discovery = OutcomeDiscovery()

    result = discovery.discover(dataframe)

    print_report(result)