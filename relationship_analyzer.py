from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr


IDENTIFIER_KEYWORDS = (
    "customer",
    "cust",
    "user",
    "id",
    "name",
    "account",
    "member",
    "subscriber",
    "student",
    "employee",
)


@dataclass
class RelationshipResult:
    feature: str
    feature_type: str
    score: float
    interpretation: str
    details: Dict[str, Any]
    method: str = ""
    direction: str = "—"


class RelationshipAnalyzer:
    """Analyze statistical relationships with a known binary outcome."""

    def __init__(self, max_categories: int = 20) -> None:
        self.max_categories = max_categories

    @staticmethod
    def _is_identifier(series: pd.Series, col_name: str) -> bool:
        lower = str(col_name).strip().lower()
        if any(keyword in lower for keyword in IDENTIFIER_KEYWORDS):
            unique_ratio = series.nunique(dropna=True) / max(len(series.dropna()), 1)
            if unique_ratio >= 0.60:
                return True
        return False

    @staticmethod
    def _encode_binary_target(target: pd.Series) -> Optional[pd.Series]:
        clean = target.dropna()

        if clean.nunique() != 2:
            return None

        if pd.api.types.is_numeric_dtype(clean):
            values = sorted(clean.unique())
            return target.map({values[0]: 0, values[1]: 1})

        values = clean.astype(str).str.strip().str.lower()
        unique = list(values.dropna().unique())

        positive_hints = {
            "yes", "true", "1", "churn", "churned", "left",
            "exit", "exited", "attrition", "cancel", "cancelled",
            "canceled", "dropout", "dropped", "inactive", "closed",
            "terminated", "resigned"
        }

        positive = [v for v in unique if v in positive_hints]
        positive_value = positive[0] if len(positive) == 1 else sorted(unique)[-1]

        return values.map(lambda x: 1 if x == positive_value else 0)

    @staticmethod
    def _strength_label(score: float) -> str:
        value = abs(score)
        if value >= 0.50:
            return "strong"
        if value >= 0.30:
            return "moderate"
        if value >= 0.10:
            return "weak"
        return "very weak"

    @staticmethod
    def to_dataframe(results: List[RelationshipResult]) -> pd.DataFrame:
        columns = [
            "Feature",
            "Type",
            "Method",
            "Strength",
            "Direction",
            "Interpretation",
        ]
        if not results:
            return pd.DataFrame(columns=columns)

        rows = []
        for r in results:
            rows.append({
                "Feature": r.feature,
                "Type": r.feature_type.capitalize(),
                "Method": r.method,
                "Strength": round(float(r.score), 3),
                "Direction": r.direction,
                "Interpretation": r.interpretation,
                # Also include lowercase keys for flexibility
                "feature": r.feature,
                "type": r.feature_type.capitalize(),
                "method": r.method,
                "strength": round(float(r.score), 3),
                "direction": r.direction,
                "interpretation": r.interpretation,
            })
        return pd.DataFrame(rows)

    def numeric_relationships(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: Optional[List[str]] = None,
    ) -> List[RelationshipResult]:

        target = self._encode_binary_target(df[target_col])
        if target is None:
            return []

        results: List[RelationshipResult] = []
        excluded = set(exclude_cols or [])

        for col in df.select_dtypes(include=np.number).columns:
            if col == target_col or col in excluded:
                continue

            if self._is_identifier(df[col], col):
                continue

            pair = pd.DataFrame({
                "feature": df[col],
                "target": target,
            }).dropna()

            if len(pair) < 10 or pair["feature"].nunique() < 2:
                continue

            try:
                correlation, p_value = pointbiserialr(
                    pair["target"],
                    pair["feature"],
                )
            except Exception:
                continue

            if not np.isfinite(correlation):
                continue

            direction = "Positive" if correlation > 0 else "Negative" if correlation < 0 else "—"

            results.append(
                RelationshipResult(
                    feature=col,
                    feature_type="numeric",
                    score=float(abs(correlation)),
                    interpretation=(
                        f"{self._strength_label(correlation)} association "
                        f"(point-biserial r={correlation:.3f})"
                    ),
                    details={
                        "correlation": float(correlation),
                        "p_value": float(p_value),
                        "sample_size": int(len(pair)),
                    },
                    method="Point-biserial",
                    direction=direction,
                )
            )

        return sorted(results, key=lambda result: result.score, reverse=True)

    def categorical_relationships(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: Optional[List[str]] = None,
    ) -> List[RelationshipResult]:

        target = self._encode_binary_target(df[target_col])
        if target is None:
            return []

        results: List[RelationshipResult] = []
        excluded = set(exclude_cols or [])

        columns = df.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns
        for col in columns:
            if col == target_col or col in excluded:
                continue

            series = df[col].astype("string")
            unique_count = series.nunique(dropna=True)

            # Prevent identifier and near-unique fields from dominating
            if self._is_identifier(series, col):
                continue

            if unique_count < 2 or unique_count > self.max_categories:
                continue

            temp = pd.DataFrame({
                "feature": series,
                "target": target,
            }).dropna()

            if len(temp) < 10:
                continue

            try:
                table = pd.crosstab(temp["feature"], temp["target"])

                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue

                chi2, p_value, _, _ = chi2_contingency(table)
                n = table.to_numpy().sum()
                min_dim = min(table.shape) - 1

                if n <= 0 or min_dim <= 0:
                    continue

                cramers_v = float(np.sqrt((chi2 / n) / min_dim))

            except Exception:
                continue

            outcome_rates = (
                temp.groupby("feature")["target"]
                .mean()
                .sort_values(ascending=False)
            )

            results.append(
                RelationshipResult(
                    feature=col,
                    feature_type="categorical",
                    score=cramers_v,
                    interpretation=(
                        f"{self._strength_label(cramers_v)} association "
                        f"(Cramer's V={cramers_v:.3f})"
                    ),
                    details={
                        "cramers_v": cramers_v,
                        "p_value": float(p_value),
                        "sample_size": int(len(temp)),
                        "outcome_rate_by_category": {
                            str(category): round(float(rate), 4)
                            for category, rate in outcome_rates.items()
                        },
                    },
                    method="Cramer's V",
                    direction="-",
                )
            )

        return sorted(results, key=lambda result: result.score, reverse=True)

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        exclude_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        if not target_col or target_col not in df.columns:
            empty_table = self.to_dataframe([])
            return {
                "status": "no_target",
                "message": (
                    "No explicit target was supplied. Relationship analysis "
                    "against a known outcome cannot be performed yet."
                ),
                "numeric": [],
                "categorical": [],
                "top_features": [],
                "table": empty_table,
            }

        if df[target_col].dropna().nunique() != 2:
            empty_table = self.to_dataframe([])
            return {
                "status": "unsupported_target",
                "message": "Relationship analysis currently requires a binary target.",
                "numeric": [],
                "categorical": [],
                "top_features": [],
                "table": empty_table,
            }

        numeric = self.numeric_relationships(df, target_col, exclude_cols=exclude_cols)
        categorical = self.categorical_relationships(df, target_col, exclude_cols=exclude_cols)

        combined = sorted(
            numeric + categorical,
            key=lambda result: result.score,
            reverse=True,
        )

        table = self.to_dataframe(combined)

        return {
            "status": "success",
            "target": target_col,
            "numeric": numeric,
            "categorical": categorical,
            "top_features": combined,
            "table": table,
        }


def print_report(analysis: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("RELATIONSHIP ANALYSIS")
    print("=" * 70)

    if analysis["status"] != "success":
        print(analysis["message"])
        return

    print(f"Target: {analysis['target']}")
    print("\nStrongest relationships:")
    print("-" * 70)

    table = analysis.get("table")
    if table is not None and not table.empty:
        display_cols = ["Feature", "Type", "Method", "Strength", "Direction"]
        print(table[display_cols].head(10).to_string(index=False))
    else:
        print("No meaningful relationships identified.")

    print("\nImportant:")
    print("These are statistical associations in the supplied dataset.")
    print("They do not prove causation between features and the outcome.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze relationships between features and a binary outcome."
    )
    parser.add_argument("--data", default="dataset.csv")
    parser.add_argument("--target", default="working_status")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    target = args.target
    if target not in data.columns:
        # Fall back to outcome discovery if target not found
        try:
            from outcome_discovery import OutcomeDiscovery
            discovered = OutcomeDiscovery().discover(data)
            if discovered.target_column:
                target = discovered.target_column
        except Exception:
            pass

    analyzer = RelationshipAnalyzer()
    analysis = analyzer.analyze(data, target)
    print_report(analysis)
