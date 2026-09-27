from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from outcome_discovery import (
    OutcomeCandidate,
    OutcomeDiscovery,
    OutcomeDiscoveryResult,
)
from relationship_analyzer import RelationshipAnalyzer, RelationshipResult


# ============================================================
# KEYWORDS USED FOR AUTOMATIC DATASET UNDERSTANDING
# ============================================================

TARGET_KEYWORDS = (
    "churn",
    "exit",
    "attrition",
    "leave",
    "left",
    "dropout",
    "drop",
    "cancel",
    "cancellation",
    "target",
    "label",
    "status",
)

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

DATE_KEYWORDS = (
    "date",
    "time",
    "month",
    "day",
    "year",
)

POSITIVE_HINTS = (
    "yes",
    "true",
    "1",
    "churn",
    "left",
    "exit",
    "attrition",
    "cancel",
    "cancelled",
    "canceled",
    "dropout",
    "dropped",
    "leave",
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    missing_cells: int
    missing_percentage: float
    duplicate_rows: int

    target_candidates: List[Dict[str, object]]
    detected_target: Optional[str]
    target_confidence: float

    identifier_column: Optional[str]
    date_column: Optional[str]

    problem_type: Optional[str]


@dataclass
class AnalysisResult:
    dataframe: pd.DataFrame
    target_col: Optional[str]
    identifier_col: Optional[str]
    date_col: Optional[str]
    model_name: Optional[str]
    metrics: Dict[str, float]
    report: str
    confusion: Optional[np.ndarray]
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame
    suggestions: List[str]
    summary: Dict[str, object]
    outcome_status: str
    outcome_target: Optional[str]
    outcome_positive_class: Optional[str]
    outcome_confidence: float
    outcome_reason: str
    relationship_analysis: Dict[str, object]
    outcome_discovery: OutcomeDiscoveryResult
    relationships: pd.DataFrame


@dataclass
class TrainedModel:
    target_col: str
    identifier_col: Optional[str]
    date_col: Optional[str]
    model_name: str
    pipeline: Pipeline
    label_encoder: Optional[LabelEncoder]
    class_names: List[str]
    positive_class_index: int
    feature_importance: pd.DataFrame
    metrics: Dict[str, float]
    report: str
    confusion: np.ndarray


# ============================================================
# MAIN ENGINE
# ============================================================

class ChurnAnalysisEngine:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_model: Optional[TrainedModel] = None
        self.outcome_discovery = OutcomeDiscovery()
        self.relationship_analyzer = RelationshipAnalyzer()

    # ========================================================
    # TARGET DETECTION
    # ========================================================

    @staticmethod
    def detect_target_candidates(
        df: pd.DataFrame,
    ) -> List[Dict[str, object]]:
        """
        Find and score possible target columns.

        The score considers:

        1. Target-related column names
        2. Number of unique values
        3. Binary / low-cardinality nature
        4. Churn/exit-like values
        5. Identifier-like behavior
        6. High-cardinality penalty
        """

        candidates: List[Dict[str, object]] = []

        for col in df.columns:

            series = df[col]
            lower = str(col).lower()

            # Ignore completely empty columns
            if series.isna().all():
                continue

            unique_values = (
                series
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
            )

            unique_count = len(unique_values)

            # Target must have at least two possible values
            if unique_count < 2:
                continue

            score = 0.0
            reasons: List[str] = []

            # ------------------------------------------------
            # 1. COLUMN NAME EVIDENCE
            # ------------------------------------------------

            keyword_matches = [
                keyword
                for keyword in TARGET_KEYWORDS
                if keyword in lower
            ]

            if keyword_matches:
                score += 0.50

                reasons.append(
                    "target-related name: "
                    + ", ".join(keyword_matches)
                )

            # ------------------------------------------------
            # 2. BINARY / LOW CARDINALITY
            # ------------------------------------------------

            if unique_count == 2:

                score += 0.25
                reasons.append("binary outcome")

            elif 2 < unique_count <= 5:

                score += 0.10
                reasons.append("low-cardinality outcome")

            # ------------------------------------------------
            # 3. POSITIVE OUTCOME VALUES
            # ------------------------------------------------

            positive_matches = [
                value
                for value in unique_values
                if any(
                    hint in value
                    for hint in POSITIVE_HINTS
                )
            ]

            if positive_matches:

                score += 0.20

                reasons.append(
                    "contains churn/exit-like values"
                )

            # ------------------------------------------------
            # 4. IDENTIFIER PENALTY
            # ------------------------------------------------

            identifier_keywords = [
                keyword
                for keyword in IDENTIFIER_KEYWORDS
                if keyword in lower
            ]

            if (
                identifier_keywords
                and series.nunique(dropna=True)
                >= len(df) * 0.8
            ):

                score -= 0.40

                reasons.append(
                    "looks like an identifier"
                )

            # ------------------------------------------------
            # 5. HIGH-CARDINALITY PENALTY
            # ------------------------------------------------

            if (
                unique_count
                > max(20, int(len(df) * 0.5))
            ):

                score -= 0.20

                reasons.append(
                    "high-cardinality column"
                )

            # Keep score between 0 and 1
            score = max(
                0.0,
                min(score, 1.0),
            )

            if score > 0:

                candidates.append(
                    {
                        "column": col,
                        "score": round(score, 3),
                        "unique_values": unique_count,
                        "reasons": reasons,
                    }
                )

        # Highest-scoring candidates first
        candidates.sort(
            key=lambda item: float(
                item["score"]
            ),
            reverse=True,
        )

        return candidates

    @staticmethod
    def detect_target_column(
        df: pd.DataFrame,
    ) -> Optional[str]:

        candidates = (
            ChurnAnalysisEngine
            .detect_target_candidates(df)
        )

        if not candidates:
            return None

        best = candidates[0]

        # Require reasonable evidence
        if float(best["score"]) < 0.40:
            return None

        return str(best["column"])

    # ========================================================
    # IDENTIFIER DETECTION
    # ========================================================

    @staticmethod
    def detect_identifier_column(
        df: pd.DataFrame,
    ) -> Optional[str]:

        for col in df.columns:

            lower = col.lower()

            if any(
                keyword in lower
                for keyword in IDENTIFIER_KEYWORDS
            ):

                unique_ratio = (
                    df[col]
                    .nunique(dropna=True)
                    / max(len(df), 1)
                )

                if unique_ratio >= 0.60:
                    return col

        return None

    # ========================================================
    # DATE DETECTION
    # ========================================================

    @staticmethod
    def detect_date_column(
        df: pd.DataFrame,
    ) -> Optional[str]:

        # --------------------------------------------------------
        # 1. Prefer columns that are already true datetime types
        # --------------------------------------------------------
        for col in df.columns:

            if pd.api.types.is_datetime64_any_dtype(
                df[col]
            ):

                return col

        # --------------------------------------------------------
        # 2. Check only strongly date-related column names
        # --------------------------------------------------------
        # Avoid generic words such as "month" because they can
        # appear in normal numeric feature names like MonthlyCharges.
        strong_date_keywords = (
            "date",
            "datetime",
            "timestamp",
            "created_at",
            "updated_at",
            "signup",
            "sign_up",
            "joined",
            "joining",
            "registration",
            "registered",
            "payment_date",
            "start_date",
            "end_date",
            "birth_date",
            "dob",
            "last_login",
            "login_date",
            "event_time",
            "event_date",
        )

        for col in df.columns:

            lower = str(col).strip().lower()

            if not any(
                keyword in lower
                for keyword in strong_date_keywords
            ):
                continue

            # Numeric columns such as MonthlyCharges, Tenure,
            # Year, etc. should not be treated as dates merely
            # because their names contain a date-related word.
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            return col

        # --------------------------------------------------------
        # 3. For string columns, only parse values when the column
        # name itself contains a safe date/time hint. This avoids
        # repeatedly calling pd.to_datetime() on normal categorical
        # columns and removes the dateutil inference warnings.
        # --------------------------------------------------------
        parse_hints = (
            "date",
            "datetime",
            "timestamp",
            "time",
            "signup",
            "sign_up",
            "joined",
            "joining",
            "registration",
            "registered",
            "login",
            "dob",
        )

        for col in df.columns:

            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                continue

            lower = str(col).strip().lower()

            if not any(
                hint in lower
                for hint in parse_hints
            ):
                continue

            non_null = series.dropna()

            if len(non_null) < 3:
                continue

            try:
                parsed = pd.to_datetime(
                    non_null,
                    format="mixed",
                    errors="coerce",
                )

                parse_ratio = parsed.notna().mean()

                # Require most values to be valid dates.
                if parse_ratio >= 0.80:
                    return col

            except (TypeError, ValueError):
                continue

        return None

    # ========================================================
    # PROBLEM TYPE DETECTION
    # ========================================================

    @staticmethod
    def detect_problem_type(
        df: pd.DataFrame,
        target_col: Optional[str],
    ) -> Optional[str]:

        if (
            not target_col
            or target_col not in df.columns
        ):
            return None

        target = df[target_col].dropna()

        if target.empty:
            return None

        unique_count = target.nunique()

        # Numeric target
        if pd.api.types.is_numeric_dtype(target):

            if unique_count <= 2:
                return "Binary Classification"

            if unique_count <= 20:
                return "Multiclass Classification"

            return "Regression"

        # Categorical/string target
        if unique_count == 2:
            return "Binary Classification"

        if unique_count <= 20:
            return "Multiclass Classification"

        return "High-Cardinality Classification"

    # ========================================================
    # DATASET PROFILING
    # ========================================================

    def profile_dataset(
        self,
        df: pd.DataFrame,
    ) -> DatasetProfile:

        numeric_columns = (
            df.select_dtypes(
                include=["number"]
            )
            .columns
            .tolist()
        )

        categorical_columns = [
            col
            for col in df.columns
            if col not in numeric_columns
        ]

        missing_cells = int(
            df.isna()
            .sum()
            .sum()
        )

        total_cells = max(
            df.shape[0]
            * max(df.shape[1], 1),
            1,
        )

        missing_percentage = (
            missing_cells
            / total_cells
        ) * 100

        duplicate_rows = int(
            df.duplicated().sum()
        )

        target_candidates = (
            self.detect_target_candidates(df)
        )

        detected_target = (
            self.detect_target_column(df)
        )

        target_confidence = 0.0

        if target_candidates:

            best_candidate = (
                target_candidates[0]
            )

            if (
                detected_target
                and best_candidate["column"]
                == detected_target
            ):

                target_confidence = float(
                    best_candidate["score"]
                )

        identifier_column = (
            self.detect_identifier_column(df)
        )

        date_column = (
            self.detect_date_column(df)
        )

        problem_type = (
            self.detect_problem_type(
                df,
                detected_target,
            )
        )

        return DatasetProfile(
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            missing_cells=missing_cells,
            missing_percentage=round(
                missing_percentage,
                2,
            ),
            duplicate_rows=duplicate_rows,
            target_candidates=target_candidates,
            detected_target=detected_target,
            target_confidence=round(
                target_confidence,
                3,
            ),
            identifier_column=identifier_column,
            date_column=date_column,
            problem_type=problem_type,
        )

    # ========================================================
    # PREPROCESSOR
    # ========================================================

    @staticmethod
    def _build_preprocessor(
        num_cols: List[str],
        cat_cols: List[str],
    ) -> ColumnTransformer:

        numeric_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipe,
                    num_cols,
                ),
                (
                    "cat",
                    categorical_pipe,
                    cat_cols,
                ),
            ]
        )

    # ========================================================
    # POSITIVE CLASS DETECTION
    # ========================================================

    @staticmethod
    def _choose_positive_class(
        class_names: List[str],
    ) -> int:

        for idx, name in enumerate(
            class_names
        ):

            lower = (
                str(name)
                .strip()
                .lower()
            )

            if any(
                hint in lower
                for hint in POSITIVE_HINTS
            ):

                return idx

        return (
            1
            if len(class_names) > 1
            else 0
        )

    # ========================================================
    # SAFE ROC-AUC
    # ========================================================

    @staticmethod
    def _safe_auc(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        positive_class_index: int = 1,
    ) -> float:

        try:

            from sklearn.metrics import (
                roc_auc_score
            )

            if (
                len(np.unique(y_true)) == 2
                and y_proba.ndim == 2
                and y_proba.shape[1] >= 2
            ):

                y_binary = (y_true == positive_class_index).astype(int)
                pos_col = min(positive_class_index, y_proba.shape[1] - 1)

                if len(np.unique(y_binary)) == 2:
                    return float(
                        roc_auc_score(
                            y_binary,
                            y_proba[:, pos_col],
                        )
                    )

        except Exception:

            return float("nan")

        return float("nan")

    # ========================================================
    # FIND COLUMN BY KEYWORDS
    # ========================================================

    @staticmethod
    def _find_column_by_keywords(
        columns: List[str],
        keywords: Tuple[str, ...],
    ) -> Optional[str]:

        for col in columns:

            lower = col.lower()

            if any(
                keyword in lower
                for keyword in keywords
            ):

                return col

        return None

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def fit(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        positive_class_hint: Optional[str] = None,
    ) -> Optional[TrainedModel]:

        target = (
            target_col
            or self.detect_target_column(df)
        )

        if (
            not target
            or target not in df.columns
        ):

            self.trained_model = None

            return None

        data = df.copy()

        # Remove duplicate rows
        data = (
            data
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Remove rows where target is missing
        data = data.dropna(
            subset=[target]
        )

        identifier_col = self.detect_identifier_column(data)
        cols_to_drop = [target]
        if identifier_col and identifier_col in data.columns:
            cols_to_drop.append(identifier_col)

        X = data.drop(
            columns=cols_to_drop
        )

        y_raw = data[target]

        # ----------------------------------------------------
        # TARGET ENCODING
        # ----------------------------------------------------

        label_encoder: Optional[
            LabelEncoder
        ] = None

        if (
            y_raw.dtype == "object"
            or str(
                y_raw.dtype
            ).startswith("category")
            or y_raw.dtype == "bool"
        ):

            label_encoder = LabelEncoder()

            y = label_encoder.fit_transform(
                y_raw.astype(str)
            )

            class_names = list(
                label_encoder.classes_
            )

        else:

            y = y_raw.to_numpy()

            class_names = [
                str(c)
                for c in np.unique(y)
            ]

        # At least two classes required
        if len(np.unique(y)) < 2:

            self.trained_model = None

            return None

        if positive_class_hint:
            pos_hint_lower = str(positive_class_hint).strip().lower()
            matched = [
                idx for idx, c in enumerate(class_names)
                if str(c).strip().lower() == pos_hint_lower
            ]
            positive_class_index = matched[0] if matched else self._choose_positive_class(class_names)
        else:
            positive_class_index = self._choose_positive_class(class_names)

        # ----------------------------------------------------
        # FEATURE TYPES
        # ----------------------------------------------------

        num_cols = (
            X.select_dtypes(
                include=["number"]
            )
            .columns
            .tolist()
        )

        cat_cols = [
            col
            for col in X.columns
            if col not in num_cols
        ]

        preprocessor = (
            self._build_preprocessor(
                num_cols,
                cat_cols,
            )
        )

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        stratify = (
            y
            if len(np.unique(y)) > 1
            else None
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=stratify,
            )
        )

        # ----------------------------------------------------
        # CANDIDATE MODELS
        # ----------------------------------------------------

        candidates = {

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=self.random_state,
                    class_weight="balanced",
                ),

            "Logistic Regression":
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                ),
        }

        best_name: Optional[str] = None

        best_pipeline: Optional[
            Pipeline
        ] = None

        best_pred = None

        best_score = -1.0

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        for name, model in candidates.items():

            pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            pred = pipeline.predict(
                X_test
            )

            score = f1_score(
                y_test,
                pred,
                average="weighted",
            )

            if score > best_score:

                best_name = name
                best_pipeline = pipeline
                best_pred = pred
                best_score = score

        if (
            best_pipeline is None
            or best_name is None
            or best_pred is None
        ):

            self.trained_model = None

            return None

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        proba = None

        if hasattr(
            best_pipeline
            .named_steps["model"],
            "predict_proba",
        ):

            proba = (
                best_pipeline
                .predict_proba(X_test)
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = {

            "accuracy":
                float(
                    accuracy_score(
                        y_test,
                        best_pred,
                    )
                ),

            "f1_weighted":
                float(
                    f1_score(
                        y_test,
                        best_pred,
                        average="weighted",
                    )
                ),
        }

        if proba is not None:

            metrics["roc_auc"] = (
                self._safe_auc(
                    y_test,
                    proba,
                    positive_class_index=positive_class_index,
                )
            )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            best_pred,
            zero_division=0,
        )

        confusion = confusion_matrix(
            y_test,
            best_pred,
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        feature_importance = (
            self._extract_feature_importance(
                best_pipeline
            )
        )

        # ----------------------------------------------------
        # SAVE TRAINED MODEL IN MEMORY
        # ----------------------------------------------------

        self.trained_model = TrainedModel(

            target_col=target,

            identifier_col=identifier_col,

            date_col=(
                self.detect_date_column(
                    df
                )
            ),

            model_name=best_name,

            pipeline=best_pipeline,

            label_encoder=label_encoder,

            class_names=[
                str(c)
                for c in class_names
            ],

            positive_class_index=(
                positive_class_index
            ),

            feature_importance=(
                feature_importance
            ),

            metrics=metrics,

            report=report,

            confusion=confusion,
        )

        return self.trained_model

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    @staticmethod
    def _extract_feature_importance(
        pipeline: Pipeline,
    ) -> pd.DataFrame:

        preprocessor = (
            pipeline
            .named_steps["preprocessor"]
        )

        model = (
            pipeline
            .named_steps["model"]
        )

        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
                .tolist()
            )

        except Exception:

            feature_names = []

        values: Optional[
            np.ndarray
        ] = None

        if hasattr(
            model,
            "feature_importances_",
        ):

            values = (
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_",
        ):

            coef = np.abs(
                model.coef_
            )

            values = (
                coef.mean(axis=0)
                if coef.ndim == 2
                else coef
            )

        if (
            values is None
            or not feature_names
        ):

            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Importance",
                ]
            )

        frame = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": values,
            }
        )

        frame["Feature"] = (
            frame["Feature"]
            .astype(str)
            .str.replace(
                r"^(num|cat)__",
                "",
                regex=True,
            )
        )

        frame["BaseFeature"] = (
            frame["Feature"]
            .str.split(
                "_",
                n=1,
            )
            .str[0]
        )

        grouped = (
            frame
            .groupby(
                "BaseFeature",
                as_index=False,
            )["Importance"]
            .sum()
            .sort_values(
                "Importance",
                ascending=False,
            )
        )

        grouped = grouped.rename(
            columns={
                "BaseFeature": "Feature"
            }
        )

        return grouped.reset_index(
            drop=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_frame(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.trained_model is None:

            raise RuntimeError(
                "No trained model is available."
            )

        target_col = (
            self.trained_model.target_col
        )

        input_df = df.copy()

        # Remove target and identifier from model input features
        cols_to_drop = [target_col]
        if (
            self.trained_model.identifier_col
            and self.trained_model.identifier_col in input_df.columns
        ):
            cols_to_drop.append(self.trained_model.identifier_col)

        feature_df = input_df.drop(
            columns=[c for c in cols_to_drop if c in input_df.columns]
        )

        # Predict encoded classes
        pred_encoded = (
            self.trained_model
            .pipeline
            .predict(feature_df)
        )

        # Convert back to original labels
        if (
            self.trained_model
            .label_encoder is not None
        ):

            pred_labels = (
                self.trained_model
                .label_encoder
                .inverse_transform(
                    pred_encoded.astype(int)
                )
            )

        else:

            pred_labels = pred_encoded

        result = input_df.copy()

        # ----------------------------------------------------
        # PREDICTION LABELS
        # ----------------------------------------------------

        result["Prediction Label"] = (
            pred_labels
        )

        pos_idx = self.trained_model.positive_class_index
        if 0 <= pos_idx < len(self.trained_model.class_names):
            pos_name = str(self.trained_model.class_names[pos_idx])
        else:
            pos_name = "Positive"

        if len(self.trained_model.class_names) == 2:
            neg_idx = 1 - pos_idx if pos_idx in (0, 1) else (0 if pos_idx != 0 else 1)
            neg_name = str(self.trained_model.class_names[neg_idx])
        else:
            neg_name = "Negative"

        pos_lower = pos_name.strip().lower()

        result["Churn Prediction"] = np.where(
            pd.Series(pred_labels).astype(str).str.strip().str.lower() == pos_lower,
            pos_name,
            neg_name,
        )

        # ----------------------------------------------------
        # CHURN PROBABILITY
        # ----------------------------------------------------

        model = (
            self.trained_model
            .pipeline
            .named_steps["model"]
        )

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = (
                self.trained_model
                .pipeline
                .predict_proba(
                    feature_df
                )
            )

            class_index = min(
                self.trained_model
                .positive_class_index,
                probabilities.shape[1] - 1,
            )

            result[
                "Churn Probability"
            ] = (
                probabilities[
                    :,
                    class_index
                ].round(4)
            )

            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            result["Risk Level"] = (
                pd.cut(
                    result[
                        "Churn Probability"
                    ],

                    bins=[
                        -0.001,
                        0.35,
                        0.65,
                        1.0,
                    ],

                    labels=[
                        "Low",
                        "Medium",
                        "High",
                    ],
                )
                .astype(str)
            )

        else:

            result[
                "Churn Probability"
            ] = np.nan

            result[
                "Risk Level"
            ] = "Unknown"

        return result

    # ========================================================
    # SUGGESTIONS
    # ========================================================

    def build_suggestions(
        self,
        df: pd.DataFrame,
        result: Optional[TrainedModel],
    ) -> List[str]:

        suggestions: List[str] = []

        # ----------------------------------------------------
        # MISSING DATA
        # ----------------------------------------------------

        missing_pct = float(
            df.isna()
            .sum()
            .sum()
            /
            max(
                df.shape[0]
                * max(df.shape[1], 1),
                1,
            )
            * 100
        )

        if missing_pct > 5:

            suggestions.append(
                f"Reduce missing data first. "
                f"About {missing_pct:.1f}% "
                f"of values are missing."
            )

        # ----------------------------------------------------
        # NO TARGET
        # ----------------------------------------------------

        if result is None:

            suggestions.append(
                "No suitable outcome target "
                "was detected. Consider "
                "selecting or naming a column "
                "that represents the outcome."
            )

            return suggestions

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        risk_pct = 0.0

        predictions = pd.DataFrame()

        try:

            predictions = (
                self.predict_frame(df)
            )

            risk_pct = float(
                (
                    predictions[
                        "Risk Level"
                    ]
                    == "High"
                ).mean()
                * 100
            )

        except Exception:

            pass

        # ----------------------------------------------------
        # HIGH-RISK ANALYSIS
        # ----------------------------------------------------

        if (
            not predictions.empty
            and "Risk Level"
            in predictions.columns
        ):

            high_mask = (
                predictions[
                    "Risk Level"
                ]
                == "High"
            )

            high_df = df.loc[
                high_mask
            ].copy()

            num_cols = (
                df
                .select_dtypes(
                    include=["number"]
                )
                .columns
                .tolist()
            )

            cat_cols = (
                df
                .select_dtypes(
                    exclude=["number"]
                )
                .columns
                .tolist()
            )

            # ------------------------------------------------
            # CHARGE ANALYSIS
            # ------------------------------------------------

            charge_col = (
                self._find_column_by_keywords(
                    num_cols,
                    (
                        "charge",
                        "bill",
                        "amount",
                        "spend",
                        "payment",
                    ),
                )
            )

            if (
                charge_col
                and not high_df.empty
            ):

                high_avg = float(
                    high_df[
                        charge_col
                    ].mean()
                )

                all_avg = float(
                    df[
                        charge_col
                    ].mean()
                )

                if (
                    pd.notna(high_avg)
                    and pd.notna(all_avg)
                    and high_avg > all_avg
                ):

                    suggestions.append(
                        f"High-risk customers "
                        f"have higher "
                        f"'{charge_col}' "
                        f"on average. "
                        f"Consider targeted "
                        f"pricing or retention offers."
                    )

            # ------------------------------------------------
            # TENURE ANALYSIS
            # ------------------------------------------------

            tenure_col = (
                self._find_column_by_keywords(
                    num_cols,
                    (
                        "tenure",
                        "duration",
                        "lifetime",
                        "months",
                        "age",
                    ),
                )
            )

            if (
                tenure_col
                and not high_df.empty
            ):

                high_avg = float(
                    high_df[
                        tenure_col
                    ].mean()
                )

                all_avg = float(
                    df[
                        tenure_col
                    ].mean()
                )

                if (
                    pd.notna(high_avg)
                    and pd.notna(all_avg)
                    and high_avg < all_avg
                ):

                    suggestions.append(
                        f"High-risk customers "
                        f"show lower "
                        f"'{tenure_col}'. "
                        f"Strengthen onboarding "
                        f"and early engagement."
                    )

            # ------------------------------------------------
            # CONTRACT / PLAN ANALYSIS
            # ------------------------------------------------

            contract_col = (
                self._find_column_by_keywords(
                    cat_cols,
                    (
                        "contract",
                        "plan",
                        "subscription",
                        "term",
                        "billing",
                    ),
                )
            )

            if (
                contract_col
                and not high_df.empty
            ):

                mode_series = (
                    high_df[
                        contract_col
                    ]
                    .astype(str)
                    .str.lower()
                    .mode()
                )

                if not mode_series.empty:

                    top_group = (
                        mode_series.iloc[0]
                    )

                    if any(
                        token in top_group
                        for token in (
                            "month",
                            "monthly",
                            "short",
                        )
                    ):

                        suggestions.append(
                            f"Most high-risk "
                            f"users are in "
                            f"'{contract_col}="
                            f"{top_group}'. "
                            f"Consider promoting "
                            f"longer-term plans."
                        )

                    else:

                        suggestions.append(
                            f"Monitor segment "
                            f"'{contract_col}="
                            f"{top_group}' "
                            f"closely and "
                            f"run targeted "
                            f"retention campaigns."
                        )

            # ------------------------------------------------
            # OVERALL RISK
            # ------------------------------------------------

            if risk_pct >= 30:

                suggestions.append(
                    "High-risk volume is "
                    "significant. Prioritize "
                    "closer monitoring for "
                    "high-risk records."
                )

            elif risk_pct >= 15:

                suggestions.append(
                    "Risk is moderate. "
                    "Schedule periodic "
                    "follow-ups and monitor "
                    "risky segments."
                )

            else:

                suggestions.append(
                    "Risk is currently "
                    "controlled. Maintain "
                    "engagement programs "
                    "and continue monitoring."
                )

        # ----------------------------------------------------
        # TOP FEATURES
        # ----------------------------------------------------

        top_features = (
            result
            .feature_importance
            .head(5)
        )

        if not top_features.empty:

            labels = ", ".join(
                top_features[
                    "Feature"
                ]
                .tolist()[:3]
            )

            suggestions.append(
                "Key churn drivers in "
                f"this dataset are: "
                f"{labels}. Use these "
                "for targeted analysis."
            )

        suggestions.append(
            f"High-risk share is "
            f"{risk_pct:.1f}%. "
            "Focus retention actions "
            "on the highest-risk "
            "customers first."
        )

        return suggestions[:6]

    # ========================================================
    # SUMMARY
    # ========================================================

    def summarize(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, object]:

        profile = (
            self.profile_dataset(df)
        )

        return {

            "rows":
                profile.rows,

            "columns":
                profile.columns,

            "numeric_columns":
                len(
                    profile.numeric_columns
                ),

            "categorical_columns":
                len(
                    profile.categorical_columns
                ),

            "missing_cells":
                profile.missing_cells,

            "missing_percentage":
                profile.missing_percentage,

            "duplicate_rows":
                profile.duplicate_rows,

            "detected_target":
                profile.detected_target,

            "target_confidence":
                profile.target_confidence,

            "detected_identifier":
                profile.identifier_column,

            "detected_date":
                profile.date_column,

            "problem_type":
                profile.problem_type,

            "target_candidates":
                profile.target_candidates,
        }

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> AnalysisResult:
        # --------------------------------------------------------
        # 1. Profile dataset
        # --------------------------------------------------------
        profile = self.profile_dataset(df)
        exclude_cols = [profile.identifier_column] if profile.identifier_column else []

        # --------------------------------------------------------
        # 2. Discover whether a defensible outcome exists
        # --------------------------------------------------------
        outcome = self.outcome_discovery.discover(df)

        # --------------------------------------------------------
        # 3. Analyze relationships against detected outcome (excluding identifiers)
        # --------------------------------------------------------
        if outcome.status == "explicit" and outcome.target_column:
            relationship_analysis = self.relationship_analyzer.analyze(
                df,
                outcome.target_column,
                exclude_cols=exclude_cols,
            )
        else:
            relationship_analysis = self.relationship_analyzer.analyze(
                df,
                None,
                exclude_cols=exclude_cols,
            )

        relationships = relationship_analysis.get("table", pd.DataFrame())

        # --------------------------------------------------------
        # 4. Train only when a defensible supervised target exists
        # --------------------------------------------------------
        trained: Optional[TrainedModel] = None

        if outcome.status == "explicit" and outcome.target_column:
            trained = self.fit(
                df,
                target_col=outcome.target_column,
                positive_class_hint=outcome.positive_class,
            )
        else:
            self.trained_model = None

        predictions = pd.DataFrame(index=df.index)
        metrics: Dict[str, float] = {}
        report = ""
        confusion = None
        feature_importance = pd.DataFrame(
            columns=["Feature", "Importance"]
        )
        suggestions: List[str] = []
        model_name = None

        if trained is not None:
            predictions = self.predict_frame(df)
            metrics = trained.metrics
            report = trained.report
            confusion = trained.confusion
            feature_importance = trained.feature_importance
            model_name = trained.model_name
            suggestions = self.build_suggestions(df, trained)
        else:
            suggestions = self.build_suggestions(df, None)

        effective_target = outcome.target_column if outcome.status == "explicit" else None

        summary = self.summarize(df)
        summary.update({
            "detected_target": effective_target,
            "target_confidence": float(outcome.confidence),
            "outcome_status": outcome.status,
            "outcome_target": outcome.target_column,
            "outcome_positive_class": outcome.positive_class,
            "outcome_confidence": float(outcome.confidence),
            "outcome_reason": outcome.reason,
            "outcome_state_columns": outcome.state_columns,
            "outcome_date_columns": outcome.date_columns,
            "relationship_top_features": relationship_analysis.get(
                "top_features", []
            ),
        })

        return AnalysisResult(
            dataframe=df,
            target_col=effective_target,
            identifier_col=profile.identifier_column,
            date_col=profile.date_column,
            model_name=model_name,
            metrics=metrics,
            report=report,
            confusion=confusion,
            feature_importance=feature_importance,
            predictions=predictions,
            suggestions=suggestions,
            summary=summary,
            outcome_status=outcome.status,
            outcome_target=outcome.target_column,
            outcome_positive_class=outcome.positive_class,
            outcome_confidence=float(outcome.confidence),
            outcome_reason=outcome.reason,
            relationship_analysis=relationship_analysis,
            outcome_discovery=outcome,
            relationships=relationships,
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save(
        self,
        path: str,
    ) -> None:

        if self.trained_model is None:

            raise RuntimeError(
                "Nothing to save."
            )

        joblib.dump(
            self.trained_model,
            path,
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load(
        self,
        path: str,
    ) -> TrainedModel:

        self.trained_model = (
            joblib.load(path)
        )

        return self.trained_model