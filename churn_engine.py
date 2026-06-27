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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

TARGET_KEYWORDS = ("churn", "exit", "attrition", "leave", "left", "target", "label", "status")
IDENTIFIER_KEYWORDS = ("customer", "cust", "user", "id", "name", "account", "member", "subscriber")
DATE_KEYWORDS = ("date", "time", "month", "day", "year")
POSITIVE_HINTS = ("yes", "true", "1", "churn", "left", "exit", "attrition", "cancel")


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


class ChurnAnalysisEngine:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_model: Optional[TrainedModel] = None

    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if any(keyword in col.lower() for keyword in TARGET_KEYWORDS):
                return col

        binary_candidates = []
        for col in df.columns:
            unique_values = df[col].dropna().astype(str).str.lower().unique()
            if 2 <= len(unique_values) <= 5:
                if any(any(hint in value for hint in POSITIVE_HINTS) for value in unique_values):
                    binary_candidates.append(col)

        return binary_candidates[0] if binary_candidates else None

    @staticmethod
    def detect_identifier_column(df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            lower = col.lower()
            if any(keyword in lower for keyword in IDENTIFIER_KEYWORDS):
                if df[col].nunique(dropna=True) >= max(int(len(df) * 0.6), 1):
                    return col
        return None

    @staticmethod
    def detect_date_column(df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            lower = col.lower()
            if any(keyword in lower for keyword in DATE_KEYWORDS):
                return col
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return None

    @staticmethod
    def _build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ]
        )

    @staticmethod
    def _choose_positive_class(class_names: List[str]) -> int:
        for idx, name in enumerate(class_names):
            lower = str(name).strip().lower()
            if any(hint in lower for hint in POSITIVE_HINTS):
                return idx
        return 1 if len(class_names) > 1 else 0

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        try:
            from sklearn.metrics import roc_auc_score

            if len(np.unique(y_true)) == 2 and y_proba.shape[1] >= 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            return float("nan")
        return float("nan")

    @staticmethod
    def _find_column_by_keywords(columns: List[str], keywords: Tuple[str, ...]) -> Optional[str]:
        for col in columns:
            lower = col.lower()
            if any(keyword in lower for keyword in keywords):
                return col
        return None

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Optional[TrainedModel]:
        target = target_col or self.detect_target_column(df)
        if not target or target not in df.columns:
            self.trained_model = None
            return None

        data = df.copy()
        data = data.drop_duplicates().reset_index(drop=True)
        data = data.dropna(subset=[target])

        X = data.drop(columns=[target])
        y_raw = data[target]

        label_encoder: Optional[LabelEncoder] = None
        if y_raw.dtype == "object" or str(y_raw.dtype).startswith("category") or y_raw.dtype == "bool":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw.astype(str))
            class_names = list(label_encoder.classes_)
        else:
            y = y_raw.to_numpy()
            class_names = [str(c) for c in np.unique(y)]

        if len(np.unique(y)) < 2:
            self.trained_model = None
            return None

        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [col for col in X.columns if col not in num_cols]
        preprocessor = self._build_preprocessor(num_cols, cat_cols)

        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=stratify,
        )

        candidates = {
            "Random Forest": RandomForestClassifier(
                n_estimators=350,
                random_state=self.random_state,
                class_weight="balanced",
            ),
            "Logistic Regression": LogisticRegression(max_iter=2500, class_weight="balanced"),
        }

        best_name: Optional[str] = None
        best_pipeline: Optional[Pipeline] = None
        best_pred = None
        best_score = -1.0

        for name, model in candidates.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            score = f1_score(y_test, pred, average="weighted")
            if score > best_score:
                best_name = name
                best_pipeline = pipeline
                best_pred = pred
                best_score = score

        if best_pipeline is None or best_name is None or best_pred is None:
            self.trained_model = None
            return None

        proba = None
        if hasattr(best_pipeline.named_steps["model"], "predict_proba"):
            proba = best_pipeline.predict_proba(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, best_pred)),
            "f1_weighted": float(f1_score(y_test, best_pred, average="weighted")),
        }
        if proba is not None:
            metrics["roc_auc"] = self._safe_auc(y_test, proba)

        report = classification_report(y_test, best_pred, zero_division=0)
        confusion = confusion_matrix(y_test, best_pred)
        feature_importance = self._extract_feature_importance(best_pipeline)
        positive_class_index = self._choose_positive_class(class_names)

        self.trained_model = TrainedModel(
            target_col=target,
            identifier_col=self.detect_identifier_column(df),
            date_col=self.detect_date_column(df),
            model_name=best_name,
            pipeline=best_pipeline,
            label_encoder=label_encoder,
            class_names=[str(c) for c in class_names],
            positive_class_index=positive_class_index,
            feature_importance=feature_importance,
            metrics=metrics,
            report=report,
            confusion=confusion,
        )
        return self.trained_model

    @staticmethod
    def _extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = []

        values: Optional[np.ndarray] = None
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            values = coef.mean(axis=0) if coef.ndim == 2 else coef

        if values is None or not feature_names:
            return pd.DataFrame(columns=["Feature", "Importance"])

        frame = pd.DataFrame({"Feature": feature_names, "Importance": values})
        frame["Feature"] = frame["Feature"].astype(str).str.replace(r"^(num|cat)__", "", regex=True)
        frame["BaseFeature"] = frame["Feature"].str.split("_", n=1).str[0]
        grouped = (
            frame.groupby("BaseFeature", as_index=False)["Importance"].sum().sort_values("Importance", ascending=False)
        )
        grouped = grouped.rename(columns={"BaseFeature": "Feature"})
        return grouped.reset_index(drop=True)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.trained_model is None:
            raise RuntimeError("No trained model is available.")

        target_col = self.trained_model.target_col
        input_df = df.copy()
        if target_col in input_df.columns:
            input_df = input_df.drop(columns=[target_col])

        pred_encoded = self.trained_model.pipeline.predict(input_df)
        if self.trained_model.label_encoder is not None:
            pred_labels = self.trained_model.label_encoder.inverse_transform(pred_encoded.astype(int))
        else:
            pred_labels = pred_encoded

        result = input_df.copy()
        result["Churn Prediction"] = np.where(
            pd.Series(pred_labels).astype(str).str.lower().isin(["yes", "1", "true", "churn", "left", "exit"]),
            "Churn ❌",
            "No Churn ✅",
        )
        result["Prediction Label"] = pred_labels

        model = self.trained_model.pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            probabilities = self.trained_model.pipeline.predict_proba(input_df)
            class_index = min(self.trained_model.positive_class_index, probabilities.shape[1] - 1)
            result["Churn Probability"] = probabilities[:, class_index].round(4)
            result["Risk Level"] = pd.cut(
                result["Churn Probability"],
                bins=[-0.001, 0.35, 0.65, 1.0],
                labels=["Low", "Medium", "High"],
            ).astype(str)
        else:
            result["Churn Probability"] = np.nan
            result["Risk Level"] = "Unknown"

        return result

    def build_suggestions(self, df: pd.DataFrame, result: Optional[TrainedModel]) -> List[str]:
        suggestions: List[str] = []
        missing_pct = float(df.isna().sum().sum() / max(df.shape[0] * max(df.shape[1], 1), 1) * 100)
        if missing_pct > 5:
            suggestions.append(f"Reduce missing data first. About {missing_pct:.1f}% of values are missing.")

        if result is None:
            suggestions.append("No churn target was detected. Rename the target column to include churn/exit/target for automated prediction.")
            return suggestions

        risk_pct = 0.0
        predictions = pd.DataFrame()
        try:
            predictions = self.predict_frame(df)
            risk_pct = float((predictions["Risk Level"] == "High").mean() * 100)
        except Exception:
            pass

        if not predictions.empty and "Risk Level" in predictions.columns:
            high_mask = predictions["Risk Level"] == "High"
            high_df = df.loc[high_mask].copy()

            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

            charge_col = self._find_column_by_keywords(num_cols, ("charge", "bill", "amount", "spend", "payment"))
            if charge_col and not high_df.empty:
                high_avg = float(high_df[charge_col].mean())
                all_avg = float(df[charge_col].mean())
                if pd.notna(high_avg) and pd.notna(all_avg) and high_avg > all_avg:
                    suggestions.append(
                        f"High-risk customers have higher '{charge_col}' on average. Offer discounts or flexible pricing plans."
                    )

            tenure_col = self._find_column_by_keywords(num_cols, ("tenure", "duration", "lifetime", "months", "age"))
            if tenure_col and not high_df.empty:
                high_avg = float(high_df[tenure_col].mean())
                all_avg = float(df[tenure_col].mean())
                if pd.notna(high_avg) and pd.notna(all_avg) and high_avg < all_avg:
                    suggestions.append(
                        f"High-risk customers show lower '{tenure_col}'. Strengthen onboarding and first-90-day engagement."
                    )

            contract_col = self._find_column_by_keywords(cat_cols, ("contract", "plan", "subscription", "term", "billing"))
            if contract_col and not high_df.empty:
                mode_series = high_df[contract_col].astype(str).str.lower().mode()
                if not mode_series.empty:
                    top_group = mode_series.iloc[0]
                    if any(token in top_group for token in ("month", "monthly", "short")):
                        suggestions.append(
                            f"Most high-risk users are in '{contract_col}={top_group}'. Promote longer-term plans with benefits."
                        )
                    else:
                        suggestions.append(
                            f"Monitor segment '{contract_col}={top_group}' closely and run targeted retention campaigns."
                        )

            if risk_pct >= 30:
                suggestions.append(
                    "High-risk volume is significant. Prioritize proactive outreach to top-risk customers this week."
                )
            elif risk_pct >= 15:
                suggestions.append(
                    "Risk is moderate. Schedule periodic follow-ups and monitor behavior changes in risky segments."
                )
            else:
                suggestions.append(
                    "Risk is currently controlled. Maintain engagement programs and continue monitoring monthly."
                )

        top_features = result.feature_importance.head(5)
        if not top_features.empty:
            labels = ", ".join(top_features["Feature"].tolist()[:3])
            suggestions.append(f"Key churn drivers in this dataset are: {labels}. Use these for campaign targeting.")

        suggestions.append(f"High-risk share is {risk_pct:.1f}%. Focus retention actions on the highest-risk customers first.")
        return suggestions[:6]

    def summarize(self, df: pd.DataFrame) -> Dict[str, object]:
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "detected_target": self.detect_target_column(df),
            "detected_identifier": self.detect_identifier_column(df),
            "detected_date": self.detect_date_column(df),
        }

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        trained = self.fit(df)
        predictions = pd.DataFrame(index=df.index)
        metrics: Dict[str, float] = {}
        report = ""
        confusion = None
        feature_importance = pd.DataFrame(columns=["Feature", "Importance"])
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

        return AnalysisResult(
            dataframe=df,
            target_col=self.detect_target_column(df),
            identifier_col=self.detect_identifier_column(df),
            date_col=self.detect_date_column(df),
            model_name=model_name,
            metrics=metrics,
            report=report,
            confusion=confusion,
            feature_importance=feature_importance,
            predictions=predictions,
            suggestions=suggestions,
            summary=self.summarize(df),
        )

    def save(self, path: str) -> None:
        if self.trained_model is None:
            raise RuntimeError("Nothing to save.")
        joblib.dump(self.trained_model, path)

    def load(self, path: str) -> TrainedModel:
        self.trained_model = joblib.load(path)
        return self.trained_model
