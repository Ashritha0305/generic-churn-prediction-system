from __future__ import annotations

import io
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from churn_engine import ChurnAnalysisEngine


st.set_page_config(page_title="Generic Churn Prediction System", layout="wide")

ENGINE = ChurnAnalysisEngine(random_state=42)


@st.cache_data(show_spinner=False)
def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


def add_dashboard_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
                color: #0f172a;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(14, 165, 233, 0.08), transparent 26%),
                    radial-gradient(circle at top right, rgba(16, 185, 129, 0.08), transparent 22%),
                    linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            }
            .hero {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 52%, #334155 100%);
                border-radius: 24px;
                padding: 22px 24px;
                color: #ffffff;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
                margin-bottom: 14px;
            }
            .hero h1 {
                font-size: 2rem;
                margin: 0;
                font-weight: 800;
                letter-spacing: -0.04em;
            }
            .hero p {
                margin: 8px 0 0 0;
                color: #cbd5e1;
                font-size: 0.96rem;
                line-height: 1.45;
                max-width: 72ch;
            }
            .panel {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-radius: 20px;
                padding: 16px 16px 14px 16px;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            }
            .panel-title {
                font-size: 0.95rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                color: #0f172a;
                margin-bottom: 6px;
            }
            .panel-caption {
                color: #64748b;
                font-size: 0.84rem;
                margin-bottom: 10px;
            }
            .metric-card {
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 14px 14px 12px 14px;
                min-height: 108px;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            }
            .metric-title {
                font-size: 0.8rem;
                color: #64748b;
                font-weight: 700;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .metric-value {
                font-size: 1.55rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.05;
            }
            .metric-sub {
                color: #475569;
                font-size: 0.84rem;
                margin-top: 6px;
                line-height: 1.35;
            }
            .insight-text {
                color: #334155;
                font-size: 0.92rem;
                line-height: 1.5;
            }
            .stDataFrame, .stTable {
                border-radius: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Generic Churn Prediction System</h1>
            <p>Upload any CSV file and the dashboard will auto-detect the churn target, identify the customer ID, train a model when possible, and surface the few visuals that matter for fast decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def wrap_panel(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def close_panel() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def get_model_label(result) -> str:
    if result.model_name:
        return result.model_name
    if result.target_col:
        return "Training pending"
    return "No model"


def format_accuracy(metrics: dict[str, float]) -> str:
    accuracy = metrics.get("accuracy")
    if accuracy is None or pd.isna(accuracy):
        return "Accuracy n/a"
    return f"Accuracy {accuracy:.1%}"


def target_counts(df: pd.DataFrame, target_col: Optional[str]) -> pd.Series:
    if not target_col or target_col not in df.columns:
        return pd.Series(dtype=int)
    return df[target_col].astype(str).value_counts(dropna=False)


def plot_target_distribution(df: pd.DataFrame, target_col: Optional[str]) -> None:
    counts = target_counts(df, target_col)
    if counts.empty:
        st.info("No churn target column was detected, so target distribution charts are unavailable.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [1.15, 0.95]})
    fig.patch.set_alpha(0)
    palette = ["#0f172a", "#38bdf8", "#94a3b8", "#cbd5e1"]

    counts.plot(kind="bar", ax=axes[0], color=palette[: len(counts)], width=0.72)
    axes[0].set_title("Target Distribution", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.16)
    axes[0].tick_params(axis="x", labelrotation=0)

    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.0f%%",
        startangle=90,
        colors=palette[: len(counts)],
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Share Split", fontsize=11, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_risk_distribution(predictions: pd.DataFrame) -> None:
    if predictions.empty or "Risk Level" not in predictions.columns:
        st.info("Risk distribution is available after the model produces predictions.")
        return

    order = ["High", "Medium", "Low"]
    counts = predictions["Risk Level"].astype(str).value_counts().reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    fig.patch.set_alpha(0)
    colors = ["#ef4444", "#f59e0b", "#22c55e"]
    ax.bar(counts.index, counts.values, color=colors, width=0.58)
    ax.set_title("Risk Distribution", fontsize=11, fontweight="bold")
    ax.set_ylabel("Customers")
    ax.grid(axis="y", alpha=0.16)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    if feature_importance.empty:
        st.info("Feature importance could not be calculated for this dataset.")
        return

    top_features = feature_importance.head(5).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    fig.patch.set_alpha(0)
    ax.barh(top_features["Feature"], top_features["Importance"], color="#2563eb")
    ax.set_title("Top 5 Feature Importance", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.16)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def choose_numeric_features(df: pd.DataFrame, feature_importance: pd.DataFrame, max_features: int = 3) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return []

    ordered: List[str] = []
    if not feature_importance.empty:
        for feature in feature_importance["Feature"].astype(str).tolist():
            if feature in numeric_cols and feature not in ordered:
                ordered.append(feature)

    for col in numeric_cols:
        if col not in ordered:
            ordered.append(col)

    return ordered[:max_features]


def plot_numeric_vs_target(df: pd.DataFrame, target_col: Optional[str], feature_importance: pd.DataFrame) -> None:
    if not target_col or target_col not in df.columns:
        st.info("Numeric-vs-target boxplots need a detected churn target column.")
        return

    features = choose_numeric_features(df, feature_importance, max_features=3)
    if not features:
        st.info("No numeric features were detected for boxplots.")
        return

    fig, axes = plt.subplots(1, len(features), figsize=(4.1 * len(features), 3.0), squeeze=False)
    fig.patch.set_alpha(0)
    for idx, feature in enumerate(features):
        ax = axes[0][idx]
        sns.boxplot(data=df, x=target_col, y=feature, ax=ax, color="#cbd5e1", width=0.45)
        ax.set_title(feature, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(axis="y", alpha=0.12)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.info("At least two numeric columns are needed for a correlation heatmap.")
        return

    if numeric_df.shape[1] > 8:
        numeric_df = numeric_df.iloc[:, :8]

    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    fig.patch.set_alpha(0)
    sns.heatmap(corr, cmap="Blues", center=0, square=True, cbar=False, ax=ax, linewidths=0.4)
    ax.set_title("Correlation Heatmap", fontsize=11, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def build_reason_summary(feature_importance: pd.DataFrame) -> str:
    if feature_importance.empty:
        return "Top factors affecting churn could not be isolated for this dataset."
    top_features = feature_importance.head(5)["Feature"].astype(str).tolist()
    return ", ".join(top_features)


def style_risk_table(frame: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def row_style(row: pd.Series) -> List[str]:
        level = str(row.get("Risk Level", "")).strip().lower()
        if level == "high":
            color = "background-color: #fee2e2; color: #7f1d1d;"
        elif level == "medium":
            color = "background-color: #fef3c7; color: #78350f;"
        elif level == "low":
            color = "background-color: #dcfce7; color: #14532d;"
        else:
            color = ""
        return [color] * len(row)

    return frame.style.apply(row_style, axis=1)


def build_prediction_table(df: pd.DataFrame, predictions: pd.DataFrame, identifier_col: Optional[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["ID", "Prediction", "Probability", "Risk Level"])

    if identifier_col and identifier_col in df.columns:
        ids = df[identifier_col].astype(str).fillna("")
    else:
        ids = pd.Series(np.arange(1, len(predictions) + 1), name="Record")

    table = pd.DataFrame({"ID": ids.values})
    prediction_col = predictions.get("Churn Prediction")
    if prediction_col is not None:
        table["Prediction"] = prediction_col.astype(str).str.replace(" ❌", "", regex=False).str.replace(" ✅", "", regex=False)
    else:
        table["Prediction"] = "Unknown"

    table["Probability"] = predictions.get("Churn Probability", pd.Series(np.nan, index=predictions.index)).astype(float)
    table["Risk Level"] = predictions.get("Risk Level", pd.Series("Unknown", index=predictions.index)).astype(str)
    table["Probability"] = table["Probability"].round(4)
    table = table.sort_values("Probability", ascending=False, na_position="last").reset_index(drop=True)
    return table


def render_app() -> None:
    add_dashboard_style()
    render_hero()

    header_left, header_right = st.columns([3.2, 1.35], vertical_alignment="center")
    with header_left:
        st.caption("Upload a CSV to automatically detect the churn target, train a model when possible, and rank customers by risk.")
    with header_right:
        file = st.file_uploader("Upload CSV dataset", type=["csv"], label_visibility="collapsed")

    if file is None:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Start here</div>
                <div class="panel-caption">Upload a CSV file to see KPI cards, compact charts, a risk table, and downloadable predictions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = read_csv(file.getvalue())
    result = ENGINE.analyze(df)
    predictions = result.predictions.copy()
    summary = result.summary
    target_col = result.target_col
    identifier_col = result.identifier_col
    model_label = get_model_label(result)
    accuracy_label = format_accuracy(result.metrics)

    high_count = 0
    medium_count = 0
    low_count = 0
    churn_rate = 0.0

    if not predictions.empty and "Risk Level" in predictions.columns:
        risk_series = predictions["Risk Level"].astype(str)
        high_count = int((risk_series == "High").sum())
        medium_count = int((risk_series == "Medium").sum())
        low_count = int((risk_series == "Low").sum())
        churn_rate = float((predictions["Churn Prediction"].astype(str) == "Churn ❌").mean() * 100) if "Churn Prediction" in predictions.columns else 0.0

    st.markdown('<div style="margin-bottom: 12px;">', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric("High Risk Customers", str(high_count), "Customers flagged as likely to churn")
    with k2:
        render_metric("Safe Customers", str(low_count), "Low-risk customers to retain and nurture")
    with k3:
        render_metric("Churn Rate", f"{churn_rate:.1f}%", "Share of predicted churn in the dataset")
    with k4:
        render_metric("Model Used", model_label, f"{accuracy_label} | Target: {target_col or 'Not detected'}")
    st.markdown("</div>", unsafe_allow_html=True)

    first_row = st.columns([1.15, 1.0, 0.92])
    with first_row[0]:
        wrap_panel("Target Distribution", "Bar and pie views of churn vs non-churn.")
        plot_target_distribution(df, target_col)
        close_panel()
    with first_row[1]:
        wrap_panel("Feature Importance", "Only the top 5 drivers are shown.")
        plot_feature_importance(result.feature_importance)
        close_panel()
    with first_row[2]:
        wrap_panel("Risk Distribution", "High, medium, and low risk counts.")
        plot_risk_distribution(predictions)
        close_panel()

    second_row = st.columns([1.15, 1.0])
    with second_row[0]:
        wrap_panel("Numeric vs Target", "Top 2–3 important numeric features shown as boxplots.")
        plot_numeric_vs_target(df, target_col, result.feature_importance)
        close_panel()
    with second_row[1]:
        wrap_panel("Correlation Heatmap", "Only shown when the dataset has multiple numeric columns.")
        plot_correlation_heatmap(df)
        close_panel()

    third_row = st.columns([1.18, 0.82])
    with third_row[0]:
        wrap_panel("High-Risk Customers", "Filter by risk level and sort by probability.")

        risk_options = ["All", "High", "Medium", "Low"]
        selected_risk = st.selectbox("Risk filter", risk_options, index=0, label_visibility="visible")
        max_rows = st.slider("Rows to display", min_value=10, max_value=20, value=10, step=5)

        table = build_prediction_table(df, predictions, identifier_col)
        if table.empty:
            st.info("No prediction table is available because no valid churn target was detected.")
        else:
            filtered = table.copy()
            if selected_risk != "All":
                filtered = filtered[filtered["Risk Level"].astype(str) == selected_risk]
            filtered = filtered.head(max_rows)
            styled = style_risk_table(filtered)
            st.dataframe(styled, use_container_width=True, height=340)
        close_panel()

    with third_row[1]:
        wrap_panel("Decision Summary", "The short version for non-technical users.")
        st.markdown(
            f'<div class="insight-text"><strong>Top factors affecting churn:</strong> {build_reason_summary(result.feature_importance)}</div>',
            unsafe_allow_html=True,
        )

        suggestions = []
        for suggestion in result.suggestions:
            cleaned = suggestion.strip()
            if cleaned and cleaned not in suggestions:
                suggestions.append(cleaned)
        if not suggestions:
            suggestions = [
                "Focus retention outreach on the highest-risk customers first.",
                "Review low-engagement or short-tenure segments for early intervention.",
                "Use the strongest churn drivers to target offers and support.",
            ]
        st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
        for suggestion in suggestions[:4]:
            st.write(f"• {suggestion}")
        st.markdown("</div>", unsafe_allow_html=True)

        export_frame = build_prediction_table(df, predictions, identifier_col)
        if export_frame.empty:
            export_frame = df.copy()
        st.download_button(
            label="Download Analysis Report",
            data=export_frame.to_csv(index=False).encode("utf-8"),
            file_name="churn_analysis_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(
            f'<div class="insight-text" style="margin-top: 10px; color: #64748b;">Rows: {summary["rows"]} | Columns: {summary["columns"]} | Missing cells: {summary["missing_cells"]} | ID: {identifier_col or "Not detected"}</div>',
            unsafe_allow_html=True,
        )
        close_panel()


if __name__ == "__main__":
    render_app()
