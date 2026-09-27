from __future__ import annotations

import io
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from churn_engine import ChurnAnalysisEngine


st.set_page_config(
    page_title="Adaptive Churn Prediction Platform",
    layout="wide",
)

ENGINE = ChurnAnalysisEngine(random_state=42)


@st.cache_data(show_spinner=False)
def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


def add_dashboard_style() -> None:
    st.markdown(
        """
        <style>
            @import url(
                'https://fonts.googleapis.com/css2?family=Inter:'
                'wght@400;500;600;700;800&display=swap'
            );

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
                color: #0f172a;
            }

            .stApp {
                background:
                    radial-gradient(
                        circle at top left,
                        rgba(14, 165, 233, 0.08),
                        transparent 26%
                    ),
                    radial-gradient(
                        circle at top right,
                        rgba(16, 185, 129, 0.08),
                        transparent 22%
                    ),
                    linear-gradient(
                        180deg,
                        #f8fafc 0%,
                        #f1f5f9 100%
                    );
            }

            .hero {
                background:
                    linear-gradient(
                        135deg,
                        #0f172a 0%,
                        #1e293b 52%,
                        #334155 100%
                    );
                border-radius: 24px;
                padding: 28px;
                color: #ffffff;
                box-shadow:
                    0 18px 40px rgba(15, 23, 42, 0.18);
                margin-bottom: 18px;
            }

            .hero h1 {
                font-size: 2.15rem;
                margin: 0;
                font-weight: 800;
                letter-spacing: -0.04em;
            }

            .hero p {
                margin: 9px 0 0 0;
                color: #cbd5e1;
                font-size: 0.98rem;
                line-height: 1.55;
                max-width: 78ch;
            }

            .panel {
                background: rgba(255, 255, 255, 0.97);
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-radius: 20px;
                padding: 18px;
                margin-bottom: 18px;
                box-shadow:
                    0 10px 28px rgba(15, 23, 42, 0.06);
            }

            .panel-title {
                font-size: 1rem;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 4px;
            }

            .panel-caption {
                color: #64748b;
                font-size: 0.84rem;
                margin-bottom: 12px;
            }

            .metric-card {
                background:
                    linear-gradient(
                        180deg,
                        #ffffff 0%,
                        #f8fafc 100%
                    );
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 15px;
                min-height: 108px;
            }

            .metric-title {
                font-size: 0.82rem;
                color: #64748b;
                font-weight: 700;
                margin-bottom: 7px;
            }

            .metric-value {
                font-size: 1.55rem;
                font-weight: 800;
                color: #0f172a;
            }

            .metric-sub {
                color: #64748b;
                font-size: 0.82rem;
                margin-top: 5px;
            }

            .status-box {
                border-radius: 14px;
                padding: 13px 15px;
                margin-bottom: 10px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
            }

            .status-title {
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 4px;
            }

            .status-text {
                color: #475569;
                font-size: 0.88rem;
                line-height: 1.45;
            }

            .stDataFrame {
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
            <h1>Adaptive Predictive Analytics Platform</h1>
            <p>
                The system automatically analyzes diverse structured/tabular datasets
                and identifies a suitable outcome when one is available, examines statistical
                relationships, trains a predictive model when defensible, and produces
                record-level risk insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_metric(
    title: str,
    value: str,
    subtitle: str,
) -> None:
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


def humanize_label(name: Optional[str]) -> str:
    if not name:
        return "None"
    s = str(name).strip()
    if s.lower() in ("none", "not detected"):
        return "Not detected"

    common = {
        "studentid": "Student ID",
        "customerid": "Customer ID",
        "userid": "User ID",
        "accountid": "Account ID",
        "employeeid": "Employee ID",
        "working_status": "Working Status",
        "previous_attendance": "Previous Attendance",
        "study_hours": "Study Hours",
        "internet_quality": "Internet Quality",
        "course_difficulty": "Course Difficulty",
        "engagement_level": "Engagement Level",
        "device_type": "Device Type",
    }
    low = s.lower().replace(" ", "_")
    if low in common:
        return common[low]

    import re
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ")
    words = s.split()
    capitalized = []
    for w in words:
        if w.upper() == "ID":
            capitalized.append("ID")
        else:
            capitalized.append(w.capitalize())
    return " ".join(capitalized)


def nice_name(name: Optional[str]) -> str:
    return humanize_label(name)


def wrap_panel(title: str, caption: str = "") -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel-title">{title}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(
            f'<div class="panel-caption">{caption}</div>',
            unsafe_allow_html=True,
        )


def close_panel() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: Optional[str],
) -> None:
    if not target_col or target_col not in df.columns:
        st.info(
            "No valid outcome column was detected, so the "
            "outcome distribution is unavailable."
        )
        return

    counts = (
        df[target_col]
        .astype(str)
        .value_counts(dropna=False)
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Outcome Distribution")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Records")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_risk_distribution(
    predictions: pd.DataFrame,
) -> None:
    if predictions.empty or "Risk Level" not in predictions.columns:
        st.info("Risk distribution is unavailable.")
        return

    counts = (
        predictions["Risk Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Predicted Risk Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Records")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_feature_importance(
    feature_importance: pd.DataFrame,
) -> None:
    if feature_importance.empty:
        st.info("Feature importance is unavailable.")
        return

    frame = (
        feature_importance
        .head(8)
        .sort_values("Importance")
        .copy()
    )
    frame["DisplayFeature"] = frame["Feature"].apply(humanize_label)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(
        frame["DisplayFeature"].astype(str),
        frame["Importance"].astype(float),
    )
    ax.set_title("Features the Model Relied on Most")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_numeric_vs_target(
    df: pd.DataFrame,
    target_col: Optional[str],
    relationships: Optional[pd.DataFrame] = None,
    feature_importance: Optional[pd.DataFrame] = None,
) -> None:
    if not target_col or target_col not in df.columns:
        st.info("Numeric-vs-outcome analysis is unavailable.")
        return

    numeric_cols: List[str] = []

    # Prefer top valid numeric relationships (identifiers already excluded)
    if relationships is not None and not relationships.empty:
        for _, row in relationships.iterrows():
            feat = row["Feature"]
            if (
                str(row.get("Type", "")).lower() == "numeric"
                and feat in df.columns
                and pd.api.types.is_numeric_dtype(df[feat])
            ):
                numeric_cols.append(feat)
            if len(numeric_cols) >= 4:
                break

    if not numeric_cols and feature_importance is not None and not feature_importance.empty:
        numeric_cols = [
            c for c in feature_importance["Feature"].head(4).tolist()
            if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    if not numeric_cols:
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if c != target_col
        ][:4]

    if not numeric_cols:
        st.info("No suitable numeric features were found.")
        return

    for feature in numeric_cols:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        sns.boxplot(
            data=df,
            x=target_col,
            y=feature,
            ax=ax,
        )
        ax.set_title(f"{humanize_label(feature)} by {humanize_label(target_col)}")
        ax.set_xlabel(humanize_label(target_col))
        ax.set_ylabel(humanize_label(feature))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def plot_categorical_vs_target(
    df: pd.DataFrame,
    target_col: Optional[str],
    relationships: Optional[pd.DataFrame] = None,
) -> None:
    if not target_col or target_col not in df.columns or relationships is None or relationships.empty:
        return

    cat_cols: List[str] = []
    for _, row in relationships.iterrows():
        feat = row["Feature"]
        if (
            str(row.get("Type", "")).lower() == "categorical"
            and feat in df.columns
        ):
            cat_cols.append(feat)
        if len(cat_cols) >= 3:
            break

    if not cat_cols:
        return

    for feature in cat_cols:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        crosstab = (
            pd.crosstab(
                df[feature].astype(str),
                df[target_col].astype(str),
                normalize="index",
            )
            * 100
        )
        crosstab.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            alpha=0.88,
        )
        ax.set_title(f"Outcome Proportion by {humanize_label(feature)}")
        ax.set_xlabel(humanize_label(feature))
        ax.set_ylabel("% Records")
        ax.legend(title=humanize_label(target_col), bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.18)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        st.info(
            "At least two numeric columns are needed "
            "for the correlation heatmap."
        )
        return

    if numeric_df.shape[1] > 10:
        numeric_df = numeric_df.iloc[:, :10]

    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(
        corr,
        cmap="Blues",
        center=0,
        square=True,
        cbar=False,
        ax=ax,
        linewidths=0.4,
    )
    ax.set_title("Numeric Correlation Heatmap")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def build_prediction_table(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    identifier_col: Optional[str],
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "ID",
                "Prediction",
                "Probability",
                "Risk Level",
            ]
        )

    if identifier_col and identifier_col in df.columns:
        ids = df[identifier_col].astype(str).fillna("")
    else:
        ids = pd.Series(
            np.arange(1, len(predictions) + 1),
            index=predictions.index,
        )

    table = pd.DataFrame({"ID": ids.values})

    if "Prediction Label" in predictions.columns:
        table["Prediction"] = (
            predictions["Prediction Label"]
            .astype(str)
            .values
        )
    elif "Churn Prediction" in predictions.columns:
        table["Prediction"] = (
            predictions["Churn Prediction"]
            .astype(str)
            .str.replace(" ❌", "", regex=False)
            .str.replace(" ✅", "", regex=False)
            .values
        )
    else:
        table["Prediction"] = "Unknown"

    if "Churn Probability" in predictions.columns:
        table["Probability"] = (
            pd.to_numeric(
                predictions["Churn Probability"],
                errors="coerce",
            )
            .values
        )
    else:
        table["Probability"] = np.nan

    if "Risk Level" in predictions.columns:
        table["Risk Level"] = (
            predictions["Risk Level"].astype(str).values
        )
    else:
        table["Risk Level"] = "Unknown"

    return table.sort_values(
        "Probability",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def style_risk_table(
    frame: pd.DataFrame,
) -> "pd.io.formats.style.Styler":
    def row_style(row: pd.Series) -> List[str]:
        level = str(
            row.get("Risk Level", "")
        ).lower()

        if level == "high":
            style = (
                "background-color: #fee2e2; "
                "color: #7f1d1d;"
            )
        elif level == "medium":
            style = (
                "background-color: #fef3c7; "
                "color: #78350f;"
            )
        elif level == "low":
            style = (
                "background-color: #dcfce7; "
                "color: #14532d;"
            )
        else:
            style = ""

        return [style] * len(row)

    return frame.style.apply(
        row_style,
        axis=1,
    )


def render_outcome_discovery(result) -> None:
    wrap_panel(
        "What outcome is the system trying to predict?",
        "The system first determines whether a defensible outcome exists before attempting supervised prediction.",
    )

    discovery = getattr(result, "outcome_discovery", None)

    if not discovery:
        st.info("Outcome discovery information is unavailable.")
        close_panel()
        return

    status = str(
        discovery.status if hasattr(discovery, "status")
        else discovery.get("status", "Unknown")
    ).upper()

    target = (
        discovery.target_column if hasattr(discovery, "target_column")
        else discovery.get("target_column")
    ) or "None"

    positive = (
        discovery.positive_class if hasattr(discovery, "positive_class")
        else discovery.get("positive_class")
    ) or "Not established"

    confidence = float(
        discovery.confidence if hasattr(discovery, "confidence")
        else discovery.get("confidence", 0.0)
    )

    reason = (
        discovery.reason if hasattr(discovery, "reason")
        else discovery.get("reason")
    ) or ""

    conf_label = "High" if confidence >= 0.8 else ("Moderate" if confidence >= 0.5 else "Low")
    target_display = humanize_label(target)

    # 1. Primary User-Facing Information
    cols = st.columns(3)
    with cols[0]:
        display_metric("Outcome", target_display, "detected target column")
    with cols[1]:
        display_metric("Positive Outcome", str(positive), "target class to predict")
    with cols[2]:
        display_metric("Confidence", conf_label, f"score: {confidence:.2f}")

    if status.lower() == "explicit":
        if target and target in result.dataframe.columns:
            u_vals = [str(v) for v in result.dataframe[target].dropna().unique().tolist()[:2]]
            vals_phrase = " and ".join(u_vals)
            explanation = (
                f"The dataset contains a clear outcome column called **{target_display}** "
                f"with two possible values: **{vals_phrase}**. "
                f"This allows the system to train a supervised prediction model."
            )
        else:
            explanation = (
                f"The dataset contains a clear outcome column called **{target_display}**. "
                f"This allows the system to train a supervised prediction model."
            )
    elif status.lower() == "derivable":
        explanation = (
            "No explicit outcome column was identified, but columns representing state or event progression "
            "exist and may contain outcome evidence. Further domain review is required before deriving a target."
        )
    elif status.lower() == "behavioral_only":
        explanation = (
            "No defensible outcome target was identified in this dataset. Behavioral and relationship analysis "
            "can still be explored, but supervised prediction cannot be trained."
        )
    else:
        explanation = (
            "There is not enough information in the dataset to identify an outcome or perform supervised learning."
        )

    st.markdown(f"**Explanation:** {explanation}")

    # 2. Expandable Technical Details
    candidates = (
        discovery.explicit_candidates if hasattr(discovery, "explicit_candidates")
        else discovery.get("explicit_candidates")
    )
    state_cols = (
        discovery.state_columns if hasattr(discovery, "state_columns")
        else discovery.get("state_columns", [])
    )
    date_cols = (
        discovery.date_columns if hasattr(discovery, "date_columns")
        else discovery.get("date_columns", [])
    )

    with st.expander("How did the system determine this?"):
        st.markdown(f"- **Outcome status:** `{status}`")
        st.markdown(f"- **Detected target column:** `{target}`")
        st.markdown(f"- **Positive class:** `{positive}`")
        st.markdown(f"- **Confidence score:** `{confidence:.2f}` ({conf_label})")
        if reason:
            st.markdown(f"- **Detection reason:** {reason}")

        if candidates:
            st.markdown("**Candidate outcome evaluation:**")
            for cand in candidates:
                col = cand.column if hasattr(cand, "column") else cand.get("column")
                score = cand.score if hasattr(cand, "score") else cand.get("score")
                r = cand.reason if hasattr(cand, "reason") else cand.get("reason")
                vals = cand.sample_values if hasattr(cand, "sample_values") else cand.get("sample_values", [])
                st.markdown(f"  • **{col}** (score: {score:.2f}) — {r}; sample values: {vals}")

        if state_cols:
            st.markdown(f"- **Additional state/event columns:** {', '.join(state_cols)}")
        else:
            st.markdown("- **Additional state/event columns:** No additional state or event columns were detected.")

        if date_cols:
            st.markdown(f"- **Date-related columns:** {', '.join(date_cols)}")
        else:
            st.markdown("- **Date-related columns:** No date-related columns were detected.")

    close_panel()


def render_relationship_analysis(result, df: Optional[pd.DataFrame] = None) -> None:
    wrap_panel(
        "What patterns are associated with the outcome?",
        "The system checks which features are most strongly associated with the detected outcome. "
        "These are statistical associations, not proof of causation.",
    )

    source_df = df if df is not None else getattr(result, "dataframe", pd.DataFrame())

    if len(source_df) < 50:
        st.info(
            "ℹ️ Small sample: association strengths may be unstable and should not "
            "be interpreted as generalizable relationships."
        )

    relationships = getattr(result, "relationships", None)

    if relationships is None or relationships.empty:
        st.info("No meaningful feature-outcome relationships were identified.")
        close_panel()
        return

    # User-Friendly Summary: top features with qualitative strength + direction
    def format_rel_label(strength_val: float, direction_val: str, type_val: str) -> str:
        s = abs(float(strength_val))
        if s >= 0.70:
            lvl = "Strong" if s < 0.85 else "Very strong"
        elif s >= 0.50:
            lvl = "Strong"
        elif s >= 0.30:
            lvl = "Moderate"
        else:
            lvl = "Weak"

        dir_clean = str(direction_val).strip().lower()
        if str(type_val).lower() == "numeric" and dir_clean in ("positive", "negative"):
            return f"{lvl} {dir_clean} association"
        return f"{lvl} association"

    top_relationships = relationships.head(6)
    cols = st.columns(2)
    for idx, (_, row) in enumerate(top_relationships.iterrows()):
        feat_raw = row["Feature"]
        feat_human = humanize_label(feat_raw)
        f_type = str(row.get("Type", ""))
        f_strength = float(row.get("Strength", 0.0))
        f_dir = str(row.get("Direction", "-"))
        rel_text = format_rel_label(f_strength, f_dir, f_type)

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">{feat_human}</div>
                    <div style="color: #475569; font-size: 0.88rem; margin-top: 2px;">{rel_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "These descriptions summarize statistical patterns observed in this dataset. "
        "Statistical association does not imply that a feature causes the outcome."
    )

    # Technical Details Expander
    with st.expander("Technical details"):
        st.markdown(
            "Statistical methods used: **Point-biserial correlation** for numeric features, "
            "and **Cramér's V** for categorical features. Features are ranked by absolute association strength."
        )
        display_cols = [
            c for c in ["Feature", "Type", "Method", "Strength", "Direction"]
            if c in relationships.columns
        ]
        if not display_cols:
            display_cols = relationships.columns.tolist()

        st.dataframe(
            relationships[display_cols].head(15),
            use_container_width=True,
            hide_index=True,
        )

    close_panel()


def render_dataset_profile(
    result,
    summary: dict,
) -> None:
    wrap_panel(
        "Dataset Understanding",
        "Summary of the uploaded dataset structure and automatically identified metadata.",
    )

    cols = st.columns(5)

    prob_type = summary.get("problem_type")
    prob_type_display = (
        "Binary Outcome"
        if prob_type and "binary" in str(prob_type).lower()
        else humanize_label(prob_type)
    )

    values = [
        (
            "Records",
            str(summary.get("rows", 0)),
            "records",
        ),
        (
            "Features",
            str(summary.get("columns", 0)),
            "features",
        ),
        (
            "Detected Outcome",
            humanize_label(result.target_col),
            "detected outcome",
        ),
        (
            "ID Column",
            humanize_label(result.identifier_col),
            "excluded from modeling",
        ),
        (
            "Prediction Type",
            prob_type_display,
            "inferred",
        ),
    ]

    for column, (title, value, subtitle) in zip(
        cols,
        values,
    ):
        with column:
            display_metric(
                title,
                value,
                subtitle,
            )

    close_panel()


def render_prediction_summary(
    result,
    predictions: pd.DataFrame,
) -> None:
    wrap_panel(
        "Prediction Summary",
        "Record-level predictions generated by the selected model.",
    )

    if predictions.empty:
        st.info(
            "No predictions were generated because a suitable "
            "supervised outcome was not available."
        )
        close_panel()
        return

    risk = (
        predictions["Risk Level"]
        if "Risk Level" in predictions.columns
        else pd.Series(dtype=str)
    )

    high = int((risk == "High").sum())
    medium = int((risk == "Medium").sum())
    low = int((risk == "Low").sum())

    rate = None
    target = result.target_col

    if target and target in result.dataframe.columns:
        target_values = (
            result.dataframe[target]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        pos_hint = str(result.outcome_positive_class or "yes").strip().lower()
        rate = float((target_values == pos_hint).mean() * 100)

    cols = st.columns(4)

    metric_values = [
        (
            "High Predicted Risk",
            f"{high:,} records",
            "highest predicted probability",
        ),
        (
            "Medium Predicted Risk",
            f"{medium:,} records",
            "moderate predicted probability",
        ),
        (
            "Low Predicted Risk",
            f"{low:,} records",
            "lowest predicted probability",
        ),
        (
            "Observed Outcome Rate",
            (
                f"{rate:.1f}%"
                if rate is not None
                else "N/A"
            ),
            "from available labels",
        ),
    ]

    for column, values in zip(
        cols,
        metric_values,
    ):
        with column:
            display_metric(*values)

    st.markdown(
        "**Risk level** represents how strongly the model predicts the positive outcome for each record."
    )
    pos_name = result.outcome_positive_class or "positive outcome"
    st.caption(
        f"Note: High predicted risk means the model assigned a higher probability to the positive outcome ({pos_name}); "
        "it does not guarantee that the outcome will definitely occur."
    )

    close_panel()


def render_app() -> None:
    add_dashboard_style()
    render_hero()

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help=(
            "Use a structured/tabular dataset. The system will "
            "automatically inspect columns and look for a suitable "
            "outcome."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Upload a CSV file to start the adaptive analysis."
        )
        return

    try:
        df = read_csv(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    if df.empty:
        st.warning("The uploaded CSV contains no records.")
        return

    with st.spinner(
        "Understanding dataset, discovering outcome, "
        "analyzing relationships, and training model..."
    ):
        try:
            result = ENGINE.analyze(df)
        except Exception as exc:
            st.error(
                "Analysis failed. Check the dataset and "
                f"the engine configuration.\n\n{exc}"
            )
            return

    render_dataset_profile(
        result,
        result.summary,
    )

    render_outcome_discovery(result)

    render_relationship_analysis(result)

    if result.model_name:
        wrap_panel(
            "How well did the model perform?",
            "Evaluation of the trained model on held-out test data.",
        )

        if len(df) < 50:
            st.warning(
                "⚠️ Small dataset: the available data is limited, so model performance "
                "and detected patterns may change substantially with more data."
            )

        metrics = result.metrics
        f1_val = metrics.get("f1_weighted", 0.0)

        metric_cols = st.columns(2)
        with metric_cols[0]:
            display_metric(
                "Model Used",
                result.model_name,
                "selected automatically",
            )
        with metric_cols[1]:
            display_metric(
                "F1 Score",
                f"{f1_val:.3f}",
                "test-set performance",
            )

        st.markdown(
            "F1 Score summarizes how well the model identified the outcome classes "
            "on the held-out test data."
        )

        with st.expander("Technical evaluation details"):
            roc_val = metrics.get("roc_auc")
            if roc_val is not None and not np.isnan(roc_val):
                roc_str = f"{roc_val:.4f}"
            else:
                roc_str = (
                    "Not available for this evaluation "
                    "(held-out test split does not contain multiple classes or sample is too small)"
                )

            acc_val = metrics.get("accuracy", 0.0)

            st.markdown(f"- **Selected algorithm:** `{result.model_name}`")
            st.markdown(f"- **Accuracy:** `{acc_val:.4f}`")
            st.markdown(f"- **Weighted F1 Score:** `{f1_val:.4f}`")
            st.markdown(f"- **ROC-AUC:** `{roc_str}`")
            st.markdown("- **Evaluation split:** 80% train / 20% test held-out evaluation")
            if result.report:
                st.markdown("**Classification Report:**")
                st.text(result.report)

        close_panel()

    render_prediction_summary(
        result,
        result.predictions,
    )

    first_row = st.columns(2)

    with first_row[0]:
        wrap_panel(
            "Outcome Distribution",
            f"Distribution of observed classes for {humanize_label(result.target_col)}.",
        )
        plot_target_distribution(
            df,
            result.target_col,
        )
        close_panel()

    with first_row[1]:
        wrap_panel(
            "Risk Distribution",
            "Distribution of predicted risk levels across records.",
        )
        plot_risk_distribution(
            result.predictions,
        )
        close_panel()

    second_row = st.columns(2)

    with second_row[0]:
        wrap_panel(
            "Features the Model Relied on Most",
            "Model feature importance scores.",
        )
        plot_feature_importance(
            result.feature_importance,
        )
        st.caption(
            "ℹ️ Model feature importance reflects which variables the trained model used most "
            "to make predictions, which can differ from direct bivariate statistical associations."
        )
        close_panel()

    with second_row[1]:
        wrap_panel(
            "Correlation Heatmap",
            "Pairwise linear correlations among numeric features.",
        )
        plot_correlation_heatmap(df)
        close_panel()

    wrap_panel(
        "How key numeric features differ by outcome",
        "Distribution of top associated numeric features across outcome classes.",
    )
    plot_numeric_vs_target(
        df,
        result.target_col,
        result.relationships,
        result.feature_importance,
    )
    close_panel()

    if result.relationships is not None and not result.relationships.empty:
        has_cats = any(
            str(t).lower() == "categorical"
            for t in result.relationships.get("Type", [])
        )
        if has_cats:
            wrap_panel(
                "How key categorical features differ by outcome",
                "Outcome distribution across strongest associated categorical segments.",
            )
            plot_categorical_vs_target(
                df,
                result.target_col,
                result.relationships,
            )
            close_panel()

    wrap_panel(
        "Records with Highest Predicted Risk",
        "These records received the highest predicted probability for the positive outcome.",
    )

    table = build_prediction_table(
        df,
        result.predictions,
        result.identifier_col,
    )

    if table.empty:
        st.info(
            "No prediction table is available."
        )
    else:
        risk_options = [
            "All",
            "High",
            "Medium",
            "Low",
        ]

        selected_risk = st.selectbox(
            "Risk filter",
            risk_options,
        )

        display_rows = st.slider(
            "Records to display",
            min_value=10,
            max_value=50,
            value=10,
            step=10,
        )

        filtered = table.copy()

        if selected_risk != "All":
            filtered = filtered[
                filtered["Risk Level"].astype(str)
                == selected_risk
            ]

        filtered = filtered.head(display_rows)

        st.dataframe(
            style_risk_table(filtered),
            use_container_width=True,
            height=360,
        )

    close_panel()

    wrap_panel(
        "What does this mean?",
        "A straightforward explanation based on the observed data and model results.",
    )

    summary_points = []

    # 1. Feature importance finding
    if not result.feature_importance.empty:
        top_model_feats = [
            humanize_label(f)
            for f in result.feature_importance.head(3)["Feature"].astype(str)
        ]
        if len(top_model_feats) == 1:
            feat_phrase = top_model_feats[0]
        elif len(top_model_feats) == 2:
            feat_phrase = f"{top_model_feats[0]} and {top_model_feats[1]}"
        else:
            feat_phrase = f"{top_model_feats[0]}, {top_model_feats[1]}, and {top_model_feats[2]}"
        summary_points.append(
            f"The model found that **{feat_phrase}** were among the features it relied on most when making predictions."
        )

    # 2. Risk distribution finding
    if not result.predictions.empty and "Risk Level" in result.predictions.columns:
        high_cnt = int((result.predictions["Risk Level"] == "High").sum())
        total_cnt = len(result.predictions)
        summary_points.append(
            f"**{high_cnt} of {total_cnt} records** received a high predicted risk level."
        )

    # 3. Practical interpretation
    summary_points.append(
        "These results can be used to identify records that may need closer attention or proactive follow-up."
    )

    for point in summary_points:
        st.write(f"• {point}")

    export_frame = table if not table.empty else df

    st.download_button(
        label="Download Analysis Report",
        data=export_frame.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="predictive_analysis_report.csv",
        mime="text/csv",
    )

    close_panel()


if __name__ == "__main__":
    render_app()
