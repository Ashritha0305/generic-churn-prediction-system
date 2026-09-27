from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from churn_engine import AnalysisResult, ChurnAnalysisEngine


EXPERIMENT_NAME = "Adaptive Predictive Analytics"


def print_outcome_discovery(result: AnalysisResult) -> None:
    print("\n" + "=" * 60)
    print("OUTCOME DISCOVERY")
    print("=" * 60)

    discovery = result.outcome_discovery

    if not discovery:
        print("Outcome discovery information is unavailable.")
        return

    print(f"Status        : {discovery.status.upper()}")
    print(f"Target        : {discovery.target_column or 'None'}")
    print(
        f"Positive class: "
        f"{discovery.positive_class or 'Not established'}"
    )
    print(f"Confidence    : {float(discovery.confidence):.2f}")

    if discovery.reason:
        print(f"\nReason:\n{discovery.reason}")

    if discovery.explicit_candidates:
        print("\nExplicit outcome candidates:")

        for candidate in discovery.explicit_candidates:
            print(
                f"  - {candidate.column}: "
                f"score={candidate.score:.2f}; "
                f"{candidate.reason}; "
                f"values={candidate.sample_values}"
            )

    print("\nState/event columns:")

    if discovery.state_columns:
        print(f"  {', '.join(discovery.state_columns)}")
    else:
        print(
            "  No additional state or event "
            "columns were detected."
        )

    print("\nDate columns:")

    if discovery.date_columns:
        print(f"  {', '.join(discovery.date_columns)}")
    else:
        print("  No date-related columns were detected.")


def print_relationships(result: AnalysisResult) -> None:
    print("\n" + "=" * 60)
    print("RELATIONSHIP ANALYSIS")
    print("=" * 60)

    relationships = result.relationships

    if relationships is None or relationships.empty:
        print(
            "No meaningful feature-outcome "
            "relationships were found."
        )
        return

    display_cols = [
        col
        for col in [
            "Feature",
            "Type",
            "Method",
            "Strength",
            "Direction",
        ]
        if col in relationships.columns
    ]

    if display_cols:
        print(
            relationships[
                display_cols
            ].head(10).to_string(index=False)
        )
    else:
        print(
            relationships.head(10)
            .to_string(index=False)
        )


def log_relationship_artifact(
    result: AnalysisResult,
) -> None:
    """
    Save relationship analysis as a CSV artifact
    for the current MLflow run.
    """

    relationships = result.relationships

    if relationships is None:
        return

    if relationships.empty:
        relationships = pd.DataFrame(
            columns=[
                "Feature",
                "Type",
                "Method",
                "Strength",
                "Direction",
            ]
        )

    artifact_path = Path(
        "mlflow_artifacts"
    )

    artifact_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        artifact_path
        / "relationship_analysis.csv"
    )

    relationships.to_csv(
        output_file,
        index=False,
    )

    mlflow.log_artifact(
        str(output_file),
        artifact_path="relationships",
    )


def log_training_metadata(
    df: pd.DataFrame,
    profile,
    result: AnalysisResult,
    trained,
    data_path: Path,
) -> None:
    """
    Save reproducibility metadata for the
    current MLflow run.
    """

    metadata = {
        "dataset_name": data_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "detected_target": profile.detected_target,
        "target_confidence": profile.target_confidence,
        "identifier_column": profile.identifier_column,
        "date_column": profile.date_column,
        "problem_type": profile.problem_type,
        "outcome_status": result.outcome_status,
        "outcome_target": result.outcome_target,
        "outcome_positive_class": (
            result.outcome_positive_class
        ),
        "outcome_confidence": (
            result.outcome_confidence
        ),
        "model_name": trained.model_name,
        "target_column": trained.target_col,
        "random_state": 42,
        "metrics": trained.metrics,
    }

    artifact_path = Path(
        "mlflow_artifacts"
    )

    artifact_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_file = (
        artifact_path
        / "training_metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=str,
        )

    mlflow.log_artifact(
        str(metadata_file),
        artifact_path="training_metadata",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a CSV, discover the outcome, "
            "train a model, and track the run "
            "with MLflow."
        )
    )

    parser.add_argument(
        "--data",
        default="dataset.csv",
        help="Path to the CSV file",
    )

    parser.add_argument(
        "--output",
        default="model.pkl",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    print("=" * 60)
    print("ADAPTIVE PREDICTIVE ANALYTICS - TRAINING")
    print("=" * 60)

    print(f"Dataset: {data_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # ENGINE
    # ---------------------------------------------------------

    engine = ChurnAnalysisEngine(
        random_state=42
    )

    # Run adaptive analysis once.
    result = engine.analyze(df)

    profile = engine.profile_dataset(df)

    # ---------------------------------------------------------
    # DATASET PROFILE
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET PROFILE")
    print("=" * 60)

    print(
        f"Detected target      : "
        f"{profile.detected_target}"
    )

    print(
        f"Target confidence    : "
        f"{profile.target_confidence:.2f}"
    )

    print(
        f"Identifier           : "
        f"{profile.identifier_column}"
    )

    print(
        f"Date column          : "
        f"{profile.date_column}"
    )

    print(
        f"Problem type         : "
        f"{profile.problem_type}"
    )

    print_outcome_discovery(result)

    print_relationships(result)

    # ---------------------------------------------------------
    # MLFLOW EXPERIMENT
    # ---------------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run() as run:

        # -----------------------------------------------------
        # DATASET PARAMETERS
        # -----------------------------------------------------

        mlflow.log_param(
            "dataset_name",
            data_path.name,
        )

        mlflow.log_param(
            "rows",
            len(df),
        )

        mlflow.log_param(
            "columns",
            len(df.columns),
        )

        mlflow.log_param(
            "numeric_columns",
            len(profile.numeric_columns),
        )

        mlflow.log_param(
            "categorical_columns",
            len(profile.categorical_columns),
        )

        mlflow.log_param(
            "missing_cells",
            profile.missing_cells,
        )

        mlflow.log_param(
            "duplicate_rows",
            profile.duplicate_rows,
        )

        mlflow.log_param(
            "detected_target",
            profile.detected_target or "None",
        )

        mlflow.log_param(
            "target_confidence",
            round(
                profile.target_confidence,
                4,
            ),
        )

        mlflow.log_param(
            "identifier_column",
            profile.identifier_column or "None",
        )

        mlflow.log_param(
            "date_column",
            profile.date_column or "None",
        )

        mlflow.log_param(
            "problem_type",
            profile.problem_type or "None",
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_param(
            "model_selection",
            "Random Forest vs Logistic Regression",
        )

        # -----------------------------------------------------
        # OUTCOME DISCOVERY PARAMETERS
        # -----------------------------------------------------

        discovery = result.outcome_discovery

        if discovery:

            mlflow.log_param(
                "outcome_status",
                discovery.status,
            )

            mlflow.log_param(
                "outcome_target",
                discovery.target_column or "None",
            )

            mlflow.log_param(
                "outcome_positive_class",
                discovery.positive_class or "None",
            )

            mlflow.log_param(
                "outcome_confidence",
                round(
                    float(
                        discovery.confidence
                    ),
                    4,
                ),
            )

        # -----------------------------------------------------
        # CHECK TRAINED MODEL
        # -----------------------------------------------------

        trained = engine.trained_model

        if trained is None:

            print("\n" + "=" * 60)
            print("NO MODEL TRAINED")
            print("=" * 60)

            print(
                "No suitable supervised "
                "outcome was detected."
            )

            mlflow.set_tag(
                "training_status",
                "no_target_detected",
            )

            return

        # -----------------------------------------------------
        # RUN TAGS
        # -----------------------------------------------------

        mlflow.set_tag(
            "training_status",
            "success",
        )

        mlflow.set_tag(
            "model_type",
            trained.model_name,
        )

        mlflow.set_tag(
            "target_column",
            trained.target_col,
        )

        mlflow.set_tag(
            "outcome_status",
            result.outcome_status,
        )

        # -----------------------------------------------------
        # MODEL PARAMETERS
        # -----------------------------------------------------

        model = (
            trained.pipeline
            .named_steps["model"]
        )

        model_params = model.get_params()

        # Log useful model configuration
        # without logging every sklearn parameter.

        for parameter in [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "class_weight",
            "max_iter",
            "solver",
        ]:

            if parameter in model_params:

                value = model_params[
                    parameter
                ]

                if value is not None:

                    mlflow.log_param(
                        parameter,
                        str(value),
                    )

        # -----------------------------------------------------
        # METRICS
        # -----------------------------------------------------

        metrics = trained.metrics

        for metric_name in (
            "accuracy",
            "f1_weighted",
            "roc_auc",
        ):

            if metric_name in metrics:

                metric_value = float(
                    metrics[metric_name]
                )

                # Do not log NaN/invalid metrics.
                if pd.notna(metric_value):

                    mlflow.log_metric(
                        metric_name,
                        metric_value,
                    )

        # -----------------------------------------------------
        # SAVE LOCAL MODEL
        # -----------------------------------------------------

        engine.save(
            args.output
        )

        # Keep existing joblib artifact.
        mlflow.log_artifact(
            args.output,
            artifact_path="model",
        )

        # -----------------------------------------------------
        # LOG RELATIONSHIP ANALYSIS
        # -----------------------------------------------------

        log_relationship_artifact(
            result
        )

        # -----------------------------------------------------
        # LOG TRAINING METADATA
        # -----------------------------------------------------

        log_training_metadata(
            df=df,
            profile=profile,
            result=result,
            trained=trained,
            data_path=data_path,
        )

        # -----------------------------------------------------
        # LOG COMPLETE SKLEARN PIPELINE
        # -----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=trained.pipeline,
            artifact_path="sklearn_model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
)
        # -----------------------------------------------------
        # FINAL OUTPUT
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        print(
            f"Target column: "
            f"{trained.target_col}"
        )

        print(
            f"Identifier column: "
            f"{trained.identifier_col}"
        )

        print(
            f"Best model: "
            f"{trained.model_name}"
        )

        print(
            f"Accuracy: "
            f"{trained.metrics.get('accuracy', 0.0):.4f}"
        )

        print(
            f"F1 weighted: "
            f"{trained.metrics.get('f1_weighted', 0.0):.4f}"
        )

        if "roc_auc" in trained.metrics:

            print(
                f"ROC-AUC: "
                f"{trained.metrics['roc_auc']:.4f}"
            )

        print(
            f"Model saved to: "
            f"{args.output}"
        )

        print(
            f"MLflow Run ID: "
            f"{run.info.run_id}"
        )

        print(
            "\nMLflow tracking complete."
        )

        print(
            "MLflow model logged at: "
            "sklearn_model"
        )

        print(
            "Relationship analysis logged."
        )

        print(
            "Training metadata logged."
        )


if __name__ == "__main__":
    main()