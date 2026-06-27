from __future__ import annotations

import argparse

import pandas as pd

from churn_engine import ChurnAnalysisEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically analyze a CSV and train a churn model when possible.")
    parser.add_argument("--data", default="dataset.csv", help="Path to the CSV file")
    parser.add_argument("--output", default="model.pkl", help="Path to save the trained model")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    engine = ChurnAnalysisEngine(random_state=42)
    trained = engine.fit(df)

    if trained is None:
        print("No churn target column was detected. No model was trained.")
        return

    engine.save(args.output)
    print("Training complete")
    print("Target column:", trained.target_col)
    print("Identifier column:", trained.identifier_col)
    print("Best model:", trained.model_name)
    print("Accuracy:", round(trained.metrics.get("accuracy", 0.0), 4))
    print("F1 weighted:", round(trained.metrics.get("f1_weighted", 0.0), 4))


if __name__ == "__main__":
    main()
