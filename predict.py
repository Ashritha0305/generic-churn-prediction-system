from __future__ import annotations

import argparse

import pandas as pd

from churn_engine import ChurnAnalysisEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate churn predictions for a CSV file.")
    parser.add_argument("--data", required=True, help="Path to the CSV file")
    parser.add_argument("--output", default="predictions.csv", help="Path to save the output CSV")
    parser.add_argument("--model", default="model.pkl", help="Saved trained model path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    engine = ChurnAnalysisEngine(random_state=42)

    try:
        engine.load(args.model)
        result = engine.predict_frame(df)
    except Exception:
        analysis = engine.analyze(df)
        result = analysis.predictions

    result.to_csv(args.output, index=False)
    print("Prediction complete")
    print("Rows processed:", len(result))
    if "Risk Level" in result.columns:
        print("High risk rows:", int((result["Risk Level"] == "High").sum()))
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()