import mlflow
from mlflow import MlflowClient

TRACKING_URI = (
    "sqlite:///C:/Users/Ashritha/"
    "OneDrive/Desktop/churn_project/mlflow.db"
)

MODEL_NAME = "AdaptivePredictiveModel"

ROLLBACK_VERSION = "1"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

print("=" * 60)
print("MLFLOW MODEL ROLLBACK")
print("=" * 60)

print(f"Registered model : {MODEL_NAME}")
print(f"Rollback version : {ROLLBACK_VERSION}")

# Move the champion alias back to Version 1.
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=ROLLBACK_VERSION,
)

print()
print("=" * 60)
print("ROLLBACK COMPLETE")
print("=" * 60)

print(
    f"{MODEL_NAME}@champion "
    f"-> Version {ROLLBACK_VERSION}"
)

print()
print("Application model URI:")
print(f"models:/{MODEL_NAME}@champion")