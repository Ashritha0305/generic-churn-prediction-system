import mlflow
from mlflow import MlflowClient

# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

TRACKING_URI = "sqlite:///C:/Users/Ashritha/OneDrive/Desktop/churn_project/mlflow.db"

RUN_ID = "99b5ad69e5e241e59cba40550ace0e28"

MODEL_NAME = "AdaptivePredictiveModel"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

# --------------------------------------------------
# Get the experiment containing our run
# --------------------------------------------------

run = client.get_run(RUN_ID)

experiment_id = run.info.experiment_id

print("=" * 60)
print("MLFLOW MODEL REGISTRATION")
print("=" * 60)

print(f"Run ID        : {RUN_ID}")
print(f"Experiment ID : {experiment_id}")
print(f"Model name    : {MODEL_NAME}")

# --------------------------------------------------
# Find the logged model belonging to this run
# --------------------------------------------------

logged_models = client.search_logged_models(
    experiment_ids=[experiment_id],
    filter_string=f"source_run_id = '{RUN_ID}'",
    max_results=10,
)

if not logged_models:
    raise RuntimeError(
        "No logged MLflow model was found for this run."
    )

print()
print(f"Logged models found: {len(logged_models)}")

# Select the logged model associated with this run.
logged_model = logged_models[0]

print(f"Logged model ID: {logged_model.model_id}")
print(f"Logged model name: {logged_model.name}")
print(f"Logged model status: {logged_model.status}")

# --------------------------------------------------
# Register the logged model
# --------------------------------------------------

model_uri = f"models:/{logged_model.model_id}"

print()
print(f"Registering model from:")
print(model_uri)

registered_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

print()
print("=" * 60)
print("REGISTRATION SUCCESSFUL")
print("=" * 60)

print(f"Registered model : {MODEL_NAME}")
print(f"Version          : {registered_version.version}")
print(f"Status           : {registered_version.status}")

# --------------------------------------------------
# Add useful model-version tags
# --------------------------------------------------

version = registered_version.version

client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key="model_type",
    value="Random Forest",
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key="target_column",
    value="working_status",
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key="training_status",
    value="validated",
)

print()
print("Model version tags added.")

# --------------------------------------------------
# Assign champion alias
# --------------------------------------------------

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=version,
)

print()
print("=" * 60)
print("CHAMPION ALIAS ASSIGNED")
print("=" * 60)

print(
    f"{MODEL_NAME}@champion -> Version {version}"
)

print()
print("Model URI:")
print(f"models:/{MODEL_NAME}@champion")