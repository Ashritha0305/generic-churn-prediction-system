import pandas as pd


# Load the original/reference dataset
df = pd.read_csv("dataset.csv")


# Create a copy representing new production data
production_df = df.copy()


# --------------------------------------------------
# Simulate numeric drift
# --------------------------------------------------

production_df["study_hours"] = (
    production_df["study_hours"] + 5
)


# --------------------------------------------------
# Simulate categorical drift
# --------------------------------------------------

production_df["device_type"] = "DifferentDevice"


# --------------------------------------------------
# Save simulated production data
# --------------------------------------------------

production_df.to_csv(
    "production_data.csv",
    index=False,
)


print("Production dataset created successfully.")

print(
    f"Rows    : {len(production_df)}"
)

print(
    f"Columns : {len(production_df.columns)}"
)

print(
    "Saved as: production_data.csv"
)