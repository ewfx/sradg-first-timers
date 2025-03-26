import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_model_prediction_pipeline():
    # Sample input
    df = pd.DataFrame({
        "GL Balance": [500000, 100000, 250000, 80000],
        "iHub Balance": [495000, 100000, 230000, 70000],
        "Match Status": ["Break", "Match", "Break", "Break"]
    })

    # Step 1: Calculate Balance Difference
    df["Balance Difference"] = df["GL Balance"] - df["iHub Balance"]

    # Step 2: Create Anomaly Labels
    df["Anomaly"] = df["Match Status"].apply(lambda x: "Yes" if str(x).strip().lower() == "break" else "No")
    df["Anomaly_Label"] = df["Anomaly"].map({"Yes": 1, "No": 0})

    X = df[["Balance Difference"]]
    y = df["Anomaly_Label"]

    # Step 3: Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Step 4: Predict new data
    preds = model.predict(X)

    # Assert predictions shape matches input
    assert preds.shape[0] == len(df)

    # Optional: check if at least one 'break' is predicted correctly
    assert (preds == y.values).sum() >= 2  # Expect at least 2 matches
