import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# === Placeholder model ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit([[0]], [0])  # Dummy fit to avoid sklearn not-fitted errors

def extract_features(df):
    """
    Generate aggregated features from angles per throw.
    Returns a single-row DataFrame suitable for ML prediction.
    """
    angle_cols = ["elbow_angle", "shoulder_angle", "hip_angle", "knee_angle"]
    features = {}

    for col in angle_cols:
        features[f"{col}_min"] = df[col].min()
        features[f"{col}_max"] = df[col].max()
        features[f"{col}_mean"] = df[col].mean()
        features[f"{col}_std"] = df[col].std()
        deltas = df[col].diff().fillna(0)
        features[f"{col}_delta_max"] = deltas.max()
        features[f"{col}_delta_mean"] = deltas.mean()
        features[f"{col}_delta_std"] = deltas.std()
        # Time to max/min
        features[f"{col}_time_to_max"] = df[col].idxmax()
        features[f"{col}_time_to_min"] = df[col].idxmin()

    # Heuristic flags for advice
    features["elbow_extended_flag"] = int(features["elbow_angle_max"] >= 160)
    features["hip_rotated_flag"] = int(features["hip_angle_max"] >= 120)
    features["knee_extended_flag"] = int(features["knee_angle_max"] >= 150)

    return pd.DataFrame([features])

def predict_throw_from_csv(csv_path):
    # 1️⃣ Load CSV
    df = pd.read_csv(csv_path)

    # 2️⃣ Extract features
    X = extract_features(df)

    # 3️⃣ Dummy prediction (replace with clf.predict(X) when trained)
    pred_label = "Good"
    pred_prob = 0.85

    # 4️⃣ Technical advice based on heuristics
    advice = []
    if not X["elbow_extended_flag"].iloc[0]:
        advice.append("Elbow not fully extended")
    if not X["hip_rotated_flag"].iloc[0]:
        advice.append("Insufficient hip rotation")
    if not X["knee_extended_flag"].iloc[0]:
        advice.append("Knee not fully extended")

    return {
        "label": pred_label,
        "probability": pred_prob,
        "advice": advice
    }
