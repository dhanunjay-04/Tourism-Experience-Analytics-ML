"""Train and save a visit-mode classification model.

The original Streamlit app expects a model named `visit_mode_model.pkl`
and a label encoder `visit_mode_label_encoder.pkl`. This script trains
a classifier to predict `VisitMode` (if available) and saves both
artifacts.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


def main():
    # Prefer cleaned dataset which contains merged columns
    df = pd.read_csv("cleaned_tourism_dataset.csv")

    if "VisitMode" not in df.columns and "VisitModeId" not in df.columns:
        raise RuntimeError("No VisitMode or VisitModeId column found for classification training.")

    # If VisitMode string exists, use it. Otherwise encode VisitModeId.
    if "VisitMode" in df.columns:
        y_raw = df["VisitMode"].fillna("Unknown")
    else:
        y_raw = df["VisitModeId"].astype(str).fillna("-1")

    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Select numeric features (drop identifiers and Rating)
    X = df.select_dtypes(include=["int64", "float64"]).copy()
    for col in ["Rating"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    X = X.fillna(X.mean())

    # Persist the feature names used for training so the app can construct
    # input arrays in the same order at prediction time.
    features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save artifacts used by the app
    joblib.dump(model, "visit_mode_model.pkl")
    joblib.dump(le, "visit_mode_label_encoder.pkl")
    joblib.dump(features, "classification_features.pkl")
    print("Saved visit mode model and label encoder")


if __name__ == "__main__":
    main()