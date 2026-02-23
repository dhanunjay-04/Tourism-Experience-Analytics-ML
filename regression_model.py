"""Train and save a regression model to predict tourism ratings.

This script reads `cleaned_tourism_dataset.csv`, selects sensible
features (if present), trains a RandomForestRegressor and saves the
trained model as `regression_model.pkl` so that `app.py` can load it.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def main():
    df = pd.read_csv("cleaned_tourism_dataset.csv")

    print("Dataset loaded for regression training. Shape:", df.shape)

    # Prefer the engineered column names produced by data_cleaning.py
    candidate_features = [
        "VisitYear", "VisitMonth",
        "TotalUserVisits", "UserAvgRating", "AttractionAvgRating",
        "IsPeakSeason", "Rating_Scaled"
    ]

    features = [c for c in candidate_features if c in df.columns]
    if not features:
        raise RuntimeError("No features found for regression training. Check cleaned dataset.")

    X = df[features]
    y = df["Rating"]

    # Simple missing value handling
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation Results:")
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)

    # Save model with the filename expected by the Streamlit app
    joblib.dump(model, "regression_model.pkl")
    # Also persist the feature names used during training so the app can
    # construct inputs in the same order when performing predictions.
    joblib.dump(features, "regression_features.pkl")
    print("Saved regression model to 'regression_model.pkl' and feature list to 'regression_features.pkl'")


if __name__ == "__main__":
    main()