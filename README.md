# Tourism Experience Analytics

## Project Overview

This repository contains a Tourism Experience Analytics project that ingests raw tourism transaction datasets, performs cleaning and feature engineering, trains models for rating prediction and visit-mode classification, and provides a Streamlit-based front-end (`app.py`) for predictions and recommendations.

## Technologies Used

- Python 3.8+
- pandas, numpy
- scikit-learn
- Streamlit
- joblib
- matplotlib / seaborn for plotting

## Files

- `app.py` - Streamlit frontend application
- `data_cleaning.py` - Data cleaning and feature engineering
- `data_preparation.py` - Merges raw data sources into `master_tourism_dataset.csv`
- `eda_analysis.py` - Exploratory data analysis visualizations
- `regression_model.py` - Train and save the regression model (`regression_model.pkl`)
- `classification_model.py` - Train and save visit-mode classifier (`visit_mode_model.pkl` and `visit_mode_label_encoder.pkl`)
- `recommendations_model.py` - Functions for collaborative filtering recommendations
- `cleaned_tourism_dataset.csv` - cleaned dataset (generated)
- `master_tourism_dataset.csv` - merged master dataset (generated)

## How to Run

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare the datasets:
- Place the raw Excel files in the `data/` folder as expected by `data_preparation.py`.
- Run `python data_preparation.py` to generate `master_tourism_dataset.csv`.
- Run `python data_cleaning.py` to generate `cleaned_tourism_dataset.csv`.

3. Train models (optional if `.pkl` already present):

```bash
python regression_model.py
python classification_model.py
```

This will produce `regression_model.pkl`, `visit_mode_model.pkl`, and `visit_mode_label_encoder.pkl`.

4. Run the Streamlit app:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit in a browser (mobile-friendly layout supported).

## Model Explanation

- Regression: `RandomForestRegressor` trained to predict `Rating` using engineered features like `TotalUserVisits`, `UserAvgRating`, `AttractionAvgRating`, `VisitYear`, and `VisitMonth`.
- Classification: `RandomForestClassifier` trained to predict `VisitMode` (or `VisitModeId`) and saved together with a `LabelEncoder` to translate labels back to original strings.
- Recommendations: Basic user-based collaborative filtering using cosine similarity over the user-item rating matrix.

## Notes & Next Steps

- Verify column names in your raw Excel files match the expected join keys used in `data_preparation.py`.
- Consider persisting feature metadata and scalers if you standardize inputs for production.
- Add unit tests and CI for automated checks.
