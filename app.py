"""Streamlit front-end for Tourism Experience Analytics.

Provides three primary functions:
- Predict Rating (regression)
- Predict Visit Mode (classification)
- Get Recommendations (collaborative filtering)

The app loads pre-trained models from .pkl files and uses the
cleaned/master datasets. It includes sidebar filters and small
visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Tourism Analytics", layout="wide")


# Small caching helpers with a backward-compatible fallback
try:
    cache_resource = st.cache_resource
except Exception:
    def cache_resource(func=None, **_kwargs):
        return st.cache(allow_output_mutation=True) if func is None else st.cache(allow_output_mutation=True)(func)


@st.cache_data
def load_dataframe(path):
    return pd.read_csv(path)


@cache_resource
def load_model(path):
    return joblib.load(path)


def safe_load_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.warning(f"Could not load model {path}: {e}")
        return None


# Load datasets
cleaned_path = "cleaned_tourism_dataset.csv"
master_path = "master_tourism_dataset.csv"

cleaned_df = load_dataframe(cleaned_path)
master_df = load_dataframe(master_path)


st.title("🌍 Smart Tourism Recommendation & Analytics")

with st.sidebar:
    st.header("Controls")
    view = st.selectbox("Choose Functionality", ["Dashboard", "Analyze Trends", "Predict Rating", "Predict Visit Mode", "Get Recommendations"]) 
    show_data = st.checkbox("Show dataset preview", value=False)
    st.markdown("---")
    # Simple filters that apply to the dashboard and recommendation
    continent_filter = st.selectbox("Filter by Continent", options=["All"] + sorted(master_df["Continent"].dropna().unique().tolist()[:50]))
    top_n = st.slider("Number of recommendations", 1, 10, 5)

if show_data:
    st.subheader("Cleaned dataset preview")
    st.dataframe(cleaned_df.head(200))


def dashboard():
    st.subheader("Dataset Overview")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Rating distribution**")
        if "Rating" in cleaned_df.columns:
            st.bar_chart(cleaned_df["Rating"].value_counts().sort_index())

        st.write("**Top attractions**")
        if "Attraction" in master_df.columns:
            top = master_df["Attraction"].value_counts().head(10)
            st.bar_chart(top)

    with col2:
        st.metric("Rows (cleaned)", cleaned_df.shape[0])
        st.metric("Rows (master)", master_df.shape[0])


def predict_rating_ui():
    st.subheader("Predict Rating")
    model = safe_load_model("regression_model.pkl")
    if model is None:
        st.error("Regression model not available. Run the training script.")
        return
    # Determine required features from model if available
    model_feature_names = None
    if hasattr(model, 'feature_names_in_'):
        model_feature_names = list(model.feature_names_in_)
    else:
        features_path = "regression_features.pkl"
        if os.path.exists(features_path):
            try:
                model_feature_names = joblib.load(features_path)
            except Exception:
                model_feature_names = None

    # Fallback candidate features
    candidate_features = ["VisitYear", "VisitMonth", "TotalUserVisits", "UserAvgRating", "AttractionAvgRating", "IsPeakSeason"]
    if model_feature_names is None:
        model_feature_names = [c for c in candidate_features if c in cleaned_df.columns]

    # Build inputs in the exact order expected by the model
    inputs = {}
    cols = st.columns(2)
    for i, f in enumerate(model_feature_names):
        with cols[i % 2]:
            if f in cleaned_df.columns:
                default = float(cleaned_df[f].median())
            else:
                default = 0.0
            val = st.number_input(f, value=default)
            inputs[f] = val

    if st.button("Predict Rating"):
        x = np.array([inputs[f] for f in model_feature_names]).reshape(1, -1)
        try:
            pred = model.predict(x)[0]
            st.success(f"Predicted Rating: {pred:.2f}")
        except ValueError as e:
            st.error(f"Model input mismatch: {e}")


def predict_visit_mode_ui():
    st.subheader("Predict Visit Mode")
    clf = safe_load_model("visit_mode_model.pkl")
    le = safe_load_model("visit_mode_label_encoder.pkl")
    if clf is None or le is None:
        st.error("Visit mode model or label encoder not available. Run the training script.")
        return
    # Determine required features from model if available
    model_feature_names = None
    if hasattr(clf, 'feature_names_in_'):
        model_feature_names = list(clf.feature_names_in_)
    else:
        features_path = "classification_features.pkl"
        if os.path.exists(features_path):
            try:
                model_feature_names = joblib.load(features_path)
            except Exception:
                model_feature_names = None

    # Fallback: numeric columns excluding Rating
    if model_feature_names is None:
        numeric_cols = cleaned_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        model_feature_names = [c for c in numeric_cols if c != "Rating"]

    inputs = {}
    cols = st.columns(2)
    for i, f in enumerate(model_feature_names):
        with cols[i % 2]:
            if f in cleaned_df.columns:
                default = float(cleaned_df[f].median())
            else:
                default = 0.0
            # Allow decimal entry for numeric features
            val = st.number_input(f, value=default)
            inputs[f] = val

    if st.button("Predict Visit Mode"):
        x = np.array([inputs[f] for f in model_feature_names]).reshape(1, -1)
        try:
            pred = clf.predict(x)
            mode = le.inverse_transform(pred)[0]
            st.success(f"Predicted Visit Mode: {mode}")
        except ValueError as e:
            st.error(f"Model input mismatch: {e}")
        except Exception as e:
            st.error(f"Could not predict: {e}")


def recommendations_ui():
    st.subheader("Recommendations")
    if not {'UserId', 'AttractionId', 'Rating'}.issubset(master_df.columns):
        st.error("Master dataset missing required columns for recommendations.")
        return

    user_id = st.number_input("Enter User ID", min_value=1, value=1)

    if st.button("Recommend"):
        df = master_df.copy()
        if continent_filter != "All":
            df = df[df["Continent"] == continent_filter]

        user_item = df.pivot_table(index='UserId', columns='AttractionId', values='Rating', fill_value=0)
        if user_id not in user_item.index:
            st.error("User not found in the filtered dataset.")
            return

        # Compute similarity only between the target user and all other users
        # to avoid creating an n_users x n_users matrix which can be huge.
        try:
            target_vec = user_item.loc[[user_id]]
            # cosine_similarity between shape (1, n_items) and (n_users, n_items)
            sim_scores = cosine_similarity(target_vec, user_item)[0]
            sim_series = pd.Series(sim_scores, index=user_item.index)
            # exclude the user themself and get top similar users
            similar_users = sim_series.drop(index=user_id).sort_values(ascending=False).head(5)

            # Score candidate items by weighted sum of similar users' ratings
            scores = {}
            for u, sim in similar_users.items():
                items = user_item.loc[u]
                for attraction_id, rating in items[items > 0].items():
                    # skip items the target user already rated
                    if user_item.loc[user_id, attraction_id] > 0:
                        continue
                    scores[attraction_id] = scores.get(attraction_id, 0.0) + sim * rating

            if not scores:
                st.info("No candidate recommendations found from similar users. Falling back to popular attractions.")
                # Fallback: recommend globally popular attractions (by count and mean rating)
                try:
                    pop = df.groupby('AttractionId').agg(Count=('Rating', 'count'), AvgRating=('Rating', 'mean')).reset_index()
                    # exclude attractions the user already rated
                    rated_by_user = user_item.loc[user_id]
                    rated_ids = rated_by_user[rated_by_user > 0].index.tolist()
                    pop = pop[~pop['AttractionId'].isin(rated_ids)]
                    pop = pop.sort_values(['Count', 'AvgRating'], ascending=[False, False]).head(top_n)

                    id_to_name = {}
                    if 'Attraction' in df.columns:
                        id_to_name = df.drop_duplicates('AttractionId').set_index('AttractionId')['Attraction'].to_dict()

                    rows = []
                    for _, r in pop.iterrows():
                        aid = int(r['AttractionId'])
                        rows.append({'AttractionId': aid, 'AttractionName': id_to_name.get(aid, ''), 'Count': int(r['Count']), 'AvgRating': float(r['AvgRating'])})

                    st.success('Popular attractions fallback')
                    st.table(pd.DataFrame(rows))
                except Exception as e:
                    st.error(f"Fallback recommendation failed: {e}")
                return

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_items = ranked[:top_n]

            # Build display table
            rows = []
            id_to_name = {}
            if 'Attraction' in df.columns:
                id_to_name = df.drop_duplicates('AttractionId').set_index('AttractionId')['Attraction'].to_dict()

            for aid, score in top_items:
                rows.append({
                    'AttractionId': int(aid),
                    'AttractionName': id_to_name.get(aid, ''),
                    'Score': float(score)
                })

            result_df = pd.DataFrame(rows)
            st.success("Top recommendations")
            st.table(result_df)
        except MemoryError:
            st.error("Not enough memory to compute recommendations for this dataset. Consider filtering dataset or using a smaller sample.")
        except Exception as e:
            st.error(f"Recommendation error: {e}")


if view == "Dashboard":
    dashboard()
elif view == "Analyze Trends":
    def analyze_trends_ui():
        st.header("Analyze Trends")

        df_clean = cleaned_df.copy()
        df_master = master_df.copy()

        if continent_filter != "All":
            df_master = df_master[df_master["Continent"] == continent_filter]
            df_clean = df_clean[df_clean["Continent"] == continent_filter]

        st.subheader("Rating Distribution")
        if "Rating" in df_clean.columns:
            st.bar_chart(df_clean["Rating"].value_counts().sort_index())

        st.subheader("Visit Mode Distribution")
        if "VisitMode" in df_clean.columns:
            st.bar_chart(df_clean["VisitMode"].value_counts())
        else:
            st.info("No `VisitMode` column available in dataset.")

        st.subheader("Top Attractions by Visits")
        if "Attraction" in df_master.columns:
            top = df_master["Attraction"].value_counts().head(10)
            st.bar_chart(top)

        st.subheader("Monthly Visit Trend")
        if "VisitMonth" in df_clean.columns:
            monthly = df_clean["VisitMonth"].value_counts().sort_index()
            st.line_chart(monthly)

        st.subheader("Average Rating by Attraction Type")
        if "AttractionType" in df_clean.columns and "Rating" in df_clean.columns:
            avg_rating = df_clean.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False)
            st.bar_chart(avg_rating.head(15))
        else:
            st.info("Attraction type or Rating column missing for this analysis.")

    analyze_trends_ui()
elif view == "Predict Rating":
    predict_rating_ui()
elif view == "Predict Visit Mode":
    predict_visit_mode_ui()
elif view == "Get Recommendations":
    recommendations_ui()
