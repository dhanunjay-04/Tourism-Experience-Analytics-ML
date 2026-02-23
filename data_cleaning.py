"""Data cleaning and feature engineering for tourism dataset.

This script reads `master_tourism_dataset.csv`, performs missing
value handling, feature engineering (user/attraction aggregates and
peak-season flags), normalizes ratings and saves a cleaned CSV
`cleaned_tourism_dataset.csv` for downstream modeling and analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==============================
# 1️⃣ LOAD DATA
# ==============================

df = pd.read_csv("master_tourism_dataset.csv")

print("Dataset Loaded Successfully!")
print("Initial Shape:", df.shape)

# ==============================
# 2️⃣ BASIC INFO
# ==============================

print("\nDataset Info:")
print(df.info())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ==============================
# 3️⃣ HANDLE MISSING VALUES
# ==============================

# Drop rows where Rating is missing (important for ML tasks)
if "Rating" in df.columns:
    df = df.dropna(subset=["Rating"])

# Fill categorical columns with 'Unknown'
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# Fill numerical columns with median
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ==============================
# 4️⃣ FIX DATA TYPES
# ==============================

if "VisitYear" in df.columns:
    df["VisitYear"] = df["VisitYear"].astype(int)

if "VisitMonth" in df.columns:
    df["VisitMonth"] = df["VisitMonth"].astype(int)

if "Rating" in df.columns:
    df["Rating"] = df["Rating"].astype(float)

# ==============================
# 5️⃣ REMOVE DUPLICATES
# ==============================

df = df.drop_duplicates()
print("\nShape After Removing Duplicates:", df.shape)

# ==============================
# 6️⃣ FEATURE ENGINEERING
# ==============================

# 🔹 Total Visits Per User
if "UserId" in df.columns:
    user_visit_count = df.groupby("UserId").size().reset_index(name="TotalUserVisits")
    df = df.merge(user_visit_count, on="UserId", how="left")

# 🔹 Average Rating Per User
if "UserId" in df.columns and "Rating" in df.columns:
    user_avg_rating = df.groupby("UserId")["Rating"].mean().reset_index(name="UserAvgRating")
    df = df.merge(user_avg_rating, on="UserId", how="left")

# 🔹 Average Rating Per Attraction
if "AttractionId" in df.columns and "Rating" in df.columns:
    attraction_avg_rating = df.groupby("AttractionId")["Rating"].mean().reset_index(name="AttractionAvgRating")
    df = df.merge(attraction_avg_rating, on="AttractionId", how="left")

# 🔹 Peak Season Feature
if "VisitMonth" in df.columns:
    df["IsPeakSeason"] = df["VisitMonth"].apply(lambda x: 1 if x in [5,6,7,11,12] else 0)

# ==============================
# 7️⃣ NORMALIZE RATING
# ==============================

if "Rating" in df.columns:
    scaler = MinMaxScaler()
    df["Rating_Scaled"] = scaler.fit_transform(df[["Rating"]])

# ==============================
# 8️⃣ FINAL CHECK
# ==============================

print("\nFinal Shape:", df.shape)
print("\nFinal Columns:")
print(df.columns)

# ==============================
# 9️⃣ SAVE CLEANED DATASET
# ==============================

df.to_csv("cleaned_tourism_dataset.csv", index=False)

print("\nCleaned dataset saved successfully as 'cleaned_tourism_dataset.csv'")