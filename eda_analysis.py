"""Exploratory Data Analysis for tourism dataset.

Generates a set of plots (distribution, top attractions, visit modes,
monthly trends, and correlation heatmap) using the cleaned dataset.
Run interactively (plots displayed) or adapt to save figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_tourism_dataset.csv")

print("Dataset Loaded for EDA")
print("Shape:", df.shape)

# Set style
sns.set(style="whitegrid")

# ==============================
# 1️⃣ Rating Distribution
# ==============================

plt.figure()
sns.histplot(df["Rating"], bins=10)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# ==============================
# 2️⃣ Top 10 Most Visited Attractions
# ==============================

top_attractions = df["Attraction"].value_counts().head(10)

plt.figure()
sns.barplot(x=top_attractions.values, y=top_attractions.index)
plt.title("Top 10 Most Visited Attractions")
plt.xlabel("Number of Visits")
plt.ylabel("Attraction")
plt.show()

# ==============================
# 3️⃣ Visit Mode Distribution
# ==============================

if "VisitMode" in df.columns:
    plt.figure()
    sns.countplot(y=df["VisitMode"], order=df["VisitMode"].value_counts().index)
    plt.title("Visit Mode Distribution")
    plt.xlabel("Count")
    plt.ylabel("Visit Mode")
    plt.show()

# ==============================
# 4️⃣ Visits by Continent
# ==============================

if "Continent" in df.columns:
    plt.figure()
    sns.countplot(y=df["Continent"], order=df["Continent"].value_counts().index)
    plt.title("Visits by Continent")
    plt.xlabel("Count")
    plt.ylabel("Continent")
    plt.show()

# ==============================
# 5️⃣ Average Rating by Attraction Type
# ==============================

if "AttractionType" in df.columns:
    avg_rating_type = df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False)

    plt.figure()
    sns.barplot(x=avg_rating_type.values, y=avg_rating_type.index)
    plt.title("Average Rating by Attraction Type")
    plt.xlabel("Average Rating")
    plt.ylabel("Attraction Type")
    plt.show()

# ==============================
# 6️⃣ Monthly Visit Trend
# ==============================

if "VisitMonth" in df.columns:
    monthly_visits = df["VisitMonth"].value_counts().sort_index()

    plt.figure()
    sns.lineplot(x=monthly_visits.index, y=monthly_visits.values)
    plt.title("Monthly Visit Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Visits")
    plt.show()

# ==============================
# 7️⃣ Correlation Heatmap
# ==============================

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure()
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("EDA Completed Successfully!")