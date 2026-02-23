"""Recommendation utilities using simple user-based collaborative filtering.

This module exposes a small helper `recommend_attractions` that computes
user-user cosine similarity over the user-item rating matrix and returns
top attractions recommended for a given user id.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def build_similarity_from_df(df):
    """Build user-item matrix and user-user similarity DataFrame.

    Returns (user_item_matrix, similarity_df).
    """
    if not {'UserId', 'AttractionId', 'Rating'}.issubset(df.columns):
        raise ValueError("DataFrame must contain UserId, AttractionId and Rating columns")

    user_item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating', fill_value=0)
    user_similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    return user_item_matrix, similarity_df


def recommend_attractions_from_matrices(user_item_matrix, similarity_df, user_id, top_n=5):
    """Return recommended attraction IDs for `user_id` given matrices."""
    if user_id not in similarity_df.index:
        return []

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
    recommended = set()
    for u in similar_users.index:
        items = user_item_matrix.loc[u]
        top = items[items > 3].index
        recommended.update(top)

    return list(recommended)[:top_n]


def average_precision(actual, predicted, k=5):
    score = 0.0
    hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            hits += 1.0
            score += hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mean_average_precision(actual_list, predicted_list, k=5):
    return np.mean([average_precision(a, p, k) for a, p in zip(actual_list, predicted_list)])
