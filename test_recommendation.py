import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MASTER = "master_tourism_dataset.csv"
TOP_N = 5

if __name__ == "__main__":
    df = pd.read_csv(MASTER)
    if not {'UserId', 'AttractionId', 'Rating'}.issubset(df.columns):
        print('Master dataset missing required columns (UserId, AttractionId, Rating).')
        raise SystemExit(1)

    # pick a user to test: first unique UserId
    user_ids = df['UserId'].dropna().unique()
    if len(user_ids) == 0:
        print('No users in dataset.')
        raise SystemExit(1)

    user_id = int(user_ids[0])
    print(f'Testing recommendations for UserId = {user_id}')

    user_item = df.pivot_table(index='UserId', columns='AttractionId', values='Rating', fill_value=0)
    if user_id not in user_item.index:
        print('Selected user not found in user-item matrix.')
        raise SystemExit(1)

    target_vec = user_item.loc[[user_id]]
    sim_scores = cosine_similarity(target_vec, user_item)[0]
    sim_series = pd.Series(sim_scores, index=user_item.index)
    similar_users = sim_series.drop(index=user_id).sort_values(ascending=False).head(10)

    recommended = []
    for u in similar_users.index:
        items = user_item.loc[u]
        top = items[items > 3].index.tolist()
        recommended.extend(top)

    # unique while preserving order
    seen = set()
    rec_unique = [x for x in recommended if not (x in seen or seen.add(x))]

    rec_final = rec_unique[:TOP_N]
    print('Recommended Attraction IDs:', rec_final)

    # try to show attraction names if exist
    if 'Attraction' in df.columns:
        id_to_name = df.drop_duplicates('AttractionId').set_index('AttractionId')['Attraction'].to_dict()
        rec_names = [id_to_name.get(a, str(a)) for a in rec_final]
        print('Recommended Attraction Names:', rec_names)
