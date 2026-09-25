import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


def preprocess_and_create_matrix(ratings_df):
    """
    Cleans ratings, splits into train/test sets, and creates
    a user-item matrix using TRAINING ratings only.
    """

    # 1. Remove null values
    ratings_df = ratings_df.dropna()

    # 2. Split into training and testing sets
    train_data, test_data = train_test_split(
        ratings_df,
        test_size=0.2,
        random_state=42
    )

    # 3. Create mappings from the TRAINING data
    user_ids = train_data['UserID'].unique()
    movie_ids = train_data['MovieID'].unique()

    user_to_idx = {
        uid: i for i, uid in enumerate(user_ids)
    }

    movie_to_idx = {
        mid: i for i, mid in enumerate(movie_ids)
    }

    # 4. Build matrix using TRAINING ratings ONLY
    rows = train_data['UserID'].map(user_to_idx)
    cols = train_data['MovieID'].map(movie_to_idx)
    data = train_data['Rating']

    sparse_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids))
    )

    # 5. Save processed data
    os.makedirs("processed", exist_ok=True)

    train_data.to_csv(
        "processed/train_ratings.csv",
        index=False
    )

    test_data.to_csv(
        "processed/test_ratings.csv",
        index=False
    )

    # 6. Save sparse matrix
    np.savez(
        "processed/sparse_user_item.npz",
        data=sparse_matrix.data,
        indices=sparse_matrix.indices,
        indptr=sparse_matrix.indptr,
        shape=sparse_matrix.shape
    )

    # 7. Save dense CSV for compatibility
    user_item_matrix = pd.DataFrame(
        sparse_matrix.toarray(),
        index=user_ids,
        columns=movie_ids
    )

    user_item_matrix.to_csv(
        "processed/user_item_matrix.csv"
    )

    print(f"Training ratings: {len(train_data)}")
    print(f"Test ratings: {len(test_data)}")
    print(f"Users in training matrix: {len(user_ids)}")
    print(f"Movies in training matrix: {len(movie_ids)}")

    return train_data, test_data, user_item_matrix
