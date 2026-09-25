import os
import pandas as pd

def load_movielens_data(data_dir="data"):
    """Loads ratings, users, and movies datasets from MovieLens 1M files with explicit column names."""
    ratings_path = os.path.join(data_dir, "ratings.dat")
    users_path = os.path.join(data_dir, "users.dat")
    movies_path = os.path.join(data_dir, "movies.dat")
    
    # Explicitly pass header=None and names so pandas doesn't misinterpret the first row
    ratings = pd.read_csv(
        ratings_path, sep='::', engine='python', header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    users = pd.read_csv(
        users_path, sep='::', engine='python', header=None,
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    )
    movies = pd.read_csv(
        movies_path, sep='::', engine='python', encoding='latin-1', header=None,
        names=['MovieID', 'Title', 'Genres']
    )
    
    return ratings, users, movies