import os
import json
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

def train_and_evaluate_svd(k=50):
    """
    Trains an SVD model using scipy.sparse.linalg.svds, evaluates RMSE on test set,
    saves model predictions, and logs metrics.
    """
    print("Loading user-item matrix and test data...")
    user_item_df = pd.read_csv("processed/user_item_matrix.csv", index_col=0)
    user_item_df.columns = user_item_df.columns.astype(int)
    
    test_data = pd.read_csv("processed/test_ratings.csv")
    train_data = pd.read_csv("processed/train_ratings.csv")
    
    matrix = user_item_df.values
    
    # Global average rating
    global_mean = train_data['Rating'].mean()
    
    # Center the data by subtracting user means (only on training non-zero ratings)
    user_ratings_mean = np.zeros(matrix.shape[0])
    for i, uid in enumerate(user_item_df.index):
        user_train = train_data[train_data['UserID'] == uid]
        if len(user_train) > 0:
            user_ratings_mean[i] = user_train['Rating'].mean()
        else:
            user_ratings_mean[i] = global_mean
            
    matrix_demeaned = np.where(matrix != 0, matrix - user_ratings_mean[:, np.newaxis], 0)
    
    # Perform SVD using scipy.sparse.linalg.svds
    print(f"Running SVD with k={k} latent factors...")
    U, sigma, Vt = svds(matrix_demeaned, k=k)
    sigma_diag = np.diag(sigma)
    
    # Reconstruct predictions adding back user means
    predicted_matrix = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean[:, np.newaxis]
    predicted_matrix = np.clip(predicted_matrix, 1.0, 5.0)
    
    user_to_idx = {uid: i for i, uid in enumerate(user_item_df.index)}
    movie_to_idx = {mid: i for i, mid in enumerate(user_item_df.columns)}
    
    # Calculate RMSE strictly on the test set pairs
    print("Evaluating RMSE on test set...")
    squared_errors = []
    for _, row in test_data.iterrows():
        u = row['UserID']
        m = row['MovieID']
        actual = row['Rating']
        
        if u in user_to_idx and m in movie_to_idx:
            ui = user_to_idx[u]
            mi = movie_to_idx[m]
            pred = predicted_matrix[ui, mi]
            squared_errors.append((pred - actual) ** 2)
        else:
            # Fallback to global mean if user/movie wasn't in matrix
            squared_errors.append((global_mean - actual) ** 2)
            
    rmse = np.sqrt(np.mean(squared_errors))
    print(f"SVD Test RMSE: {rmse:.4f}")
    
    # Save reports
    os.makedirs("reports", exist_ok=True)
    np.save("reports/svd_predictions.npy", predicted_matrix)
    
    metrics_path = "reports/model_metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                metrics = {}
                
    metrics["SVD_RMSE"] = round(float(rmse), 4)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("SVD Model training complete!")
    return rmse, predicted_matrix

if __name__ == "__main__":
    train_and_evaluate_svd()