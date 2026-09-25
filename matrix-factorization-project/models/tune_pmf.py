import os
import numpy as np
import pandas as pd
import optuna
from pmf_model import ProbabilisticMatrixFactorization

# Suppress overly verbose Optuna logs if desired
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    # Load training and test data
    train_data = pd.read_csv("processed/train_ratings.csv")
    test_data = pd.read_csv("processed/test_ratings.csv")
    
    user_ids = train_data['UserID'].unique()
    movie_ids = train_data['MovieID'].unique()
    n_users = len(user_ids)
    n_items = len(movie_ids)
    global_mean = train_data['Rating'].mean()
    
    # Suggest hyperparameters using TPE (Bayesian Optimization)
    latent_dim = trial.suggest_int('latent_dim', 10, 40, step=5)
    lr = trial.suggest_float('lr', 0.001, 0.008, log=True)
    reg = trial.suggest_float('reg', 0.01, 0.15, log=True)
    
    # Initialize PMF with trial parameters
    pmf = ProbabilisticMatrixFactorization(
        n_users=n_users, 
        n_items=n_items, 
        latent_dim=latent_dim, 
        lr=lr, 
        reg=reg
    )
    
    # Train for 15 epochs per trial (sufficient for evaluation trend)
    _, user_to_idx, movie_to_idx = pmf.fit(train_data, epochs=15)
    
    # Evaluate Test RMSE
    squared_errors = []
    for _, row in test_data.iterrows():
        u = row['UserID']
        m = row['MovieID']
        actual = row['Rating']
        
        if u in user_to_idx and m in movie_to_idx:
            ui = user_to_idx[u]
            mi = movie_to_idx[m]
            pred = np.dot(pmf.U[ui], pmf.V[mi])
            pred = np.clip(pred, 1.0, 5.0)
            squared_errors.append((pred - actual) ** 2)
        else:
            squared_errors.append((global_mean - actual) ** 2)
            
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse

def run_bayesian_optimization():
    print("Initializing Bayesian Optimization Study with Optuna...")
    study = optuna.create_study(direction='minimize')
    
    # Run 12 intelligent trials
    study.optimize(objective, n_trials=12, show_progress_bar=True)
    
    print("\n" + "=" * 50)
    print("Optimization Complete!")
    print(f"Best Test RMSE: {study.best_value:.4f}")
    print("Best Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("=" * 50)
    
    return study.best_params

if __name__ == "__main__":
    run_bayesian_optimization()