
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ProbabilisticMatrixFactorization:
    def __init__(self, n_users, n_items, latent_dim=30, lr=0.005, reg=0.02):
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.lr = lr
        self.reg = reg

        # Fixed seed for reproducible initialization
        np.random.seed(42)

        # Initialize latent factor matrices
        self.U = np.random.normal(0, 0.1, (n_users, latent_dim))
        self.V = np.random.normal(0, 0.1, (n_items, latent_dim))

        # Initialize biases and global mean
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.mu = 0.0

    def fit(self, train_data, epochs=30):
        self.mu = train_data["Rating"].mean()
        mse_history = []

        user_ids = train_data["UserID"].unique()
        movie_ids = train_data["MovieID"].unique()

        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

        # Map ratings to matrix indices
        train_u = train_data["UserID"].map(user_to_idx).values
        train_m = train_data["MovieID"].map(movie_to_idx).values
        train_r = train_data["Rating"].values

        for epoch in range(epochs):
            error = 0.0

            indices = np.random.permutation(len(train_r))

            for idx in indices:
                u = train_u[idx]
                m = train_m[idx]
                r = train_r[idx]

                prediction = (
                    self.mu
                    + self.b_u[u]
                    + self.b_i[m]
                    + np.dot(self.U[u], self.V[m])
                )

                err = r - prediction
                error += err ** 2

                # Update biases
                self.b_u[u] += self.lr * (
                    err - self.reg * self.b_u[u]
                )

                self.b_i[m] += self.lr * (
                    err - self.reg * self.b_i[m]
                )

                # Save old user vector before updating
                u_old = self.U[u].copy()

                # Update latent factors
                self.U[u] += self.lr * (
                    err * self.V[m] - self.reg * self.U[u]
                )

                self.V[m] += self.lr * (
                    err * u_old - self.reg * self.V[m]
                )

            mse = error / len(train_r)
            mse_history.append(mse)

            print(
                f"Epoch {epoch + 1}/{epochs} - MSE: {mse:.4f}"
            )

        return mse_history, user_to_idx, movie_to_idx


def train_and_evaluate_pmf():

    print("Loading training and test data...")

    train_data = pd.read_csv("processed/train_ratings.csv")
    test_data = pd.read_csv("processed/test_ratings.csv")

    user_ids = train_data["UserID"].unique()
    movie_ids = train_data["MovieID"].unique()

    n_users = len(user_ids)
    n_items = len(movie_ids)

    print(
        f"Training PMF on {len(train_data)} ratings "
        f"({n_users} users, {n_items} movies)..."
    )

    # Slightly tuned parameters
    pmf = ProbabilisticMatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        latent_dim=20,
        lr=0.005,
        reg=0.05
    )

    mse_history, user_to_idx, movie_to_idx = pmf.fit(
        train_data,
        epochs=30
    )

    # Save convergence plot
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(mse_history) + 1),
        mse_history,
        marker="o"
    )

    plt.title("PMF Convergence (MSE vs. Iteration)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)

    plt.savefig("reports/pmf_convergence.png")
    plt.close()

    # Save latent factors
    os.makedirs("reports/pmf_factors", exist_ok=True)

    np.save(
        "reports/pmf_factors/U_matrix.npy",
        pmf.U
    )

    np.save(
        "reports/pmf_factors/V_matrix.npy",
        pmf.V
    )

    # Evaluate PMF on test set
    print("Evaluating PMF RMSE on test set...")

    squared_errors = []

    known_ratings = 0
    fallback_ratings = 0

    for _, row in test_data.iterrows():

        u = row["UserID"]
        m = row["MovieID"]
        actual = row["Rating"]

        if u in user_to_idx and m in movie_to_idx:

            ui = user_to_idx[u]
            mi = movie_to_idx[m]

            pred = (
                pmf.mu
                + pmf.b_u[ui]
                + pmf.b_i[mi]
                + np.dot(pmf.U[ui], pmf.V[mi])
            )

            pred = np.clip(pred, 1.0, 5.0)

            squared_errors.append(
                (pred - actual) ** 2
            )

            known_ratings += 1

        else:

            squared_errors.append(
                (pmf.mu - actual) ** 2
            )

            fallback_ratings += 1

    pmf_rmse = np.sqrt(np.mean(squared_errors))

    print(f"PMF Test RMSE: {pmf_rmse:.4f}")
    print(f"Known test ratings: {known_ratings}")
    print(f"Fallback ratings: {fallback_ratings}")

    # Load existing metrics
    metrics_path = "reports/model_metrics.json"

    metrics = {}

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    svd_rmse = metrics.get("SVD_RMSE", 0.8557)

    # Calculate improvement over SVD
    improvement = (
        (svd_rmse - pmf_rmse)
        / svd_rmse
    ) * 100

    metrics["PMF_RMSE"] = round(
        float(pmf_rmse),
        4
    )

    metrics["PMF_vs_SVD_improvement_%"] = round(
        float(improvement),
        2
    )

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\nPMF training and evaluation complete!")
    print(f"SVD RMSE: {svd_rmse:.4f}")
    print(f"PMF RMSE: {pmf_rmse:.4f}")
    print(f"PMF vs SVD: {improvement:.2f}%")

    return pmf_rmse


if __name__ == "__main__":
    train_and_evaluate_pmf()
