import matplotlib.pyplot as plt
import numpy as np


def plot_explained_variance(explained_variance, save_path=None):
    """
    Plot explained variance for each principal component.
    """

    components = np.arange(1, len(explained_variance) + 1)

    plt.figure(figsize=(8, 5))

    plt.bar(
        components,
        explained_variance
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")

    plt.xticks(components)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_cumulative_variance(cumulative_variance, save_path=None):
    """
    Plot cumulative explained variance.
    """

    components = np.arange(1, len(cumulative_variance) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        components,
        cumulative_variance,
        marker="o"
    )

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")

    plt.xticks(components)
    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_pca_projection(X_reduced, save_path=None):
    """
    Plot the first two principal components.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        alpha=0.6
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Wine Dataset in PCA Space")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_clusters(X_reduced, labels, save_path=None):
    """
    Plot K-Means clusters in PCA space.
    """

    plt.figure(figsize=(8, 6))

    for cluster in np.unique(labels):
        points = X_reduced[labels == cluster]

        plt.scatter(
            points[:, 0],
            points[:, 1],
            alpha=0.6,
            label=f"Cluster {cluster}"
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means Clusters in PCA Space")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_k_evaluation(results, save_dir=None):
    """
    Plot inertia and silhouette score across K values.
    """

    k_values = [result["k"] for result in results]
    inertia = [result["inertia"] for result in results]
    silhouette = [result["silhouette"] for result in results]

    # Inertia
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        inertia,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("K-Means Inertia")

    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        plt.savefig(
            f"{save_dir}/kmeans_inertia.png",
            dpi=300
        )

    plt.show()

    # Silhouette
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        silhouette,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means Silhouette Score")

    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        plt.savefig(
            f"{save_dir}/kmeans_silhouette.png",
            dpi=300
        )

    plt.show()