from utils.data_loader import load_data, preprocess_data
from models.pca_model import PCA
from models.svd_model import SVD
from utils.clustering import KMeans, evaluate_k_values
from utils.visualization import (
    plot_explained_variance,
    plot_cumulative_variance,
    plot_pca_projection,
    plot_clusters,
    plot_k_evaluation,
)
import numpy as np


DATA_PATH = "data/winequality-red.csv"


def main():
    # --------------------------------------------------
    # 1. Load and preprocess data
    # --------------------------------------------------

    print("=" * 60)
    print("SPECTRAL LEARNING PROJECT")
    print("=" * 60)

    data = load_data(DATA_PATH)

    X, feature_names = preprocess_data(data)

    print("\nDataset shape:", data.shape)
    print("Feature matrix shape:", X.shape)

    # --------------------------------------------------
    # 2. PCA
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PCA")
    print("=" * 60)

    pca = PCA()

    X_pca = pca.fit_transform(X)

    print("\nNumber of components:", len(pca.eigenvalues))

    print("\nExplained variance ratio:")

    for i, variance in enumerate(
        pca.explained_variance_ratio,
        start=1
    ):
        print(
            f"PC{i}: {variance:.4f} "
            f"({variance * 100:.2f}%)"
        )

    print(
        "\nVariance explained by first 2 components:",
        f"{pca.cumulative_variance()[1] * 100:.2f}%"
    )

    # --------------------------------------------------
    # 3. PCA visualizations
    # --------------------------------------------------

    plot_explained_variance(
    pca.explained_variance_ratio,
    "results/pca_explained_variance.png"
    )

    plot_cumulative_variance(
        pca.cumulative_variance(),
        "results/pca_cumulative_variance.png"
    )

    # Use first two principal components
    pca_2d = PCA(n_components=2)

    X_pca_2d = pca_2d.fit_transform(X)

    plot_pca_projection(
        X_pca_2d,
        "results/pca_projection.png"
    )
    # --------------------------------------------------
    # 4. SVD
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SVD")
    print("=" * 60)

    svd = SVD()

    X_svd = svd.fit_transform(X)

    print("\nSingular values:")

    for i, value in enumerate(
        svd.singular_values,
        start=1
    ):
        print(f"σ{i}: {value:.4f}")

    print("\nSVD explained variance:")

    for i, variance in enumerate(
        svd.explained_variance_ratio,
        start=1
    ):
        print(
            f"Component {i}: "
            f"{variance:.4f} "
            f"({variance * 100:.2f}%)"
        )

    # --------------------------------------------------
    # 5. Verify PCA and SVD relationship
    # --------------------------------------------------

    difference = np.max(
        np.abs(
            pca.explained_variance_ratio
            - svd.explained_variance_ratio
        )
    )

    print(
        "\nMaximum PCA/SVD variance difference:",
        f"{difference:.10f}"
    )

    # --------------------------------------------------
    # 6. K-Means clustering
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    labels = kmeans.fit_predict(X_pca_2d)

    print("\nCluster sizes:")

    for cluster in range(3):
        count = np.sum(labels == cluster)
        print(f"Cluster {cluster}: {count}")

    print("\nInertia:", kmeans.inertia_)

    # --------------------------------------------------
    # 7. Cluster visualization
    # --------------------------------------------------

    plot_clusters(
        X_pca_2d,
        labels,
        "results/kmeans_clusters.png"
    )

    # --------------------------------------------------
    # 8. Evaluate different K values
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("K-MEANS EVALUATION")
    print("=" * 60)

    results = evaluate_k_values(
        X_pca_2d,
        range(2, 7)
    )

    print("\nK | Inertia | Silhouette")
    print("-" * 30)

    for result in results:
        print(
            f"{result['k']} | "
            f"{result['inertia']:.2f} | "
            f"{result['silhouette']:.4f}"
        )

    plot_k_evaluation(
        results,
        "results"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()