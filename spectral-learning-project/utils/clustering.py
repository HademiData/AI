import numpy as np


class KMeans:
    """
    K-Means clustering implemented from scratch.
    """

    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        """Randomly select initial centroids from the dataset."""

        rng = np.random.default_rng(self.random_state)

        indices = rng.choice(
            X.shape[0],
            size=self.n_clusters,
            replace=False
        )

        return X[indices].copy()

    def _assign_clusters(self, X, centroids):
        """Assign each sample to its nearest centroid."""

        distances = np.linalg.norm(
            X[:, np.newaxis] - centroids,
            axis=2
        )

        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Calculate new centroid positions."""

        centroids = np.zeros(
            (self.n_clusters, X.shape[1])
        )

        for cluster in range(self.n_clusters):
            points = X[labels == cluster]

            if len(points) > 0:
                centroids[cluster] = np.mean(points, axis=0)
            else:
                # Keep the previous centroid if cluster is empty
                centroids[cluster] = X[
                    np.random.randint(X.shape[0])
                ]

        return centroids

    def _calculate_inertia(self, X, labels, centroids):
        """Calculate within-cluster sum of squared distances."""

        inertia = 0.0

        for cluster in range(self.n_clusters):
            points = X[labels == cluster]

            if len(points) > 0:
                distances = points - centroids[cluster]
                inertia += np.sum(distances ** 2)

        return inertia

    def fit(self, X):
        """
        Fit K-Means to the dataset.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            self
        """

        centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):

            labels = self._assign_clusters(X, centroids)

            new_centroids = self._update_centroids(
                X,
                labels
            )

            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break

            centroids = new_centroids

        labels = self._assign_clusters(X, centroids)

        self.centroids = centroids
        self.labels_ = labels
        self.inertia_ = self._calculate_inertia(
            X,
            labels,
            centroids
        )

        return self

    def predict(self, X):
        """
        Assign new samples to the nearest cluster.
        """

        if self.centroids is None:
            raise ValueError(
                "K-Means must be fitted before calling predict()."
            )

        return self._assign_clusters(
            X,
            self.centroids
        )

    def fit_predict(self, X):
        """
        Fit K-Means and return cluster labels.
        """

        self.fit(X)

        return self.labels_


def silhouette_score(X, labels):
    """
    Calculate the average silhouette score.

    The silhouette score measures how well each point
    fits within its own cluster compared with the
    nearest neighboring cluster.

    Score range:
        -1 → poor clustering
         0 → overlapping clusters
         1 → well-separated clusters
    """

    n_samples = X.shape[0]

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Silhouette score requires at least 2 clusters."
        )

    scores = np.zeros(n_samples)

    for i in range(n_samples):

        current_cluster = labels[i]

        # Points in the same cluster
        same_cluster = labels == current_cluster

        # Exclude the point itself
        same_cluster[i] = False

        if np.sum(same_cluster) > 0:
            intra_distances = np.linalg.norm(
                X[same_cluster] - X[i],
                axis=1
            )

            a = np.mean(intra_distances)
        else:
            a = 0.0

        # Calculate average distance to every other cluster
        other_cluster_distances = []

        for cluster in np.unique(labels):

            if cluster == current_cluster:
                continue

            cluster_points = X[labels == cluster]

            distances = np.linalg.norm(
                cluster_points - X[i],
                axis=1
            )

            other_cluster_distances.append(
                np.mean(distances)
            )

        b = min(other_cluster_distances)

        # Silhouette coefficient
        scores[i] = (b - a) / max(a, b)

    return np.mean(scores)

def evaluate_k_values(X, k_values):
    """
    Evaluate K-Means clustering across multiple values of K.

    Returns:
        list: Results containing K, inertia, and silhouette score.
    """

    results = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)

        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)

        results.append({
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette": score
        })

    return results