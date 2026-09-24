import numpy as np


class SVD:
    """
    Singular Value Decomposition implemented from scratch
    using eigenvalue decomposition of X^T X.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None
        self.singular_values = None
        self.Vt = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Fit SVD to the input matrix.

        Parameters:
            X (np.ndarray): Input matrix.

        Returns:
            self
        """

        # Calculate X^T X
        covariance_like = X.T @ X

        # Eigenvalue decomposition
        eigenvalues, V = np.linalg.eigh(covariance_like)

        # Sort from largest to smallest
        indices = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[indices]
        V = V[:, indices]

        # Numerical errors can produce tiny negative values
        eigenvalues = np.maximum(eigenvalues, 0)

        # Singular values are square roots of eigenvalues
        singular_values = np.sqrt(eigenvalues)

        # Calculate U = X V / sigma
        U = np.zeros((X.shape[0], len(singular_values)))

        for i, sigma in enumerate(singular_values):
            if sigma > 1e-12:
                U[:, i] = (X @ V[:, i]) / sigma

        self.U = U
        self.singular_values = singular_values
        self.Vt = V.T

        # Variance explained by each singular value
        squared_singular_values = singular_values ** 2
        total = np.sum(squared_singular_values)

        self.explained_variance_ratio = (
            squared_singular_values / total
        )

        return self

    def transform(self, X):
        """
        Project X into the reduced SVD space.

        Parameters:
            X (np.ndarray): Input matrix.

        Returns:
            np.ndarray: Reduced representation.
        """

        if self.Vt is None:
            raise ValueError("SVD must be fitted before calling transform().")

        if self.n_components is None:
            k = len(self.singular_values)
        else:
            k = self.n_components

        return X @ self.Vt[:k].T

    def fit_transform(self, X):
        """
        Fit SVD and transform the data.

        Parameters:
            X (np.ndarray): Input matrix.

        Returns:
            np.ndarray: Reduced representation.
        """

        self.fit(X)

        return self.transform(X)

    def cumulative_variance(self):
        """
        Calculate cumulative explained variance.

        Returns:
            np.ndarray: Cumulative variance.
        """

        if self.explained_variance_ratio is None:
            raise ValueError("SVD must be fitted first.")

        return np.cumsum(self.explained_variance_ratio)