import numpy as np

from utils.matrix_operations import (
    covariance_matrix,
    sort_eigenpairs,
    explained_variance_ratio,
)


class PCA:
    """
    Principal Component Analysis implemented from scratch.
    """

    def __init__(self, n_components=None):
        """
        Parameters:
            n_components (int, optional):
                Number of principal components to retain.
                If None, retain all components.
        """
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance_ratio = None
        self.components_ = None

    def fit(self, X):
        """
        Fit PCA to the input data.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            self
        """

        # Calculate covariance matrix
        covariance = covariance_matrix(X)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Sort eigenvalues and eigenvectors
        eigenvalues, eigenvectors = sort_eigenpairs(
            eigenvalues,
            eigenvectors
        )

        # Calculate explained variance
        variance_ratio = explained_variance_ratio(eigenvalues)

        # Store results
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.explained_variance_ratio = variance_ratio

        # Select components
        if self.n_components is None:
            self.components_ = eigenvectors
        else:
            self.components_ = eigenvectors[:, :self.n_components]

        return self

    def transform(self, X):
        """
        Transform data into principal component space.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Reduced feature matrix.
        """

        if self.components_ is None:
            raise ValueError("PCA must be fitted before calling transform().")

        return X @ self.components_

    def fit_transform(self, X):
        """
        Fit PCA and transform the data.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Transformed feature matrix.
        """

        self.fit(X)

        return self.transform(X)

    def cumulative_variance(self):
        """
        Calculate cumulative explained variance.

        Returns:
            np.ndarray: Cumulative variance ratio.
        """

        if self.explained_variance_ratio is None:
            raise ValueError("PCA must be fitted first.")

        return np.cumsum(self.explained_variance_ratio)

    def get_loadings(self, n_components=None):
        """
        Return the feature loadings for the selected components.

        Parameters:
            n_components (int, optional):
                Number of components to return.

        Returns:
            np.ndarray: Component loadings.
        """

        if self.components_ is None:
            raise ValueError("PCA must be fitted first.")

        if n_components is None:
            return self.components_

        return self.components_[:, :n_components]
