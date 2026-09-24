import numpy as np


def center_matrix(X):
    """
    Center a matrix by subtracting the mean of each column.

    Parameters:
        X (np.ndarray): Input feature matrix.

    Returns:
        np.ndarray: Centered matrix.
    """
    mean = np.mean(X, axis=0)
    return X - mean


def covariance_matrix(X):
    """
    Calculate the covariance matrix of a centered dataset.

    Parameters:
        X (np.ndarray): Centered feature matrix.

    Returns:
        np.ndarray: Covariance matrix.
    """
    n_samples = X.shape[0]

    return (X.T @ X) / (n_samples - 1)


def sort_eigenpairs(eigenvalues, eigenvectors):
    """
    Sort eigenvalues and corresponding eigenvectors
    from largest to smallest eigenvalue.

    Parameters:
        eigenvalues (np.ndarray): Eigenvalues.
        eigenvectors (np.ndarray): Eigenvectors.

    Returns:
        tuple: Sorted eigenvalues and eigenvectors.
    """
    indices = np.argsort(eigenvalues)[::-1]

    sorted_values = eigenvalues[indices]
    sorted_vectors = eigenvectors[:, indices]

    return sorted_values, sorted_vectors


def explained_variance_ratio(eigenvalues):
    """
    Calculate the proportion of total variance
    explained by each eigenvalue.

    Parameters:
        eigenvalues (np.ndarray): Eigenvalues.

    Returns:
        np.ndarray: Explained variance ratio.
    """
    total_variance = np.sum(eigenvalues)

    return eigenvalues / total_variance
