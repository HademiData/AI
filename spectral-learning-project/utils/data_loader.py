import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load the Wine Quality dataset.

    Parameters:
        file_path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path, sep=";")

    return data


def preprocess_data(data):
    """
    Clean and standardize the dataset for spectral analysis.

    The target column 'quality' is excluded because PCA and SVD
    will operate on the input feature matrix.

    Parameters:
        data (pd.DataFrame): Raw dataset.

    Returns:
        np.ndarray: Standardized feature matrix.
        list: Names of the selected features.
    """

    # Remove rows containing missing values
    data = data.dropna()

    # Select numerical features and exclude the target
    features = data.drop(columns=["quality"])

    # Convert to NumPy array
    X = features.to_numpy(dtype=float)

    # Standardize each feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_standardized = (X - mean) / std

    return X_standardized, features.columns.tolist()