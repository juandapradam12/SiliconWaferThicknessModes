"""Exploratory metrics for thickness correlations and PCA diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wafer_pca.pca import PrincipalComponentAnalysis


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation among gauge groups."""
    return frame.corr(method="pearson")


def pairwise_sensor_distance(loadings: np.ndarray, k: int = 2) -> pd.DataFrame:
    """Euclidean distance between sensors in the first ``k`` loading axes."""
    coords = loadings[:, :k]
    n = coords.shape[0]
    labels = [f"G{i}" for i in range(1, n + 1)]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
    return pd.DataFrame(dist, index=labels, columns=labels)


def most_correlated_sensors(
    loadings: np.ndarray, k: int = 2
) -> tuple[str, str, float]:
    """Return the closest distinct sensor pair in loading space."""
    dist = pairwise_sensor_distance(loadings, k=k)
    values = dist.values.copy()
    np.fill_diagonal(values, np.inf)
    i, j = np.unravel_index(np.argmin(values), values.shape)
    return dist.index[i], dist.columns[j], float(values[i, j])


def reconstruction_error(
    X: np.ndarray, model: PrincipalComponentAnalysis, n_components: int
) -> float:
    """RMSE of reconstructing ``X`` from the first ``n_components`` PCs."""
    V = model.eigenvectors[:, :n_components]
    scores = X @ V
    X_hat = scores @ V.T
    return float(np.sqrt(np.mean((X - X_hat) ** 2)))


def align_loading_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Flip each column of ``candidate`` to match the sign of ``reference``."""
    aligned = candidate.copy()
    for k in range(min(reference.shape[1], aligned.shape[1])):
        if np.dot(reference[:, k], aligned[:, k]) < 0:
            aligned[:, k] *= -1
    return aligned
