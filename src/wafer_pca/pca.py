"""Principal component analysis from the sample covariance matrix.

The implementation follows the classical eigendecomposition route so that
every algebraic step in the wafer-thickness analysis remains explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrincipalComponentAnalysis:
    """PCA fit on an already-centered (and optionally scaled) data matrix.

    Parameters
    ----------
    n_components :
        Number of leading components to retain. If ``None``, all components
        are kept.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_samples: int
    n_features: int

    @classmethod
    def fit(
        cls, X: np.ndarray, n_components: int | None = None
    ) -> "PrincipalComponentAnalysis":
        """Fit PCA via the unbiased sample covariance matrix.

        For an :math:`n \\times p` matrix ``X`` with zero column means,

        .. math::

            C = \\frac{1}{n-1} X^{\\mathsf{T}} X,

        and we solve :math:`C v_k = \\lambda_k v_k` with :math:`\\lambda_1
        \\ge \\cdots \\ge \\lambda_p`.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("Need at least two samples to estimate covariance")

        covariance = (X.T @ X) / (n_samples - 1)
        # eigh returns ascending eigenvalues; reverse for PCA convention.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[order])
        eigenvectors = np.real(eigenvectors[:, order])

        # Fix sign ambiguity: largest absolute loading is positive.
        for k in range(eigenvectors.shape[1]):
            idx = int(np.argmax(np.abs(eigenvectors[:, k])))
            if eigenvectors[idx, k] < 0:
                eigenvectors[:, k] *= -1

        if n_components is not None:
            if not 1 <= n_components <= n_features:
                raise ValueError(
                    f"n_components must be in [1, {n_features}], got {n_components}"
                )
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = eigenvectors[:, :n_components]

        return cls(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            n_samples=n_samples,
            n_features=n_features,
        )

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance captured by each retained component."""
        total = self.eigenvalues.sum()
        if total <= 0:
            return np.zeros_like(self.eigenvalues)
        return self.eigenvalues / total

    @property
    def cumulative_variance_ratio(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project observations onto the principal axes: ``T = X V``."""
        return np.asarray(X, dtype=float) @ self.eigenvectors

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """Map scores back to the original feature space: ``X ≈ T Vᵀ``."""
        return np.asarray(scores, dtype=float) @ self.eigenvectors.T

    def loadings(self) -> np.ndarray:
        """Sensor loadings (columns of ``V``)."""
        return self.eigenvectors.copy()
