"""Principal component analysis from the sample covariance matrix.

The implementation follows the classical eigendecomposition route so that
every algebraic step in the wafer-thickness analysis remains explicit.
An SVD cross-check and bootstrap eigenvalue intervals are included as
numerical safeguards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _fix_eigenvector_signs(eigenvectors: np.ndarray) -> np.ndarray:
    """Make the largest absolute loading of each column positive."""
    V = eigenvectors.copy()
    for k in range(V.shape[1]):
        idx = int(np.argmax(np.abs(V[:, k])))
        if V[idx, k] < 0:
            V[:, k] *= -1
    return V


@dataclass(frozen=True)
class PrincipalComponentAnalysis:
    """PCA fit on an already-centered (and optionally scaled) data matrix."""

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
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[order])
        eigenvectors = _fix_eigenvector_signs(np.real(eigenvectors[:, order]))

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

    @classmethod
    def fit_svd(
        cls, X: np.ndarray, n_components: int | None = None
    ) -> "PrincipalComponentAnalysis":
        """Fit PCA from the thin SVD of ``X / sqrt(n-1)``.

        If ``X = U Σ Vᵀ`` for the scaled matrix, then the right singular
        vectors ``V`` are the principal loadings and ``λ_k = Σ_{kk}²``.
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        scaled = X / np.sqrt(n_samples - 1)
        _, singular_values, vt = np.linalg.svd(scaled, full_matrices=False)
        eigenvalues = singular_values**2
        eigenvectors = _fix_eigenvector_signs(vt.T)

        if n_components is not None:
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


def bootstrap_variance_ratios(
    X: np.ndarray,
    n_bootstrap: int = 500,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap percentiles of the explained-variance ratios.

    Returns ``(lower, median, upper)`` arrays of shape ``(n_features,)``
    for the 2.5%, 50%, and 97.5% percentiles across row-resamples of ``X``.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    ratios = np.empty((n_bootstrap, p), dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        model = PrincipalComponentAnalysis.fit(X[idx])
        ratios[b] = model.explained_variance_ratio

    lower = np.percentile(ratios, 2.5, axis=0)
    median = np.percentile(ratios, 50.0, axis=0)
    upper = np.percentile(ratios, 97.5, axis=0)
    return lower, median, upper
