"""Secondary diagnostics: pruning, MP edge, loading CIs, scaling contrast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafer_pca.data import standardize
from wafer_pca.exploration import align_loading_sign, reconstruction_error
from wafer_pca.pca import PrincipalComponentAnalysis


def marchenko_pastur_edges(n_samples: int, n_features: int) -> tuple[float, float]:
    """Bulk edges of the Marchenko–Pastur law for unit-variance noise.

    For aspect ratio ``γ = p / n`` and population covariance ``I``, sample
    eigenvalues of the noise concentrate in
    ``[ (1 - √γ)² , (1 + √γ)² ]``. Components above the upper edge are
    candidates for signal.
    """
    if n_samples <= 0 or n_features <= 0:
        raise ValueError("n_samples and n_features must be positive")
    gamma = n_features / n_samples
    root = np.sqrt(gamma)
    return float((1.0 - root) ** 2), float((1.0 + root) ** 2)


def n_signal_components(eigenvalues: np.ndarray, upper_edge: float) -> int:
    """Count eigenvalues strictly above the MP upper edge."""
    return int(np.sum(np.asarray(eigenvalues, dtype=float) > upper_edge))


@dataclass(frozen=True)
class PruningResult:
    dropped: tuple[str, ...]
    n_gauges: int
    explained_pc1: float
    explained_pc1_pc2: float
    rmse_2pc: float


def sensor_pruning_study(X_raw: np.ndarray, gauge_names: list[str]) -> list[PruningResult]:
    """Re-fit PCA after dropping redundant / contrast-driving gauges.

    Scenarios:
    - all gauges
    - drop G5 (nearest neighbour of G1 in the loading plane)
    - drop G6 (dominant PC2 loading)
    - drop G5 and G6
    """
    name_to_idx = {name: i for i, name in enumerate(gauge_names)}
    scenarios: list[tuple[str, ...]] = [
        (),
        ("G5",),
        ("G6",),
        ("G5", "G6"),
    ]
    results: list[PruningResult] = []
    for dropped in scenarios:
        keep_idx = [i for name, i in name_to_idx.items() if name not in dropped]
        X, _, _ = standardize(X_raw[:, keep_idx])
        model = PrincipalComponentAnalysis.fit(X)
        r = min(2, model.n_features)
        results.append(
            PruningResult(
                dropped=dropped,
                n_gauges=len(keep_idx),
                explained_pc1=float(model.explained_variance_ratio[0]),
                explained_pc1_pc2=float(model.cumulative_variance_ratio[r - 1]),
                rmse_2pc=reconstruction_error(X, model, r),
            )
        )
    return results


def bootstrap_loadings(
    X: np.ndarray,
    n_components: int = 2,
    n_bootstrap: int = 400,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap percentile CIs for the leading loading matrix.

    Returns ``(lower, median, upper)`` with shape ``(n_features, n_components)``.
    Each resample is sign-aligned to the full-sample loadings.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    reference = PrincipalComponentAnalysis.fit(X, n_components=n_components).loadings()
    rng = np.random.default_rng(random_state)
    samples = np.empty((n_bootstrap, p, n_components), dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot = PrincipalComponentAnalysis.fit(X[idx], n_components=n_components)
        samples[b] = align_loading_sign(reference, boot.loadings())

    lower = np.percentile(samples, 2.5, axis=0)
    median = np.percentile(samples, 50.0, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)
    return lower, median, upper


def covariance_vs_correlation_spectra(
    X_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Explained-variance ratios for centered-only vs z-scored PCA.

    Centered-only PCA diagonalizes the raw covariance (gauge scales matter).
    Z-scored PCA diagonalizes the correlation structure.
    """
    X_raw = np.asarray(X_raw, dtype=float)
    centered = X_raw - X_raw.mean(axis=0)
    X_corr, _, _ = standardize(X_raw)
    cov_model = PrincipalComponentAnalysis.fit(centered)
    corr_model = PrincipalComponentAnalysis.fit(X_corr)
    return cov_model.explained_variance_ratio, corr_model.explained_variance_ratio


def reconstruct_with_r(
    X: np.ndarray, model: PrincipalComponentAnalysis, n_components: int
) -> np.ndarray:
    """Return the rank-``n_components`` reconstruction of ``X``."""
    V = model.eigenvectors[:, :n_components]
    return (X @ V) @ V.T


def spc_limits(scores_1d: np.ndarray, k: float = 3.0) -> tuple[float, float, float]:
    """Shewhart center line and ±kσ control limits for a score series."""
    center = float(np.mean(scores_1d))
    sigma = float(np.std(scores_1d, ddof=1))
    return center - k * sigma, center, center + k * sigma
