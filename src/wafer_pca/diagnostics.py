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


def mean_abs_residual_by_gauge(
    X: np.ndarray, model: PrincipalComponentAnalysis, n_components: int = 2
) -> np.ndarray:
    """Mean absolute residual per gauge after a truncated PCA reconstruction."""
    X_hat = reconstruct_with_r(X, model, n_components)
    return np.mean(np.abs(X - X_hat), axis=0)


def cross_validated_rmse(
    X: np.ndarray,
    max_components: int | None = None,
    n_folds: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """K-fold out-of-sample RMSE versus number of retained components.

    For each fold, PCA is fit on the training sections and the held-out
    sections are reconstructed in the training loading basis.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    max_components = p if max_components is None else max_components
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)
    errors = np.zeros(max_components, dtype=float)

    for r in range(1, max_components + 1):
        sq = []
        for fold in folds:
            test = np.zeros(n, dtype=bool)
            test[fold] = True
            train_X = X[~test]
            test_X = X[test]
            model = PrincipalComponentAnalysis.fit(train_X, n_components=r)
            V = model.eigenvectors
            recon = (test_X @ V) @ V.T
            sq.append(np.mean((test_X - recon) ** 2))
        errors[r - 1] = float(np.sqrt(np.mean(sq)))
    return errors


def physical_plane_design_matrix(gauge_names: list[str]) -> np.ndarray:
    """Design matrix ``[1, x, y]`` on the schematic gauge layout."""
    from wafer_pca.data import SCHEMATIC_GAUGE_XY

    rows = []
    for name in gauge_names:
        x, y = SCHEMATIC_GAUGE_XY[name]
        rows.append([1.0, x, y])
    return np.asarray(rows, dtype=float)


def fit_physical_plane(
    X: np.ndarray, gauge_names: list[str]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-section global offset + linear tilt on the schematic wafer.

    For each section ``i`` solve
    ``x_i ≈ a_i · 1 + b_i · x + c_i · y`` in the least-squares sense.
    Returns ``(coefficients, reconstruction, rmse)``.
    """
    X = np.asarray(X, dtype=float)
    A = physical_plane_design_matrix(gauge_names)  # (p, 3)
    # β = (AᵀA)⁻¹ Aᵀ xᵀ  →  coefficients shape (n, 3)
    coeffs, _, _, _ = np.linalg.lstsq(A, X.T, rcond=None)
    coeffs = coeffs.T
    X_hat = coeffs @ A.T
    rmse = float(np.sqrt(np.mean((X - X_hat) ** 2)))
    return coeffs, X_hat, rmse


def permutation_null_variance_ratios(
    X: np.ndarray,
    n_permutations: int = 500,
    random_state: int = 0,
) -> np.ndarray:
    """Null distribution of explained-variance ratios under column shuffles.

    Independently permuting each gauge column destroys cross-gauge
    correlation while preserving marginal variances. Useful for testing
    whether ``ρ_1`` exceeds chance; ``ρ_2`` is *deflated* when ``ρ_1`` is
    large, so a spatial test is preferred for PC2.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    ratios = np.empty((n_permutations, p), dtype=float)
    for b in range(n_permutations):
        shuffled = np.column_stack(
            [rng.permutation(X[:, j]) for j in range(p)]
        )
        ratios[b] = PrincipalComponentAnalysis.fit(shuffled).explained_variance_ratio
    return ratios


def permutation_pvalue(
    observed_ratio: float, null_ratios: np.ndarray, component: int = 0
) -> float:
    """One-sided p-value: fraction of null ρ_k ≥ observed ρ_k."""
    null_k = null_ratios[:, component]
    return float(np.mean(null_k >= observed_ratio))


def loading_spatial_r2(loadings_1d: np.ndarray, gauge_names: list[str]) -> float:
    """R² of regressing a loading vector on schematic gauge coordinates (x, y)."""
    A = physical_plane_design_matrix(gauge_names)[:, 1:]  # drop intercept
    y = np.asarray(loadings_1d, dtype=float)
    y = y - y.mean()
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ beta
    ss_tot = float(np.sum(y**2))
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - np.sum((y - y_hat) ** 2) / ss_tot)


def permutation_spatial_r2_null(
    loadings_1d: np.ndarray,
    gauge_names: list[str],
    n_permutations: int = 500,
    random_state: int = 0,
) -> tuple[float, np.ndarray, float]:
    """Permute gauge labels to test spatial organization of a loading vector.

    Returns ``(observed_R2, null_R2_samples, p_value)``.
    """
    observed = loading_spatial_r2(loadings_1d, gauge_names)
    rng = np.random.default_rng(random_state)
    names = list(gauge_names)
    null = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        shuffled = list(rng.permutation(names))
        null[b] = loading_spatial_r2(loadings_1d, shuffled)
    p_value = float(np.mean(null >= observed))
    return observed, null, p_value
