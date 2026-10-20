"""Unit tests for standardization, PCA invariants, SVD, and outliers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafer_pca.data import (
    load_thickness,
    mahalanobis_mask,
    robust_standardize,
    standardize,
)
from wafer_pca.pca import PrincipalComponentAnalysis, bootstrap_variance_ratios


def test_load_thickness_shape():
    frame = load_thickness()
    assert frame.shape == (184, 9)
    assert list(frame.columns) == [f"G{i}" for i in range(1, 10)]


def test_standardize_zero_mean_unit_variance():
    frame = load_thickness()
    X, mu, sigma = standardize(frame.to_numpy())
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X.std(axis=0, ddof=1), 1.0, atol=1e-10)
    assert mu.shape == (9,)
    assert sigma.shape == (9,)


def test_robust_standardize_finite():
    X, center, scale = robust_standardize(load_thickness().to_numpy())
    assert np.isfinite(X).all()
    assert np.all(scale > 0)
    assert center.shape == (9,)


def test_pca_orthonormal_eigenvectors():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    V = model.eigenvectors
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)


def test_pca_eigenvalues_descending_and_nonnegative():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    assert np.all(model.eigenvalues >= -1e-12)
    assert np.all(np.diff(model.eigenvalues) <= 1e-12)


def test_explained_variance_sums_to_one():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    assert np.isclose(model.explained_variance_ratio.sum(), 1.0)


def test_transform_inverse_full_rank_recovers_data():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    scores = model.transform(X)
    X_hat = model.inverse_transform(scores)
    assert np.allclose(X, X_hat, atol=1e-10)


def test_n_components_truncation():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X, n_components=2)
    assert model.eigenvalues.shape == (2,)
    assert model.eigenvectors.shape == (9, 2)
    assert model.transform(X).shape == (184, 2)


def test_matches_numpy_covariance_spectrum():
    X, _, _ = standardize(load_thickness().to_numpy())
    cov = np.cov(X, rowvar=False, ddof=1)
    evals_ref = np.sort(np.linalg.eigvalsh(cov))[::-1]
    model = PrincipalComponentAnalysis.fit(X)
    assert np.allclose(model.eigenvalues, evals_ref, rtol=1e-10)


def test_svd_matches_covariance_eigenvalues():
    X, _, _ = standardize(load_thickness().to_numpy())
    cov_model = PrincipalComponentAnalysis.fit(X)
    svd_model = PrincipalComponentAnalysis.fit_svd(X)
    assert np.allclose(cov_model.eigenvalues, svd_model.eigenvalues, rtol=1e-10)
    # Loadings agree up to a global sign per column (already fixed in fit).
    for k in range(9):
        assert np.allclose(
            cov_model.eigenvectors[:, k], svd_model.eigenvectors[:, k], atol=1e-8
        ) or np.allclose(
            cov_model.eigenvectors[:, k], -svd_model.eigenvectors[:, k], atol=1e-8
        )


def test_bootstrap_bands_ordered():
    X, _, _ = standardize(load_thickness().to_numpy())
    lower, median, upper = bootstrap_variance_ratios(
        X, n_bootstrap=40, random_state=1
    )
    assert lower.shape == (9,)
    assert np.all(lower <= median + 1e-12)
    assert np.all(median <= upper + 1e-12)
    assert np.isclose(median.sum(), 1.0, atol=0.05)


def test_mahalanobis_mask_flags_some_rows():
    X, _, _ = standardize(load_thickness().to_numpy())
    keep = mahalanobis_mask(X)
    assert keep.dtype == bool
    assert keep.shape == (184,)
    assert keep.sum() < 184
    assert keep.sum() > 100
