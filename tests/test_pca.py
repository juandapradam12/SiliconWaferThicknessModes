"""Unit tests for standardization and PCA invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafer_pca.data import load_thickness, standardize
from wafer_pca.pca import PrincipalComponentAnalysis


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


def test_pca_orthonormal_eigenvectors():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    V = model.eigenvectors
    gram = V.T @ V
    assert np.allclose(gram, np.eye(V.shape[1]), atol=1e-10)


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
