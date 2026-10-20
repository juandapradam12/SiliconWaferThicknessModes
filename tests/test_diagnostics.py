"""Tests for MP edge, pruning, SPC limits, and loading bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafer_pca.data import load_thickness, standardize
from wafer_pca.diagnostics import (
    bootstrap_loadings,
    covariance_vs_correlation_spectra,
    cross_validated_rmse,
    fit_physical_plane,
    marchenko_pastur_edges,
    mean_abs_residual_by_gauge,
    n_signal_components,
    permutation_null_variance_ratios,
    permutation_pvalue,
    permutation_spatial_r2_null,
    reconstruct_with_r,
    sensor_pruning_study,
    spc_limits,
)
from wafer_pca.pca import PrincipalComponentAnalysis


def test_marchenko_pastur_edges_ordered():
    lo, hi = marchenko_pastur_edges(184, 9)
    assert 0 <= lo < hi
    # γ = 9/184 ≈ 0.049 → λ₊ ≈ (1+√γ)² ≈ 1.49
    assert 1.3 < hi < 1.7


def test_signal_components_at_least_pc1():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    _, hi = marchenko_pastur_edges(model.n_samples, model.n_features)
    assert n_signal_components(model.eigenvalues, hi) >= 1
    assert model.eigenvalues[0] > hi


def test_pruning_drop_g5_keeps_pc1():
    frame = load_thickness()
    results = sensor_pruning_study(frame.to_numpy(), list(frame.columns))
    full = results[0]
    drop_g5 = next(r for r in results if r.dropped == ("G5",))
    assert abs(full.explained_pc1 - drop_g5.explained_pc1) < 0.02


def test_spc_limits_symmetric():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    lo, mid, hi = spc_limits(x, k=3.0)
    assert lo < mid < hi
    assert np.isclose(hi - mid, mid - lo)


def test_bootstrap_loadings_shape_and_order():
    X, _, _ = standardize(load_thickness().to_numpy())
    lo, med, hi = bootstrap_loadings(X, n_components=2, n_bootstrap=30, random_state=0)
    assert lo.shape == (9, 2)
    assert np.all(lo <= med + 1e-12)
    assert np.all(med <= hi + 1e-12)


def test_cov_vs_corr_differ():
    ratios_cov, ratios_corr = covariance_vs_correlation_spectra(
        load_thickness().to_numpy()
    )
    assert ratios_cov.shape == ratios_corr.shape == (9,)
    assert not np.allclose(ratios_cov, ratios_corr)


def test_reconstruction_rank():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    X_hat = reconstruct_with_r(X, model, 2)
    assert X_hat.shape == X.shape
    # Rank of reconstruction is at most 2.
    assert np.linalg.matrix_rank(X_hat, tol=1e-8) <= 2


def test_mean_abs_residual_nonnegative():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    resid = mean_abs_residual_by_gauge(X, model, 2)
    assert resid.shape == (9,)
    assert np.all(resid >= 0)


def test_cv_rmse_decreases_initially():
    X, _, _ = standardize(load_thickness().to_numpy())
    cv = cross_validated_rmse(X, max_components=4, n_folds=4, random_state=0)
    assert cv.shape == (4,)
    assert cv[1] < cv[0]


def test_physical_plane_competitive_with_pca2():
    frame = load_thickness()
    X, _, _ = standardize(frame.to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    _, _, rmse_phys = fit_physical_plane(X, list(frame.columns))
    rmse_pca2 = float(np.sqrt(np.mean((X - reconstruct_with_r(X, model, 2)) ** 2)))
    # Plane model should be in the same ballpark as 2-PC PCA.
    assert rmse_phys < 2.0 * rmse_pca2


def test_permutation_pc1_significant():
    X, _, _ = standardize(load_thickness().to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    null = permutation_null_variance_ratios(X, n_permutations=80, random_state=0)
    p = permutation_pvalue(float(model.explained_variance_ratio[0]), null, component=0)
    assert p < 0.05


def test_pc2_spatial_r2_moderate():
    frame = load_thickness()
    X, _, _ = standardize(frame.to_numpy())
    model = PrincipalComponentAnalysis.fit(X)
    r2, null, p = permutation_spatial_r2_null(
        model.loadings()[:, 1], list(frame.columns), n_permutations=120, random_state=0
    )
    assert r2 > 0.2
    assert null.shape == (120,)
    assert 0.0 <= p <= 1.0
