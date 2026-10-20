"""CLI entry point: ``python -m wafer_pca``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from wafer_pca.data import (
    load_thickness,
    mahalanobis_mask,
    robust_standardize,
    standardize,
    summary_statistics,
)
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
from wafer_pca.exploration import (
    correlation_matrix,
    most_correlated_sensors,
    reconstruction_error,
)
from wafer_pca.pca import PrincipalComponentAnalysis, bootstrap_variance_ratios
from wafer_pca.plotting import (
    plot_correlation_heatmap,
    plot_cov_vs_corr_spectra,
    plot_cv_rmse,
    plot_loading_bars,
    plot_loading_intervals,
    plot_outlier_comparison,
    plot_pc1_spc,
    plot_permutation_null,
    plot_physical_vs_pca_rmse,
    plot_spatial_r2_null,
    plot_pruning_comparison,
    plot_reconstruction_curve,
    plot_reconstruction_panels,
    plot_residual_wafer_map,
    plot_score_scatter,
    plot_score_scatter_outliers,
    plot_scree_with_mp,
    plot_sensor_loadings,
    plot_thickness_profiles,
    plot_wafer_mode_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover dominant thickness modes of a silicon wafer from "
            "nine-gauge metrology via sample-covariance PCA."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the thickness CSV (default: package data file)",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("figures"),
        help="Directory for output figures (default: ./figures)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=400,
        help="Number of bootstrap resamples for variance-ratio / loading CIs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for bootstrap resampling",
    )
    return parser


def _representative_sections(scores: np.ndarray) -> list[int]:
    """Median-|PC1|, max-PC1, and max-|PC2| section indices."""
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]
    med = int(np.argsort(np.abs(pc1))[len(pc1) // 2])
    hi1 = int(np.argmax(pc1))
    hi2 = int(np.argmax(np.abs(pc2)))
    # Ensure uniqueness while preserving intent.
    picks = [med]
    for idx in (hi1, hi2):
        if idx not in picks:
            picks.append(idx)
    while len(picks) < 3:
        cand = int(np.argsort(np.abs(pc1))[-(len(picks) + 1)])
        if cand not in picks:
            picks.append(cand)
    return picks[:3]


def run(data: Path | None, figures: Path, bootstrap: int, seed: int) -> dict:
    figures.mkdir(parents=True, exist_ok=True)

    frame = load_thickness(data)
    gauge_names = list(frame.columns)
    X_raw = frame.to_numpy()
    X, mu, sigma = standardize(X_raw)
    X_robust, _, _ = robust_standardize(X_raw)
    keep = mahalanobis_mask(X)
    X_filtered, _, _ = standardize(X_raw[keep])

    classical = PrincipalComponentAnalysis.fit(X)
    svd_model = PrincipalComponentAnalysis.fit_svd(X)
    robust_model = PrincipalComponentAnalysis.fit(X_robust)
    filtered_model = PrincipalComponentAnalysis.fit(X_filtered)
    model_2 = PrincipalComponentAnalysis.fit(X, n_components=2)

    scores = model_2.transform(X)
    loadings = model_2.loadings()
    corr = correlation_matrix(frame)
    g_a, g_b, dist = most_correlated_sensors(classical.loadings(), k=2)
    rmse_2 = reconstruction_error(X, classical, 2)
    bands = bootstrap_variance_ratios(X, n_bootstrap=bootstrap, random_state=seed)
    load_lo, load_med, load_hi = bootstrap_loadings(
        X, n_components=2, n_bootstrap=bootstrap, random_state=seed
    )

    mp_lo, mp_hi = marchenko_pastur_edges(classical.n_samples, classical.n_features)
    n_signal = n_signal_components(classical.eigenvalues, mp_hi)
    pruning = sensor_pruning_study(X_raw, gauge_names)
    ratios_cov, ratios_corr = covariance_vs_correlation_spectra(X_raw)
    X_hat = reconstruct_with_r(X, classical, 2)
    section_ids = _representative_sections(scores)
    limits = spc_limits(scores[:, 0], k=3.0)
    n_outside = int(np.sum((scores[:, 0] < limits[0]) | (scores[:, 0] > limits[2])))

    resid_by_gauge = mean_abs_residual_by_gauge(X, classical, 2)
    in_sample_rmse = np.array(
        [reconstruction_error(X, classical, r) for r in range(1, 10)]
    )
    cv_rmse = cross_validated_rmse(X, max_components=9, n_folds=5, random_state=seed)
    _, _, rmse_physical = fit_physical_plane(X, gauge_names)
    rmse_pca = {
        1: reconstruction_error(X, classical, 1),
        2: rmse_2,
        3: reconstruction_error(X, classical, 3),
    }
    null_ratios = permutation_null_variance_ratios(
        X, n_permutations=max(bootstrap, 200), random_state=seed
    )
    p_pc1 = permutation_pvalue(
        float(classical.explained_variance_ratio[0]), null_ratios, component=0
    )
    pc2_spatial_r2, pc2_spatial_null, p_pc2_spatial = permutation_spatial_r2_null(
        classical.loadings()[:, 1],
        gauge_names,
        n_permutations=max(bootstrap, 200),
        random_state=seed,
    )

    plot_thickness_profiles(frame, figures / "01_thickness_profiles.png")
    plot_correlation_heatmap(corr, figures / "02_correlation_heatmap.png")
    plot_cov_vs_corr_spectra(
        ratios_cov, ratios_corr, figures / "03_cov_vs_corr_spectra.png"
    )
    plot_scree_with_mp(
        classical.explained_variance_ratio,
        classical.eigenvalues,
        mp_hi,
        figures / "04_scree_marchenko_pastur.png",
        bootstrap_bands=bands,
    )
    plot_score_scatter(scores, figures / "05_pc_scores.png")
    plot_pc1_spc(scores[:, 0], limits, figures / "06_pc1_spc_chart.png")
    plot_sensor_loadings(loadings, figures / "07_sensor_loadings.png")
    plot_loading_bars(loadings, figures / "08_loading_bars.png")
    plot_loading_intervals(
        loadings, load_lo, load_hi, 0, figures / "09_pc1_loading_intervals.png"
    )
    plot_loading_intervals(
        loadings, load_lo, load_hi, 1, figures / "10_pc2_loading_intervals.png"
    )
    plot_wafer_mode_map(classical.loadings(), 0, figures / "11_wafer_pc1_map.png")
    plot_wafer_mode_map(classical.loadings(), 1, figures / "12_wafer_pc2_map.png")
    plot_pruning_comparison(pruning, figures / "13_sensor_pruning.png")
    plot_score_scatter_outliers(
        scores, keep, figures / "14_scores_with_outliers.png"
    )
    plot_outlier_comparison(
        classical.explained_variance_ratio,
        robust_model.explained_variance_ratio,
        filtered_model.explained_variance_ratio,
        figures / "15_outlier_spectrum_comparison.png",
    )
    plot_reconstruction_panels(
        X, X_hat, section_ids, figures / "16_reconstruction_panels.png"
    )
    plot_reconstruction_curve(
        X, classical, figures / "17_reconstruction_rmse.png"
    )
    plot_residual_wafer_map(resid_by_gauge, figures / "18_residual_wafer_map.png")
    plot_cv_rmse(in_sample_rmse, cv_rmse, figures / "19_cv_rmse.png")
    plot_physical_vs_pca_rmse(
        rmse_physical, rmse_pca, figures / "20_physical_vs_pca.png"
    )
    plot_permutation_null(
        float(classical.explained_variance_ratio[0]),
        null_ratios,
        0,
        figures / "21_pc1_permutation_null.png",
    )
    plot_spatial_r2_null(
        pc2_spatial_r2, pc2_spatial_null, figures / "22_pc2_spatial_permutation.png"
    )

    summary = {
        "n_sections": int(frame.shape[0]),
        "n_gauges": int(frame.shape[1]),
        "n_outliers_mahalanobis": int((~keep).sum()),
        "n_spc_alarms_pc1": n_outside,
        "spc_limits_pc1": {"lcl": limits[0], "cl": limits[1], "ucl": limits[2]},
        "marchenko_pastur": {
            "lower": mp_lo,
            "upper": mp_hi,
            "n_signal_components": n_signal,
        },
        "sensor_means": mu.tolist(),
        "sensor_stds": sigma.tolist(),
        "eigenvalues": classical.eigenvalues.tolist(),
        "explained_variance_ratio": classical.explained_variance_ratio.tolist(),
        "cumulative_variance_ratio": classical.cumulative_variance_ratio.tolist(),
        "bootstrap_variance_ratio_ci": {
            "lower": bands[0].tolist(),
            "median": bands[1].tolist(),
            "upper": bands[2].tolist(),
        },
        "bootstrap_loading_ci": {
            "lower": load_lo.tolist(),
            "median": load_med.tolist(),
            "upper": load_hi.tolist(),
        },
        "svd_eigenvalues": svd_model.eigenvalues.tolist(),
        "svd_matches_covariance": bool(
            np.allclose(classical.eigenvalues, svd_model.eigenvalues, rtol=1e-10)
        ),
        "robust_explained_variance_ratio": robust_model.explained_variance_ratio.tolist(),
        "filtered_explained_variance_ratio": filtered_model.explained_variance_ratio.tolist(),
        "covariance_pca_variance_ratio": ratios_cov.tolist(),
        "correlation_pca_variance_ratio": ratios_corr.tolist(),
        "pruning": [
            {
                "dropped": list(r.dropped),
                "n_gauges": r.n_gauges,
                "explained_pc1": r.explained_pc1,
                "explained_pc1_pc2": r.explained_pc1_pc2,
                "rmse_2pc": r.rmse_2pc,
            }
            for r in pruning
        ],
        "reconstruction_panel_sections": section_ids,
        "mean_abs_residual_by_gauge": resid_by_gauge.tolist(),
        "cv_rmse": cv_rmse.tolist(),
        "in_sample_rmse": in_sample_rmse.tolist(),
        "physical_plane_rmse": rmse_physical,
        "pc1_permutation_pvalue": p_pc1,
        "pc2_spatial_r2": pc2_spatial_r2,
        "pc2_spatial_permutation_pvalue": p_pc2_spatial,
        "pc1_pc2_variance": float(classical.cumulative_variance_ratio[1]),
        "closest_sensors_in_loading_plane": {
            "pair": [g_a, g_b],
            "distance": dist,
        },
        "reconstruction_rmse_2pc": rmse_2,
        "summary_statistics": summary_statistics(frame).to_dict(),
    }

    out = figures / "analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print("Silicon wafer thickness — PCA summary")
    print(f"  sections × gauges      : {frame.shape[0]} × {frame.shape[1]}")
    print(f"  MP signal components   : {n_signal} (λ₊ = {mp_hi:.3f})")
    print(f"  Mahalanobis outliers   : {summary['n_outliers_mahalanobis']}")
    print(f"  PC1 SPC alarms (±3σ)   : {n_outside}")
    print(
        "  variance in PC1+PC2     : "
        f"{100 * summary['pc1_pc2_variance']:.1f}%"
    )
    print(
        "  PC1 bootstrap 95% CI    : "
        f"[{100 * bands[0][0]:.1f}%, {100 * bands[2][0]:.1f}%]"
    )
    print(
        "  closest gauges (PC plane): "
        f"{g_a}–{g_b} (distance {dist:.3f})"
    )
    print(f"  SVD ↔ covariance match : {summary['svd_matches_covariance']}")
    print(f"  RMSE with 2 PCs        : {rmse_2:.4f}")
    print(f"  CV RMSE (2 PCs)        : {cv_rmse[1]:.4f}")
    print(f"  Physical plane RMSE    : {rmse_physical:.4f}")
    print(f"  PC1 permutation p-value: {p_pc1:.4f}")
    print(
        f"  PC2 spatial R² / p     : {pc2_spatial_r2:.3f} / {p_pc2_spatial:.4f}"
    )
    print(f"  figures written to     : {figures.resolve()}")
    print(f"  summary JSON           : {out.resolve()}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.data, args.figures, args.bootstrap, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
