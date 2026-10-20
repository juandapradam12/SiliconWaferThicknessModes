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
from wafer_pca.exploration import (
    correlation_matrix,
    most_correlated_sensors,
    reconstruction_error,
)
from wafer_pca.pca import PrincipalComponentAnalysis, bootstrap_variance_ratios
from wafer_pca.plotting import (
    plot_correlation_heatmap,
    plot_loading_bars,
    plot_outlier_comparison,
    plot_reconstruction_curve,
    plot_score_scatter,
    plot_score_scatter_outliers,
    plot_scree,
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
        help="Number of bootstrap resamples for variance-ratio CIs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for bootstrap resampling",
    )
    return parser


def run(data: Path | None, figures: Path, bootstrap: int, seed: int) -> dict:
    figures.mkdir(parents=True, exist_ok=True)

    frame = load_thickness(data)
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

    plot_thickness_profiles(frame, figures / "01_thickness_profiles.png")
    plot_correlation_heatmap(corr, figures / "02_correlation_heatmap.png")
    plot_scree(
        classical,
        figures / "03_scree_plot.png",
        classical.eigenvalues,
        bootstrap_bands=bands,
    )
    plot_score_scatter(scores, figures / "04_pc_scores.png")
    plot_sensor_loadings(loadings, figures / "05_sensor_loadings.png")
    plot_loading_bars(loadings, figures / "06_loading_bars.png")
    plot_wafer_mode_map(classical.loadings(), 0, figures / "07_wafer_pc1_map.png")
    plot_wafer_mode_map(classical.loadings(), 1, figures / "08_wafer_pc2_map.png")
    plot_score_scatter_outliers(
        scores, keep, figures / "09_scores_with_outliers.png"
    )
    plot_outlier_comparison(
        classical.explained_variance_ratio,
        robust_model.explained_variance_ratio,
        filtered_model.explained_variance_ratio,
        figures / "10_outlier_spectrum_comparison.png",
    )
    plot_reconstruction_curve(
        X, classical, figures / "11_reconstruction_rmse.png"
    )

    summary = {
        "n_sections": int(frame.shape[0]),
        "n_gauges": int(frame.shape[1]),
        "n_outliers_mahalanobis": int((~keep).sum()),
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
        "svd_eigenvalues": svd_model.eigenvalues.tolist(),
        "svd_matches_covariance": bool(
            np.allclose(classical.eigenvalues, svd_model.eigenvalues, rtol=1e-10)
        ),
        "robust_explained_variance_ratio": robust_model.explained_variance_ratio.tolist(),
        "filtered_explained_variance_ratio": filtered_model.explained_variance_ratio.tolist(),
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
    print(f"  Mahalanobis outliers   : {summary['n_outliers_mahalanobis']}")
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
    print(f"  figures written to     : {figures.resolve()}")
    print(f"  summary JSON           : {out.resolve()}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.data, args.figures, args.bootstrap, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
