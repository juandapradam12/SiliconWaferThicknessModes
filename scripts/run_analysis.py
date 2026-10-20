#!/usr/bin/env python3
"""End-to-end silicon wafer thickness PCA pipeline.

Produces figures under ``figures/`` and prints a concise numerical summary
aligned with the narrative in ``README.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafer_pca.data import load_thickness, standardize, summary_statistics
from wafer_pca.exploration import (
    correlation_matrix,
    most_correlated_sensors,
    reconstruction_error,
)
from wafer_pca.pca import PrincipalComponentAnalysis
from wafer_pca.plotting import (
    plot_correlation_heatmap,
    plot_loading_bars,
    plot_reconstruction_curve,
    plot_score_scatter,
    plot_sensor_loadings,
    plot_scree,
    plot_thickness_profiles,
)


def main() -> None:
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    frame = load_thickness()
    X_raw = frame.to_numpy()
    X, mu, sigma = standardize(X_raw)

    # Full eigen-spectrum for scree / cumulative variance plots.
    full_model = PrincipalComponentAnalysis.fit(X)
    model = PrincipalComponentAnalysis.fit(X, n_components=2)
    scores = model.transform(X)
    loadings = model.loadings()

    corr = correlation_matrix(frame)
    g_a, g_b, dist = most_correlated_sensors(full_model.loadings(), k=2)
    rmse_2 = reconstruction_error(X, full_model, 2)

    plot_thickness_profiles(frame, figures / "01_thickness_profiles.png")
    plot_correlation_heatmap(corr, figures / "02_correlation_heatmap.png")
    plot_scree(full_model, figures / "03_scree_plot.png", full_model.eigenvalues)
    plot_score_scatter(scores, figures / "04_pc_scores.png")
    plot_sensor_loadings(loadings, figures / "05_sensor_loadings.png")
    plot_loading_bars(loadings, figures / "06_loading_bars.png")
    plot_reconstruction_curve(X, full_model, figures / "07_reconstruction_rmse.png")

    summary = {
        "n_sections": int(frame.shape[0]),
        "n_gauges": int(frame.shape[1]),
        "sensor_means": mu.tolist(),
        "sensor_stds": sigma.tolist(),
        "eigenvalues": full_model.eigenvalues.tolist(),
        "explained_variance_ratio": full_model.explained_variance_ratio.tolist(),
        "cumulative_variance_ratio": full_model.cumulative_variance_ratio.tolist(),
        "pc1_pc2_variance": float(full_model.cumulative_variance_ratio[1]),
        "closest_sensors_in_loading_plane": {
            "pair": [g_a, g_b],
            "distance": dist,
        },
        "reconstruction_rmse_2pc": rmse_2,
        "summary_statistics": summary_statistics(frame).to_dict(),
    }

    out = ROOT / "figures" / "analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print("Silicon wafer thickness — PCA summary")
    print(f"  sections × gauges : {frame.shape[0]} × {frame.shape[1]}")
    print(
        "  variance in PC1+PC2: "
        f"{100 * summary['pc1_pc2_variance']:.1f}%"
    )
    print(
        "  closest gauges (PC plane): "
        f"{g_a}–{g_b} (distance {dist:.3f})"
    )
    print(f"  RMSE with 2 PCs   : {rmse_2:.4f}")
    print(f"  figures written to: {figures}")
    print(f"  summary JSON      : {out}")


if __name__ == "__main__":
    main()
