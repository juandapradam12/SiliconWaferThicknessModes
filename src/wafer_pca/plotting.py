"""Publication-style figures for the wafer-thickness PCA workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from wafer_pca.pca import PrincipalComponentAnalysis

# Cool steel / silicon visual language (avoid purple/cream AI defaults).
PALETTE = {
    "ink": "#1a2332",
    "steel": "#4a6fa5",
    "cyan": "#2a9d8f",
    "amber": "#c4783a",
    "slate": "#6b7c93",
    "bg": "#f7f5f2",
    "grid": "#d8d4ce",
}

SENSOR_COLORS = [
    "#1a2332",
    "#2a4a6f",
    "#4a6fa5",
    "#2a9d8f",
    "#3d8b6e",
    "#c4783a",
    "#a65d3a",
    "#8b5a3c",
    "#6b7c93",
]

SENSOR_CMAP = LinearSegmentedColormap.from_list(
    "wafer_corr",
    ["#1a2332", "#4a6fa5", "#f7f5f2", "#c4783a", "#8b2e1a"],
)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["slate"])
    ax.spines["bottom"].set_color(PALETTE["slate"])
    ax.tick_params(colors=PALETTE["ink"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    ax.title.set_color(PALETTE["ink"])


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    return path


def plot_thickness_profiles(frame: pd.DataFrame, path: Path) -> Path:
    """Overlay of the nine gauge traces across wafer sections."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    sections = np.arange(len(frame))
    for i, col in enumerate(frame.columns):
        ax.plot(
            sections,
            frame[col],
            color=SENSOR_COLORS[i],
            lw=1.1,
            alpha=0.85,
            label=col,
        )
    ax.set_xlabel("Wafer section index")
    ax.set_ylabel("Thickness deviation")
    ax.set_title("Nine-gauge thickness profiles across wafer sections")
    ax.legend(
        ncol=9,
        loc="upper right",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
    )
    ax.axhline(0, color=PALETTE["slate"], lw=0.8, ls="--", alpha=0.6)
    _style_axes(ax)
    return _save(fig, path)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> Path:
    """Sensor–sensor Pearson correlation heatmap."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    fig.patch.set_facecolor(PALETTE["bg"])
    im = ax.imshow(corr.values, cmap=SENSOR_CMAP, vmin=-1, vmax=1, origin="upper")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.values[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(val) > 0.55 else PALETTE["ink"],
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", color=PALETTE["ink"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["ink"])
    ax.set_title("Cross-gauge thickness correlation")
    _style_axes(ax)
    return _save(fig, path)


def plot_scree(
    model: PrincipalComponentAnalysis,
    path: Path,
    full_eigenvalues: np.ndarray,
    bootstrap_bands: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Path:
    """Scree plot of explained variance for all principal components."""
    ratios = full_eigenvalues / full_eigenvalues.sum()
    cumulative = np.cumsum(ratios)
    pcs = np.arange(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.bar(pcs, ratios, color=PALETTE["steel"], width=0.65, label="Per-component")
    if bootstrap_bands is not None:
        lower, _, upper = bootstrap_bands
        ax.vlines(
            pcs,
            lower,
            upper,
            color=PALETTE["ink"],
            lw=1.4,
            label="95% bootstrap CI",
        )
    ax.plot(
        pcs,
        cumulative,
        color=PALETTE["amber"],
        marker="o",
        lw=1.8,
        label="Cumulative",
    )
    ax.axhline(0.9, color=PALETTE["cyan"], ls="--", lw=1, label="90% threshold")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Fraction of variance")
    ax.set_title("Eigenvalue spectrum of the thickness covariance")
    ax.set_xticks(pcs)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    return _save(fig, path)


def plot_score_scatter(scores: np.ndarray, path: Path) -> Path:
    """Wafer sections in the PC1–PC2 plane."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=PALETTE["steel"],
        s=28,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.axhline(0, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.axvline(0, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("Wafer sections in the leading principal plane")
    _style_axes(ax)
    return _save(fig, path)


def plot_sensor_loadings(loadings: np.ndarray, path: Path) -> Path:
    """Gauge groups as points in the PC1–PC2 loading plane."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    labels = [f"G{i}" for i in range(1, loadings.shape[0] + 1)]
    for i, label in enumerate(labels):
        ax.scatter(
            loadings[i, 0],
            loadings[i, 1],
            color=SENSOR_COLORS[i],
            s=90,
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            label,
            (loadings[i, 0], loadings[i, 1]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            color=PALETTE["ink"],
        )
    ax.axhline(0, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.axvline(0, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("PC2 loading")
    ax.set_title("Gauge groups projected onto principal loadings")
    _style_axes(ax)
    return _save(fig, path)


def plot_loading_bars(loadings: np.ndarray, path: Path) -> Path:
    """Bar chart of PC1 and PC2 loadings for each gauge."""
    labels = [f"G{i}" for i in range(1, loadings.shape[0] + 1)]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.bar(
        x - width / 2,
        loadings[:, 0],
        width,
        color=PALETTE["steel"],
        label="PC1",
    )
    ax.bar(
        x + width / 2,
        loadings[:, 1],
        width,
        color=PALETTE["amber"],
        label="PC2",
    )
    ax.axhline(0, color=PALETTE["slate"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Loading weight")
    ax.set_title("Contribution of each gauge to PC1 and PC2")
    ax.legend(frameon=False)
    _style_axes(ax)
    return _save(fig, path)


def plot_reconstruction_curve(
    X: np.ndarray, model: PrincipalComponentAnalysis, path: Path
) -> Path:
    """RMSE versus number of retained principal components."""
    from wafer_pca.exploration import reconstruction_error

    ks = np.arange(1, model.n_features + 1)
    errors = [reconstruction_error(X, model, int(k)) for k in ks]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.plot(ks, errors, color=PALETTE["cyan"], marker="o", lw=1.8)
    ax.set_xlabel("Number of retained components")
    ax.set_ylabel("Reconstruction RMSE (standardized units)")
    ax.set_title("Thickness-field reconstruction error versus model order")
    ax.set_xticks(ks)
    _style_axes(ax)
    return _save(fig, path)


def plot_wafer_mode_map(loadings: np.ndarray, component: int, path: Path) -> Path:
    """Schematic 3×3 wafer disk colored by a chosen loading vector."""
    from wafer_pca.data import SCHEMATIC_GAUGE_XY

    labels = [f"G{i}" for i in range(1, loadings.shape[0] + 1)]
    weights = loadings[:, component]
    vmax = float(np.max(np.abs(weights))) or 1.0

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    fig.patch.set_facecolor(PALETTE["bg"])
    wafer = plt.Circle((0, 0), 1.0, fill=False, color=PALETTE["slate"], lw=1.6)
    ax.add_patch(wafer)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)

    for label, w in zip(labels, weights):
        x, y = SCHEMATIC_GAUGE_XY[label]
        # Map loading ∈ [-vmax, vmax] onto the steel→amber correlation cmap.
        t = 0.5 * (w / vmax + 1.0)
        color = SENSOR_CMAP(t)
        ax.scatter(
            x,
            y,
            s=520,
            c=[color],
            edgecolors=PALETTE["ink"],
            linewidths=0.8,
            zorder=3,
        )
        ax.text(
            x,
            y,
            f"{label}\n{w:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=PALETTE["ink"],
            zorder=4,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Schematic wafer map of PC{component + 1} loadings")
    ax.set_facecolor(PALETTE["bg"])
    return _save(fig, path)


def plot_outlier_comparison(
    ratios_classical: np.ndarray,
    ratios_robust: np.ndarray,
    ratios_filtered: np.ndarray,
    path: Path,
) -> Path:
    """Compare explained-variance spectra under classical vs robust pipelines."""
    pcs = np.arange(1, len(ratios_classical) + 1)
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.bar(
        pcs - width,
        ratios_classical,
        width,
        color=PALETTE["steel"],
        label="Classical z-score",
    )
    ax.bar(
        pcs,
        ratios_robust,
        width,
        color=PALETTE["amber"],
        label="Median / MAD",
    )
    ax.bar(
        pcs + width,
        ratios_filtered,
        width,
        color=PALETTE["cyan"],
        label="Mahalanobis-filtered",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Fraction of variance")
    ax.set_title("Spectrum stability under outlier-resistant preprocessing")
    ax.set_xticks(pcs)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    return _save(fig, path)


def plot_score_scatter_outliers(
    scores: np.ndarray, keep_mask: np.ndarray, path: Path
) -> Path:
    """PC scores with Mahalanobis outliers highlighted."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.scatter(
        scores[keep_mask, 0],
        scores[keep_mask, 1],
        c=PALETTE["steel"],
        s=28,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        label="Retained sections",
    )
    ax.scatter(
        scores[~keep_mask, 0],
        scores[~keep_mask, 1],
        c=PALETTE["amber"],
        s=42,
        alpha=0.9,
        edgecolors=PALETTE["ink"],
        linewidths=0.5,
        marker="X",
        label="Mahalanobis outliers",
    )
    ax.axhline(0, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.axvline(0, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("Thickness outliers in the leading principal plane")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    return _save(fig, path)


def plot_pc1_spc(
    scores_pc1: np.ndarray,
    limits: tuple[float, float, float],
    path: Path,
) -> Path:
    """Shewhart control chart of the global-thickness (PC1) score."""
    lower, center, upper = limits
    sections = np.arange(len(scores_pc1))
    outside = (scores_pc1 < lower) | (scores_pc1 > upper)

    fig, ax = plt.subplots(figsize=(10, 4.0))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.plot(sections, scores_pc1, color=PALETTE["steel"], lw=1.1, label="PC1 score")
    ax.scatter(
        sections[outside],
        scores_pc1[outside],
        color=PALETTE["amber"],
        s=36,
        zorder=3,
        label="Outside ±3σ",
    )
    ax.axhline(center, color=PALETTE["ink"], lw=1.0, label="Center line")
    ax.axhline(upper, color=PALETTE["cyan"], lw=1.0, ls="--", label="UCL / LCL (±3σ)")
    ax.axhline(lower, color=PALETTE["cyan"], lw=1.0, ls="--")
    ax.set_xlabel("Wafer section index")
    ax.set_ylabel("PC1 score")
    ax.set_title("SPC chart of the global thickness mode (PC1)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _style_axes(ax)
    return _save(fig, path)


def plot_pruning_comparison(results: list, path: Path) -> Path:
    """Bar chart of PC1 variance and 2-PC RMSE across gauge-pruning scenarios."""
    labels = []
    for r in results:
        labels.append("all gauges" if not r.dropped else "drop " + "+".join(r.dropped))
    pc1 = [r.explained_pc1 for r in results]
    rmse = [r.rmse_2pc for r in results]
    x = np.arange(len(results))

    fig, ax1 = plt.subplots(figsize=(8.0, 4.4))
    fig.patch.set_facecolor(PALETTE["bg"])
    bars = ax1.bar(x - 0.18, pc1, 0.36, color=PALETTE["steel"], label="PC1 variance fraction")
    ax1.set_ylabel("PC1 explained-variance fraction", color=PALETTE["steel"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, rmse, color=PALETTE["amber"], marker="o", lw=1.8, label="2-PC RMSE")
    ax2.set_ylabel("Reconstruction RMSE (std. units)", color=PALETTE["amber"])
    ax2.spines["top"].set_visible(False)

    ax1.set_title("Metrology impact of dropping redundant or contrast gauges")
    _style_axes(ax1)
    ax2.tick_params(colors=PALETTE["ink"], labelsize=9)
    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")
    del bars
    return _save(fig, path)


def plot_reconstruction_panels(
    X: np.ndarray,
    X_hat: np.ndarray,
    section_indices: list[int],
    path: Path,
) -> Path:
    """Original vs 2-PC reconstruction for selected wafer sections."""
    labels = [f"G{i}" for i in range(1, X.shape[1] + 1)]
    x = np.arange(len(labels))
    width = 0.36
    n = len(section_indices)

    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, idx in zip(axes, section_indices):
        ax.bar(x - width / 2, X[idx], width, color=PALETTE["steel"], label="Observed")
        ax.bar(x + width / 2, X_hat[idx], width, color=PALETTE["amber"], label="2-PC recon.")
        ax.axhline(0, color=PALETTE["slate"], lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"Section {idx}", fontsize=10)
        _style_axes(ax)

    axes[0].set_ylabel("Standardized thickness")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Observed vs two-component reconstruction for representative sections",
        color=PALETTE["ink"],
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_loading_intervals(
    loadings: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    component: int,
    path: Path,
) -> Path:
    """PC loading point estimates with bootstrap 95% intervals."""
    labels = [f"G{i}" for i in range(1, loadings.shape[0] + 1)]
    y = loadings[:, component]
    err_lo = y - lower[:, component]
    err_hi = upper[:, component] - y
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.bar(x, y, color=PALETTE["steel"], width=0.65)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([err_lo, err_hi]),
        fmt="none",
        ecolor=PALETTE["ink"],
        elinewidth=1.2,
        capsize=3,
    )
    ax.axhline(0, color=PALETTE["slate"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Loading weight")
    ax.set_title(f"PC{component + 1} loadings with 95% bootstrap intervals")
    _style_axes(ax)
    return _save(fig, path)


def plot_scree_with_mp(
    ratios: np.ndarray,
    eigenvalues: np.ndarray,
    mp_upper: float,
    path: Path,
    bootstrap_bands: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Path:
    """Scree plot with Marchenko–Pastur upper edge marked on a twin axis."""
    cumulative = np.cumsum(ratios)
    pcs = np.arange(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.bar(pcs, ratios, color=PALETTE["steel"], width=0.65, label="Per-component ρ")
    if bootstrap_bands is not None:
        lower, _, upper = bootstrap_bands
        ax.vlines(pcs, lower, upper, color=PALETTE["ink"], lw=1.3, label="95% CI (ρ)")
    ax.plot(pcs, cumulative, color=PALETTE["amber"], marker="o", lw=1.8, label="Cumulative")

    ax2 = ax.twinx()
    ax2.plot(pcs, eigenvalues, color=PALETTE["cyan"], marker="s", lw=1.4, label="Eigenvalue λ")
    ax2.axhline(mp_upper, color=PALETTE["cyan"], ls="--", lw=1.1, label=f"MP edge λ₊={mp_upper:.2f}")
    ax2.set_ylabel("Eigenvalue", color=PALETTE["cyan"])
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(colors=PALETTE["ink"], labelsize=9)

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Fraction of variance")
    ax.set_title("Spectrum versus Marchenko–Pastur noise edge")
    ax.set_xticks(pcs)
    ax.set_ylim(0, 1.05)
    _style_axes(ax)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="center right")
    return _save(fig, path)


def plot_cov_vs_corr_spectra(
    ratios_cov: np.ndarray, ratios_corr: np.ndarray, path: Path
) -> Path:
    """Compare raw-covariance PCA against correlation (z-scored) PCA."""
    pcs = np.arange(1, len(ratios_cov) + 1)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.bar(pcs - width / 2, ratios_cov, width, color=PALETTE["amber"], label="Covariance (centered)")
    ax.bar(pcs + width / 2, ratios_corr, width, color=PALETTE["steel"], label="Correlation (z-scored)")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Fraction of variance")
    ax.set_title("Why standardization matters: covariance vs correlation PCA")
    ax.set_xticks(pcs)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    return _save(fig, path)


def plot_residual_wafer_map(mean_abs_resid: np.ndarray, path: Path) -> Path:
    """Schematic wafer map of mean |residual| after a 2-PC reconstruction."""
    from wafer_pca.data import SCHEMATIC_GAUGE_XY

    labels = [f"G{i}" for i in range(1, len(mean_abs_resid) + 1)]
    vmax = float(np.max(mean_abs_resid)) or 1.0

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    fig.patch.set_facecolor(PALETTE["bg"])
    wafer = plt.Circle((0, 0), 1.0, fill=False, color=PALETTE["slate"], lw=1.6)
    ax.add_patch(wafer)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)

    for label, r in zip(labels, mean_abs_resid):
        x, y = SCHEMATIC_GAUGE_XY[label]
        t = r / vmax
        color = SENSOR_CMAP(0.5 + 0.5 * t)
        ax.scatter(
            x, y, s=520, c=[color], edgecolors=PALETTE["ink"], linewidths=0.8, zorder=3
        )
        ax.text(
            x,
            y,
            f"{label}\n{r:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=PALETTE["ink"],
            zorder=4,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Mean |residual| per gauge after 2-PC reconstruction")
    ax.set_facecolor(PALETTE["bg"])
    return _save(fig, path)


def plot_cv_rmse(
    in_sample: np.ndarray, cv_rmse: np.ndarray, path: Path
) -> Path:
    """In-sample versus cross-validated reconstruction RMSE."""
    ks = np.arange(1, len(in_sample) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.plot(ks, in_sample, color=PALETTE["cyan"], marker="o", lw=1.8, label="In-sample")
    ax.plot(ks, cv_rmse, color=PALETTE["amber"], marker="s", lw=1.8, label="5-fold CV")
    ax.set_xlabel("Number of retained components")
    ax.set_ylabel("Reconstruction RMSE (standardized units)")
    ax.set_title("In-sample vs out-of-sample reconstruction error")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    return _save(fig, path)


def plot_physical_vs_pca_rmse(
    rmse_physical: float,
    rmse_pca: dict[int, float],
    path: Path,
) -> Path:
    """Bar comparison of physical plane model vs PCA reconstructions."""
    labels = ["Physical\n(offset+tilt)"] + [f"PCA\nr={k}" for k in sorted(rmse_pca)]
    values = [rmse_physical] + [rmse_pca[k] for k in sorted(rmse_pca)]
    colors = [PALETTE["amber"]] + [PALETTE["steel"]] * len(rmse_pca)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.bar(x, values, color=colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reconstruction RMSE (standardized units)")
    ax.set_title("Physical plane model versus PCA reconstructions")
    _style_axes(ax)
    return _save(fig, path)


def plot_permutation_null(
    observed_ratio: float,
    null_ratios: np.ndarray,
    component: int,
    path: Path,
) -> Path:
    """Histogram of permuted ρ_k against the observed value."""
    null_k = null_ratios[:, component]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.hist(null_k, bins=30, color=PALETTE["steel"], alpha=0.85, edgecolor="white")
    ax.axvline(
        observed_ratio,
        color=PALETTE["amber"],
        lw=2.0,
        label=f"Observed ρ_{component + 1} = {observed_ratio:.3f}",
    )
    ax.set_xlabel(f"Explained-variance fraction ρ_{component + 1}")
    ax.set_ylabel("Permutation count")
    ax.set_title(f"Permutation null for PC{component + 1} variance fraction")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    return _save(fig, path)


def plot_spatial_r2_null(
    observed_r2: float, null_r2: np.ndarray, path: Path
) -> Path:
    """Histogram of permuted spatial R² for a loading vector."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.hist(null_r2, bins=30, color=PALETTE["steel"], alpha=0.85, edgecolor="white")
    ax.axvline(
        observed_r2,
        color=PALETTE["amber"],
        lw=2.0,
        label=f"Observed R² = {observed_r2:.3f}",
    )
    ax.set_xlabel("Spatial R² of loadings on schematic (x, y)")
    ax.set_ylabel("Permutation count")
    ax.set_title("Gauge-label permutation null for PC2 spatial structure")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    return _save(fig, path)
