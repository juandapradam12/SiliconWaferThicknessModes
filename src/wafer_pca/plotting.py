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
