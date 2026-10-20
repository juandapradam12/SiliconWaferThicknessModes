"""Data loading and preprocessing for silicon wafer thickness measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "silicon_wafer_thickness.csv"
)

# Schematic 3×3 gauge layout on the wafer disk (row-major G1…G9).
# Physical survey coordinates were not shipped with the table; this grid is
# the conventional metrology arrangement used only for spatial interpretation.
SCHEMATIC_GAUGE_XY: dict[str, tuple[float, float]] = {
    "G1": (-0.55, 0.55),
    "G2": (0.0, 0.55),
    "G3": (0.55, 0.55),
    "G4": (-0.55, 0.0),
    "G5": (0.0, 0.0),
    "G6": (0.55, 0.0),
    "G7": (-0.55, -0.55),
    "G8": (0.0, -0.55),
    "G9": (0.55, -0.55),
}


def load_thickness(path: str | Path | None = None) -> pd.DataFrame:
    """Load the nine-sensor thickness table.

    Each column ``G1``…``G9`` is a spatial gauge group; each row is one
    wafer section (or scan index). Values are thickness deviations from a
    manufacturing target, in the same arbitrary length unit used by the
    metrology tool.
    """
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    frame = pd.read_csv(csv_path)
    expected = [f"G{i}" for i in range(1, 10)]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return frame[expected].astype(float)


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score: ``(X - μ) / σ``.

    Returns the standardized matrix together with the per-column mean and
    standard deviation so that reconstructions can be mapped back to the
    original thickness scale.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    if np.any(sigma == 0):
        raise ValueError("Cannot standardize a constant sensor column")
    return (X - mu) / sigma, mu, sigma


def robust_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median / MAD scaling, resistant to heavy-tailed thickness spikes.

    Uses the consistency factor ``1.4826`` so that the MAD matches the
    standard deviation for Gaussian noise.
    """
    X = np.asarray(X, dtype=float)
    center = np.median(X, axis=0)
    mad = np.median(np.abs(X - center), axis=0)
    scale = 1.4826 * mad
    if np.any(scale == 0):
        raise ValueError("Cannot robust-standardize a constant sensor column")
    return (X - center) / scale, center, scale


# χ² critical values at the 97.5% level for common gauge counts (df = p).
_CHI2_0975 = {9: 19.02276779864163}


def mahalanobis_mask(X: np.ndarray, chi2_quantile: float = 0.975) -> np.ndarray:
    """Boolean keep-mask from a Gaussian Mahalanobis gate on rows of ``X``.

    ``X`` should already be column-standardized. Rows whose squared
    Mahalanobis distance exceeds the ``chi2_quantile`` critical value of
    a :math:`\\chi^2_p` law are flagged as thickness outliers.
    """
    if chi2_quantile != 0.975:
        raise ValueError("Only the 97.5% χ² gate is tabulated in this module")

    X = np.asarray(X, dtype=float)
    _, p = X.shape
    if p not in _CHI2_0975:
        raise ValueError(f"No tabulated χ² critical value for p={p}")

    cov = np.cov(X, rowvar=False, ddof=1)
    # Ridge for numerical stability when gauges are nearly collinear.
    cov = cov + 1e-8 * np.eye(p)
    precision = np.linalg.inv(cov)
    delta = X - X.mean(axis=0)
    d2 = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return d2 <= _CHI2_0975[p]


def summary_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mean, std, min, and max for each gauge group."""
    return frame.agg(["mean", "std", "min", "max"]).T
