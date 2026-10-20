"""Data loading and preprocessing for silicon wafer thickness measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "silicon_wafer_thickness.csv"
)


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


def summary_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mean, std, min, and max for each gauge group."""
    return frame.agg(["mean", "std", "min", "max"]).T
