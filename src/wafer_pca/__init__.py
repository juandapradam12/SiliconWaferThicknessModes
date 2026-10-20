"""Principal component analysis of silicon wafer thickness measurements."""

from wafer_pca.data import load_thickness, robust_standardize, standardize
from wafer_pca.diagnostics import marchenko_pastur_edges, sensor_pruning_study
from wafer_pca.pca import PrincipalComponentAnalysis, bootstrap_variance_ratios

__all__ = [
    "load_thickness",
    "standardize",
    "robust_standardize",
    "PrincipalComponentAnalysis",
    "bootstrap_variance_ratios",
    "marchenko_pastur_edges",
    "sensor_pruning_study",
]
__version__ = "1.3.0"
