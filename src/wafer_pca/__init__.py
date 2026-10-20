"""Principal component analysis of silicon wafer thickness measurements."""

from wafer_pca.data import load_thickness, standardize
from wafer_pca.pca import PrincipalComponentAnalysis

__all__ = ["load_thickness", "standardize", "PrincipalComponentAnalysis"]
__version__ = "1.0.0"
