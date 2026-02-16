"""
Standard Errors Tutorial Utilities
===================================

Utility functions for standard errors and robust inference tutorials.

Modules
-------
plotting : Visualization helpers for standard error comparisons
diagnostics : Diagnostic tests for heteroskedasticity, autocorrelation, and clustering
data_generators : Generate synthetic data with various error structures

Usage
-----
Import specific functions from submodules:

    from utils.plotting import plot_se_comparison, plot_residuals
    from utils.diagnostics import test_heteroskedasticity
    from utils.data_generators import generate_heteroskedastic_data

Or import entire modules:

    from utils import plotting, diagnostics, data_generators

Version
-------
1.0.0 (2026-02-16)

Dependencies
------------
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels (optional, for some diagnostic tests)

"""

__version__ = "1.0.0"
__author__ = "PanelBox Development Team"
__license__ = "MIT"

# Import submodules for convenient access
from . import data_generators, diagnostics, plotting

__all__ = [
    "plotting",
    "diagnostics",
    "data_generators",
]
