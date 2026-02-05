"""
PanelBox - Panel Data Econometrics in Python

PanelBox provides comprehensive tools for panel data econometrics,
inspired by Stata (xtabond2), R (plm), and statsmodels.

Features:
- Static panel models: Pooled OLS, Fixed Effects, Random Effects
- Dynamic panel GMM: Arellano-Bond (1991), Blundell-Bond (1998)
- Robust to unbalanced panels
- Comprehensive specification tests
- Publication-ready reporting

Quick Start:
    >>> from panelbox import DifferenceGMM
    >>> gmm = DifferenceGMM(data=df, dep_var='y', lags=1, id_var='id', time_var='year')
    >>> results = gmm.fit()
    >>> print(results.summary())
"""

from panelbox.__version__ import __version__, __author__, __email__, __license__

# Core classes
from panelbox.core.panel_data import PanelData
from panelbox.core.formula_parser import FormulaParser, parse_formula
from panelbox.core.results import PanelResults

# Static panel models
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.models.static.between import BetweenEstimator
from panelbox.models.static.first_difference import FirstDifferenceEstimator

# Dynamic panel GMM models
from panelbox.gmm.difference_gmm import DifferenceGMM
from panelbox.gmm.system_gmm import SystemGMM
from panelbox.gmm.results import GMMResults

# Tests
from panelbox.validation.specification.hausman import HausmanTest, HausmanTestResult

# Robustness analysis
from panelbox.validation.robustness.bootstrap import PanelBootstrap
from panelbox.validation.robustness.sensitivity import SensitivityAnalysis, SensitivityResults
from panelbox.validation.robustness.cross_validation import TimeSeriesCV, CVResults
from panelbox.validation.robustness.jackknife import PanelJackknife, JackknifeResults
from panelbox.validation.robustness.outliers import OutlierDetector, OutlierResults
from panelbox.validation.robustness.influence import InfluenceDiagnostics, InfluenceResults
from panelbox.validation.robustness.checks import RobustnessChecker

# Datasets
from panelbox.datasets import (
    load_grunfeld,
    load_abdata,
    list_datasets,
    get_dataset_info
)

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__email__',
    '__license__',

    # Core
    'PanelData',
    'FormulaParser',
    'parse_formula',
    'PanelResults',

    # Static Models
    'PooledOLS',
    'FixedEffects',
    'RandomEffects',
    'BetweenEstimator',
    'FirstDifferenceEstimator',

    # GMM Models
    'DifferenceGMM',
    'SystemGMM',
    'GMMResults',

    # Tests
    'HausmanTest',
    'HausmanTestResult',

    # Robustness
    'PanelBootstrap',
    'SensitivityAnalysis',
    'SensitivityResults',
    'TimeSeriesCV',
    'CVResults',
    'PanelJackknife',
    'JackknifeResults',
    'OutlierDetector',
    'OutlierResults',
    'InfluenceDiagnostics',
    'InfluenceResults',
    'RobustnessChecker',

    # Datasets
    'load_grunfeld',
    'load_abdata',
    'list_datasets',
    'get_dataset_info',
]
