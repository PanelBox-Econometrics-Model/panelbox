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

from panelbox.__version__ import __author__, __email__, __license__, __version__
from panelbox.core.formula_parser import FormulaParser, parse_formula

# Core classes
from panelbox.core.panel_data import PanelData
from panelbox.core.results import PanelResults

# Datasets
from panelbox.datasets import get_dataset_info, list_datasets, load_abdata, load_grunfeld

# Dynamic panel GMM models
from panelbox.gmm.difference_gmm import DifferenceGMM
from panelbox.gmm.results import GMMResults
from panelbox.gmm.system_gmm import SystemGMM

# IV models
from panelbox.models.iv.panel_iv import PanelIV
from panelbox.models.static.between import BetweenEstimator
from panelbox.models.static.first_difference import FirstDifferenceEstimator
from panelbox.models.static.fixed_effects import FixedEffects

# Static panel models
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.cointegration.kao import KaoTest, KaoTestResult

# Cointegration tests
from panelbox.validation.cointegration.pedroni import PedroniTest, PedroniTestResult

# Robustness analysis
from panelbox.validation.robustness.bootstrap import PanelBootstrap
from panelbox.validation.robustness.checks import RobustnessChecker
from panelbox.validation.robustness.cross_validation import CVResults, TimeSeriesCV
from panelbox.validation.robustness.influence import InfluenceDiagnostics, InfluenceResults
from panelbox.validation.robustness.jackknife import JackknifeResults, PanelJackknife
from panelbox.validation.robustness.outliers import OutlierDetector, OutlierResults
from panelbox.validation.robustness.sensitivity import SensitivityAnalysis, SensitivityResults

# Tests
from panelbox.validation.specification.hausman import HausmanTest, HausmanTestResult
from panelbox.validation.unit_root.fisher import FisherTest, FisherTestResult
from panelbox.validation.unit_root.ips import IPSTest, IPSTestResult

# Unit root tests
from panelbox.validation.unit_root.llc import LLCTest, LLCTestResult

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core
    "PanelData",
    "FormulaParser",
    "parse_formula",
    "PanelResults",
    # Static Models
    "PooledOLS",
    "FixedEffects",
    "RandomEffects",
    "BetweenEstimator",
    "FirstDifferenceEstimator",
    # IV Models
    "PanelIV",
    # GMM Models
    "DifferenceGMM",
    "SystemGMM",
    "GMMResults",
    # Tests
    "HausmanTest",
    "HausmanTestResult",
    # Unit Root Tests
    "LLCTest",
    "LLCTestResult",
    "IPSTest",
    "IPSTestResult",
    "FisherTest",
    "FisherTestResult",
    # Cointegration Tests
    "PedroniTest",
    "PedroniTestResult",
    "KaoTest",
    "KaoTestResult",
    # Robustness
    "PanelBootstrap",
    "SensitivityAnalysis",
    "SensitivityResults",
    "TimeSeriesCV",
    "CVResults",
    "PanelJackknife",
    "JackknifeResults",
    "OutlierDetector",
    "OutlierResults",
    "InfluenceDiagnostics",
    "InfluenceResults",
    "RobustnessChecker",
    # Datasets
    "load_grunfeld",
    "load_abdata",
    "list_datasets",
    "get_dataset_info",
]
