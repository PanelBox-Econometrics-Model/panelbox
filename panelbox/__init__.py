"""
PanelBox - Panel Data Econometrics in Python

PanelBox provides comprehensive tools for panel data econometrics,
inspired by Stata (xtabond2), R (plm), and statsmodels.

Features:
- Static panel models: Pooled OLS, Fixed Effects, Random Effects
- Dynamic panel GMM: Arellano-Bond (1991), Blundell-Bond (1998)
- Experiment Pattern: Factory-based model management with result containers
- Interactive HTML reports with Plotly visualizations
- Robust to unbalanced panels
- Comprehensive specification tests
- Publication-ready reporting

Quick Start (Traditional):
    >>> from panelbox import FixedEffects
    >>> fe = FixedEffects("y ~ x1 + x2", data, "firm", "year")
    >>> results = fe.fit()
    >>> print(results.summary())

Quick Start (Experiment Pattern):
    >>> from panelbox import PanelExperiment
    >>> experiment = PanelExperiment(data, "y ~ x1 + x2", "firm", "year")
    >>> experiment.fit_all_models(names=['pooled', 'fe', 're'])
    >>> val_result = experiment.validate_model('fe')
    >>> val_result.save_html('validation.html', test_type='validation')
"""

from panelbox.__version__ import __author__, __email__, __license__, __version__
from panelbox.core.formula_parser import FormulaParser, parse_formula

# Core classes
from panelbox.core.panel_data import PanelData
from panelbox.core.results import PanelResults

# Datasets
from panelbox.datasets import get_dataset_info, list_datasets, load_abdata, load_grunfeld

# Experiment Pattern (Sprints 3-5)
from panelbox.experiment import PanelExperiment
from panelbox.experiment.results import (
    BaseResult,
    ComparisonResult,
    ResidualResult,
    ValidationResult,
)

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
from panelbox.validation.cross_sectional_dependence.breusch_pagan_lm import BreuschPaganLMTest
from panelbox.validation.cross_sectional_dependence.frees import FreesTest

# Cross-Sectional Dependence Tests
from panelbox.validation.cross_sectional_dependence.pesaran_cd import PesaranCDTest
from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest

# Heteroskedasticity Tests
from panelbox.validation.heteroskedasticity.modified_wald import ModifiedWaldTest
from panelbox.validation.heteroskedasticity.white import WhiteTest

# Robustness analysis
from panelbox.validation.robustness.bootstrap import PanelBootstrap
from panelbox.validation.robustness.checks import RobustnessChecker
from panelbox.validation.robustness.cross_validation import CVResults, TimeSeriesCV
from panelbox.validation.robustness.influence import InfluenceDiagnostics, InfluenceResults
from panelbox.validation.robustness.jackknife import JackknifeResults, PanelJackknife
from panelbox.validation.robustness.outliers import OutlierDetector, OutlierResults
from panelbox.validation.robustness.sensitivity import SensitivityAnalysis, SensitivityResults
from panelbox.validation.serial_correlation.baltagi_wu import BaltagiWuTest
from panelbox.validation.serial_correlation.breusch_godfrey import BreuschGodfreyTest

# Serial Correlation Tests
from panelbox.validation.serial_correlation.wooldridge_ar import WooldridgeARTest
from panelbox.validation.specification.chow import ChowTest

# Specification Tests
from panelbox.validation.specification.hausman import HausmanTest, HausmanTestResult
from panelbox.validation.specification.mundlak import MundlakTest
from panelbox.validation.specification.reset import RESETTest
from panelbox.validation.unit_root.fisher import FisherTest, FisherTestResult
from panelbox.validation.unit_root.ips import IPSTest, IPSTestResult

# Unit Root Tests
from panelbox.validation.unit_root.llc import LLCTest, LLCTestResult

# Panel VAR models
from panelbox.var import (
    CointegrationRankTest,
    ForecastResult,
    LagOrderResult,
    PanelVAR,
    PanelVARData,
    PanelVARResult,
    PanelVECM,
    PanelVECMResult,
    RankSelectionResult,
    RankTestResult,
    plot_causality_network,
)

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
    # Panel VAR Models
    "PanelVAR",
    "PanelVARData",
    "PanelVARResult",
    "ForecastResult",
    "LagOrderResult",
    "PanelVECM",
    "PanelVECMResult",
    "CointegrationRankTest",
    "RankSelectionResult",
    "RankTestResult",
    "plot_causality_network",
    # Specification Tests
    "HausmanTest",
    "HausmanTestResult",
    "MundlakTest",
    "RESETTest",
    "ChowTest",
    # Serial Correlation Tests
    "WooldridgeARTest",
    "BreuschGodfreyTest",
    "BaltagiWuTest",
    # Heteroskedasticity Tests
    "ModifiedWaldTest",
    "BreuschPaganTest",
    "WhiteTest",
    # Cross-Sectional Dependence Tests
    "PesaranCDTest",
    "BreuschPaganLMTest",
    "FreesTest",
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
    # Experiment Pattern
    "PanelExperiment",
    "BaseResult",
    "ValidationResult",
    "ComparisonResult",
    "ResidualResult",
]
