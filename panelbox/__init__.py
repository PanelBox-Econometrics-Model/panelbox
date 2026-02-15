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
from panelbox.diagnostics.cointegration import (
    KaoResult,
    PedroniResult,
    WesterlundResult,
    kao_test,
    pedroni_test,
    westerlund_test,
)

# Quantile Regression Diagnostics
from panelbox.diagnostics.quantile import QuantileRegressionDiagnostics
from panelbox.diagnostics.specification import (
    EncompassingResult,
    JTestResult,
    cox_test,
    j_test,
    likelihood_ratio_test,
    wald_encompassing_test,
)
from panelbox.diagnostics.unit_root import (
    BreitungResult,
    HadriResult,
    PanelUnitRootResult,
    breitung_test,
    hadri_test,
    panel_unit_root_test,
)

# Experiment Pattern (Sprints 3-5)
from panelbox.experiment import PanelExperiment
from panelbox.experiment.results import (
    BaseResult,
    ComparisonResult,
    ResidualResult,
    ValidationResult,
)

# Stochastic Frontier Analysis (FASE 1)
from panelbox.frontier import (
    DistributionType,
    FrontierType,
    ModelType,
    SFResult,
    StochasticFrontier,
)

# Advanced GMM estimators (FASE 1)
from panelbox.gmm import BiasCorrectedGMM, ContinuousUpdatedGMM, GMMDiagnostics

# Dynamic panel GMM models
from panelbox.gmm.difference_gmm import DifferenceGMM
from panelbox.gmm.results import GMMResults
from panelbox.gmm.system_gmm import SystemGMM

# Quantile Regression Inference
from panelbox.inference.quantile import BootstrapResult, QuantileBootstrap

# Count Data Models (FASE 5)
from panelbox.models.count import (
    PPML,
    FixedEffectsNegativeBinomial,
    NegativeBinomial,
    PoissonFixedEffects,
    PoissonQML,
    PooledPoisson,
    PPMLResult,
    RandomEffectsPoisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedNegativeBinomialResult,
    ZeroInflatedPoisson,
    ZeroInflatedPoissonResult,
)

# Discrete Choice Models (FASE 5)
from panelbox.models.discrete import (
    ConditionalLogit,
    FixedEffectsLogit,
    MultinomialLogit,
    MultinomialLogitResult,
    NonlinearPanelModel,
    OrderedLogit,
    OrderedProbit,
    PooledLogit,
    PooledProbit,
    RandomEffectsOrderedLogit,
    RandomEffectsProbit,
)

# IV models
from panelbox.models.iv.panel_iv import PanelIV

# Quantile Regression Models
from panelbox.models.quantile import PooledQuantile, PooledQuantileResults

# Selection Models (FASE 2)
from panelbox.models.selection import (
    PanelHeckman,
    PanelHeckmanResult,
    compute_imr,
    imr_derivative,
    imr_diagnostics,
    test_selection_effect,
)
from panelbox.models.static.between import BetweenEstimator
from panelbox.models.static.first_difference import FirstDifferenceEstimator
from panelbox.models.static.fixed_effects import FixedEffects

# Static panel models
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.cointegration.kao import KaoTest, KaoTestResult

# Cointegration tests (existing + FASE 3 advanced tests)
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

# Specification Tests (existing + FASE 5 specification tests)
from panelbox.validation.specification.hausman import HausmanTest, HausmanTestResult
from panelbox.validation.specification.mundlak import MundlakTest
from panelbox.validation.specification.reset import RESETTest
from panelbox.validation.unit_root.fisher import FisherTest, FisherTestResult
from panelbox.validation.unit_root.ips import IPSTest, IPSTestResult

# Unit Root Tests (existing + FASE 4 advanced tests)
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

# Quantile Regression Visualization
from panelbox.visualization.quantile import qq_plot, quantile_process_plot, residual_plot

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
    # Advanced GMM (FASE 1)
    "ContinuousUpdatedGMM",
    "BiasCorrectedGMM",
    "GMMDiagnostics",
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
    # Specification Tests (existing + FASE 5)
    "HausmanTest",
    "HausmanTestResult",
    "MundlakTest",
    "RESETTest",
    "ChowTest",
    "j_test",
    "JTestResult",
    "cox_test",
    "wald_encompassing_test",
    "likelihood_ratio_test",
    "EncompassingResult",
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
    # Unit Root Tests (existing + FASE 4)
    "LLCTest",
    "LLCTestResult",
    "IPSTest",
    "IPSTestResult",
    "FisherTest",
    "FisherTestResult",
    "hadri_test",
    "HadriResult",
    "breitung_test",
    "BreitungResult",
    "panel_unit_root_test",
    "PanelUnitRootResult",
    # Cointegration Tests (existing + FASE 3)
    "PedroniTest",
    "PedroniTestResult",
    "KaoTest",
    "KaoTestResult",
    "westerlund_test",
    "WesterlundResult",
    "kao_test",
    "KaoResult",
    "pedroni_test",
    "PedroniResult",
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
    # Quantile Regression Models
    "PooledQuantile",
    "PooledQuantileResults",
    # Quantile Regression Inference
    "QuantileBootstrap",
    "BootstrapResult",
    # Selection Models (FASE 2)
    "PanelHeckman",
    "PanelHeckmanResult",
    "compute_imr",
    "imr_derivative",
    "imr_diagnostics",
    "test_selection_effect",
    # Discrete Choice Models (FASE 5)
    "NonlinearPanelModel",
    "PooledLogit",
    "PooledProbit",
    "FixedEffectsLogit",
    "RandomEffectsProbit",
    "OrderedLogit",
    "OrderedProbit",
    "RandomEffectsOrderedLogit",
    "MultinomialLogit",
    "MultinomialLogitResult",
    "ConditionalLogit",
    # Count Data Models (FASE 5)
    "PooledPoisson",
    "PoissonFixedEffects",
    "RandomEffectsPoisson",
    "PoissonQML",
    "PPML",
    "PPMLResult",
    "NegativeBinomial",
    "FixedEffectsNegativeBinomial",
    "ZeroInflatedPoisson",
    "ZeroInflatedPoissonResult",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedNegativeBinomialResult",
    # Quantile Regression Diagnostics
    "QuantileRegressionDiagnostics",
    # Quantile Regression Visualization
    "quantile_process_plot",
    "residual_plot",
    "qq_plot",
    # Stochastic Frontier Analysis (FASE 1)
    "StochasticFrontier",
    "SFResult",
    "FrontierType",
    "DistributionType",
    "ModelType",
]
