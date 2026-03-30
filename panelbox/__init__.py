"""
PanelBox - Panel Data Econometrics in Python.

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
    >>> experiment.fit_all_models(names=["pooled", "fe", "re"])
    >>> val_result = experiment.validate_model("fe")
    >>> val_result.save_html("validation.html", test_type="validation")
"""

from __future__ import annotations

import contextlib
import logging

# Library default: no output unless user configures logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

from panelbox.__version__ import __author__, __email__, __license__, __version__
from panelbox._logging import set_verbosity

# AutoExperiment
from panelbox.autoexperiment import AutoExperiment, AutoExperimentResults
from panelbox.core.formula_parser import FormulaParser, parse_formula

# Core classes
from panelbox.core.panel_data import PanelData
from panelbox.core.results import PanelResults
from panelbox.core.serialization import SerializableMixin, load_model

# Datasets
from panelbox.datasets import (
    get_dataset_info,
    list_categories,
    list_datasets,
    load_abdata,
    load_dataset,
    load_grunfeld,
)
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
from panelbox.models.static.mean_group import (
    MeanGroupEstimator,
    PooledMeanGroupEstimator,
    hausman_mg_pmg,
)

# Static panel models
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.models.static.random_effects import RandomEffects
from panelbox.models.static.sur import SUR, SURResults
from panelbox.validation.cointegration.kao import KaoTest, KaoTestResult

# Cointegration tests (existing + FASE 3 advanced tests)
from panelbox.validation.cointegration.pedroni import PedroniTest, PedroniTestResult
from panelbox.validation.cross_sectional_dependence.breusch_pagan_lm import (
    BreuschPaganLMTest,
)
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
from panelbox.validation.robustness.influence import (
    InfluenceDiagnostics,
    InfluenceResults,
)
from panelbox.validation.robustness.jackknife import JackknifeResults, PanelJackknife
from panelbox.validation.robustness.outliers import OutlierDetector, OutlierResults
from panelbox.validation.robustness.sensitivity import (
    SensitivityAnalysis,
    SensitivityResults,
)
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

# Quantile Regression Visualization (optional - requires matplotlib)
with contextlib.suppress(ImportError):
    from panelbox.visualization.quantile import (
        qq_plot,
        quantile_process_plot,
        residual_plot,
    )

__all__ = [
    "PPML",
    "SUR",
    "AutoExperiment",
    "AutoExperimentResults",
    "BaltagiWuTest",
    "BaseResult",
    "BetweenEstimator",
    "BiasCorrectedGMM",
    "BootstrapResult",
    "BreitungResult",
    "BreuschGodfreyTest",
    "BreuschPaganLMTest",
    "BreuschPaganTest",
    "CVResults",
    "ChowTest",
    "CointegrationRankTest",
    "ComparisonResult",
    "ConditionalLogit",
    "ContinuousUpdatedGMM",
    "DifferenceGMM",
    "DistributionType",
    "EncompassingResult",
    "FirstDifferenceEstimator",
    "FisherTest",
    "FisherTestResult",
    "FixedEffects",
    "FixedEffectsLogit",
    "FixedEffectsNegativeBinomial",
    "ForecastResult",
    "FormulaParser",
    "FreesTest",
    "FrontierType",
    "GMMDiagnostics",
    "GMMResults",
    "HadriResult",
    "HausmanTest",
    "HausmanTestResult",
    "IPSTest",
    "IPSTestResult",
    "InfluenceDiagnostics",
    "InfluenceResults",
    "JTestResult",
    "JackknifeResults",
    "KaoResult",
    "KaoTest",
    "KaoTestResult",
    "LLCTest",
    "LLCTestResult",
    "LagOrderResult",
    "MeanGroupEstimator",
    "ModelType",
    "ModifiedWaldTest",
    "MultinomialLogit",
    "MultinomialLogitResult",
    "MundlakTest",
    "NegativeBinomial",
    "NonlinearPanelModel",
    "OrderedLogit",
    "OrderedProbit",
    "OutlierDetector",
    "OutlierResults",
    "PPMLResult",
    "PanelBootstrap",
    "PanelData",
    "PanelExperiment",
    "PanelHeckman",
    "PanelHeckmanResult",
    "PanelIV",
    "PanelJackknife",
    "PanelResults",
    "PanelUnitRootResult",
    "PanelVAR",
    "PanelVARData",
    "PanelVARResult",
    "PanelVECM",
    "PanelVECMResult",
    "PedroniResult",
    "PedroniTest",
    "PedroniTestResult",
    "PesaranCDTest",
    "PoissonFixedEffects",
    "PoissonQML",
    "PooledLogit",
    "PooledMeanGroupEstimator",
    "PooledOLS",
    "PooledPoisson",
    "PooledProbit",
    "PooledQuantile",
    "PooledQuantileResults",
    "QuantileBootstrap",
    "QuantileRegressionDiagnostics",
    "RESETTest",
    "RandomEffects",
    "RandomEffectsOrderedLogit",
    "RandomEffectsPoisson",
    "RandomEffectsProbit",
    "RankSelectionResult",
    "RankTestResult",
    "ResidualResult",
    "RobustnessChecker",
    "SFResult",
    "SURResults",
    "SensitivityAnalysis",
    "SensitivityResults",
    "SerializableMixin",
    "StochasticFrontier",
    "SystemGMM",
    "TimeSeriesCV",
    "ValidationResult",
    "WesterlundResult",
    "WhiteTest",
    "WooldridgeARTest",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedNegativeBinomialResult",
    "ZeroInflatedPoisson",
    "ZeroInflatedPoissonResult",
    "__author__",
    "__email__",
    "__license__",
    "__version__",
    "breitung_test",
    "compute_imr",
    "cox_test",
    "get_dataset_info",
    "hadri_test",
    "hausman_mg_pmg",
    "imr_derivative",
    "imr_diagnostics",
    "j_test",
    "kao_test",
    "likelihood_ratio_test",
    "list_categories",
    "list_datasets",
    "load_abdata",
    "load_dataset",
    "load_grunfeld",
    "load_model",
    "panel_unit_root_test",
    "parse_formula",
    "pedroni_test",
    "plot_causality_network",
    "qq_plot",
    "quantile_process_plot",
    "residual_plot",
    "set_verbosity",
    "test_selection_effect",
    "wald_encompassing_test",
    "westerlund_test",
]
