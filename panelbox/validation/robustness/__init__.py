"""
Robustness analysis tools for panel data models.

This module provides various methods for assessing the robustness of panel
data estimation results, including:

- Bootstrap inference (various methods)
- Sensitivity analysis
- Cross-validation
- Jackknife resampling
- Outlier detection
- Influence diagnostics
- Robustness checks

Examples
--------
>>> import panelbox as pb
>>>
>>> # Fit model
>>> fe = pb.FixedEffects("y ~ x1 + x2", data, "id", "time")
>>> results = fe.fit()
>>>
>>> # Bootstrap inference
>>> bootstrap = pb.PanelBootstrap(results, n_bootstrap=1000, method='pairs')
>>> bootstrap.run()
>>> ci = bootstrap.conf_int()
>>> print(ci)
"""

from panelbox.validation.robustness.bootstrap import PanelBootstrap
from panelbox.validation.robustness.checks import RobustnessChecker
from panelbox.validation.robustness.cross_validation import CVResults, TimeSeriesCV
from panelbox.validation.robustness.influence import InfluenceDiagnostics, InfluenceResults
from panelbox.validation.robustness.jackknife import JackknifeResults, PanelJackknife
from panelbox.validation.robustness.outliers import OutlierDetector, OutlierResults
from panelbox.validation.robustness.sensitivity import SensitivityAnalysis, SensitivityResults

__all__ = [
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
]
