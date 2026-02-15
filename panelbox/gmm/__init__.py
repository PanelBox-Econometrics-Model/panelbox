"""
PanelBox GMM Module
===================

Dynamic panel data models using Generalized Method of Moments (GMM).

This module implements:
- Difference GMM (Arellano-Bond, 1991)
- System GMM (Blundell-Bond, 1998)
- One-step, two-step, and iterative estimation
- Windmeijer (2005) finite-sample correction
- Specification tests: Hansen J, Sargan, AR(1), AR(2)

Classes
-------
DifferenceGMM : Arellano-Bond (1991) Difference GMM estimator
SystemGMM : Blundell-Bond (1998) System GMM estimator
GMMResults : Results from GMM estimation
InstrumentBuilder : Generates instrument matrices
GMMEstimator : Low-level GMM estimation routines
GMMTests : Specification tests

Examples
--------
>>> from panelbox.gmm import DifferenceGMM
>>> model = DifferenceGMM(
...     data=abdata,
...     dep_var='n',
...     lags=[1],
...     exog_vars=['w', 'k'],
...     time_dummies=True
... )
>>> results = model.fit()
>>> print(results.summary())

References
----------
.. [1] Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel
       Data: Monte Carlo Evidence and an Application to Employment Equations."
       Review of Economic Studies, 58(2), 277-297.

.. [2] Blundell, R., & Bond, S. (1998). "Initial Conditions and Moment
       Restrictions in Dynamic Panel Data Models." Journal of Econometrics,
       87(1), 115-143.

.. [3] Roodman, D. (2009). "How to do xtabond2: An Introduction to Difference
       and System GMM in Stata." Stata Journal, 9(1), 86-136.

.. [4] Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of
       Linear Efficient Two-Step GMM Estimators." Journal of Econometrics,
       126(1), 25-51.
"""

from panelbox.gmm.bias_corrected import BiasCorrectedGMM
from panelbox.gmm.cue_gmm import ContinuousUpdatedGMM
from panelbox.gmm.diagnostics import GMMDiagnostics
from panelbox.gmm.difference_gmm import DifferenceGMM
from panelbox.gmm.results import GMMResults, TestResult
from panelbox.gmm.system_gmm import SystemGMM

__all__ = [
    "BiasCorrectedGMM",
    "ContinuousUpdatedGMM",
    "GMMDiagnostics",
    "DifferenceGMM",
    "SystemGMM",
    "GMMResults",
    "TestResult",
]

__version__ = "0.1.0"
