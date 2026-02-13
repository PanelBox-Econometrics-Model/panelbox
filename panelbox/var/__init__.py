"""
Panel Vector Autoregression (VAR) module for PanelBox.

This module provides tools for estimating and analyzing Panel VAR models,
including OLS estimation, GMM estimation, impulse response functions,
forecast error variance decomposition, and Granger causality tests.

Classes
-------
PanelVARData
    Data container for Panel VAR models
PanelVAR
    Panel VAR model estimation
PanelVARResult
    Panel VAR estimation results
LagOrderResult
    Lag order selection results

Examples
--------
>>> import panelbox as pb
>>> from panelbox.var import PanelVARData, PanelVAR
>>>
>>> # Prepare data
>>> data = PanelVARData(
...     df,
...     endog_vars=['gdp', 'inflation', 'interest_rate'],
...     entity_col='country',
...     time_col='year',
...     lags=2
... )
>>>
>>> # Estimate Panel VAR
>>> model = PanelVAR(data)
>>> results = model.fit(method='ols', cov_type='clustered')
>>> print(results.summary())
>>>
>>> # Select optimal lag order
>>> lag_results = model.select_lag_order(max_lags=8)
>>> print(lag_results.summary())

References
----------
.. [1] Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector
       autoregressions with panel data. Econometrica, 56(6), 1371-1395.
.. [2] Love, I., & Zicchino, L. (2006). Financial development and dynamic
       investment behavior: Evidence from panel VAR. The Quarterly Review of
       Economics and Finance, 46(2), 190-210.
.. [3] Abrigo, M. R., & Love, I. (2016). Estimation of panel vector
       autoregression in Stata. The Stata Journal, 16(3), 778-804.
"""

from panelbox.var.causality_network import plot_causality_network
from panelbox.var.data import PanelVARData
from panelbox.var.forecast import ForecastResult
from panelbox.var.model import PanelVAR
from panelbox.var.result import LagOrderResult, PanelVARResult
from panelbox.var.vecm import (
    CointegrationRankTest,
    PanelVECM,
    PanelVECMResult,
    RankSelectionResult,
    RankTestResult,
)

__all__ = [
    "PanelVARData",
    "PanelVAR",
    "PanelVARResult",
    "ForecastResult",
    "LagOrderResult",
    "plot_causality_network",
    "CointegrationRankTest",
    "PanelVECM",
    "PanelVECMResult",
    "RankSelectionResult",
    "RankTestResult",
]
