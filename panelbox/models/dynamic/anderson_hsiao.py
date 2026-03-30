"""
Anderson-Hsiao (1982) estimator for dynamic panel data.

This estimator provides consistent (but inefficient) estimation of dynamic
panel models using instrumental variables. It is primarily used as an
initial consistent estimator for the LSDVC bias correction.

References
----------
.. [1] Anderson, T. W., & Hsiao, C. (1982). "Formulation and Estimation of
       Dynamic Models Using Panel Data." Journal of Econometrics, 18(1), 47-82.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from panelbox.models.dynamic.lag_utils import build_dynamic_design


class AndersonHsiao:
    """
    Anderson-Hsiao IV estimator for dynamic panel models.

    Estimates the model:
        Dy_it = rho * Dy_{i,t-1} + beta * DX_it + De_it

    using y_{i,t-2} (levels) or Dy_{i,t-2} (differences) as an instrument
    for the endogenous Dy_{i,t-1} via 2SLS.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    dep_var : str
        Name of the dependent variable.
    entity_col : str
        Column identifying entities (individuals/firms).
    time_col : str
        Column identifying time periods.
    lags : int, default=1
        Number of AR lags (currently only 1 supported).
    exog_vars : list of str, optional
        Names of exogenous regressors.
    instrument : str, default='levels'
        Type of instrument for the lagged dependent variable:
        - 'levels': uses y_{t-2} (more efficient, default)
        - 'differences': uses Dy_{t-2}

    Examples
    --------
    >>> from panelbox.models.dynamic import AndersonHsiao
    >>> from panelbox import load_grunfeld
    >>> data = load_grunfeld()
    >>> ah = AndersonHsiao(data, "invest", "firm", "year", exog_vars=["value", "capital"])
    >>> result = ah.fit()
    >>> print(f"rho = {result['rho']:.4f}")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dep_var: str,
        entity_col: str,
        time_col: str,
        lags: int = 1,
        exog_vars: list[str] | None = None,
        instrument: str = "levels",
    ):
        self.data = data
        self.dep_var = dep_var
        self.entity_col = entity_col
        self.time_col = time_col
        self.lags = lags
        self.exog_vars = exog_vars
        self.instrument = instrument

    def fit(self) -> dict:
        """
        Estimate the model via 2SLS.

        Returns
        -------
        dict
            - 'rho': float, AR(1) coefficient
            - 'beta': np.ndarray or None, exogenous coefficients
            - 'params': np.ndarray, all coefficients [rho, beta_1, ..., beta_k]
            - 'resid': np.ndarray, residuals from the differenced equation
            - 'param_names': list of str, parameter names
            - 'nobs': int, number of observations used
        """
        design = build_dynamic_design(
            self.data,
            self.dep_var,
            self.entity_col,
            self.time_col,
            lags=self.lags,
            exog_vars=self.exog_vars,
            instrument=self.instrument,
        )

        Dy = design["Dy"]
        Dy_lag = design["Dy_lag"]
        Z_iv = design["Z"]
        DX = design["DX"]
        n = len(Dy)

        # Build endogenous regressor matrix: [Dy_lag, DX]
        if DX is not None:
            W = np.column_stack([Dy_lag, DX])
        else:
            W = Dy_lag.reshape(-1, 1)

        # Build instrument matrix: [Z_iv, DX]
        # DX is assumed exogenous, so it instruments itself
        if DX is not None:
            Z = np.column_stack([Z_iv, DX])
        else:
            Z = Z_iv.reshape(-1, 1)

        # 2SLS estimation
        # Stage 1: project W onto Z
        # W_hat = Z (Z'Z)^{-1} Z' W
        ZtZ = Z.T @ Z
        ZtZ_inv = np.linalg.pinv(ZtZ)
        P_Z = Z @ ZtZ_inv @ Z.T
        W_hat = P_Z @ W

        # Stage 2: regress Dy on W_hat
        # params = (W_hat' W_hat)^{-1} W_hat' Dy
        WhtWh = W_hat.T @ W_hat
        WhtWh_inv = np.linalg.pinv(WhtWh)
        params = WhtWh_inv @ (W_hat.T @ Dy)

        # Residuals (using actual W, not W_hat)
        resid = Dy - W @ params

        # Build parameter names
        param_names = [f"L.{self.dep_var}"]
        if self.exog_vars is not None:
            param_names.extend(self.exog_vars)

        rho = float(params[0])
        beta = params[1:] if len(params) > 1 else None

        return {
            "rho": rho,
            "beta": beta,
            "params": params,
            "resid": resid,
            "param_names": param_names,
            "nobs": n,
        }
