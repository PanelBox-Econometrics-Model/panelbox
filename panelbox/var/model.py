"""
Panel VAR model estimation.

This module provides the main PanelVAR class for model estimation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.var.data import PanelVARData
from panelbox.var.inference import (
    compute_covariance_matrix,
    compute_ols_equation,
    within_transformation,
)
from panelbox.var.result import LagOrderResult, PanelVARResult


class PanelVAR:
    """
    Panel Vector Autoregression model.

    This class provides estimation methods for Panel VAR models, including
    OLS equation-by-equation estimation with fixed effects.

    Parameters
    ----------
    data : PanelVARData
        Panel VAR data container

    Attributes
    ----------
    data : PanelVARData
        Panel VAR data
    K : int
        Number of endogenous variables
    p : int
        Number of lags
    N : int
        Number of entities

    Examples
    --------
    >>> from panelbox.var import PanelVARData, PanelVAR
    >>>
    >>> # Prepare data
    >>> data = PanelVARData(
    ...     df,
    ...     endog_vars=['gdp', 'inflation'],
    ...     entity_col='country',
    ...     time_col='year',
    ...     lags=2
    ... )
    >>>
    >>> # Estimate
    >>> model = PanelVAR(data)
    >>> results = model.fit(method='ols', cov_type='clustered')
    >>> print(results.summary())
    >>>
    >>> # Check stability
    >>> print(f"Stable: {results.is_stable()}")
    >>>
    >>> # Granger causality
    >>> test = results.test_granger_causality('gdp', 'inflation')
    >>> print(test)

    Notes
    -----
    **Estimation Methods:**

    Currently implements OLS equation-by-equation estimation with within
    transformation (entity fixed effects). Future versions will add:
    - GMM estimation (Arellano-Bond, Arellano-Bover)
    - System GMM
    - Maximum likelihood

    **Within Transformation:**

    The within transformation removes entity-specific fixed effects by
    demeaning each variable within each entity:

    ỹ_it = y_it - ȳ_i

    This is applied to both dependent and independent variables before OLS.

    References
    ----------
    .. [1] Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating
           vector autoregressions with panel data. Econometrica, 1371-1395.
    .. [2] Abrigo, M. R., & Love, I. (2016). Estimation of panel vector
           autoregression in Stata. The Stata Journal, 16(3), 778-804.
    """

    def __init__(self, data: PanelVARData):
        """
        Initialize Panel VAR model.

        Parameters
        ----------
        data : PanelVARData
            Panel VAR data container
        """
        self.data = data
        self.K = data.K
        self.p = data.p
        self.N = data.N

    def fit(self, method: str = "ols", cov_type: str = "clustered", **cov_kwds) -> PanelVARResult:
        """
        Fit the Panel VAR model.

        Parameters
        ----------
        method : str, default='ols'
            Estimation method:
            - 'ols': OLS equation-by-equation with fixed effects
        cov_type : str, default='clustered'
            Type of covariance estimator:
            - 'nonrobust': Classical standard errors
            - 'hc1': Heteroskedasticity-robust (HC1)
            - 'clustered': Cluster-robust by entity (recommended)
            - 'driscoll_kraay': Driscoll-Kraay HAC
            - 'sur': SUR (Seemingly Unrelated Regressions)
        **cov_kwds
            Additional arguments for covariance estimation:
            - max_lags: For Driscoll-Kraay (default: auto)
            - kernel: For HAC estimators ('bartlett', 'parzen', 'quadratic_spectral')

        Returns
        -------
        PanelVARResult
            Estimation results

        Notes
        -----
        **OLS Estimation:**

        For each equation k = 1, ..., K:

        1. Apply within transformation to y_k and X
        2. Estimate β_k = (X'X)^(-1) X'y_k
        3. Compute residuals ε̂_k = y_k - X β_k
        4. Compute covariance matrix using specified method

        **Recommended Covariance Type:**

        For panel data, 'clustered' is recommended to account for within-entity
        correlation. 'driscoll_kraay' is appropriate when cross-sectional
        dependence is suspected.

        Examples
        --------
        >>> # OLS with cluster-robust SE
        >>> results = model.fit(method='ols', cov_type='clustered')
        >>>
        >>> # OLS with Driscoll-Kraay SE
        >>> results = model.fit(method='ols', cov_type='driscoll_kraay', max_lags=3)
        """
        if method != "ols":
            raise NotImplementedError(f"Method '{method}' not implemented yet. Use method='ols'.")

        return self._fit_ols(cov_type=cov_type, **cov_kwds)

    def _fit_ols(self, cov_type: str = "clustered", **cov_kwds) -> PanelVARResult:
        """
        Fit Panel VAR using OLS equation-by-equation with fixed effects.

        Parameters
        ----------
        cov_type : str
            Covariance type
        **cov_kwds
            Additional covariance arguments

        Returns
        -------
        PanelVARResult
            Estimation results
        """
        # Get entity and time identifiers
        df = self.data.to_stacked()
        entities = df[self.data.entity_col].values
        times = df[self.data.time_col].values

        # Storage for results by equation
        params_by_eq = []
        std_errors_by_eq = []
        cov_by_eq = []
        resid_by_eq = []
        fitted_by_eq = []
        X_demeaned_by_eq = []  # Store for later use

        # Estimate each equation separately (first pass)
        for k in range(self.K):
            # Get y and X for equation k
            # IMPORTANT: include_constant=False because within transformation removes constant
            y, X = self.data.equation_data(k, include_constant=False)

            # Apply within transformation (entity demeaning)
            y_demeaned, _ = within_transformation(y, entities)
            X_demeaned, _ = within_transformation(X, entities)

            # OLS estimation on demeaned data
            beta, resid_demeaned, fitted_demeaned = compute_ols_equation(y_demeaned, X_demeaned)

            # Store results (covariance computed later for SUR)
            params_by_eq.append(beta)
            resid_by_eq.append(resid_demeaned)
            fitted_by_eq.append(fitted_demeaned)
            X_demeaned_by_eq.append(X_demeaned)

        # Compute covariance matrices
        if cov_type == "sur":
            # SUR: compute system covariance using all residuals
            residuals_all = np.column_stack(resid_by_eq)  # n x K matrix

            # Use first equation's X (same for all equations in Panel VAR)
            X_demeaned_common = X_demeaned_by_eq[0]

            # Compute SUR system covariance
            vcov_system = compute_covariance_matrix(
                X_demeaned_common,
                None,  # not used for SUR
                cov_type="sur",
                entities=entities,
                times=times,
                residuals_all=residuals_all,
                K=self.K,
                **cov_kwds,
            )

            # Extract per-equation covariances and standard errors
            k_params = X_demeaned_common.shape[1]  # number of parameters per equation
            for k in range(self.K):
                # Extract k-th diagonal block (k_params x k_params)
                start_idx = k * k_params
                end_idx = (k + 1) * k_params
                vcov_k = vcov_system[start_idx:end_idx, start_idx:end_idx]
                std_errors_k = np.sqrt(np.diag(vcov_k))

                cov_by_eq.append(vcov_k)
                std_errors_by_eq.append(std_errors_k)
        else:
            # Non-SUR: compute covariance separately for each equation
            for k in range(self.K):
                # Use stored X_demeaned
                X_demeaned = X_demeaned_by_eq[k]

                # Compute covariance matrix
                vcov = compute_covariance_matrix(
                    X_demeaned,
                    resid_by_eq[k],
                    cov_type,
                    entities=entities,
                    times=times,
                    **cov_kwds,
                )

                # Standard errors
                std_errors = np.sqrt(np.diag(vcov))

                cov_by_eq.append(vcov)
                std_errors_by_eq.append(std_errors)

        # Model information
        model_info = {
            "method": "ols",
            "cov_type": cov_type,
            "cov_kwds": cov_kwds,
            "lags": self.p,
            "trend": self.data.trend,
            "n_exog": len(self.data.exog_vars),
        }

        # Data information
        data_info = {
            "n_entities": self.N,
            "n_obs": self.data.n_obs,
            "is_balanced": self.data.is_balanced,
        }

        # Create result object
        result = PanelVARResult(
            params_by_eq=params_by_eq,
            std_errors_by_eq=std_errors_by_eq,
            cov_by_eq=cov_by_eq,
            resid_by_eq=resid_by_eq,
            fitted_by_eq=fitted_by_eq,
            endog_names=self.data.endog_vars,
            exog_names=self.data.get_regressor_names(include_constant=False),
            model_info=model_info,
            data_info=data_info,
        )

        return result

    def select_lag_order(
        self,
        max_lags: int = 8,
        criteria: Optional[List[str]] = None,
        cov_type: str = "clustered",
        **cov_kwds,
    ) -> LagOrderResult:
        """
        Select optimal lag order using information criteria.

        Parameters
        ----------
        max_lags : int, default=8
            Maximum number of lags to test
        criteria : List[str], optional
            Criteria to use. Default: ['AIC', 'BIC', 'HQIC', 'MBIC']
        cov_type : str, default='clustered'
            Covariance type for estimation
        **cov_kwds
            Additional covariance arguments

        Returns
        -------
        LagOrderResult
            Lag order selection results

        Notes
        -----
        **Information Criteria:**

        For each p = 1, 2, ..., max_lags, compute:

        - AIC  = log|Σ̂| + (2K²p)/NT
        - BIC  = log|Σ̂| + (K²p log(NT))/NT
        - HQIC = log|Σ̂| + (2K²p log(log(NT)))/NT
        - MBIC = log|Σ̂| + (K²p log(NT) log(log(NT)))/NT  (Modified BIC, Andrews & Lu 2001)

        where Σ̂ is the residual covariance matrix.

        **Warning:**

        If max_lags * K > T / 3, a warning is issued because too many
        observations are lost to lag construction.

        Examples
        --------
        >>> # Select lag order
        >>> lag_results = model.select_lag_order(max_lags=8)
        >>> print(lag_results.summary())
        >>>
        >>> # Get BIC-selected lag
        >>> optimal_p = lag_results.selected['BIC']
        >>>
        >>> # Re-estimate with optimal lag
        >>> data_optimal = PanelVARData(df, endog_vars=['y1', 'y2'],
        ...                             entity_col='entity', time_col='time',
        ...                             lags=optimal_p)
        >>> model_optimal = PanelVAR(data_optimal)
        >>> results_optimal = model_optimal.fit()
        """
        if criteria is None:
            criteria = ["AIC", "BIC", "HQIC", "MBIC"]

        # Check if max_lags is too large
        T_avg = self.data.T_avg
        if max_lags * self.K > T_avg / 3:
            import warnings

            warnings.warn(
                f"max_lags={max_lags} is large relative to average T={T_avg:.1f}. "
                f"This will result in loss of many observations. "
                f"Consider reducing max_lags.",
                UserWarning,
            )

        # Store original data
        orig_df = self.data.data
        orig_lags = self.data._lags

        # Test each lag order
        results_list = []

        for p in range(1, max_lags + 1):
            # Create data with p lags
            try:
                data_p = PanelVARData(
                    orig_df,
                    endog_vars=self.data.endog_vars,
                    entity_col=self.data.entity_col,
                    time_col=self.data.time_col,
                    exog_vars=self.data.exog_vars if len(self.data.exog_vars) > 0 else None,
                    lags=p,
                    trend=self.data.trend,
                    dropna=self.data.dropna,
                )
            except Exception as e:
                # If we can't create data (e.g., too few observations), skip
                import warnings

                warnings.warn(f"Could not test p={p}: {e}", UserWarning)
                continue

            # Fit model
            model_p = PanelVAR(data_p)
            result_p = model_p.fit(method="ols", cov_type=cov_type, **cov_kwds)

            # Calculate MBIC (Modified BIC) - Andrews & Lu (2001)
            # MBIC = log|Σ̂| + (K²p · log(NT) · log(log(NT)))/NT
            n_obs = result_p.n_obs
            K = result_p.K
            log_det_sigma = np.log(np.linalg.det(result_p.Sigma))
            n_params_total = K * K * p  # Total parameters in system
            mbic = log_det_sigma + (n_params_total * np.log(n_obs) * np.log(np.log(n_obs))) / n_obs

            # Store criteria
            results_list.append(
                {
                    "lags": p,
                    "AIC": result_p.aic,
                    "BIC": result_p.bic,
                    "HQIC": result_p.hqic,
                    "MBIC": mbic,
                }
            )

        # Create DataFrame
        criteria_df = pd.DataFrame(results_list)

        # Select best lag for each criterion
        selected = {}
        for criterion in criteria:
            if criterion in criteria_df.columns:
                best_idx = criteria_df[criterion].idxmin()
                selected[criterion] = int(criteria_df.loc[best_idx, "lags"])

        return LagOrderResult(criteria_df=criteria_df, selected=selected)

    def __repr__(self) -> str:
        """String representation."""
        return f"PanelVAR(K={self.K}, p={self.p}, N={self.N}, n_obs={self.data.n_obs})"
