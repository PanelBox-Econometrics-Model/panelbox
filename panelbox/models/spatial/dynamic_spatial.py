"""
Dynamic Spatial Panel Model.

This module implements the Dynamic Spatial Panel Model following
Yu, de Jong, and Lee (2008) using Generalized Method of Moments (GMM).

Model specification:
    y_it = γ y_{i,t-1} + ρ W y_it + X_it β + α_i + ε_it

References
----------
Yu, J., de Jong, R., & Lee, L.F. (2008). "Quasi-maximum likelihood estimators
    for spatial dynamic panel data with fixed effects when both n and T are large."
    Journal of Econometrics, 146(1), 118-134.

Lee, L.F. & Yu, J. (2010). "Estimation of spatial autoregressive panel data models
    with fixed effects." Journal of Econometrics, 154(2), 165-185.
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg, optimize
from scipy.sparse import issparse

from .base_spatial import SpatialPanelModel
from .spatial_lag import SpatialPanelResults


class DynamicSpatialPanel(SpatialPanelModel):
    """
    Dynamic Spatial Panel Model with spatial and temporal dependencies.

    This model combines temporal dynamics (lagged dependent variable) with
    spatial spillovers (spatial lag of dependent variable).

    Model:
        y_it = γ y_{i,t-1} + ρ W y_it + X_it β + α_i + ε_it

    where:
        - γ: temporal persistence parameter
        - ρ: spatial spillover parameter
        - W: spatial weight matrix
        - α_i: entity fixed effects
        - ε_it: idiosyncratic errors

    Parameters
    ----------
    formula : str
        Patsy formula for the model
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time)
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    W : np.ndarray
        Spatial weight matrix (N×N)
    weights : np.ndarray, optional
        Observation weights for weighted estimation

    Attributes
    ----------
    gamma : float
        Temporal persistence parameter
    rho : float
        Spatial spillover parameter
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        W: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ):
        """Initialize Dynamic Spatial Panel model."""
        super().__init__(formula, data, entity_col, time_col, W, weights)
        self.model_type = "Dynamic Spatial Panel"
        self.gamma = None
        self.rho = None

    def fit(
        self,
        effects: Literal["fixed", "random"] = "fixed",
        method: Literal["gmm", "qml"] = "gmm",
        lags: int = 1,
        spatial_lags: int = 1,
        time_lags: int = 2,
        initial_values: Optional[Dict] = None,
        maxiter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
        **kwargs,
    ) -> SpatialPanelResults:
        """
        Fit Dynamic Spatial Panel Model.

        Parameters
        ----------
        effects : {'fixed', 'random'}
            Type of panel effects
        method : {'gmm', 'qml'}
            Estimation method
        lags : int
            Number of temporal lags of dependent variable
        spatial_lags : int
            Number of spatial lags to include
        time_lags : int
            Maximum time lag for instruments
        initial_values : dict, optional
            Initial values for parameters {'gamma': ..., 'rho': ..., 'beta': ...}
        maxiter : int
            Maximum iterations for optimization
        tol : float
            Convergence tolerance
        verbose : bool
            Print optimization progress

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        if method == "gmm":
            return self._fit_gmm(
                effects,
                lags,
                spatial_lags,
                time_lags,
                initial_values,
                maxiter,
                tol,
                verbose,
                **kwargs,
            )
        elif method == "qml":
            return self._fit_qml(effects, lags, initial_values, maxiter, tol, verbose, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _fit_gmm(
        self,
        effects,
        lags,
        spatial_lags,
        time_lags,
        initial_values,
        maxiter,
        tol,
        verbose,
        **kwargs,
    ):
        """
        GMM estimation for Dynamic Spatial Panel.

        Uses moment conditions based on:
        - Lagged values as instruments for temporal lag
        - Spatial lags of X as instruments for spatial lag
        """
        # Prepare data
        y, X = self.prepare_data(effects)
        N = self.n_entities
        T = self.n_periods

        if T <= time_lags + lags:
            raise ValueError(
                f"Need T > {time_lags + lags} for GMM with {lags} lags "
                f"and {time_lags} instrument lags"
            )

        # Create lagged dependent variable
        y_lag = self._create_temporal_lag(y, N, T, lags)

        # Create spatial lag of y
        Wy = self._create_spatial_lag(y, N, T)

        # Remove first time periods (lost to lagging)
        valid_periods = T - lags
        valid_obs = valid_periods * N

        # Indices for valid observations
        start_idx = lags * N
        y_valid = y[start_idx:]
        X_valid = X[start_idx:]
        y_lag_valid = y_lag[start_idx:]
        Wy_valid = Wy[start_idx:]

        # Construct instruments
        Z = self._construct_instruments(y, X, N, T, lags, spatial_lags, time_lags)

        # Ensure Z has same number of rows as valid observations
        Z_valid = Z[start_idx : start_idx + valid_obs]

        if verbose:
            print(f"GMM estimation with {valid_obs} observations, {Z_valid.shape[1]} instruments")

        # First stage: 2SLS for initial consistent estimates
        initial_params = self._gmm_first_stage(
            y_valid, X_valid, y_lag_valid, Wy_valid, Z_valid, initial_values
        )

        if verbose:
            print(
                f"First stage estimates: γ={initial_params['gamma']:.3f}, "
                f"ρ={initial_params['rho']:.3f}"
            )

        # Second stage: Optimal GMM with efficient weight matrix
        final_params = self._gmm_second_stage(
            y_valid, X_valid, y_lag_valid, Wy_valid, Z_valid, initial_params, maxiter, tol, verbose
        )

        # Compute standard errors
        cov_matrix = self._gmm_covariance_matrix(
            y_valid, X_valid, y_lag_valid, Wy_valid, Z_valid, final_params
        )

        # Extract parameters
        self.gamma = final_params["gamma"]
        self.rho = final_params["rho"]
        beta = final_params["beta"]

        # Organize results
        k_vars = len(beta)
        param_names = ["gamma", "rho"] + [f"beta_{i}" for i in range(k_vars)]
        param_values = np.concatenate([[self.gamma, self.rho], beta])

        std_errors = np.sqrt(np.diag(cov_matrix))
        t_stats = param_values / std_errors

        params_df = pd.DataFrame(
            {
                "coefficient": param_values,
                "std_error": std_errors,
                "t_statistic": t_stats,
                "p_value": 2
                * (
                    1
                    - pd.Series(np.abs(t_stats)).apply(
                        lambda x: pd.Series([x]).apply(
                            lambda z: 1
                            - 0.5 * (1 + np.sign(z) * (1 - np.exp(-2 * z**2 / np.pi) ** 0.5))
                        )[0]
                    )
                ),
            },
            index=param_names,
        )

        # Compute fitted values and residuals
        y_fitted = self.gamma * y_lag_valid + self.rho * Wy_valid + X_valid @ beta
        residuals = y_valid - y_fitted

        # Hansen J-test for overidentifying restrictions
        j_stat, j_pval = self._hansen_j_test(residuals, Z_valid, cov_matrix)

        # Create results object
        result = SpatialPanelResults(
            model=self,
            params=params_df,
            fitted_values=y_fitted,
            residuals=residuals,
            entity_effects=None,
            time_effects=None,
            cov_params=cov_matrix,
            method="GMM",
            effects=effects,
            nobs=valid_obs,
            df_model=k_vars + 2,  # gamma, rho, betas
            df_resid=valid_obs - k_vars - 2,
            llf=None,  # Not applicable for GMM
            sigma2=np.var(residuals),
        )

        # Add GMM-specific diagnostics
        result.j_statistic = j_stat
        result.j_pvalue = j_pval
        result.n_instruments = Z_valid.shape[1]
        result.gamma = self.gamma
        result.rho = self.rho

        return result

    def _create_temporal_lag(self, y: np.ndarray, N: int, T: int, lags: int) -> np.ndarray:
        """Create temporal lag of dependent variable."""
        y_reshape = y.reshape(T, N).T  # (N, T)
        y_lag = np.zeros_like(y_reshape)

        for lag in range(1, lags + 1):
            if lag < T:
                y_lag[:, lag:] = y_reshape[:, :-lag]

        return y_lag.T.reshape(-1)  # Back to (NT, 1) format

    def _create_spatial_lag(self, y: np.ndarray, N: int, T: int) -> np.ndarray:
        """Create spatial lag of dependent variable."""
        Wy = np.zeros_like(y)

        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            y_t = y[start_idx:end_idx]
            Wy[start_idx:end_idx] = self.W_normalized @ y_t

        return Wy

    def _construct_instruments(
        self,
        y: np.ndarray,
        X: np.ndarray,
        N: int,
        T: int,
        lags: int,
        spatial_lags: int,
        time_lags: int,
    ) -> np.ndarray:
        """
        Construct GMM instruments.

        Instruments include:
        - Lagged values of y (y_{i,t-2}, y_{i,t-3}, ...)
        - Spatial lags of X (WX, W²X, ...)
        - Lagged values of X
        - Interactions of spatial and temporal lags
        """
        instruments = []

        # Time lags of y as instruments (starting from t-2 to avoid endogeneity)
        for lag in range(2, min(time_lags + 1, T)):
            y_lag = self._create_temporal_lag(y, N, T, lag)
            instruments.append(y_lag.reshape(-1, 1))

        # Current X
        instruments.append(X)

        # Spatial lags of X
        W_power = self.W_normalized.copy()
        for s in range(1, spatial_lags + 1):
            WX = np.zeros_like(X)
            for t in range(T):
                start_idx = t * N
                end_idx = (t + 1) * N
                X_t = X[start_idx:end_idx]
                WX[start_idx:end_idx] = W_power @ X_t

            instruments.append(WX)

            # Update W_power for next iteration
            if s < spatial_lags:
                W_power = W_power @ self.W_normalized

        # Combine instruments
        Z = np.hstack(instruments)

        return Z

    def _gmm_first_stage(self, y, X, y_lag, Wy, Z, initial_values):
        """First stage 2SLS estimation for initial consistent estimates."""

        # Stack endogenous regressors
        W_endo = np.column_stack([y_lag, Wy])

        # First stage regression: W_endo = Z * Pi + v
        Pi_hat = np.linalg.lstsq(Z, W_endo, rcond=None)[0]
        W_endo_hat = Z @ Pi_hat

        # Stack all regressors
        X_all = np.column_stack([W_endo_hat, X])

        # Second stage regression: y = X_all * theta + e
        theta_hat = np.linalg.lstsq(X_all, y, rcond=None)[0]

        # Parse parameters
        gamma_hat = theta_hat[0]
        rho_hat = theta_hat[1]
        beta_hat = theta_hat[2:]

        return {"gamma": gamma_hat, "rho": rho_hat, "beta": beta_hat}

    def _gmm_second_stage(self, y, X, y_lag, Wy, Z, initial_params, maxiter, tol, verbose):
        """Second stage optimal GMM estimation."""

        # Compute initial residuals
        gamma_init = initial_params["gamma"]
        rho_init = initial_params["rho"]
        beta_init = initial_params["beta"]

        residuals_init = y - gamma_init * y_lag - rho_init * Wy - X @ beta_init

        # Estimate optimal weight matrix
        W_opt = self._gmm_weight_matrix(residuals_init, Z)

        # GMM objective function
        def gmm_objective(params):
            gamma, rho = params[:2]
            beta = params[2:]

            # Residuals
            resid = y - gamma * y_lag - rho * Wy - X @ beta

            # Moment conditions
            moments = Z.T @ resid / len(resid)

            # GMM objective
            obj = moments.T @ W_opt @ moments

            return obj

        # Initial values for optimization
        params_init = np.concatenate([[gamma_init, rho_init], beta_init])

        # Bounds
        bounds = [(-0.99, 0.99), (-0.99, 0.99)]  # gamma, rho
        bounds += [(None, None)] * len(beta_init)  # beta

        # Optimize
        result = optimize.minimize(
            gmm_objective,
            params_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": tol, "disp": verbose},
        )

        if not result.success and verbose:
            warnings.warn(f"GMM optimization did not converge: {result.message}")

        # Extract optimized parameters
        gamma_opt = result.x[0]
        rho_opt = result.x[1]
        beta_opt = result.x[2:]

        return {"gamma": gamma_opt, "rho": rho_opt, "beta": beta_opt}

    def _gmm_weight_matrix(self, residuals: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Compute optimal GMM weight matrix."""

        n_obs = len(residuals)

        # Moment conditions at observation level
        g_i = Z * residuals.reshape(-1, 1)

        # Covariance of moments
        S = g_i.T @ g_i / n_obs

        # Optimal weight matrix
        try:
            W_opt = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            warnings.warn("Moment covariance matrix is singular, using pseudo-inverse")
            W_opt = np.linalg.pinv(S)

        return W_opt

    def _gmm_covariance_matrix(self, y, X, y_lag, Wy, Z, params):
        """Compute GMM covariance matrix for parameter estimates."""

        gamma = params["gamma"]
        rho = params["rho"]
        beta = params["beta"]

        n_obs = len(y)
        k_params = 2 + len(beta)  # gamma, rho, beta

        # Residuals
        residuals = y - gamma * y_lag - rho * Wy - X @ beta

        # Gradient of moment conditions w.r.t. parameters
        # G = d(Z'g)/dθ where g = residuals
        G = np.zeros((Z.shape[1], k_params))

        # Derivatives w.r.t. gamma
        G[:, 0] = -Z.T @ y_lag / n_obs

        # Derivatives w.r.t. rho
        G[:, 1] = -Z.T @ Wy / n_obs

        # Derivatives w.r.t. beta
        for j in range(len(beta)):
            G[:, 2 + j] = -Z.T @ X[:, j] / n_obs

        # Weight matrix
        W = self._gmm_weight_matrix(residuals, Z)

        # Covariance matrix: (G'WG)^{-1}
        GWG = G.T @ W @ G

        try:
            cov_matrix = np.linalg.inv(GWG) / n_obs
        except np.linalg.LinAlgError:
            warnings.warn("Cannot compute covariance matrix, using identity")
            cov_matrix = np.eye(k_params)

        return cov_matrix

    def _hansen_j_test(
        self, residuals: np.ndarray, Z: np.ndarray, cov_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """
        Hansen J-test for overidentifying restrictions.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        Z : np.ndarray
            Instrument matrix
        cov_matrix : np.ndarray
            Covariance matrix of parameters

        Returns
        -------
        j_statistic : float
            J-statistic
        p_value : float
            P-value from chi-squared distribution
        """
        n_obs = len(residuals)
        n_instruments = Z.shape[1]
        n_params = cov_matrix.shape[0]

        if n_instruments <= n_params:
            # Model is exactly identified or underidentified
            return np.nan, np.nan

        # Moment conditions
        moments = Z.T @ residuals / n_obs

        # Weight matrix
        W = self._gmm_weight_matrix(residuals, Z)

        # J-statistic
        j_stat = n_obs * moments.T @ W @ moments

        # Degrees of freedom
        df = n_instruments - n_params

        # P-value from chi-squared distribution
        from scipy.stats import chi2

        p_value = 1 - chi2.cdf(j_stat, df)

        return j_stat, p_value

    def _fit_qml(self, effects, lags, initial_values, maxiter, tol, verbose, **kwargs):
        """
        Quasi-Maximum Likelihood estimation for Dynamic Spatial Panel.

        This is more complex than GMM and follows Lee & Yu (2010).
        """
        raise NotImplementedError(
            "QML estimation for Dynamic Spatial Panel " "is not yet implemented. Use method='gmm'."
        )

    def predict(self, steps: int = 1, X_future: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Multi-step ahead prediction for dynamic spatial model.

        Parameters
        ----------
        steps : int
            Number of steps ahead to predict
        X_future : np.ndarray, optional
            Future values of exogenous variables

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if self.gamma is None or self.rho is None:
            raise ValueError("Model must be fitted before prediction")

        # Get last period values
        y_last = self.endog.values[-self.n_entities :].flatten()

        predictions = []

        for step in range(steps):
            # Temporal component
            y_lag_component = self.gamma * y_last

            # Spatial component
            Wy = self.W_normalized @ y_last
            spatial_component = self.rho * Wy

            # Exogenous component
            if X_future is not None and step < X_future.shape[0]:
                X_step = X_future[step]
                exog_component = X_step @ self.last_result.params.values[2:]  # Skip gamma, rho
            else:
                exog_component = 0

            # Prediction
            y_pred = y_lag_component + spatial_component + exog_component

            predictions.append(y_pred)
            y_last = y_pred  # Update for next step

        return np.array(predictions)

    def compute_impulse_response(self, shock_entity: int, periods: int = 10) -> np.ndarray:
        """
        Compute spatial-temporal impulse response function.

        Shows how a shock to one entity propagates through space and time.

        Parameters
        ----------
        shock_entity : int
            Entity index receiving the initial shock
        periods : int
            Number of periods to compute response

        Returns
        -------
        np.ndarray
            Impulse responses (periods × N)
        """
        if self.gamma is None or self.rho is None:
            raise ValueError("Model must be fitted first")

        N = self.n_entities
        responses = np.zeros((periods, N))

        # Initial shock
        shock = np.zeros(N)
        shock[shock_entity] = 1
        responses[0] = shock

        # Propagate through time
        for t in range(1, periods):
            # Temporal persistence + spatial spillover
            responses[t] = (
                self.gamma * responses[t - 1] + self.rho * self.W_normalized @ responses[t - 1]
            )

        return responses
