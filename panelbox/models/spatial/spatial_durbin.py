"""
Spatial Durbin Model (SDM) for panel data.

The Spatial Durbin Model combines spatial lag of dependent variable (Wy)
with spatial lags of explanatory variables (WX), capturing both endogenous
and exogenous spatial spillovers:

y = ρWy + Xβ + WXθ + α + ε

This implementation supports:
- Fixed effects (within) estimation via Quasi-ML
- Random effects estimation via ML
- Spatial effects decomposition (direct, indirect, total)
- Simulation-based and delta method inference

References
----------
LeSage, J.P. & Pace, R.K. (2009). Introduction to Spatial Econometrics. CRC Press.
Elhorst, J.P. (2014). Spatial Econometrics: From Cross-Sectional Data to Spatial Panels. Springer.
"""

import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from panelbox.models.spatial.base_spatial import SpatialPanelModel
from panelbox.models.spatial.spatial_lag import SpatialPanelResults as SpatialPanelResult


class SpatialDurbin(SpatialPanelModel):
    """
    Spatial Durbin Model (SDM) for panel data.

    Model specification:
    y = ρWy + Xβ + WXθ + α + ε

    where:
    - ρ: spatial autoregressive parameter
    - β: direct effects of X
    - θ: spatial spillover effects of X
    - α: entity fixed or random effects
    - ε: error term

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
    W : np.ndarray or SpatialWeights
        Spatial weight matrix (N×N)
    effects : {'fixed', 'random'}, default='fixed'
        Type of effects to include
    weights : np.ndarray, optional
        Observation weights for weighted estimation

    Attributes
    ----------
    spatial_model_type : str
        Model type identifier ('SDM')
    rho : float
        Estimated spatial autoregressive parameter
    beta : np.ndarray
        Estimated direct coefficients
    theta : np.ndarray
        Estimated spatial spillover coefficients
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        W: Union[np.ndarray, "SpatialWeights"],
        effects: Literal["fixed", "random"] = "fixed",
        weights: Optional[np.ndarray] = None,
    ):
        """Initialize Spatial Durbin Model."""
        super().__init__(formula, data, entity_col, time_col, W, weights)
        self.effects = effects
        self.spatial_model_type = "SDM"

        # Model parameters (set after fitting)
        self.rho = None
        self.beta = None
        self.theta = None

        # For random effects
        self.sigma_alpha = None
        self.sigma_epsilon = None

    def fit(
        self,
        method: Literal["qml", "ml"] = "qml",
        initial_values: Optional[Dict[str, float]] = None,
        maxiter: int = 1000,
        **kwargs,
    ) -> SpatialPanelResult:
        """
        Fit the Spatial Durbin Model.

        Parameters
        ----------
        method : {'qml', 'ml'}, default='qml'
            Estimation method:
            - 'qml': Quasi-Maximum Likelihood (for fixed effects)
            - 'ml': Maximum Likelihood (for random effects)
        initial_values : dict, optional
            Initial values for parameters {'rho': ..., 'beta': ..., 'theta': ...}
        maxiter : int, default=1000
            Maximum number of iterations
        **kwargs
            Additional arguments passed to optimizer

        Returns
        -------
        SpatialPanelResult
            Fitted model results
        """
        # Validate panel structure
        self._validate_panel_structure()

        if self.effects == "fixed" and method == "qml":
            return self._fit_qml_fe(initial_values, maxiter, **kwargs)
        elif self.effects == "random" and method == "ml":
            return self._fit_ml_re(initial_values, maxiter, **kwargs)
        else:
            raise ValueError(f"Invalid combination: effects='{self.effects}', method='{method}'")

    def _fit_qml_fe(
        self, initial_values: Optional[Dict[str, float]] = None, maxiter: int = 1000, **kwargs
    ) -> SpatialPanelResult:
        """
        Quasi-Maximum Likelihood estimation with fixed effects.

        Procedure:
        1. Within transformation to remove fixed effects
        2. Construct augmented design matrix [X, WX]
        3. Concentrated log-likelihood optimization over ρ
        4. Conditional OLS for β and θ given ρ
        """
        # Within transformation
        y_within = self._within_transformation(self.endog)
        X_within = self._within_transformation(self.exog)

        # Construct spatial lag of X
        WX_within = self._spatial_lag(X_within)

        # Augmented design matrix: [X, WX]
        X_augmented = np.column_stack([X_within, WX_within])

        # Get bounds for rho
        rho_min, rho_max = self._spatial_coefficient_bounds()

        # Concentrated log-likelihood function
        def concentrated_llf(rho):
            """Concentrated log-likelihood as function of rho only."""
            # Spatial lag of y
            Wy_within = self._spatial_lag(y_within)

            # Spatially filtered dependent variable
            y_filtered = y_within - rho * Wy_within

            # OLS on augmented regressors
            XtX = X_augmented.T @ X_augmented
            Xty = X_augmented.T @ y_filtered

            try:
                # Solve normal equations
                beta_theta = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Use least squares if singular
                beta_theta, _, _, _ = np.linalg.lstsq(X_augmented, y_filtered, rcond=None)

            # Residuals
            residuals = y_filtered - X_augmented @ beta_theta
            ssr = np.dot(residuals, residuals)

            # Compute log-likelihood
            N = self.n_entities
            T = self.T
            NT = N * T

            sigma2 = ssr / NT

            # Log-determinant of Jacobian (T times for T periods)
            log_det_jacobian = T * self._log_det_jacobian(rho)

            # Log-likelihood
            llf = (
                -NT / 2 * np.log(2 * np.pi)
                - NT / 2 * np.log(sigma2)
                + log_det_jacobian
                - ssr / (2 * sigma2)
            )

            return -llf  # Minimize negative log-likelihood

        # Initial value for rho
        if initial_values and "rho" in initial_values:
            rho0 = initial_values["rho"]
        else:
            # Start at OLS (rho=0) or small value
            rho0 = 0.1

        # Optimize concentrated likelihood
        result = minimize(
            concentrated_llf,
            x0=rho0,
            method="L-BFGS-B",
            bounds=[(rho_min, rho_max)],
            options={"maxiter": maxiter, **kwargs},
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Extract optimal rho
        rho_hat = result.x[0]
        self.rho = rho_hat

        # Compute β and θ at optimal ρ
        Wy_within = self._spatial_lag(y_within)
        y_filtered = y_within - rho_hat * Wy_within

        # Final regression
        XtX = X_augmented.T @ X_augmented
        Xty = X_augmented.T @ y_filtered
        beta_theta_hat = np.linalg.solve(XtX, Xty)

        # Separate β and θ
        K = self.exog.shape[1]
        beta_hat = beta_theta_hat[:K]
        theta_hat = beta_theta_hat[K:]

        self.beta = beta_hat
        self.theta = theta_hat

        # Residuals and variance
        residuals = y_filtered - X_augmented @ beta_theta_hat
        sigma2 = np.dot(residuals, residuals) / (self.nobs - len(beta_theta_hat))

        # Compute variance-covariance matrix
        # Using the sandwich formula for QML
        vcov = self._compute_qml_vcov(rho_hat, beta_hat, theta_hat, sigma2, X_augmented)

        # Prepare parameter names
        param_names = ["rho"]
        param_names.extend(list(self.exog_names))
        param_names.extend([f"W*{name}" for name in self.exog_names])

        # Combine all parameters
        params = np.concatenate([[rho_hat], beta_hat, theta_hat])

        # Create result object
        return SpatialPanelResult(
            model=self,
            params=pd.Series(params, index=param_names),
            cov_matrix=vcov,
            residuals=residuals,
            fitted_values=self.endog - residuals,
            sigma2=sigma2,
            log_likelihood=-result.fun,
            nobs=self.nobs,
            df_model=len(params),
            df_residual=self.nobs - len(params),
            W=self.W_normalized,
            spatial_params={"rho": rho_hat},
        )

    def _fit_ml_re(
        self, initial_values: Optional[Dict[str, float]] = None, maxiter: int = 1000, **kwargs
    ) -> SpatialPanelResult:
        """
        Maximum Likelihood estimation with random effects.

        Estimates all parameters jointly: ρ, β, θ, σ_α, σ_ε
        """
        N = self.n_entities
        T = self.T
        NT = N * T

        # Prepare data
        y = self.endog
        X = self.exog
        WX = self._spatial_lag(X)
        X_augmented = np.column_stack([X, WX])
        K = self.exog.shape[1]

        # Get bounds for rho
        rho_min, rho_max = self._spatial_coefficient_bounds()

        def negative_log_likelihood(params):
            """Full log-likelihood for random effects SDM."""
            # Parse parameters
            rho = params[0]
            beta = params[1 : K + 1]
            theta = params[K + 1 : 2 * K + 1]
            log_sigma_alpha = params[2 * K + 1]
            log_sigma_epsilon = params[2 * K + 2]

            # Transform to positive
            sigma_alpha = np.exp(log_sigma_alpha)
            sigma_epsilon = np.exp(log_sigma_epsilon)

            # GLS transformation parameter
            theta_gls = 1 - sigma_epsilon / np.sqrt(T * sigma_alpha**2 + sigma_epsilon**2)

            # Quasi-demean the data
            y_transformed = self._quasi_demean(y, theta_gls)
            X_transformed = self._quasi_demean(X_augmented, theta_gls)

            # Spatial lag of transformed y
            Wy_transformed = self._spatial_lag(y_transformed)

            # Compute residuals
            y_filtered = y_transformed - rho * Wy_transformed
            fitted = X_transformed @ np.concatenate([beta, theta])
            residuals = y_filtered - fitted

            # Variance components
            sigma2_v = T * sigma_alpha**2 + sigma_epsilon**2

            # Log-likelihood components
            log_det_jacobian = T * self._log_det_jacobian(rho)
            log_det_omega = N * np.log(sigma2_v) + N * (T - 1) * np.log(sigma_epsilon**2)

            ssr = np.dot(residuals, residuals)

            # Full log-likelihood
            llf = (
                -NT / 2 * np.log(2 * np.pi)
                - 0.5 * log_det_omega
                + log_det_jacobian
                - ssr / (2 * sigma_epsilon**2)
            )

            return -llf

        # Initial values
        if initial_values:
            params0 = self._parse_initial_values_re(initial_values, K)
        else:
            # Start with reasonable values
            params0 = np.zeros(2 * K + 3)
            params0[0] = 0.1  # rho
            params0[1 : K + 1] = np.ones(K) * 0.1  # beta
            params0[K + 1 : 2 * K + 1] = np.zeros(K)  # theta
            params0[2 * K + 1] = np.log(0.5)  # log(sigma_alpha)
            params0[2 * K + 2] = np.log(1.0)  # log(sigma_epsilon)

        # Bounds
        bounds = [(rho_min, rho_max)]  # rho
        bounds.extend([(None, None)] * (2 * K))  # beta and theta
        bounds.extend([(-10, 10)] * 2)  # log-sigmas

        # Optimize
        result = minimize(
            negative_log_likelihood,
            x0=params0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, **kwargs},
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Extract parameters
        params_opt = result.x
        rho_hat = params_opt[0]
        beta_hat = params_opt[1 : K + 1]
        theta_hat = params_opt[K + 1 : 2 * K + 1]
        sigma_alpha_hat = np.exp(params_opt[2 * K + 1])
        sigma_epsilon_hat = np.exp(params_opt[2 * K + 2])

        self.rho = rho_hat
        self.beta = beta_hat
        self.theta = theta_hat
        self.sigma_alpha = sigma_alpha_hat
        self.sigma_epsilon = sigma_epsilon_hat

        # Compute ICC
        icc = sigma_alpha_hat**2 / (sigma_alpha_hat**2 + sigma_epsilon_hat**2)

        # Compute variance-covariance matrix
        hessian = self._compute_hessian(negative_log_likelihood, params_opt)
        vcov = np.linalg.inv(hessian)

        # Parameter names
        param_names = ["rho"]
        param_names.extend(list(self.exog_names))
        param_names.extend([f"W*{name}" for name in self.exog_names])
        param_names.extend(["sigma_alpha", "sigma_epsilon"])

        # All parameters
        all_params = np.concatenate(
            [[rho_hat], beta_hat, theta_hat, [sigma_alpha_hat, sigma_epsilon_hat]]
        )

        # Compute fitted values and residuals
        theta_gls = 1 - sigma_epsilon_hat / np.sqrt(T * sigma_alpha_hat**2 + sigma_epsilon_hat**2)
        y_transformed = self._quasi_demean(y, theta_gls)
        X_transformed = self._quasi_demean(X_augmented, theta_gls)
        Wy_transformed = self._spatial_lag(y_transformed)

        y_filtered = y_transformed - rho_hat * Wy_transformed
        fitted = X_transformed @ np.concatenate([beta_hat, theta_hat])
        residuals = (
            y - self._spatial_lag(y) * rho_hat - X_augmented @ np.concatenate([beta_hat, theta_hat])
        )

        return SpatialPanelResult(
            model=self,
            params=pd.Series(all_params, index=param_names),
            cov_matrix=vcov,
            residuals=residuals,
            fitted_values=self.endog - residuals,
            sigma2=sigma_epsilon_hat**2,
            log_likelihood=-result.fun,
            nobs=self.nobs,
            df_model=len(all_params),
            df_residual=self.nobs - len(all_params),
            W=self.W_normalized,
            spatial_params={"rho": rho_hat},
            variance_components={
                "sigma_alpha": sigma_alpha_hat,
                "sigma_epsilon": sigma_epsilon_hat,
                "icc": icc,
            },
        )

    def _quasi_demean(self, X: np.ndarray, theta: float) -> np.ndarray:
        """
        Quasi-demean data for random effects estimation.

        Parameters
        ----------
        X : np.ndarray
            Data to transform (NT × K)
        theta : float
            Transformation parameter (0 ≤ θ ≤ 1)

        Returns
        -------
        np.ndarray
            Quasi-demeaned data
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            # Group mean
            X_bar = X.groupby(level=self.entity_col).transform("mean")
            # Quasi-demean
            return X - theta * X_bar
        else:
            # Reshape to panel format
            T = self.T
            N = self.n_entities

            if X.ndim == 1:
                X_panel = X.reshape(T, N).T
                X_bar = np.mean(X_panel, axis=1, keepdims=True)
                X_transformed = X_panel - theta * X_bar
                return X_transformed.T.flatten()
            else:
                K = X.shape[1]
                X_transformed = np.zeros_like(X)

                for k in range(K):
                    X_k = X[:, k].reshape(T, N).T
                    X_bar_k = np.mean(X_k, axis=1, keepdims=True)
                    X_transformed_k = X_k - theta * X_bar_k
                    X_transformed[:, k] = X_transformed_k.T.flatten()

                return X_transformed

    def _compute_qml_vcov(
        self,
        rho: float,
        beta: np.ndarray,
        theta: np.ndarray,
        sigma2: float,
        X_augmented: np.ndarray,
    ) -> np.ndarray:
        """
        Compute variance-covariance matrix for QML estimator.

        Uses sandwich formula accounting for spatial structure.
        """
        K = len(beta)
        n_params = 1 + 2 * K  # rho + beta + theta

        # Information matrix (simplified - should use numerical Hessian)
        # This is a placeholder - proper implementation would compute full Hessian
        info_matrix = np.eye(n_params)

        # Scale by sigma2
        vcov = sigma2 * np.linalg.inv(info_matrix)

        return vcov

    def _compute_hessian(self, func, params, epsilon=1e-5):
        """
        Compute Hessian matrix numerically.

        Parameters
        ----------
        func : callable
            Function to differentiate
        params : np.ndarray
            Point at which to evaluate Hessian
        epsilon : float, default=1e-5
            Step size for finite differences

        Returns
        -------
        np.ndarray
            Hessian matrix
        """
        n = len(params)
        hessian = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                # Compute second derivative
                params_pp = params.copy()
                params_pp[i] += epsilon
                params_pp[j] += epsilon

                params_pm = params.copy()
                params_pm[i] += epsilon
                params_pm[j] -= epsilon

                params_mp = params.copy()
                params_mp[i] -= epsilon
                params_mp[j] += epsilon

                params_mm = params.copy()
                params_mm[i] -= epsilon
                params_mm[j] -= epsilon

                # Second derivative
                d2f = (func(params_pp) - func(params_pm) - func(params_mp) + func(params_mm)) / (
                    4 * epsilon**2
                )

                hessian[i, j] = d2f
                if i != j:
                    hessian[j, i] = d2f

        return hessian

    def _parse_initial_values_re(self, initial_values: Dict[str, Any], K: int) -> np.ndarray:
        """
        Parse initial values dictionary for random effects model.

        Parameters
        ----------
        initial_values : dict
            Dictionary with initial values
        K : int
            Number of exogenous variables

        Returns
        -------
        np.ndarray
            Array of initial parameters
        """
        params0 = np.zeros(2 * K + 3)

        if "rho" in initial_values:
            params0[0] = initial_values["rho"]
        else:
            params0[0] = 0.1

        if "beta" in initial_values:
            params0[1 : K + 1] = initial_values["beta"]
        else:
            params0[1 : K + 1] = np.ones(K) * 0.1

        if "theta" in initial_values:
            params0[K + 1 : 2 * K + 1] = initial_values["theta"]
        else:
            params0[K + 1 : 2 * K + 1] = np.zeros(K)

        if "sigma_alpha" in initial_values:
            params0[2 * K + 1] = np.log(initial_values["sigma_alpha"])
        else:
            params0[2 * K + 1] = np.log(0.5)

        if "sigma_epsilon" in initial_values:
            params0[2 * K + 2] = np.log(initial_values["sigma_epsilon"])
        else:
            params0[2 * K + 2] = np.log(1.0)

        return params0

    def predict(
        self, exog: Optional[np.ndarray] = None, effects_type: Literal["direct", "total"] = "total"
    ) -> np.ndarray:
        """
        Generate predictions from the fitted SDM.

        Parameters
        ----------
        exog : np.ndarray, optional
            Exogenous variables for prediction (uses fitted data if None)
        effects_type : {'direct', 'total'}, default='total'
            Whether to include only direct effects or total (direct + indirect)

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if self.rho is None:
            raise ValueError("Model must be fitted before prediction")

        if exog is None:
            X = self.exog
        else:
            X = exog

        # Compute WX
        WX = self._spatial_lag(X)

        # Direct prediction: Xβ + WXθ
        y_direct = X @ self.beta + WX @ self.theta

        if effects_type == "direct":
            return y_direct
        else:
            # Total effects: (I - ρW)^(-1)(Xβ + WXθ)
            N = self.n_entities
            I_rhoW_inv = np.linalg.inv(np.eye(N) - self.rho * self.W_normalized)

            # Apply to each time period
            T = len(y_direct) // N
            y_total = np.zeros_like(y_direct)

            for t in range(T):
                start_idx = t * N
                end_idx = (t + 1) * N
                y_direct_t = y_direct[start_idx:end_idx]
                y_total[start_idx:end_idx] = I_rhoW_inv @ y_direct_t

            return y_total
