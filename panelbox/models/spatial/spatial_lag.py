"""
Spatial Lag Model (SAR) for panel data.

This module implements the Spatial Autoregressive Model (SAR) with
fixed effects using Quasi-Maximum Likelihood estimation following
Lee & Yu (2010).
"""

from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.models.discrete.results import PanelResults

from .base_spatial import SpatialPanelModel


class SpatialLag(SpatialPanelModel):
    """
    Spatial Lag Model (SAR) for panel data.

    The model is specified as:
        y = ρWy + Xβ + α + ε

    where:
        - y is the dependent variable
        - W is the spatial weight matrix
        - ρ is the spatial autoregressive parameter
        - X are exogenous variables
        - β are coefficients
        - α are fixed/random effects
        - ε are errors

    Parameters
    ----------
    endog : array-like
        Dependent variable
    exog : array-like
        Independent variables
    W : np.ndarray or SpatialWeights
        Spatial weight matrix (N×N)
    entity_id : array-like, optional
        Entity identifiers
    time_id : array-like, optional
        Time identifiers
    weights : array-like, optional
        Observation weights
    """

    def __init__(self, *args, **kwargs):
        """Initialize SAR model."""
        super().__init__(*args, **kwargs)
        self.model_type = "SAR"

    def fit(
        self,
        effects: str = "fixed",
        method: str = "qml",
        rho_grid_size: int = 20,
        optimizer: str = "brent",
        maxiter: int = 1000,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Fit the Spatial Lag Model.

        Parameters
        ----------
        effects : str
            Type of effects: 'fixed', 'random', or 'pooled'
        method : str
            Estimation method: 'qml' (Quasi-ML), 'ml' (Full ML), or 'gmm'
        rho_grid_size : int
            Number of grid points for initial ρ search
        optimizer : str
            Optimization method: 'brent' or 'l-bfgs-b'
        maxiter : int
            Maximum iterations for optimization
        verbose : bool
            Print optimization progress

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        if effects == "fixed" and method == "qml":
            return self._fit_qml_fe(rho_grid_size, optimizer, maxiter, verbose, **kwargs)
        elif effects == "pooled" and method == "qml":
            return self._fit_qml_pooled(rho_grid_size, optimizer, maxiter, verbose, **kwargs)
        elif effects == "random" and method == "ml":
            return self._fit_ml_re(**kwargs)
        else:
            raise NotImplementedError(
                f"Combination effects='{effects}' and method='{method}' " "not yet implemented"
            )

    def _fit_qml_fe(
        self,
        rho_grid_size: int = 20,
        optimizer: str = "brent",
        maxiter: int = 1000,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Quasi-ML estimation for SAR with fixed effects.

        Based on Lee & Yu (2010) approach:
        1. Within transformation to remove fixed effects
        2. Concentrated log-likelihood for ρ
        3. Compute β conditional on ρ

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        if verbose:
            print("Estimating SAR-FE using Quasi-ML (Lee & Yu 2010)")

        # Apply within transformation
        y_within = self._within_transformation(self.endog.values.reshape(-1, 1)).flatten()
        X_within = self._within_transformation(self.exog)

        # Get bounds for spatial parameter
        rho_bounds = self._spatial_coefficient_bounds()

        # Define concentrated log-likelihood function
        def concentrated_llf(rho):
            """Concentrated log-likelihood as function of ρ only."""
            # Compute spatial lag of y
            Wy = self._spatial_lag(y_within.reshape(-1, 1)).flatten()

            # Transform y: (I - ρW)y
            y_rho = y_within - rho * Wy

            # OLS regression of transformed y on X
            XtX = X_within.T @ X_within
            Xty = X_within.T @ y_rho

            try:
                beta_rho = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Use least squares if singular
                beta_rho, _, _, _ = np.linalg.lstsq(X_within, y_rho, rcond=None)

            # Compute residuals
            residuals = y_rho - X_within @ beta_rho
            ssr = residuals @ residuals
            sigma2 = ssr / self.n_obs

            # Log-determinant term
            log_det = self._log_det_jacobian(rho)

            # Concentrated log-likelihood
            # For panel: T × log|I - ρW| term
            llf = (-self.n_obs / 2) * np.log(2 * np.pi * sigma2)
            llf += self.n_periods * log_det
            llf -= ssr / (2 * sigma2)

            return llf

        # Grid search for initial value
        if verbose:
            print(
                f"Grid search over {rho_grid_size} points in [{rho_bounds[0]:.3f}, {rho_bounds[1]:.3f}]"
            )

        rho_grid = np.linspace(rho_bounds[0] * 0.95, rho_bounds[1] * 0.95, rho_grid_size)
        llf_grid = []

        for rho in rho_grid:
            try:
                llf = concentrated_llf(rho)
                llf_grid.append(llf)
            except:
                llf_grid.append(-np.inf)

        llf_grid = np.array(llf_grid)
        rho_init = rho_grid[np.argmax(llf_grid)]

        if verbose:
            print(f"Initial ρ from grid search: {rho_init:.4f}")

        # Optimization
        if optimizer == "brent":
            # Scalar optimization using Brent's method
            result = minimize_scalar(
                lambda r: -concentrated_llf(r),
                bounds=rho_bounds,
                method="bounded",
                options={"xatol": 1e-8, "maxiter": maxiter},
            )
            rho_hat = result.x
            llf_opt = -result.fun

        elif optimizer == "l-bfgs-b":
            # Use L-BFGS-B
            result = minimize(
                lambda r: -concentrated_llf(r[0]),
                x0=[rho_init],
                method="L-BFGS-B",
                bounds=[rho_bounds],
                options={"maxiter": maxiter},
            )
            rho_hat = result.x[0]
            llf_opt = -result.fun

        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        if verbose:
            print(f"Optimized ρ: {rho_hat:.6f}")
            print(f"Log-likelihood: {llf_opt:.2f}")

        # Compute β at optimal ρ
        Wy = self._spatial_lag(y_within.reshape(-1, 1)).flatten()
        y_rho = y_within - rho_hat * Wy

        XtX = X_within.T @ X_within
        Xty = X_within.T @ y_rho
        beta_hat = np.linalg.solve(XtX, Xty)

        # Compute residuals and variance
        residuals = y_rho - X_within @ beta_hat
        sigma2_hat = (residuals @ residuals) / (self.n_obs - len(beta_hat))

        # Compute standard errors
        cov_matrix = self._qml_fe_covariance(rho_hat, beta_hat, y_within, X_within, Wy, sigma2_hat)

        # Prepare results
        params = np.concatenate([[rho_hat], beta_hat])

        # Get parameter names
        if hasattr(self.exog, "columns"):
            param_names = ["rho"] + list(self.exog.columns)
        else:
            param_names = ["rho"] + [f"x{i}" for i in range(self.exog.shape[1])]

        # Create results object
        results = SpatialPanelResults(
            model=self,
            params=pd.Series(params, index=param_names),
            cov_params=pd.DataFrame(cov_matrix, index=param_names, columns=param_names),
            llf=llf_opt,
            nobs=self.n_obs,
            df_model=len(params),
            df_resid=self.n_obs - len(params) - self.n_entities,  # Account for FE
            method="Quasi-ML (Lee & Yu 2010)",
            effects=effects,
            resid=residuals,
            sigma2=sigma2_hat,
        )

        # Compute spillover effects
        results.spillover_effects = self._compute_spillover_effects(
            dict(zip(param_names, params)), param_names[1:], "rho"  # Exclude rho
        )

        self.results = results
        self.fitted = True

        return results

    def _qml_fe_covariance(
        self,
        rho: float,
        beta: np.ndarray,
        y: np.ndarray,
        X: np.ndarray,
        Wy: np.ndarray,
        sigma2: float,
    ) -> np.ndarray:
        """
        Compute covariance matrix for Quasi-ML fixed effects estimator.

        Uses sandwich estimator for robustness.

        Parameters
        ----------
        rho : float
            Spatial parameter estimate
        beta : np.ndarray
            Coefficient estimates
        y : np.ndarray
            Within-transformed dependent variable
        X : np.ndarray
            Within-transformed independent variables
        Wy : np.ndarray
            Spatial lag of y
        sigma2 : float
            Error variance estimate

        Returns
        -------
        np.ndarray
            Covariance matrix for [ρ, β']
        """
        k = len(beta)

        # Build augmented design matrix Z = [Wy, X]
        Z = np.column_stack([Wy, X])

        # Information matrix approximation
        # V ≈ σ² (Z'Z)⁻¹
        try:
            ZtZ_inv = inv(Z.T @ Z)
            cov_simple = sigma2 * ZtZ_inv
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            ZtZ_inv = np.linalg.pinv(Z.T @ Z)
            cov_simple = sigma2 * ZtZ_inv

        # Add correction for finite sample
        # Based on Lee & Yu (2010) bias correction
        n = self.n_entities
        T = self.n_periods
        correction = n * T / (n * T - n - k - 1)

        cov_matrix = cov_simple * correction

        return cov_matrix

    def _fit_qml_pooled(
        self,
        rho_grid_size: int = 20,
        optimizer: str = "brent",
        maxiter: int = 1000,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Quasi-ML estimation for pooled SAR model (no effects).

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        if verbose:
            print("Estimating pooled SAR using Quasi-ML")

        y = self.endog.values.flatten()
        X = self.exog

        # Add constant if not present
        if not np.any(np.all(X == X[0, :], axis=0)):
            X = np.column_stack([np.ones(len(y)), X])

        # Get bounds
        rho_bounds = self._spatial_coefficient_bounds()

        # Define concentrated log-likelihood
        def concentrated_llf(rho):
            """Concentrated log-likelihood for pooled model."""
            Wy = self._spatial_lag(y.reshape(-1, 1)).flatten()
            y_rho = y - rho * Wy

            # OLS
            beta_rho, _, _, _ = np.linalg.lstsq(X, y_rho, rcond=None)
            residuals = y_rho - X @ beta_rho
            ssr = residuals @ residuals
            sigma2 = ssr / self.n_obs

            # Log-det
            log_det = self._log_det_jacobian(rho)

            # Log-likelihood
            llf = -self.n_obs / 2 * np.log(2 * np.pi * sigma2)
            llf += self.n_periods * log_det
            llf -= ssr / (2 * sigma2)

            return llf

        # Grid search
        rho_grid = np.linspace(rho_bounds[0] * 0.95, rho_bounds[1] * 0.95, rho_grid_size)
        llf_grid = [concentrated_llf(r) for r in rho_grid]
        rho_init = rho_grid[np.argmax(llf_grid)]

        # Optimization
        result = minimize_scalar(
            lambda r: -concentrated_llf(r),
            bounds=rho_bounds,
            method="bounded",
            options={"xatol": 1e-8},
        )

        rho_hat = result.x

        # Compute β at optimal ρ
        Wy = self._spatial_lag(y.reshape(-1, 1)).flatten()
        y_rho = y - rho_hat * Wy
        beta_hat, _, _, _ = np.linalg.lstsq(X, y_rho, rcond=None)

        # Results
        residuals = y_rho - X @ beta_hat
        sigma2_hat = (residuals @ residuals) / (self.n_obs - len(beta_hat))

        # Covariance
        Z = np.column_stack([Wy, X])
        cov_matrix = sigma2_hat * np.linalg.inv(Z.T @ Z)

        # Parameters
        params = np.concatenate([[rho_hat], beta_hat])

        if hasattr(self.exog, "columns"):
            param_names = ["rho", "const"] + list(self.exog.columns)
        else:
            param_names = ["rho", "const"] + [f"x{i}" for i in range(self.exog.shape[1])]

        results = SpatialPanelResults(
            model=self,
            params=pd.Series(params, index=param_names),
            cov_params=pd.DataFrame(cov_matrix, index=param_names, columns=param_names),
            llf=-result.fun,
            nobs=self.n_obs,
            df_model=len(params),
            df_resid=self.n_obs - len(params),
            method="Quasi-ML",
            effects="pooled",
            resid=residuals,
            sigma2=sigma2_hat,
        )

        self.results = results
        self.fitted = True

        return results

    def _fit_ml_re(self, **kwargs):
        """
        Maximum likelihood estimation for random effects SAR.

        Not yet implemented.
        """
        raise NotImplementedError("Random effects SAR not yet implemented")

    def predict(
        self,
        params: Optional[Dict[str, float]] = None,
        exog: Optional[np.ndarray] = None,
        effects: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate predictions from SAR model.

        For SAR: y = (I - ρW)⁻¹(Xβ + α + ε)

        Parameters
        ----------
        params : dict, optional
            Parameter values (uses fitted values if None)
        exog : np.ndarray, optional
            Exogenous variables (uses training data if None)
        effects : np.ndarray, optional
            Fixed/random effects

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if not self.fitted and params is None:
            raise ValueError("Model must be fitted before prediction")

        if params is None:
            params = self.results.params

        # Extract parameters
        rho = params["rho"] if isinstance(params, dict) else params[0]
        beta = params.drop("rho") if hasattr(params, "drop") else params[1:]

        # Use provided or training exog
        if exog is None:
            exog = self.exog

        # Linear prediction
        Xbeta = exog @ beta

        # Add effects if provided
        if effects is not None:
            Xbeta += effects

        # Apply spatial multiplier: (I - ρW)⁻¹
        # For panel, need to handle time dimension
        N = self.n_entities
        T = self.n_periods

        predictions = np.zeros_like(Xbeta)

        for t in range(T):
            # Get slice for time t
            start_idx = t * N
            end_idx = (t + 1) * N

            Xbeta_t = Xbeta[start_idx:end_idx]

            # Compute (I - ρW)⁻¹ Xβ
            I_rhoW = np.eye(N) - rho * self.W.to_dense()
            y_pred_t = np.linalg.solve(I_rhoW, Xbeta_t)

            predictions[start_idx:end_idx] = y_pred_t

        return predictions


class SpatialPanelResults(PanelResults):
    """
    Results class for spatial panel models.

    Extends PanelResults with spatial-specific diagnostics.
    """

    def __init__(
        self,
        model,
        params,
        cov_params,
        llf,
        nobs,
        df_model,
        df_resid,
        method,
        effects,
        resid,
        sigma2,
        **kwargs,
    ):
        """Initialize spatial panel results."""
        # Store basic attributes
        self.model = model
        self.params = params
        self.cov_params = cov_params
        self.llf = llf
        self.nobs = nobs
        self.df_model = df_model
        self.df_resid = df_resid
        self.method = method
        self.effects = effects
        self.resid = resid
        self.sigma2 = sigma2

        # Standard errors
        self.bse = pd.Series(np.sqrt(np.diag(cov_params)), index=params.index)

        # T-statistics
        self.tvalues = params / self.bse

        # P-values (two-tailed)
        from scipy.stats import t

        self.pvalues = pd.Series(
            2 * (1 - t.cdf(np.abs(self.tvalues), df_resid)), index=params.index
        )

        # Store additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Compute fit statistics
        self._compute_fit_statistics()

    def _compute_fit_statistics(self):
        """Compute model fit statistics."""
        # AIC and BIC
        self.aic = -2 * self.llf + 2 * self.df_model
        self.bic = -2 * self.llf + np.log(self.nobs) * self.df_model

        # Pseudo R-squared (for spatial models)
        # Compare with null model (no spatial, no covariates)
        y_mean = self.model.endog.mean()
        tss = ((self.model.endog - y_mean) ** 2).sum()
        rss = (self.resid**2).sum()
        self.rsquared_pseudo = 1 - rss / tss

    def summary(self):
        """Generate summary of results."""
        print("\n" + "=" * 78)
        print(f"Spatial Panel Model Results ({self.model.model_type})")
        print("=" * 78)
        print(f"Method:              {self.method}")
        print(f"Effects:             {self.effects}")
        print(f"Number of obs:       {self.nobs}")
        print(f"Number of entities:  {self.model.n_entities}")
        print(f"Number of periods:   {self.model.n_periods}")
        print(f"Log-likelihood:      {self.llf:.4f}")
        print(f"AIC:                 {self.aic:.4f}")
        print(f"BIC:                 {self.bic:.4f}")
        print(f"Pseudo R²:           {self.rsquared_pseudo:.4f}")
        print("-" * 78)

        # Parameter estimates table
        results_df = pd.DataFrame(
            {
                "Coefficient": self.params,
                "Std. Error": self.bse,
                "t-value": self.tvalues,
                "P>|t|": self.pvalues,
            }
        )

        # Add significance stars
        results_df["Signif"] = ""
        results_df.loc[self.pvalues < 0.001, "Signif"] = "***"
        results_df.loc[(self.pvalues >= 0.001) & (self.pvalues < 0.01), "Signif"] = "**"
        results_df.loc[(self.pvalues >= 0.01) & (self.pvalues < 0.05), "Signif"] = "*"
        results_df.loc[(self.pvalues >= 0.05) & (self.pvalues < 0.1), "Signif"] = "."

        print(results_df.to_string())
        print("-" * 78)
        print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

        # Spatial parameter bounds
        if "rho" in self.params:
            bounds = self.model._spatial_coefficient_bounds()
            print(f"\nSpatial parameter (ρ) bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
            print(f"Estimated ρ: {self.params['rho']:.6f}")

        # Spillover effects if available
        if hasattr(self, "spillover_effects"):
            print("\n" + "-" * 78)
            print("Spillover Effects:")
            print("-" * 78)
            for var, effects in self.spillover_effects.items():
                if var != "rho":  # Skip spatial parameter
                    print(f"\n{var}:")
                    print(f"  Direct effect:   {effects['direct']:.6f}")
                    print(f"  Indirect effect: {effects['indirect']:.6f}")
                    print(f"  Total effect:    {effects['total']:.6f}")

        print("=" * 78)
