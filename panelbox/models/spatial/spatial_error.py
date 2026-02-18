"""
Spatial Error Model (SEM) for panel data.

GMM-based estimation with spatial instruments.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.optimize import minimize, minimize_scalar

from .base_spatial import SpatialPanelModel
from .spatial_lag import SpatialPanelResults
from .spatial_weights import SpatialWeights


class SpatialError(SpatialPanelModel):
    """
    Spatial Error Model (SEM) for panel data.

    The model is specified as:
        y = Xβ + α + u
        u = λWu + ε

    where:
        - y is the dependent variable
        - X are exogenous variables
        - β are coefficients
        - α are fixed/random effects
        - u are spatially autocorrelated errors
        - W is the spatial weight matrix
        - λ is the spatial error parameter
        - ε are i.i.d. errors

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
        """Initialize SEM model."""
        super().__init__(*args, **kwargs)
        self.model_type = "SEM"

    def _estimate_coefficients(self) -> np.ndarray:
        """
        Placeholder required by abstract base class.

        Actual estimation is performed in ``fit()`` which returns full results.
        """
        return np.array([])

    def fit(
        self,
        effects: str = "fixed",
        method: str = "gmm",
        n_lags: int = 2,
        maxiter: int = 1000,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Fit the Spatial Error Model.

        Parameters
        ----------
        effects : str
            Type of effects: 'fixed', 'random', or 'pooled'
        method : str
            Estimation method: 'gmm' (default), 'ml'
        n_lags : int
            Number of spatial lags to use as instruments (for GMM)
        maxiter : int
            Maximum iterations
        verbose : bool
            Print optimization progress

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        if effects == "fixed" and method == "gmm":
            return self._fit_gmm_fe(n_lags, maxiter, verbose, **kwargs)
        elif effects == "pooled" and method == "gmm":
            return self._fit_gmm_pooled(n_lags, maxiter, verbose, **kwargs)
        elif effects in ["fixed", "random"] and method == "ml":
            return self._fit_ml(effects, maxiter, verbose, **kwargs)
        else:
            raise NotImplementedError(
                f"Combination effects='{effects}' and method='{method}' " "not yet implemented"
            )

    def _fit_gmm_fe(self, n_lags: int = 2, maxiter: int = 1000, verbose: bool = False, **kwargs):
        """
        GMM estimation for SEM with fixed effects.

        Uses spatial instruments: Z = [X, WX, W²X, ...]

        Parameters
        ----------
        n_lags : int
            Number of spatial lags for instruments
        maxiter : int
            Maximum iterations
        verbose : bool
            Print progress

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        if verbose:
            print("Estimating SEM-FE using GMM with spatial instruments")

        # Apply within transformation
        y_within = self._within_transformation(self.endog.values.reshape(-1, 1)).flatten()
        X_within = self._within_transformation(self.exog)

        # Construct instrument matrix Z = [X, WX, W²X, ...]
        Z_list = [X_within]
        WkX = X_within.copy()

        for lag in range(1, n_lags + 1):
            WkX = self._spatial_lag(WkX)
            Z_list.append(WkX)

        Z = np.hstack(Z_list)

        if verbose:
            print(f"Instruments: X and {n_lags} spatial lag(s)")
            print(f"Number of instruments: {Z.shape[1]}")
            print(f"Number of parameters: {X_within.shape[1]}")

        # Two-step GMM estimation

        # Step 1: Initial GMM with W = I
        if verbose:
            print("\nStep 1: Initial GMM (W = I)")

        # First-stage regression: project X onto Z
        ZtZ = Z.T @ Z
        ZtX = Z.T @ X_within
        try:
            P_Z = Z @ inv(ZtZ) @ Z.T  # Projection matrix
            X_hat = P_Z @ X_within
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            ZtZ_inv = np.linalg.pinv(ZtZ)
            P_Z = Z @ ZtZ_inv @ Z.T
            X_hat = P_Z @ X_within

        # GMM estimator: β = (X'P_Z X)⁻¹ X'P_Z y
        XtPZX = X_hat.T @ X_within
        XtPZy = X_hat.T @ y_within

        try:
            beta_1 = inv(XtPZX) @ XtPZy
        except np.linalg.LinAlgError:
            beta_1 = np.linalg.pinv(XtPZX) @ XtPZy

        # Compute residuals
        u_1 = y_within - X_within @ beta_1

        # Estimate λ from residuals using spatial correlation
        Wu = self._spatial_lag(u_1.reshape(-1, 1)).flatten()
        lambda_1 = (Wu @ u_1) / (Wu @ Wu)

        # Clip λ to bounds
        lambda_bounds = self._spatial_coefficient_bounds()
        lambda_1 = np.clip(lambda_1, lambda_bounds[0], lambda_bounds[1])

        if verbose:
            print(f"Initial β: {beta_1}")
            print(f"Initial λ: {lambda_1:.4f}")

        # Step 2: Efficient GMM with optimal weighting matrix
        if verbose:
            print("\nStep 2: Efficient GMM")

        # Compute optimal weighting matrix
        # Ω = E[Z'uu'Z] ≈ Z'ûû'Z
        u_mat = u_1.reshape(-1, 1)
        Omega = (Z.T @ (u_mat @ u_mat.T) @ Z) / self.n_obs

        try:
            W_gmm = inv(Omega)
        except np.linalg.LinAlgError:
            W_gmm = np.linalg.pinv(Omega)
            warnings.warn("Singular weighting matrix, using pseudo-inverse")

        # Two-step GMM estimator
        XtZWZ = X_within.T @ Z @ W_gmm @ Z.T
        beta_2 = inv(XtZWZ @ X_within) @ (XtZWZ @ y_within)

        # Final residuals
        u_2 = y_within - X_within @ beta_2

        # Re-estimate λ
        Wu_2 = self._spatial_lag(u_2.reshape(-1, 1)).flatten()

        # ML-like estimation of λ given β
        def neg_concentrated_llf_lambda(lam):
            """Negative concentrated log-likelihood for λ."""
            # Transform residuals: v = u - λWu
            v = u_2 - lam * Wu_2
            sigma2 = (v @ v) / self.n_obs

            # Log-determinant
            log_det = self._log_det_jacobian(lam)

            # Log-likelihood
            llf = -self.n_obs / 2 * np.log(2 * np.pi * sigma2)
            llf += self.n_periods * log_det
            llf -= (v @ v) / (2 * sigma2)

            return -llf

        # Optimize λ
        result = minimize_scalar(
            neg_concentrated_llf_lambda,
            bounds=lambda_bounds,
            method="bounded",
            options={"xatol": 1e-8},
        )

        lambda_hat = result.x

        if verbose:
            print(f"Final β: {beta_2}")
            print(f"Final λ: {lambda_hat:.6f}")

        # Compute variance of final estimator
        sigma2_hat = ((u_2 - lambda_hat * Wu_2) ** 2).mean()

        # Covariance matrix (GMM sandwich formula)
        cov_matrix = self._gmm_covariance(beta_2, lambda_hat, X_within, Z, W_gmm, sigma2_hat)

        # Prepare results
        params = np.concatenate([[lambda_hat], beta_2])

        # Parameter names
        if hasattr(self.exog, "columns"):
            param_names = ["lambda"] + list(self.exog.columns)
        else:
            param_names = ["lambda"] + [f"x{i}" for i in range(self.exog.shape[1])]

        # Create results object
        results = SpatialPanelResults(
            model=self,
            params=pd.Series(params, index=param_names),
            cov_params=pd.DataFrame(cov_matrix, index=param_names, columns=param_names),
            llf=-result.fun,  # From λ optimization
            nobs=self.n_obs,
            df_model=len(params),
            df_resid=self.n_obs - len(params) - self.n_entities,
            method=f"GMM (spatial instruments, {n_lags} lags)",
            effects=effects,
            resid=u_2,
            sigma2=sigma2_hat,
        )

        self.results = results
        self.fitted = True

        return results

    def _gmm_covariance(
        self,
        beta: np.ndarray,
        lambda_param: float,
        X: np.ndarray,
        Z: np.ndarray,
        W_gmm: np.ndarray,
        sigma2: float,
    ) -> np.ndarray:
        """
        Compute GMM covariance matrix.

        Uses sandwich formula for robust standard errors.

        Parameters
        ----------
        beta : np.ndarray
            Coefficient estimates
        lambda_param : float
            Spatial error parameter
        X : np.ndarray
            Regressors
        Z : np.ndarray
            Instruments
        W_gmm : np.ndarray
            GMM weighting matrix
        sigma2 : float
            Error variance

        Returns
        -------
        np.ndarray
            Covariance matrix for [λ, β']
        """
        # Simplified covariance for GMM
        # V = (G'W G)⁻¹ G'W Ω W G (G'W G)⁻¹
        # where G = ∂moments/∂params

        # For now, use simplified formula
        # Augmented design: [Wu, X]
        Wu = self._spatial_lag(X @ beta).reshape(-1, 1)
        augmented_X = np.column_stack([Wu, X])

        # Project onto instruments
        ZtZ_inv = np.linalg.pinv(Z.T @ Z)
        P_Z = Z @ ZtZ_inv @ Z.T
        augmented_X_hat = P_Z @ augmented_X

        # Covariance approximation
        try:
            cov = sigma2 * inv(augmented_X_hat.T @ augmented_X_hat)
        except np.linalg.LinAlgError:
            cov = sigma2 * np.linalg.pinv(augmented_X_hat.T @ augmented_X_hat)

        # Finite sample correction
        n = self.n_entities
        T = self.n_periods
        k = beta.shape[0]
        correction = n * T / (n * T - n - k - 1)

        return cov * correction

    def _fit_gmm_pooled(
        self, n_lags: int = 2, maxiter: int = 1000, verbose: bool = False, **kwargs
    ):
        """
        GMM estimation for pooled SEM (no effects).

        Parameters
        ----------
        n_lags : int
            Number of spatial lags for instruments
        maxiter : int
            Maximum iterations
        verbose : bool
            Print progress

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        if verbose:
            print("Estimating pooled SEM using GMM")

        y = self.endog.values.flatten()
        X = np.asarray(self.exog)  # ensure numpy array (self.exog may be a DataFrame)

        # Add constant if not present; track whether we added one
        _const_added = not np.any(np.all(X == X[0, :], axis=0))
        if _const_added:
            X = np.column_stack([np.ones(len(y)), X])

        # Construct instruments
        Z_list = [X]
        WkX = X.copy()

        for lag in range(1, n_lags + 1):
            WkX = self._spatial_lag(WkX)
            Z_list.append(WkX)

        Z = np.hstack(Z_list)

        # Two-step GMM (similar to FE case but without within transformation)
        # Step 1
        P_Z = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
        X_hat = P_Z @ X

        beta_1 = np.linalg.pinv(X_hat.T @ X) @ (X_hat.T @ y)
        u_1 = y - X @ beta_1

        # Estimate λ
        Wu = self._spatial_lag(u_1.reshape(-1, 1)).flatten()
        lambda_1 = (Wu @ u_1) / (Wu @ Wu)

        # Step 2 with optimal weights
        u_mat = u_1.reshape(-1, 1)
        Omega = (Z.T @ (u_mat @ u_mat.T) @ Z) / len(y)
        W_gmm = np.linalg.pinv(Omega)

        XtZWZ = X.T @ Z @ W_gmm @ Z.T
        beta_2 = np.linalg.pinv(XtZWZ @ X) @ (XtZWZ @ y)

        # Final λ estimation
        u_2 = y - X @ beta_2
        Wu_2 = self._spatial_lag(u_2.reshape(-1, 1)).flatten()

        # Get bounds
        lambda_bounds = self._spatial_coefficient_bounds()

        # Optimize λ
        def neg_llf_lambda(lam):
            v = u_2 - lam * Wu_2
            sigma2 = (v @ v) / len(y)
            log_det = self._log_det_jacobian(lam)
            llf = -len(y) / 2 * np.log(2 * np.pi * sigma2)
            llf += self.n_periods * log_det
            llf -= (v @ v) / (2 * sigma2)
            return -llf

        result = minimize_scalar(neg_llf_lambda, bounds=lambda_bounds, method="bounded")

        lambda_hat = result.x

        # Covariance
        sigma2_hat = ((u_2 - lambda_hat * Wu_2) ** 2).mean()
        Wu = self._spatial_lag(X @ beta_2).reshape(-1, 1)
        augmented_X = np.column_stack([Wu, X])
        cov = sigma2_hat * np.linalg.pinv(augmented_X.T @ P_Z @ augmented_X)

        # Results
        params = np.concatenate([[lambda_hat], beta_2])

        # Build param_names to match [lambda, beta...] where beta has X.shape[1] entries
        if _const_added:
            if hasattr(self.exog, "columns"):
                param_names = ["lambda", "const"] + list(self.exog.columns)
            else:
                param_names = ["lambda", "const"] + [f"x{i}" for i in range(self.exog.shape[1])]
        else:
            if hasattr(self.exog, "columns"):
                param_names = ["lambda"] + list(self.exog.columns)
            else:
                param_names = ["lambda"] + [f"x{i}" for i in range(X.shape[1])]

        results = SpatialPanelResults(
            model=self,
            params=pd.Series(params, index=param_names),
            cov_params=pd.DataFrame(cov, index=param_names, columns=param_names),
            llf=-result.fun,
            nobs=len(y),
            df_model=len(params),
            df_resid=len(y) - len(params),
            method=f"GMM ({n_lags} spatial lags)",
            effects="pooled",
            resid=u_2,
            sigma2=sigma2_hat,
        )

        self.results = results
        self.fitted = True

        return results

    def _fit_ml(self, effects: str, maxiter: int = 1000, verbose: bool = False, **kwargs):
        """
        Maximum likelihood estimation for SEM.

        Parameters
        ----------
        effects : str
            'fixed' or 'random'
        maxiter : int
            Maximum iterations
        verbose : bool
            Print progress

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        if verbose:
            print(f"Estimating SEM-{effects.upper()} using ML")

        # Apply transformations
        if effects == "fixed":
            y = self._within_transformation(self.endog.values.reshape(-1, 1)).flatten()
            X = self._within_transformation(self.exog)
        else:
            y = self.endog.values.flatten()
            X = self.exog

        # Get bounds
        lambda_bounds = self._spatial_coefficient_bounds()

        # Full log-likelihood function
        def neg_log_likelihood(params):
            """Negative log-likelihood for SEM."""
            lambda_param = params[0]
            beta = params[1:]

            # Residuals
            u = y - X @ beta

            # Transform: v = (I - λW)u
            Wu = self._spatial_lag(u.reshape(-1, 1)).flatten()
            v = u - lambda_param * Wu

            # Variance
            sigma2 = (v @ v) / self.n_obs

            # Log-determinant
            log_det = self._log_det_jacobian(lambda_param)

            # Log-likelihood
            llf = -self.n_obs / 2 * np.log(2 * np.pi * sigma2)
            llf += self.n_periods * log_det
            llf -= (v @ v) / (2 * sigma2)

            return -llf

        # Starting values: OLS for β, 0 for λ
        beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        params_init = np.concatenate([[0.0], beta_init])

        # Bounds
        bounds = [lambda_bounds] + [(None, None)] * len(beta_init)

        # Optimization
        from scipy.optimize import minimize

        result = minimize(
            neg_log_likelihood,
            x0=params_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Extract results
        lambda_hat = result.x[0]
        beta_hat = result.x[1:]

        if verbose:
            print(f"λ: {lambda_hat:.6f}")
            print(f"β: {beta_hat}")

        # Compute residuals and variance
        u = y - X @ beta_hat
        Wu = self._spatial_lag(u.reshape(-1, 1)).flatten()
        v = u - lambda_hat * Wu
        sigma2_hat = (v @ v) / self.n_obs

        # Hessian for standard errors
        # Use numerical approximation
        from scipy.optimize import approx_fprime

        def grad_func(p):
            eps = 1e-8
            grad = np.zeros_like(p)
            for i in range(len(p)):
                p_plus = p.copy()
                p_plus[i] += eps
                p_minus = p.copy()
                p_minus[i] -= eps
                grad[i] = (neg_log_likelihood(p_plus) - neg_log_likelihood(p_minus)) / (2 * eps)
            return grad

        # Hessian approximation
        hessian = np.zeros((len(result.x), len(result.x)))
        eps = 1e-5
        for i in range(len(result.x)):
            p_plus = result.x.copy()
            p_plus[i] += eps
            p_minus = result.x.copy()
            p_minus[i] -= eps

            grad_plus = grad_func(p_plus)
            grad_minus = grad_func(p_minus)

            hessian[i, :] = (grad_plus - grad_minus) / (2 * eps)

        # Covariance matrix
        try:
            cov_matrix = inv(hessian)
        except np.linalg.LinAlgError:
            cov_matrix = np.linalg.pinv(hessian)
            warnings.warn("Singular Hessian, using pseudo-inverse")

        # Parameter names
        if hasattr(self.exog, "columns"):
            param_names = ["lambda"] + list(self.exog.columns)
        else:
            param_names = ["lambda"] + [f"x{i}" for i in range(X.shape[1])]

        # Results
        params = result.x

        results = SpatialPanelResults(
            model=self,
            params=pd.Series(params, index=param_names),
            cov_params=pd.DataFrame(cov_matrix, index=param_names, columns=param_names),
            llf=-result.fun,
            nobs=self.n_obs,
            df_model=len(params),
            df_resid=self.n_obs - len(params) - (self.n_entities if effects == "fixed" else 0),
            method="Maximum Likelihood",
            effects=effects,
            resid=u,
            sigma2=sigma2_hat,
        )

        self.results = results
        self.fitted = True

        return results

    def predict(
        self,
        params: Optional[Dict[str, float]] = None,
        exog: Optional[np.ndarray] = None,
        effects: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate predictions from SEM model.

        For SEM: y = Xβ + α
        (errors are spatially correlated but don't affect prediction)

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

        # Extract β (skip λ)
        if isinstance(params, pd.Series):
            beta = params.drop("lambda")
        else:
            beta = params[1:]

        # Use provided or training exog
        if exog is None:
            exog = self.exog

        # Linear prediction (SEM doesn't have spatial multiplier in mean)
        predictions = exog @ beta

        # Add effects if provided
        if effects is not None:
            predictions += effects

        return predictions
