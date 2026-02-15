"""
General Nesting Spatial (GNS) Model for panel data.

This module implements the General Nesting Spatial model that nests
all common spatial specifications (SAR, SEM, SDM, SAC) as special cases.

References
----------
Elhorst, J.P. (2010). "Applied Spatial Econometrics: Raising the Bar."
    Spatial Economic Analysis, 5(1), 9-28.
LeSage, J.P. and Pace, R.K. (2009). "Introduction to Spatial Econometrics."
    Chapman & Hall/CRC.
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.sparse import issparse
from scipy.stats import chi2

from .base_spatial import SpatialPanelModel
from .spatial_lag import SpatialPanelResults


class GeneralNestingSpatial(SpatialPanelModel):
    """
    General Nesting Spatial (GNS) Model.

    The GNS model generalizes all common spatial specifications:

    y = ρW₁y + Xβ + W₂Xθ + u
    u = λW₃u + ε

    Special cases:
    - ρ≠0, θ=0, λ=0 → SAR (Spatial Autoregressive)
    - ρ=0, θ=0, λ≠0 → SEM (Spatial Error Model)
    - ρ≠0, θ≠0, λ=0 → SDM (Spatial Durbin Model)
    - ρ≠0, θ=0, λ≠0 → SAC/SARAR (Spatial Autoregressive with Spatial Errors)
    - ρ=0, θ≠0, λ=0 → SDEM (Spatial Durbin Error Model)
    - ρ=0, θ≠0, λ≠0 → SDEM-SEM
    - ρ≠0, θ≠0, λ≠0 → GNS (full model)

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
    W1 : np.ndarray or None
        Weight matrix for Wy (defaults to W)
    W2 : np.ndarray or None
        Weight matrix for WX (defaults to W)
    W3 : np.ndarray or None
        Weight matrix for Wu (defaults to W)
    weights : np.ndarray, optional
        Observation weights for weighted estimation

    Attributes
    ----------
    W1 : np.ndarray
        Weight matrix for spatial lag of y
    W2 : np.ndarray
        Weight matrix for spatial lag of X
    W3 : np.ndarray
        Weight matrix for spatial error
    model_type : str
        Detected model type based on estimated parameters
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        W1: Optional[np.ndarray] = None,
        W2: Optional[np.ndarray] = None,
        W3: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        """Initialize GNS model with multiple weight matrices."""

        # Use first non-null W matrix for base initialization
        W_base = W1 if W1 is not None else (W2 if W2 is not None else W3)
        if W_base is None:
            raise ValueError("At least one weight matrix (W1, W2, or W3) must be provided")

        super().__init__(formula, data, entity_col, time_col, W_base, weights)

        # Store individual weight matrices (default to base W if not provided)
        self.W1 = W1 if W1 is not None else self.W_normalized
        self.W2 = W2 if W2 is not None else self.W_normalized
        self.W3 = W3 if W3 is not None else self.W_normalized

        # Ensure all matrices have correct dimensions
        n = self.n_entities
        for name, W in [("W1", self.W1), ("W2", self.W2), ("W3", self.W3)]:
            if W.shape != (n, n):
                raise ValueError(f"{name} must be {n}×{n}, got {W.shape}")

        # Row-normalize if needed
        self.W1 = self._row_normalize(self.W1)
        self.W2 = self._row_normalize(self.W2)
        self.W3 = self._row_normalize(self.W3)

        self.model_type = None

    def fit(
        self,
        effects: Literal["fixed", "random", "pooled"] = "fixed",
        method: Literal["ml", "gmm"] = "ml",
        rho_init: float = 0.0,
        lambda_init: float = 0.0,
        include_wx: bool = True,
        maxiter: int = 1000,
        optim_method: str = "L-BFGS-B",
        **kwargs,
    ) -> SpatialPanelResults:
        """
        Fit the General Nesting Spatial model.

        Parameters
        ----------
        effects : {'fixed', 'random', 'pooled'}
            Type of panel effects
        method : {'ml', 'gmm'}
            Estimation method (ML or GMM)
        rho_init : float
            Initial value for spatial lag parameter ρ
        lambda_init : float
            Initial value for spatial error parameter λ
        include_wx : bool
            Whether to include spatial lag of X (θ parameters)
        maxiter : int
            Maximum iterations for optimization
        optim_method : str
            Optimization method for scipy.optimize
        **kwargs
            Additional arguments for optimization

        Returns
        -------
        SpatialPanelResults
            Estimation results
        """
        # Prepare data
        y, X = self.prepare_data(effects)
        n_obs, k_vars = X.shape
        T = self.n_periods
        N = self.n_entities

        # Create spatial lag variables if needed
        if include_wx:
            # Compute WX for each time period
            WX_list = []
            for t in range(T):
                start_idx = t * N
                end_idx = (t + 1) * N
                X_t = X[start_idx:end_idx]
                WX_t = self.W2 @ X_t
                WX_list.append(WX_t)
            WX = np.vstack(WX_list)

            # Combine X and WX
            X_full = np.hstack([X, WX])
            k_full = X_full.shape[1]
        else:
            X_full = X
            k_full = k_vars
            WX = None

        if method == "ml":
            result = self._fit_ml(
                y,
                X,
                X_full,
                WX,
                rho_init,
                lambda_init,
                include_wx,
                effects,
                maxiter,
                optim_method,
                **kwargs,
            )
        elif method == "gmm":
            result = self._fit_gmm(
                y, X, X_full, WX, rho_init, lambda_init, include_wx, effects, **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Identify model type based on estimated parameters
        self.model_type = self.identify_model_type(result)
        result.model_type = self.model_type

        return result

    def _fit_ml(
        self,
        y,
        X,
        X_full,
        WX,
        rho_init,
        lambda_init,
        include_wx,
        effects,
        maxiter,
        optim_method,
        **kwargs,
    ):
        """Maximum Likelihood estimation for GNS model."""

        n_obs = len(y)
        T = self.n_periods
        N = self.n_entities
        k_full = X_full.shape[1]

        # Compute spatial lags of y
        Wy = self._compute_spatial_lag_panel(y, self.W1, T, N)

        def negative_log_likelihood(params):
            """Negative log-likelihood for GNS model."""

            # Parse parameters
            rho = params[0]
            lambda_ = params[1]
            beta_theta = params[2 : 2 + k_full]
            sigma2 = params[2 + k_full]

            # Ensure valid parameter ranges
            if abs(rho) >= 1 or abs(lambda_) >= 1 or sigma2 <= 0:
                return 1e10

            # Spatial transformation of y
            y_star = y - rho * Wy

            # Predicted values
            y_pred = X_full @ beta_theta

            # Residuals before spatial error transformation
            u = y_star - y_pred

            # Spatial error transformation: (I - λW₃)u = ε
            # For log-likelihood we need: ε'ε = u'(I-λW₃)'(I-λW₃)u
            I_N = np.eye(N)
            A_lambda = I_N - lambda_ * self.W3

            # Compute residuals for each time period
            ssr = 0
            for t in range(T):
                start_idx = t * N
                end_idx = (t + 1) * N
                u_t = u[start_idx:end_idx]

                # Transform residuals
                eps_t = A_lambda @ u_t
                ssr += eps_t.T @ eps_t

            # Log-determinants
            log_det_rho = T * self._compute_log_det(I_N - rho * self.W1)
            log_det_lambda = T * self._compute_log_det(A_lambda)

            # Negative log-likelihood
            nll = (
                -log_det_rho
                - log_det_lambda
                + 0.5 * n_obs * np.log(2 * np.pi * sigma2)
                + 0.5 * ssr / sigma2
            )

            return nll

        # Initial parameter values
        beta_init = np.linalg.lstsq(X_full, y, rcond=None)[0]
        residuals_init = y - X_full @ beta_init
        sigma2_init = np.sum(residuals_init**2) / n_obs

        params_init = np.concatenate([[rho_init, lambda_init], beta_init, [sigma2_init]])

        # Parameter bounds
        bounds = [(-0.99, 0.99), (-0.99, 0.99)]  # rho, lambda
        bounds += [(None, None)] * k_full  # beta and theta
        bounds += [(1e-6, None)]  # sigma2

        # Optimize
        opt_result = optimize.minimize(
            negative_log_likelihood,
            params_init,
            method=optim_method,
            bounds=bounds,
            options={"maxiter": maxiter, "disp": False},
        )

        if not opt_result.success:
            warnings.warn(f"Optimization did not converge: {opt_result.message}")

        # Extract results
        params_opt = opt_result.x
        rho = params_opt[0]
        lambda_ = params_opt[1]
        beta_theta = params_opt[2 : 2 + k_full]
        sigma2 = params_opt[2 + k_full]

        # Split beta and theta if WX included
        if include_wx:
            k_vars = X.shape[1]
            beta = beta_theta[:k_vars]
            theta = beta_theta[k_vars:]
        else:
            beta = beta_theta
            theta = np.array([])

        # Compute standard errors using information matrix
        hessian = self._compute_hessian_ml(params_opt, y, X, X_full, WX, T, N)

        try:
            cov_matrix = np.linalg.inv(-hessian)
            std_errors = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            warnings.warn("Hessian is singular, using identity matrix for covariance")
            std_errors = np.ones(len(params_opt))
            cov_matrix = np.eye(len(params_opt))

        # Organize results
        param_names = ["rho", "lambda"]
        param_names += [f"beta_{i}" for i in range(len(beta))]
        if len(theta) > 0:
            param_names += [f"theta_{i}" for i in range(len(theta))]
        param_names += ["sigma2"]

        params_df = pd.DataFrame(
            {
                "coefficient": params_opt,
                "std_error": std_errors,
                "t_statistic": params_opt / std_errors,
                "p_value": 2
                * (
                    1
                    - pd.Series(np.abs(params_opt / std_errors)).apply(
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
        y_fitted = rho * Wy + X_full @ beta_theta
        residuals = y - y_fitted

        # Compute log-likelihood at optimum
        log_likelihood = -opt_result.fun

        # Create results object
        result = SpatialPanelResults(
            params=params_df,
            fitted_values=y_fitted,
            residuals=residuals,
            entity_effects=None,  # Could be computed if needed
            time_effects=None,
            n_obs=n_obs,
            n_entities=N,
            n_periods=T,
            effects_type=effects,
            cov_matrix=cov_matrix,
            log_likelihood=log_likelihood,
            rho=rho,
            lambda_=lambda_,
            spatial_model_type="GNS",
        )

        return result

    def _fit_gmm(self, y, X, X_full, WX, rho_init, lambda_init, include_wx, effects, **kwargs):
        """GMM estimation for GNS model (not yet implemented)."""
        raise NotImplementedError("GMM estimation for GNS model is not yet implemented")

    def _compute_spatial_lag_panel(self, y, W, T, N):
        """Compute spatial lag for panel data."""
        Wy = np.zeros_like(y)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            y_t = y[start_idx:end_idx]
            Wy[start_idx:end_idx] = W @ y_t
        return Wy

    def _compute_log_det(self, A):
        """Compute log determinant of matrix A."""
        if issparse(A):
            # For sparse matrices, use LU decomposition
            lu = splu(A.tocsc())
            log_det = np.sum(np.log(np.abs(lu.U.diagonal())))
        else:
            # For dense matrices
            sign, log_det = np.linalg.slogdet(A)
            if sign <= 0:
                warnings.warn("Matrix has non-positive determinant")
                return -1e10
        return log_det

    def _compute_hessian_ml(self, params, y, X, X_full, WX, T, N):
        """Compute Hessian matrix for ML estimation (numerical approximation)."""

        eps = 1e-5
        n_params = len(params)
        hessian = np.zeros((n_params, n_params))

        # Define objective function
        def obj_func(p):
            return self._ml_objective_for_hessian(p, y, X, X_full, WX, T, N)

        # Numerical Hessian using finite differences
        f0 = obj_func(params)

        for i in range(n_params):
            params_i_plus = params.copy()
            params_i_plus[i] += eps

            params_i_minus = params.copy()
            params_i_minus[i] -= eps

            for j in range(i, n_params):
                params_j_plus = params.copy()
                params_j_plus[j] += eps

                params_ij_plus = params_i_plus.copy()
                params_ij_plus[j] += eps

                # Second derivative approximation
                f_ij = obj_func(params_ij_plus)
                f_i = obj_func(params_i_plus)
                f_j = obj_func(params_j_plus)

                hessian[i, j] = (f_ij - f_i - f_j + f0) / (eps * eps)
                if i != j:
                    hessian[j, i] = hessian[i, j]

        return hessian

    def _ml_objective_for_hessian(self, params, y, X, X_full, WX, T, N):
        """ML objective function for Hessian computation."""
        # Similar to negative_log_likelihood but simplified for Hessian
        n_obs = len(y)
        k_full = X_full.shape[1]

        rho = params[0]
        lambda_ = params[1]
        beta_theta = params[2 : 2 + k_full]
        sigma2 = max(params[2 + k_full], 1e-6)

        # Compute spatial transformations
        Wy = self._compute_spatial_lag_panel(y, self.W1, T, N)
        y_star = y - rho * Wy
        y_pred = X_full @ beta_theta
        u = y_star - y_pred

        # Compute SSR with spatial error
        I_N = np.eye(N)
        A_lambda = I_N - lambda_ * self.W3

        ssr = 0
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            u_t = u[start_idx:end_idx]
            eps_t = A_lambda @ u_t
            ssr += eps_t.T @ eps_t

        # Log-likelihood components
        log_det_rho = T * self._compute_log_det(I_N - rho * self.W1)
        log_det_lambda = T * self._compute_log_det(A_lambda)

        # Return negative log-likelihood
        nll = (
            -log_det_rho
            - log_det_lambda
            + 0.5 * n_obs * np.log(2 * np.pi * sigma2)
            + 0.5 * ssr / sigma2
        )

        return nll

    def test_restrictions(
        self, restrictions: Dict[str, float], full_model: Optional[SpatialPanelResults] = None
    ) -> Dict:
        """
        Likelihood Ratio test for parameter restrictions.

        Parameters
        ----------
        restrictions : dict
            Parameter restrictions, e.g., {'rho': 0, 'theta': 0}
        full_model : SpatialPanelResults, optional
            Full model results (if not provided, will fit)

        Returns
        -------
        dict
            Test results with LR statistic, p-value, and conclusion
        """
        # Fit full model if not provided
        if full_model is None:
            full_model = self.fit(include_wx=True)

        # Determine which parameters to restrict
        restrict_rho = "rho" in restrictions
        restrict_lambda = "lambda" in restrictions
        restrict_theta = "theta" in restrictions

        # Set initial values and options for restricted model
        rho_init = restrictions.get("rho", 0.0) if restrict_rho else full_model.rho
        lambda_init = restrictions.get("lambda", 0.0) if restrict_lambda else full_model.lambda_
        include_wx = not restrict_theta

        # Fit restricted model
        if restrict_rho and restrict_lambda and restrict_theta:
            # This is just OLS
            from ..linear.fixed_effects import FixedEffects

            fe_model = FixedEffects(self.formula, self.data, self.entity_col, self.time_col)
            restricted_model = fe_model.fit()
            restricted_ll = restricted_model.log_likelihood
        else:
            # Fit with restrictions
            restricted_model = self.fit(
                rho_init=rho_init if not restrict_rho else 0,
                lambda_init=lambda_init if not restrict_lambda else 0,
                include_wx=include_wx,
            )
            restricted_ll = restricted_model.log_likelihood

        # Compute LR statistic
        lr_statistic = 2 * (full_model.log_likelihood - restricted_ll)

        # Degrees of freedom = number of restrictions
        df = len(restrictions)

        # P-value from chi-squared distribution
        p_value = 1 - chi2.cdf(lr_statistic, df)

        # Determine model type from restrictions
        if restrict_rho and not restrict_lambda and restrict_theta:
            model_type = "SEM"
        elif not restrict_rho and restrict_lambda and restrict_theta:
            model_type = "SAR"
        elif not restrict_rho and restrict_lambda and not restrict_theta:
            model_type = "SDM"
        elif not restrict_rho and restrict_theta and not restrict_lambda:
            model_type = "SAC"
        else:
            model_type = "Mixed"

        return {
            "lr_statistic": lr_statistic,
            "p_value": p_value,
            "df": df,
            "reject_restrictions": p_value < 0.05,
            "full_ll": full_model.log_likelihood,
            "restricted_ll": restricted_ll,
            "restricted_model_type": model_type,
            "conclusion": f"{'Reject' if p_value < 0.05 else 'Cannot reject'} restrictions at 5% level",
        }

    def identify_model_type(self, result: SpatialPanelResults) -> str:
        """
        Identify which nested spatial model based on estimated parameters.

        Parameters
        ----------
        result : SpatialPanelResults
            Estimation results

        Returns
        -------
        str
            Model type: 'SAR', 'SEM', 'SDM', 'SAC', 'SDEM', 'SDEM-SEM', 'GNS', or 'OLS'
        """
        # Get parameters and their significance
        params = result.params

        # Check rho significance
        rho_sig = False
        if "rho" in params.index:
            rho = params.loc["rho", "coefficient"]
            rho_se = params.loc["rho", "std_error"]
            rho_sig = abs(rho / rho_se) > 1.96 if rho_se > 0 else False

        # Check lambda significance
        lambda_sig = False
        if "lambda" in params.index:
            lambda_ = params.loc["lambda", "coefficient"]
            lambda_se = params.loc["lambda", "std_error"]
            lambda_sig = abs(lambda_ / lambda_se) > 1.96 if lambda_se > 0 else False

        # Check theta (WX parameters) significance
        theta_params = [p for p in params.index if p.startswith("theta_")]
        theta_sig = False
        if theta_params:
            for p in theta_params:
                coef = params.loc[p, "coefficient"]
                se = params.loc[p, "std_error"]
                if se > 0 and abs(coef / se) > 1.96:
                    theta_sig = True
                    break

        # Determine model type based on significance
        if rho_sig and not theta_sig and not lambda_sig:
            return "SAR"
        elif not rho_sig and not theta_sig and lambda_sig:
            return "SEM"
        elif rho_sig and theta_sig and not lambda_sig:
            return "SDM"
        elif rho_sig and not theta_sig and lambda_sig:
            return "SAC"
        elif not rho_sig and theta_sig and not lambda_sig:
            return "SDEM"
        elif not rho_sig and theta_sig and lambda_sig:
            return "SDEM-SEM"
        elif rho_sig and theta_sig and lambda_sig:
            return "GNS"
        else:
            return "OLS"

    def _row_normalize(self, W):
        """Row-normalize a weight matrix."""
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        return W / row_sums[:, np.newaxis]
