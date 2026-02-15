"""
Panel Heckman selection model.

This module implements the Heckman two-step correction for sample selection bias
in panel data models.

The model consists of:
1. Selection equation: s*_it = Z_it'γ + u_it, s_it = 1[s*_it > 0]
2. Outcome equation: y_it = X_it'β + ε_it if s_it = 1

Where (u_it, ε_it) ~ Bivariate Normal with correlation ρ.
If ρ ≠ 0, OLS estimation of the outcome equation is biased.

References
----------
Heckman, J.J. (1979). "Sample Selection Bias as a Specification Error."
    Econometrica, 47(1), 153-161.
Wooldridge, J.M. (1995). "Selection Corrections for Panel Data Models Under
    Conditional Mean Independence Assumptions." Journal of Econometrics, 68(1), 115-132.
"""

import warnings
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from panelbox.models.base import NonlinearPanelModel, PanelModelResults


class PanelHeckman(NonlinearPanelModel):
    """
    Panel Heckman selection model.

    Corrects for sample selection bias using either two-step or MLE estimation.

    Parameters
    ----------
    endog : array-like
        Outcome variable (observed only if selected)
    exog : array-like
        Regressors for outcome equation
    selection : array-like
        Binary selection indicator (1 if observed, 0 otherwise)
    exog_selection : array-like
        Regressors for selection equation
    entity : array-like, optional
        Entity/individual identifiers
    time : array-like, optional
        Time period identifiers
    method : str, default='two_step'
        Estimation method ('two_step' or 'mle')
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        selection: np.ndarray,
        exog_selection: np.ndarray,
        entity: Optional[np.ndarray] = None,
        time: Optional[np.ndarray] = None,
        method: Literal["two_step", "mle"] = "two_step",
    ):
        # Store entity and time before calling super().__init__
        self.entity = entity
        self.time = time

        # Initialize parent with outcome data
        super().__init__(endog, exog)

        self.selection = selection
        self.exog_selection = exog_selection
        self.method = method

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate input data."""
        # Check dimensions
        if len(self.selection) != len(self.endog):
            raise ValueError("Selection and outcome must have same length")

        if len(self.exog_selection) != len(self.endog):
            raise ValueError("Selection regressors must match data length")

        # Check selection is binary
        unique_sel = np.unique(self.selection)
        if not np.array_equal(unique_sel, [0, 1]) and not np.array_equal(unique_sel, [1]):
            raise ValueError("Selection must be binary (0/1)")

        # Check for exclusion restriction (at least one variable in Z not in X)
        if self.exog_selection.shape[1] <= self.exog.shape[1]:
            warnings.warn(
                "Selection equation should include at least one exclusion restriction "
                "(variable not in outcome equation) for identification"
            )

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for MLE estimation.

        Parameters
        ----------
        params : np.ndarray
            Combined parameter vector [beta, gamma, sigma, rho]

        Returns
        -------
        float
            Negative log-likelihood
        """
        # Extract parameters
        k_outcome = self.exog.shape[1]
        k_selection = self.exog_selection.shape[1]

        beta = params[:k_outcome]
        gamma = params[k_outcome : k_outcome + k_selection]
        sigma = np.exp(params[-2])  # Ensure positive
        rho = np.tanh(params[-1])  # Ensure in [-1, 1]

        # Compute linear predictions
        Xb = self.exog @ beta
        Zg = self.exog_selection @ gamma

        # Log-likelihood contributions
        llf = 0

        for i in range(len(self.selection)):
            if self.selection[i] == 1:
                # Observed outcome
                residual = (self.endog[i] - Xb[i]) / sigma
                z_star = (Zg[i] + rho * residual) / np.sqrt(1 - rho**2)

                llf += (
                    -0.5 * np.log(2 * np.pi)
                    - np.log(sigma)
                    - 0.5 * residual**2
                    + np.log(stats.norm.cdf(z_star))
                )
            else:
                # Not selected
                llf += np.log(stats.norm.cdf(-Zg[i]))

        return -llf  # Return negative for minimization

    def predict(self, params=None, exog=None):
        """Generate predictions (required by base class)."""
        if params is None:
            if self.results is not None:
                params = self.results.params
            else:
                raise ValueError("Model not fitted yet")

        k_outcome = self.exog.shape[1]
        beta = params[:k_outcome]

        if exog is None:
            return self.exog @ beta
        else:
            return exog @ beta

    def fit(self, method: Optional[str] = None, **kwargs) -> "PanelHeckmanResult":
        """
        Estimate Heckman model.

        Parameters
        ----------
        method : str, optional
            Override default estimation method
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        PanelHeckmanResult
            Fitted model results
        """
        if method is None:
            method = self.method

        # Performance warnings
        n_obs = len(self.selection)
        if method == "mle" and n_obs > 500:
            warnings.warn(
                "MLE with N>500 may take several minutes. "
                "Consider two-step estimation for large samples.",
                UserWarning,
            )

        if method == "mle" and kwargs.get("quadrature_points", 10) > 15:
            warnings.warn(
                "MLE with >15 quadrature points may be very slow. "
                "Consider q=10 for exploratory analysis.",
                UserWarning,
            )

        # Selection rate warnings
        selection_rate = np.mean(self.selection)
        if selection_rate < 0.05 or selection_rate > 0.95:
            warnings.warn(
                f"Extreme selection rate ({selection_rate:.1%}). "
                "Inverse Mills ratios may be unstable. Check model specification.",
                UserWarning,
            )

        if method == "two_step":
            return self._two_step_estimation()
        elif method == "mle":
            return self._mle_estimation(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _two_step_estimation(self) -> "PanelHeckmanResult":
        """
        Two-step Heckman procedure.

        Step 1: Probit for selection equation
        Step 2: OLS with inverse Mills ratio correction
        """
        # Check if all observations are selected
        selected = self.selection == 1
        n_selected = np.sum(selected)

        # If all selected, no selection bias - return OLS results
        if n_selected == len(self.selection):
            warnings.warn(
                "All observations are selected. No selection bias to correct. Returning OLS estimates."
            )

            # Simple OLS
            XtX_inv = np.linalg.inv(self.exog.T @ self.exog)
            beta_hat = XtX_inv @ self.exog.T @ self.endog
            residuals = self.endog - self.exog @ beta_hat
            sigma_hat = np.sqrt(np.mean(residuals**2))

            # Dummy gamma values (not meaningful when all selected)
            gamma_hat = np.zeros(self.exog_selection.shape[1])
            rho_hat = 0.0  # No selection bias
            lambda_imr = np.zeros(len(self.selection))

            params = np.concatenate([beta_hat, gamma_hat, [sigma_hat, rho_hat]])

            return PanelHeckmanResult(
                model=self,
                params=params,
                method="two_step",
                probit_params=gamma_hat,
                outcome_params=beta_hat,
                sigma=sigma_hat,
                rho=rho_hat,
                lambda_imr=lambda_imr,
            )

        # Step 1: Estimate selection equation (Probit)
        from scipy.optimize import minimize

        def probit_llf(params):
            linear_pred = self.exog_selection @ params
            prob = stats.norm.cdf(linear_pred)
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            llf = np.sum(self.selection * np.log(prob) + (1 - self.selection) * np.log(1 - prob))
            return -llf

        # Initial values for probit
        gamma_init = np.zeros(self.exog_selection.shape[1])
        probit_result = minimize(probit_llf, gamma_init, method="BFGS")
        gamma_hat = probit_result.x

        # Compute inverse Mills ratio
        linear_pred_sel = self.exog_selection @ gamma_hat
        lambda_imr = np.zeros_like(linear_pred_sel)

        # IMR = phi(Zg) / Phi(Zg) for selected observations
        # Clip CDF to avoid division issues
        cdf_vals = np.clip(stats.norm.cdf(linear_pred_sel[selected]), 1e-10, 1 - 1e-10)
        lambda_imr[selected] = stats.norm.pdf(linear_pred_sel[selected]) / cdf_vals

        # Step 2: Augmented OLS on selected sample
        y_selected = self.endog[selected]
        X_selected = self.exog[selected]
        lambda_selected = lambda_imr[selected].reshape(-1, 1)
        X_augmented = np.column_stack([X_selected, lambda_selected])

        # OLS estimation
        XtX_inv = np.linalg.inv(X_augmented.T @ X_augmented)
        beta_augmented = XtX_inv @ X_augmented.T @ y_selected

        # Extract coefficients
        beta_hat = beta_augmented[:-1]
        lambda_coef = beta_augmented[-1]

        # Estimate rho and sigma
        sigma_hat = np.sqrt(
            np.mean(
                (y_selected - X_selected @ beta_hat - lambda_coef * lambda_selected.ravel()) ** 2
            )
        )

        # Clip rho to valid range
        rho_hat = np.clip(lambda_coef / sigma_hat, -0.99, 0.99)

        # Combine parameters
        params = np.concatenate([beta_hat, gamma_hat, [sigma_hat, rho_hat]])

        return PanelHeckmanResult(
            model=self,
            params=params,
            method="two_step",
            probit_params=gamma_hat,
            outcome_params=beta_hat,
            sigma=sigma_hat,
            rho=rho_hat,
            lambda_imr=lambda_imr,
        )

    def _mle_estimation(self, **kwargs) -> "PanelHeckmanResult":
        """
        Full information maximum likelihood estimation.
        """
        # Initial values from two-step
        two_step = self._two_step_estimation()

        k_outcome = self.exog.shape[1]
        k_selection = self.exog_selection.shape[1]

        # Initial parameters for MLE
        init_params = np.concatenate(
            [
                two_step.outcome_params,
                two_step.probit_params,
                [np.log(two_step.sigma)],  # Log transformation
                [np.arctanh(np.clip(two_step.rho, -0.99, 0.99))],  # Fisher transformation
            ]
        )

        # Optimize
        result = minimize(
            self._log_likelihood, init_params, method="BFGS", options={"maxiter": 1000, **kwargs}
        )

        if not result.success:
            warnings.warn(f"MLE did not converge: {result.message}")

        # Extract parameters
        params = result.x
        beta = params[:k_outcome]
        gamma = params[k_outcome : k_outcome + k_selection]
        sigma = np.exp(params[-2])
        rho = np.tanh(params[-1])

        # Compute IMR for consistency
        linear_pred_sel = self.exog_selection @ gamma
        lambda_imr = stats.norm.pdf(linear_pred_sel) / stats.norm.cdf(linear_pred_sel)

        return PanelHeckmanResult(
            model=self,
            params=np.concatenate([beta, gamma, [sigma, rho]]),
            method="mle",
            probit_params=gamma,
            outcome_params=beta,
            sigma=sigma,
            rho=rho,
            lambda_imr=lambda_imr,
            llf=-result.fun,
            converged=result.success,
        )


class PanelHeckmanResult(PanelModelResults):
    """Results class for Panel Heckman model."""

    def __init__(
        self,
        model: PanelHeckman,
        params: np.ndarray,
        method: str,
        probit_params: np.ndarray,
        outcome_params: np.ndarray,
        sigma: float,
        rho: float,
        lambda_imr: np.ndarray,
        llf: Optional[float] = None,
        converged: bool = True,
    ):
        self.model = model
        self.params = params
        self.method = method
        self.probit_params = probit_params
        self.outcome_params = outcome_params
        self.sigma = sigma
        self.rho = rho
        self.lambda_imr = lambda_imr
        self.llf = llf
        self.converged = converged

        # Number of observations
        self.n_selected = np.sum(model.selection)
        self.n_total = len(model.selection)

    def summary(self) -> str:
        """Generate summary of results."""
        output = [
            "Panel Heckman Selection Model Results",
            "=" * 60,
            f"Method: {self.method.upper()}",
            f"Total observations: {self.n_total}",
            f"Selected observations: {self.n_selected}",
            f"Censored observations: {self.n_total - self.n_selected}",
            "",
        ]

        if self.llf is not None:
            output.append(f"Log-likelihood: {self.llf:.4f}")

        output.extend(["", "Selection Equation (Probit):", "-" * 40])

        for i, coef in enumerate(self.probit_params):
            output.append(f"gamma_{i}: {coef:.4f}")

        output.extend(["", "Outcome Equation:", "-" * 40])

        for i, coef in enumerate(self.outcome_params):
            output.append(f"beta_{i}: {coef:.4f}")

        output.extend(
            [
                "",
                "Selection Parameters:",
                "-" * 40,
                f"sigma: {self.sigma:.4f}",
                f"rho: {self.rho:.4f}",
            ]
        )

        # Test for selection bias
        if abs(self.rho) > 0.1:
            output.append("")
            if self.rho > 0:
                output.append("Note: Positive selection (rho > 0)")
            else:
                output.append("Note: Negative selection (rho < 0)")
            output.append("Selection bias is present. OLS would be biased.")
        else:
            output.append("\nNote: Little evidence of selection bias (rho ≈ 0)")

        return "\n".join(output)

    def predict(
        self,
        exog: Optional[np.ndarray] = None,
        exog_selection: Optional[np.ndarray] = None,
        type: Literal["unconditional", "conditional"] = "unconditional",
    ) -> np.ndarray:
        """
        Predict outcomes.

        Parameters
        ----------
        exog : np.ndarray, optional
            Outcome equation regressors
        exog_selection : np.ndarray, optional
            Selection equation regressors
        type : str
            'unconditional': E[y*] (latent outcome)
            'conditional': E[y|selected] (observed outcome)

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if exog is None:
            exog = self.model.exog
        if exog_selection is None:
            exog_selection = self.model.exog_selection

        # Linear prediction for outcome
        y_pred = exog @ self.outcome_params

        if type == "conditional":
            # Add selection correction
            linear_pred_sel = exog_selection @ self.probit_params
            lambda_pred = stats.norm.pdf(linear_pred_sel) / stats.norm.cdf(linear_pred_sel)
            y_pred += self.rho * self.sigma * lambda_pred

        return y_pred

    def selection_test(self) -> Dict[str, float]:
        """
        Test for presence of selection bias.

        Returns
        -------
        dict
            Test statistics for selection bias
        """
        # Wald test on rho
        # Under H0: rho = 0 (no selection bias)
        # This is a simplified version; proper standard errors needed

        z_stat = self.rho / 0.1  # Simplified; should use proper SE
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "rho": self.rho,
            "z_statistic": z_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    def selection_effect(self, alpha: float = 0.05) -> dict:
        """
        Test for selection bias (H0: ρ = 0).

        This is equivalent to testing whether the IMR coefficient is significant
        in the outcome equation.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for test

        Returns
        -------
        dict
            Dictionary with:
            - 'statistic': test statistic
            - 'pvalue': two-sided p-value
            - 'reject': bool indicating rejection at alpha level
            - 'interpretation': str describing result

        Examples
        --------
        >>> result = model.fit()
        >>> test = result.selection_effect()
        >>> print(test['interpretation'])

        Notes
        -----
        For two-step estimator, this tests H0: θ = 0 where θ = ρσ_ε.
        Since θ = 0 ⟺ ρ = 0, this directly tests for selection bias.
        """
        from .inverse_mills import test_selection_effect

        # For two-step, we can test using IMR coefficient
        # θ̂ = ρ̂ σ̂_ε
        theta_hat = self.rho * self.sigma

        # Simplified SE (should be from Murphy-Topel correction)
        # Using approximation: SE(θ) ≈ 0.1 * σ
        theta_se = 0.1 * self.sigma  # TODO: Replace with proper SE

        result = test_selection_effect(theta_hat, theta_se, alpha)
        return result

    def imr_diagnostics(self) -> dict:
        """
        Compute diagnostic statistics for Inverse Mills Ratio.

        Returns
        -------
        dict
            Dictionary with:
            - 'imr_mean': mean IMR for selected observations
            - 'imr_std': std dev of IMR
            - 'imr_min': minimum IMR
            - 'imr_max': maximum IMR
            - 'high_imr_count': count of observations with very high IMR (> 2)
            - 'selection_rate': fraction of observations selected

        Examples
        --------
        >>> result = model.fit()
        >>> diag = result.imr_diagnostics()
        >>> print(f"Mean IMR: {diag['imr_mean']:.3f}")
        >>> print(f"High selection observations: {diag['high_imr_count']}")

        Notes
        -----
        High IMR values (> 2) indicate strong selection effects.
        """
        from .inverse_mills import imr_diagnostics

        linear_pred = self.model.exog_selection @ self.probit_params
        diag = imr_diagnostics(linear_pred, self.model.selection)
        return diag

    def plot_imr(self, figsize=(12, 5)):
        """
        Plot Inverse Mills Ratio diagnostics.

        Creates two plots:
        1. Scatter: IMR vs predicted selection probability
        2. Histogram of IMR for selected observations

        Parameters
        ----------
        figsize : tuple, default=(12, 5)
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            Figure object

        Examples
        --------
        >>> result = model.fit()
        >>> fig = result.plot_imr()
        >>> plt.show()

        Notes
        -----
        This helps identify:
        - Observations with strong selection effects (high IMR)
        - Relationship between selection probability and IMR
        - Distribution of selection correction
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Compute selection probabilities
        linear_pred = self.model.exog_selection @ self.probit_params
        selection_prob = stats.norm.cdf(linear_pred)

        # Get IMR for selected observations
        selected_mask = self.model.selection == 1
        imr_selected = self.lambda_imr[selected_mask]
        prob_selected = selection_prob[selected_mask]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: IMR vs Selection Probability
        axes[0].scatter(prob_selected, imr_selected, alpha=0.5, s=10)
        axes[0].set_xlabel("Predicted Selection Probability")
        axes[0].set_ylabel("Inverse Mills Ratio")
        axes[0].set_title("IMR vs Selection Probability")
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=2, color="r", linestyle="--", alpha=0.5, label="High IMR threshold")
        axes[0].legend()

        # Plot 2: Histogram of IMR
        axes[1].hist(imr_selected, bins=30, edgecolor="black", alpha=0.7)
        axes[1].axvline(x=2, color="r", linestyle="--", label="High IMR threshold")
        axes[1].set_xlabel("Inverse Mills Ratio")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of IMR (Selected Sample)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def compare_ols_heckman(self) -> dict:
        """
        Compare OLS (biased) vs Heckman (corrected) estimates.

        Estimates OLS on the selected sample and compares coefficients
        to Heckman estimates.

        Returns
        -------
        dict
            Dictionary with:
            - 'beta_ols': OLS coefficients
            - 'beta_heckman': Heckman coefficients
            - 'difference': beta_ols - beta_heckman
            - 'pct_difference': percentage difference
            - 'interpretation': str describing results

        Examples
        --------
        >>> result = model.fit()
        >>> comparison = result.compare_ols_heckman()
        >>> print(comparison['interpretation'])

        Notes
        -----
        Large differences indicate substantial selection bias.
        If ρ ≈ 0, OLS and Heckman should be similar.
        """
        # Extract selected sample
        selected_mask = self.model.selection == 1
        y_selected = self.model.endog[selected_mask]
        X_selected = self.model.exog[selected_mask]

        # OLS estimation
        beta_ols = np.linalg.lstsq(X_selected, y_selected, rcond=None)[0]
        beta_heckman = self.outcome_params

        # Compute differences
        difference = beta_ols - beta_heckman
        pct_difference = 100 * difference / (np.abs(beta_heckman) + 1e-10)

        # Interpretation
        max_abs_diff = np.max(np.abs(difference))
        if max_abs_diff > 0.1:
            interpretation = (
                f"Substantial selection bias detected (max diff: {max_abs_diff:.3f}). "
                f"OLS estimates are biased. Heckman correction is necessary."
            )
        else:
            interpretation = (
                f"Minimal selection bias (max diff: {max_abs_diff:.3f}). "
                f"OLS and Heckman yield similar results."
            )

        return {
            "beta_ols": beta_ols,
            "beta_heckman": beta_heckman,
            "difference": difference,
            "pct_difference": pct_difference,
            "max_abs_difference": max_abs_diff,
            "interpretation": interpretation,
        }
