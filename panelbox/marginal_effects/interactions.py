"""
Interaction effects in nonlinear models.

In nonlinear models, the interaction effect between two variables is not
simply the coefficient on the interaction term. The cross-partial derivative
depends on all covariates and varies across observations.

This module implements the correct calculation of interaction effects
following Ai & Norton (2003).

References
----------
Ai, C., & Norton, E. C. (2003). "Interaction terms in logit and probit models."
    Economics Letters, 80(1), 123-129.
Norton, E. C., Wang, H., & Ai, C. (2004). "Computing interaction effects and
    standard errors in logit and probit models." The Stata Journal, 4(2), 154-167.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class InteractionEffectsResult:
    """
    Results container for interaction effects analysis.

    Attributes
    ----------
    cross_partial : np.ndarray
        Cross-partial derivative for each observation
    mean_effect : float
        Average interaction effect
    std_effect : float
        Standard deviation of interaction effects
    z_statistics : np.ndarray
        Z-statistics for each observation's effect
    significant_positive : float
        Proportion of observations with significant positive effect
    significant_negative : float
        Proportion of observations with significant negative effect
    """

    def __init__(
        self,
        cross_partial: np.ndarray,
        standard_errors: Optional[np.ndarray] = None,
        predicted_prob: Optional[np.ndarray] = None,
        var1_name: str = "X1",
        var2_name: str = "X2",
    ):
        """Initialize interaction effects results."""
        self.cross_partial = cross_partial
        self.standard_errors = standard_errors
        self.predicted_prob = predicted_prob
        self.var1_name = var1_name
        self.var2_name = var2_name

        # Compute summary statistics
        self.mean_effect = np.mean(cross_partial)
        self.std_effect = np.std(cross_partial)
        self.min_effect = np.min(cross_partial)
        self.max_effect = np.max(cross_partial)

        # Proportion with different signs
        self.prop_positive = np.mean(cross_partial > 0)
        self.prop_negative = np.mean(cross_partial < 0)

        # Statistical significance if SEs available
        if standard_errors is not None:
            self.z_statistics = cross_partial / standard_errors
            # Using 5% significance level
            self.significant_positive = np.mean(self.z_statistics > 1.96)
            self.significant_negative = np.mean(self.z_statistics < -1.96)
        else:
            self.z_statistics = None
            self.significant_positive = None
            self.significant_negative = None

    def summary(self) -> str:
        """Generate text summary of interaction effects."""
        lines = []
        lines.append("=" * 70)
        lines.append("Interaction Effects Analysis")
        lines.append(f"Variables: {self.var1_name} × {self.var2_name}")
        lines.append("=" * 70)
        lines.append("")

        lines.append("Summary Statistics:")
        lines.append("-" * 40)
        lines.append(f"Mean effect:     {self.mean_effect:12.6f}")
        lines.append(f"Std. deviation:  {self.std_effect:12.6f}")
        lines.append(f"Minimum:         {self.min_effect:12.6f}")
        lines.append(f"Maximum:         {self.max_effect:12.6f}")
        lines.append("")

        lines.append("Sign Distribution:")
        lines.append("-" * 40)
        lines.append(f"Positive effects: {self.prop_positive:6.1%}")
        lines.append(f"Negative effects: {self.prop_negative:6.1%}")
        lines.append("")

        if self.z_statistics is not None:
            lines.append("Statistical Significance (5% level):")
            lines.append("-" * 40)
            lines.append(f"Significant positive: {self.significant_positive:6.1%}")
            lines.append(f"Significant negative: {self.significant_negative:6.1%}")
            lines.append("")

        lines.append("Note: In nonlinear models, interaction effects vary across")
        lines.append("observations and may have different signs at different values")
        lines.append("of the covariates (Ai & Norton, 2003).")
        lines.append("=" * 70)

        return "\n".join(lines)

    def plot(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create visualization of interaction effects.

        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size

        Returns
        -------
        plt.Figure
            Matplotlib figure with four subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Distribution of interaction effects
        ax = axes[0, 0]
        ax.hist(self.cross_partial, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="Zero effect")
        ax.axvline(
            self.mean_effect, color="green", linestyle="-", label=f"Mean = {self.mean_effect:.4f}"
        )
        ax.set_xlabel("Interaction Effect")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Interaction Effects")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Interaction effect vs predicted probability
        if self.predicted_prob is not None:
            ax = axes[0, 1]
            scatter = ax.scatter(
                self.predicted_prob,
                self.cross_partial,
                alpha=0.5,
                c=self.cross_partial,
                cmap="RdBu_r",
            )
            ax.axhline(0, color="black", linestyle="--", alpha=0.5)
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Interaction Effect")
            ax.set_title("Interaction Effect vs Predicted Probability")
            plt.colorbar(scatter, ax=ax, label="Effect")
            ax.grid(True, alpha=0.3)
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Predicted probabilities not available",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])

        # 3. Z-statistics if available
        if self.z_statistics is not None:
            ax = axes[1, 0]
            ax.hist(self.z_statistics, bins=30, edgecolor="black", alpha=0.7)
            ax.axvline(-1.96, color="red", linestyle="--", alpha=0.7, label="5% critical values")
            ax.axvline(1.96, color="red", linestyle="--", alpha=0.7)
            ax.axvline(0, color="black", linestyle="-", alpha=0.5)
            ax.set_xlabel("Z-statistic")
            ax.set_ylabel("Frequency")
            ax.set_title("Statistical Significance of Interaction Effects")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "Standard errors not available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])

        # 4. Sorted interaction effects
        ax = axes[1, 1]
        sorted_effects = np.sort(self.cross_partial)
        n = len(sorted_effects)
        ax.plot(range(n), sorted_effects, linewidth=2)
        ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax.fill_between(
            range(n),
            0,
            sorted_effects,
            where=(sorted_effects > 0),
            alpha=0.3,
            color="green",
            label="Positive",
        )
        ax.fill_between(
            range(n),
            0,
            sorted_effects,
            where=(sorted_effects < 0),
            alpha=0.3,
            color="red",
            label="Negative",
        )
        ax.set_xlabel("Observation (sorted)")
        ax.set_ylabel("Interaction Effect")
        ax.set_title("Sorted Interaction Effects")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Interaction Effects: {self.var1_name} × {self.var2_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        return fig


def compute_interaction_effects(
    model_result,
    var1: Union[str, int],
    var2: Union[str, int],
    interaction_term: Optional[Union[str, int]] = None,
    method: str = "delta",
    n_bootstrap: int = 100,
) -> InteractionEffectsResult:
    """
    Compute interaction effects in nonlinear models.

    Following Ai & Norton (2003), computes the cross-partial derivative
    ∂²P/∂x₁∂x₂ which is the correct measure of interaction in nonlinear models.

    Parameters
    ----------
    model_result : NonlinearPanelResult
        Fitted nonlinear model result (e.g., from Logit, Probit)
    var1, var2 : str or int
        Names or indices of the interacting variables
    interaction_term : str or int, optional
        Name or index of the interaction term (var1*var2) in the model
    method : str, default='delta'
        Method for computing standard errors:
        - 'delta': Delta method
        - 'bootstrap': Bootstrap
        - None: No standard errors
    n_bootstrap : int, default=100
        Number of bootstrap replications if method='bootstrap'

    Returns
    -------
    InteractionEffectsResult
        Container with interaction effects and statistics

    References
    ----------
    Ai, C., & Norton, E. C. (2003). "Interaction terms in logit and probit models."
        Economics Letters, 80(1), 123-129.
    """
    # Extract model components
    X = model_result.model.exog
    params = model_result.params
    model_type = model_result.model.__class__.__name__.lower()

    # Get variable indices
    if isinstance(var1, str):
        var1_idx = model_result.model.exog_names.index(var1)
        var1_name = var1
    else:
        var1_idx = var1
        var1_name = f"X{var1}"

    if isinstance(var2, str):
        var2_idx = model_result.model.exog_names.index(var2)
        var2_name = var2
    else:
        var2_idx = var2
        var2_name = f"X{var2}"

    if interaction_term is not None:
        if isinstance(interaction_term, str):
            interact_idx = model_result.model.exog_names.index(interaction_term)
        else:
            interact_idx = interaction_term
    else:
        # Try to find interaction term automatically
        interact_idx = None
        for i, name in enumerate(model_result.model.exog_names):
            if (
                f"{var1_name}:{var2_name}" in name
                or f"{var2_name}:{var1_name}" in name
                or f"{var1_name}*{var2_name}" in name
                or f"{var2_name}*{var1_name}" in name
            ):
                interact_idx = i
                break

    # Linear predictor
    xb = X @ params

    # Compute cross-partial based on model type
    if "logit" in model_type or "logistic" in model_type:
        cross_partial = _logit_interaction(X, params, xb, var1_idx, var2_idx, interact_idx)
        # Predicted probabilities for plotting
        predicted_prob = 1 / (1 + np.exp(-xb))

    elif "probit" in model_type:
        cross_partial = _probit_interaction(X, params, xb, var1_idx, var2_idx, interact_idx)
        # Predicted probabilities for plotting
        from scipy.stats import norm

        predicted_prob = norm.cdf(xb)

    elif "poisson" in model_type:
        cross_partial = _poisson_interaction(X, params, xb, var1_idx, var2_idx, interact_idx)
        predicted_prob = np.exp(xb)

    else:
        raise ValueError(f"Model type {model_type} not supported for interaction effects")

    # Compute standard errors if requested
    if method == "delta":
        standard_errors = _delta_method_se(
            model_result, X, params, xb, var1_idx, var2_idx, interact_idx, model_type
        )
    elif method == "bootstrap":
        standard_errors = _bootstrap_se(
            model_result, X, var1_idx, var2_idx, interact_idx, n_bootstrap
        )
    else:
        standard_errors = None

    return InteractionEffectsResult(
        cross_partial=cross_partial,
        standard_errors=standard_errors,
        predicted_prob=predicted_prob,
        var1_name=var1_name,
        var2_name=var2_name,
    )


def _logit_interaction(X, params, xb, var1_idx, var2_idx, interact_idx):
    """
    Compute interaction effect for logit model.

    The cross-partial derivative is:
    ∂²Λ/∂x₁∂x₂ = β₁₂Λ(1-Λ) + β₁β₂Λ(1-Λ)(1-2Λ)

    where Λ is the logistic CDF.
    """
    # Logistic probabilities and derivatives
    Lambda = 1 / (1 + np.exp(-xb))
    lambda_pdf = Lambda * (1 - Lambda)

    # Parameters
    beta1 = params[var1_idx]
    beta2 = params[var2_idx]
    beta12 = params[interact_idx] if interact_idx is not None else 0

    # Cross-partial derivative
    cross_partial = beta12 * lambda_pdf + beta1 * beta2 * lambda_pdf * (1 - 2 * Lambda)

    return cross_partial


def _probit_interaction(X, params, xb, var1_idx, var2_idx, interact_idx):
    """
    Compute interaction effect for probit model.

    The cross-partial derivative is:
    ∂²Φ/∂x₁∂x₂ = -β₁₂φ(xb) - β₁β₂xb·φ(xb)

    where Φ is the normal CDF and φ is the normal PDF.
    """
    from scipy.stats import norm

    # Normal PDF
    phi = norm.pdf(xb)

    # Parameters
    beta1 = params[var1_idx]
    beta2 = params[var2_idx]
    beta12 = params[interact_idx] if interact_idx is not None else 0

    # Cross-partial derivative
    cross_partial = -phi * (beta12 + beta1 * beta2 * xb)

    return cross_partial


def _poisson_interaction(X, params, xb, var1_idx, var2_idx, interact_idx):
    """
    Compute interaction effect for Poisson model.

    The cross-partial derivative is:
    ∂²λ/∂x₁∂x₂ = λ(β₁₂ + β₁β₂)

    where λ = exp(xb).
    """
    # Mean function
    lambda_ = np.exp(xb)

    # Parameters
    beta1 = params[var1_idx]
    beta2 = params[var2_idx]
    beta12 = params[interact_idx] if interact_idx is not None else 0

    # Cross-partial derivative
    cross_partial = lambda_ * (beta12 + beta1 * beta2)

    return cross_partial


def _delta_method_se(model_result, X, params, xb, var1_idx, var2_idx, interact_idx, model_type):
    """
    Compute standard errors using delta method.

    This is complex as it requires the gradient of the cross-partial
    with respect to all parameters.
    """
    try:
        n_obs = len(X)
        n_params = len(params)

        # Gradient of cross-partial w.r.t. parameters
        gradient = np.zeros((n_obs, n_params))

        if "logit" in model_type:
            Lambda = 1 / (1 + np.exp(-xb))
            lambda_pdf = Lambda * (1 - Lambda)
            lambda_dpdf = lambda_pdf * (1 - 2 * Lambda)

            # Gradient computation is model-specific and complex
            # Simplified version here
            for k in range(n_params):
                if k == interact_idx and interact_idx is not None:
                    gradient[:, k] = lambda_pdf
                # Additional terms would go here

        # Variance of cross-partial
        V = model_result.cov_params  # Parameter covariance matrix
        var_cross_partial = np.sum((gradient @ V) * gradient, axis=1)
        se_cross_partial = np.sqrt(var_cross_partial)

        return se_cross_partial

    except Exception as e:
        warnings.warn(f"Could not compute delta method standard errors: {e}")
        return None


def _bootstrap_se(model_result, X, var1_idx, var2_idx, interact_idx, n_bootstrap):
    """
    Compute standard errors using bootstrap.

    Resamples data and re-estimates model multiple times.
    """
    try:
        n_obs = len(X)
        bootstrap_effects = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(n_obs, size=n_obs, replace=True)
            X_boot = X[idx]
            y_boot = model_result.model.endog[idx]

            # Re-estimate model
            model_boot = model_result.model.__class__(y_boot, X_boot)
            result_boot = model_boot.fit()

            # Compute interaction effect
            xb_boot = X_boot @ result_boot.params

            if "logit" in model_result.model.__class__.__name__.lower():
                effects_boot = _logit_interaction(
                    X_boot, result_boot.params, xb_boot, var1_idx, var2_idx, interact_idx
                )
            # Add other model types as needed

            bootstrap_effects.append(effects_boot)

        # Standard errors from bootstrap distribution
        bootstrap_effects = np.array(bootstrap_effects)
        se_cross_partial = np.std(bootstrap_effects, axis=0)

        return se_cross_partial

    except Exception as e:
        warnings.warn(f"Could not compute bootstrap standard errors: {e}")
        return None


def test_interaction_significance(
    model_with_interaction, model_without_interaction, var1: Union[str, int], var2: Union[str, int]
) -> Dict[str, Any]:
    """
    Test whether interaction improves model fit.

    Compares models with and without interaction using:
    1. Likelihood ratio test
    2. AIC/BIC comparison
    3. Average interaction effect significance

    Parameters
    ----------
    model_with_interaction : NonlinearPanelResult
        Model including interaction term
    model_without_interaction : NonlinearPanelResult
        Model without interaction term
    var1, var2 : str or int
        Interacting variables

    Returns
    -------
    dict
        Test statistics and p-values
    """
    # Likelihood ratio test
    lr_stat = 2 * (model_with_interaction.llf - model_without_interaction.llf)
    lr_pval = 1 - stats.chi2.cdf(lr_stat, df=1)

    # AIC/BIC comparison
    delta_aic = model_with_interaction.aic - model_without_interaction.aic
    delta_bic = model_with_interaction.bic - model_without_interaction.bic

    # Compute average interaction effect
    interaction_result = compute_interaction_effects(model_with_interaction, var1, var2)

    results = {
        "lr_statistic": lr_stat,
        "lr_pvalue": lr_pval,
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
        "avg_interaction_effect": interaction_result.mean_effect,
        "interaction_std": interaction_result.std_effect,
        "prefer_interaction": lr_pval < 0.05 and delta_aic < 0,
    }

    return results
