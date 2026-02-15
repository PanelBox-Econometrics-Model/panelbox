"""
Marginal effects for ordered choice models.

This module implements marginal effects computations for ordered logit/probit models
where effects vary by outcome category.

Author: PanelBox Developers
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from scipy.stats import norm

from panelbox.marginal_effects.delta_method import delta_method_se, numerical_gradient


class OrderedMarginalEffectsResult:
    """
    Container for ordered model marginal effects results.

    For ordered models, marginal effects differ by outcome category.
    The sum of marginal effects across categories equals zero.

    Attributes
    ----------
    marginal_effects : pd.DataFrame
        Marginal effects for each variable and category
    std_errors : pd.DataFrame
        Standard errors via delta method
    parent_result : object
        Parent model result object
    me_type : str
        Type of marginal effect ('ame', 'mem', 'mer')
    at_values : dict or None
        Values used for MER computation
    """

    def __init__(
        self,
        marginal_effects: pd.DataFrame,
        std_errors: pd.DataFrame,
        parent_result,
        me_type: str = "ame",
        at_values: Optional[dict] = None,
    ):
        """Initialize ordered marginal effects result."""
        self.marginal_effects = marginal_effects
        self.std_errors = std_errors
        self.parent_result = parent_result
        self.me_type = me_type.upper()
        self.at_values = at_values
        self.n_categories = marginal_effects.shape[1]

    @property
    def z_stats(self) -> pd.DataFrame:
        """Z-statistics for marginal effects."""
        return self.marginal_effects / self.std_errors

    @property
    def pvalues(self) -> pd.DataFrame:
        """P-values from two-sided z-tests."""
        z_stats = self.z_stats
        return z_stats.applymap(lambda z: 2 * norm.cdf(-abs(z)))

    def verify_sum_to_zero(self, tol: float = 1e-10) -> bool:
        """
        Verify that marginal effects sum to zero across categories.

        This is a key property of ordered models: the sum of marginal
        effects across all categories must equal zero for each variable.

        Parameters
        ----------
        tol : float, default=1e-10
            Tolerance for checking sum equals zero

        Returns
        -------
        bool
            True if all sums are within tolerance of zero
        """
        sums = self.marginal_effects.sum(axis=1)
        return np.all(np.abs(sums) < tol)

    def plot(self, var: str, ax=None, **kwargs):
        """
        Plot marginal effects across categories for a variable.

        Parameters
        ----------
        var : str
            Variable name to plot
        ax : matplotlib axis, optional
            Axis to plot on
        **kwargs : dict
            Additional arguments for matplotlib.pyplot.bar

        Returns
        -------
        ax : matplotlib axis
            The plot axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if var not in self.marginal_effects.index:
            raise ValueError(f"Variable '{var}' not found in marginal effects")

        # Get effects and standard errors for variable
        effects = self.marginal_effects.loc[var]
        errors = self.std_errors.loc[var]

        categories = range(len(effects))
        x_pos = np.arange(len(categories))

        # Bar plot with error bars
        bars = ax.bar(x_pos, effects, yerr=1.96 * errors, **kwargs)

        # Color positive and negative differently
        for i, bar in enumerate(bars):
            if effects.iloc[i] > 0:
                bar.set_color("green")
            else:
                bar.set_color("red")

        ax.set_xlabel("Outcome Category")
        ax.set_ylabel(f"{self.me_type} Marginal Effect")
        ax.set_title(f"Marginal Effects of {var} Across Categories")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Cat {j}" for j in categories])
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        return ax

    def summary(self, alpha: float = 0.05) -> None:
        """
        Print formatted summary of ordered marginal effects.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals
        """
        print(f"\n{self.me_type} Marginal Effects for Ordered Model")
        print("=" * 70)
        print(f"Number of categories: {self.n_categories}")

        if self.at_values is not None:
            print("\nEvaluated at:")
            for var, val in self.at_values.items():
                print(f"  {var} = {val:.4f}")

        print("\nMarginal Effects by Category:")
        print("-" * 70)

        # Format as a nice table
        pd.options.display.float_format = "{:.4f}".format

        # Add significance stars
        def add_stars(pval):
            if pval < 0.001:
                return "***"
            elif pval < 0.01:
                return "**"
            elif pval < 0.05:
                return "*"
            elif pval < 0.1:
                return "."
            else:
                return ""

        pvals = self.pvalues
        stars_df = pvals.applymap(add_stars)

        # Combine effects with significance stars
        output_df = self.marginal_effects.copy()
        for col in output_df.columns:
            output_df[col] = output_df[col].astype(str) + stars_df[col]

        print(output_df)

        print("\nStandard Errors:")
        print(self.std_errors)

        # Verify sum to zero property
        if self.verify_sum_to_zero():
            print("\n✓ Marginal effects sum to zero across categories (verified)")
        else:
            print("\n⚠ Warning: Marginal effects do not sum to zero!")

        print("\nSignificance codes: '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")
        print("=" * 70)

    def __repr__(self) -> str:
        """String representation."""
        n_vars = len(self.marginal_effects)
        return (
            f"OrderedMarginalEffectsResult(type='{self.me_type}', "
            f"n_variables={n_vars}, n_categories={self.n_categories})"
        )


def compute_ordered_ame(
    result, varlist: Optional[List[str]] = None
) -> OrderedMarginalEffectsResult:
    """
    Compute Average Marginal Effects (AME) for ordered models.

    For ordered models, marginal effects vary by outcome category j:
    ∂P(y=j|x)/∂x_k = [f(κ_{j-1} - x'β) - f(κ_j - x'β)] * β_k

    where f is the PDF of the error distribution.

    Parameters
    ----------
    result : OrderedChoiceModel result
        Fitted ordered model result
    varlist : list of str, optional
        Variables to compute AME for (default: all exogenous)

    Returns
    -------
    OrderedMarginalEffectsResult
        Container with marginal effects and standard errors for each category
    """
    model = result.model

    # Get data
    X = model.exog
    n_obs, n_vars = X.shape
    n_categories = model.n_categories

    # Extract parameters
    beta = result.beta
    cutpoints = result.cutpoints

    # Extended cutpoints with boundaries
    cutpoints_extended = np.concatenate([[-np.inf], cutpoints, [np.inf]])

    # Determine PDF based on model type
    if hasattr(model, "_pdf"):
        pdf_func = model._pdf
    elif "Logit" in model.__class__.__name__:
        pdf_func = lambda z: expit(z) * (1 - expit(z))
    else:  # Probit
        pdf_func = stats.norm.pdf

    # Get variable names
    if varlist is None:
        if hasattr(model, "exog_names"):
            varlist = model.exog_names
        else:
            varlist = [f"x{i}" for i in range(n_vars)]

    # Compute AME for each variable and category
    ame_dict = {}
    ame_se_dict = {}

    for k, var in enumerate(varlist):
        if k >= n_vars:
            continue

        # Effects for each category
        category_effects = []
        category_ses = []

        for j in range(n_categories):
            # Compute marginal effect for category j
            effects_ij = []

            for i in range(n_obs):
                linear_pred = X[i] @ beta

                # ∂P(y=j|x)/∂x_k
                z_lower = cutpoints_extended[j] - linear_pred
                z_upper = cutpoints_extended[j + 1] - linear_pred

                pdf_lower = pdf_func(z_lower) if np.isfinite(z_lower) else 0
                pdf_upper = pdf_func(z_upper) if np.isfinite(z_upper) else 0

                me_ijk = beta[k] * (pdf_lower - pdf_upper)
                effects_ij.append(me_ijk)

            # Average marginal effect for category j
            ame_j = np.mean(effects_ij)
            category_effects.append(ame_j)

            # Compute standard error via delta method
            # Define function for this specific category effect
            def category_j_ame(params):
                beta_temp = params[:n_vars]
                cutpoint_params_temp = params[n_vars : n_vars + len(cutpoints)]

                # Transform cutpoints
                if hasattr(model, "_transform_cutpoints"):
                    cutpoints_temp = model._transform_cutpoints(cutpoint_params_temp)
                else:
                    cutpoints_temp = cutpoint_params_temp

                cutpoints_ext_temp = np.concatenate([[-np.inf], cutpoints_temp, [np.inf]])

                effects_temp = []
                for i in range(n_obs):
                    lp = X[i] @ beta_temp
                    z_l = cutpoints_ext_temp[j] - lp
                    z_u = cutpoints_ext_temp[j + 1] - lp

                    pdf_l = pdf_func(z_l) if np.isfinite(z_l) else 0
                    pdf_u = pdf_func(z_u) if np.isfinite(z_u) else 0

                    effects_temp.append(beta_temp[k] * (pdf_l - pdf_u))

                return np.mean(effects_temp)

            # Get all parameters including cutpoints
            all_params = result.params

            # Numerical gradient
            gradient = numerical_gradient(category_j_ame, all_params)

            # Compute SE via delta method
            cov_matrix = (
                result.cov_params
                if hasattr(result, "cov_params")
                else np.eye(len(all_params)) * 0.01
            )
            se_result = delta_method_se(gradient, cov_matrix)
            se_j = se_result["std_error"] if "std_error" in se_result else se_result["std_errors"]
            category_ses.append(se_j)

        ame_dict[var] = category_effects
        ame_se_dict[var] = category_ses

    # Create DataFrames
    category_names = [f"Category_{j}" for j in range(n_categories)]
    ame_df = pd.DataFrame(ame_dict, index=category_names).T
    se_df = pd.DataFrame(ame_se_dict, index=category_names).T

    return OrderedMarginalEffectsResult(ame_df, se_df, result, me_type="ame")


def compute_ordered_mem(
    result, varlist: Optional[List[str]] = None
) -> OrderedMarginalEffectsResult:
    """
    Compute Marginal Effects at Means (MEM) for ordered models.

    Evaluates marginal effects at the mean of all covariates.

    Parameters
    ----------
    result : OrderedChoiceModel result
        Fitted ordered model result
    varlist : list of str, optional
        Variables to compute MEM for (default: all exogenous)

    Returns
    -------
    OrderedMarginalEffectsResult
        Container with marginal effects and standard errors
    """
    model = result.model

    # Get data
    X = model.exog
    n_vars = X.shape[1]
    n_categories = model.n_categories

    # Compute mean of X
    X_mean = X.mean(axis=0)

    # Extract parameters
    beta = result.beta
    cutpoints = result.cutpoints

    # Extended cutpoints
    cutpoints_extended = np.concatenate([[-np.inf], cutpoints, [np.inf]])

    # Linear prediction at mean
    linear_pred_mean = X_mean @ beta

    # Determine PDF based on model type
    if hasattr(model, "_pdf"):
        pdf_func = model._pdf
    elif "Logit" in model.__class__.__name__:
        pdf_func = lambda z: expit(z) * (1 - expit(z))
    else:  # Probit
        pdf_func = stats.norm.pdf

    # Get variable names
    if varlist is None:
        if hasattr(model, "exog_names"):
            varlist = model.exog_names
        else:
            varlist = [f"x{i}" for i in range(n_vars)]

    # Compute MEM for each variable and category
    mem_dict = {}
    mem_se_dict = {}

    for k, var in enumerate(varlist):
        if k >= n_vars:
            continue

        category_effects = []
        category_ses = []

        for j in range(n_categories):
            # MEM for category j
            z_lower = cutpoints_extended[j] - linear_pred_mean
            z_upper = cutpoints_extended[j + 1] - linear_pred_mean

            pdf_lower = pdf_func(z_lower) if np.isfinite(z_lower) else 0
            pdf_upper = pdf_func(z_upper) if np.isfinite(z_upper) else 0

            mem_j = beta[k] * (pdf_lower - pdf_upper)
            category_effects.append(mem_j)

            # Standard error computation
            def category_j_mem(params):
                beta_temp = params[:n_vars]
                cutpoint_params_temp = params[n_vars : n_vars + len(cutpoints)]

                if hasattr(model, "_transform_cutpoints"):
                    cutpoints_temp = model._transform_cutpoints(cutpoint_params_temp)
                else:
                    cutpoints_temp = cutpoint_params_temp

                cutpoints_ext_temp = np.concatenate([[-np.inf], cutpoints_temp, [np.inf]])

                lp = X_mean @ beta_temp
                z_l = cutpoints_ext_temp[j] - lp
                z_u = cutpoints_ext_temp[j + 1] - lp

                pdf_l = pdf_func(z_l) if np.isfinite(z_l) else 0
                pdf_u = pdf_func(z_u) if np.isfinite(z_u) else 0

                return beta_temp[k] * (pdf_l - pdf_u)

            all_params = result.params
            gradient = numerical_gradient(category_j_mem, all_params)

            cov_matrix = (
                result.cov_params
                if hasattr(result, "cov_params")
                else np.eye(len(all_params)) * 0.01
            )
            se_result = delta_method_se(gradient, cov_matrix)
            se_j = se_result["std_error"] if "std_error" in se_result else se_result["std_errors"]
            category_ses.append(se_j)

        mem_dict[var] = category_effects
        mem_se_dict[var] = category_ses

    # Create DataFrames
    category_names = [f"Category_{j}" for j in range(n_categories)]
    mem_df = pd.DataFrame(mem_dict, index=category_names).T
    se_df = pd.DataFrame(mem_se_dict, index=category_names).T

    # Store mean values
    at_values = {var: X_mean[i] for i, var in enumerate(varlist[:n_vars])}

    return OrderedMarginalEffectsResult(mem_df, se_df, result, me_type="mem", at_values=at_values)
