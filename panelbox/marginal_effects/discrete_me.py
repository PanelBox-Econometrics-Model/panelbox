"""
Marginal effects for discrete choice models.

This module implements marginal effects computations for binary choice models
including Average Marginal Effects (AME), Marginal Effects at Means (MEM),
and Marginal Effects at Representative values (MER).
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from panelbox.marginal_effects.delta_method import (
    compute_me_gradient,
    delta_method_se,
    numerical_gradient,
)

# Import ordered models for type checking
try:
    from panelbox.models.discrete.ordered import OrderedLogit, OrderedProbit
except ImportError:
    OrderedLogit = None
    OrderedProbit = None


class MarginalEffectsResult:
    """
    Container for marginal effects results.

    Attributes
    ----------
    marginal_effects : pd.Series
        Estimated marginal effects
    std_errors : pd.Series
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
        marginal_effects: Union[dict, pd.Series],
        std_errors: Union[dict, pd.Series],
        parent_result,
        me_type: str = "ame",
        at_values: Optional[dict] = None,
    ):
        """Initialize marginal effects result."""
        self.marginal_effects = pd.Series(marginal_effects)
        self.std_errors = pd.Series(std_errors)
        self.parent_result = parent_result
        self.me_type = me_type.upper()
        self.at_values = at_values

    @property
    def z_stats(self) -> pd.Series:
        """Z-statistics for marginal effects."""
        return self.marginal_effects / self.std_errors

    @property
    def pvalues(self) -> pd.Series:
        """P-values from two-sided z-tests."""
        z_stats = self.z_stats
        return 2 * norm.cdf(-np.abs(z_stats))

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Confidence intervals for marginal effects.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        pd.DataFrame
            DataFrame with 'lower' and 'upper' columns
        """
        z_crit = norm.ppf(1 - alpha / 2)
        lower = self.marginal_effects - z_crit * self.std_errors
        upper = self.marginal_effects + z_crit * self.std_errors
        return pd.DataFrame({"lower": lower, "upper": upper}, index=self.marginal_effects.index)

    def summary(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Formatted summary table of marginal effects.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals

        Returns
        -------
        pd.DataFrame
            Summary table with effects, SEs, z-stats, p-values, and CIs
        """
        ci = self.conf_int(alpha)

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
        stars = pvals.apply(add_stars)

        df = pd.DataFrame(
            {
                f"{self.me_type}": self.marginal_effects,
                "Std. Err.": self.std_errors,
                "z": self.z_stats,
                "P>|z|": pvals,
                f"[{alpha/2:.3f}": ci["lower"],
                f"{1-alpha/2:.3f}]": ci["upper"],
            }
        )

        # Add significance column
        df[""] = stars

        # Format for display
        pd.options.display.float_format = "{:.4f}".format

        print(f"\n{self.me_type} Marginal Effects")
        print("=" * 60)

        if self.at_values is not None:
            print("\nEvaluated at:")
            for var, val in self.at_values.items():
                print(f"  {var} = {val:.4f}")

        return df

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MarginalEffectsResult(type='{self.me_type}', n_effects={len(self.marginal_effects)})"
        )


def _is_binary(x: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a variable is binary (0/1)."""
    unique_vals = np.unique(x)
    if len(unique_vals) == 2:
        return np.allclose(sorted(unique_vals), [0, 1], atol=tol)
    elif len(unique_vals) == 1:
        return unique_vals[0] in [0, 1]
    return False


def _logit_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """PDF of standard logistic distribution: exp(x) / (1 + exp(x))^2."""
    # Use stable computation to avoid overflow
    exp_x = np.exp(np.minimum(x, 20))  # Cap to avoid overflow
    return exp_x / (1 + exp_x) ** 2


def _logit_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """CDF of standard logistic distribution: 1 / (1 + exp(-x))."""
    # Use stable computation
    return 1 / (1 + np.exp(-np.minimum(x, 20)))


def compute_ame(
    result, varlist: Optional[List[str]] = None, dummy_method: str = "diff"
) -> MarginalEffectsResult:
    """
    Compute Average Marginal Effects (AME).

    AME averages the marginal effect across all observations:
    - For continuous variables: average of ∂P/∂x evaluated at each obs
    - For binary variables: average of P(y=1|x=1) - P(y=1|x=0)

    Parameters
    ----------
    result : NonlinearPanelResult
        Fitted model result
    varlist : list of str, optional
        Variables to compute AME for (default: all exogenous)
    dummy_method : str, default='diff'
        Method for binary variables ('diff' for discrete difference)

    Returns
    -------
    MarginalEffectsResult
        Container with marginal effects and standard errors
    """
    model = result.model

    # Get exogenous variables
    if hasattr(model, "exog_df"):
        X = model.exog_df.values
        exog_names = model.exog_df.columns.tolist()
    else:
        X = model.exog
        exog_names = (
            model.exog_names
            if hasattr(model, "exog_names")
            else [f"x{i}" for i in range(X.shape[1])]
        )

    params = result.params
    if isinstance(params, pd.Series):
        params = params.values

    if varlist is None:
        varlist = exog_names

    # Compute AME for each variable
    ame = {}
    ame_gradients = {}

    for var in varlist:
        if var not in exog_names:
            continue

        var_idx = exog_names.index(var)
        x_var = X[:, var_idx]

        if _is_binary(x_var) and dummy_method == "diff":
            # Discrete difference for binary variable
            ame[var] = _ame_discrete(result, X, var_idx, params)
            # Gradient computation for SE
            ame_gradients[var] = _ame_discrete_gradient(result, X, var_idx, params)
        else:
            # Continuous marginal effect
            ame[var] = _ame_continuous(result, X, var_idx, params)
            # Gradient for SE
            ame_gradients[var] = compute_me_gradient(model, params, var_idx, X, "ame")

    # Compute standard errors via delta method
    std_errors = {}
    cov_matrix = result.cov_params if hasattr(result, "cov_params") else result.cov

    for var in ame.keys():
        gradient = ame_gradients[var]
        se_results = delta_method_se(gradient, cov_matrix)
        std_errors[var] = (
            se_results["std_error"] if "std_error" in se_results else se_results["std_errors"]
        )

    return MarginalEffectsResult(ame, std_errors, result, me_type="ame")


def _ame_continuous(result, X: np.ndarray, var_idx: int, params: np.ndarray) -> float:
    """
    AME for continuous variable.

    For variable x_k, AME = (1/N) Σᵢ ∂P(yᵢ=1|Xᵢ)/∂x_k

    Parameters
    ----------
    result : model result
    X : np.ndarray
        Covariate matrix (N x K)
    var_idx : int
        Index of variable
    params : np.ndarray
        Parameter estimates

    Returns
    -------
    float
        Average marginal effect
    """
    model = result.model
    n_obs = X.shape[0]

    # Linear predictions
    linear_pred = X @ params
    param_k = params[var_idx]

    if hasattr(model, "family"):
        if model.family == "probit":
            # For Probit: ME = β_k * φ(X'β)
            me_i = param_k * norm.pdf(linear_pred)
        elif model.family == "logit":
            # For Logit: ME = β_k * Λ(X'β)[1 - Λ(X'β)]
            me_i = param_k * _logit_pdf(linear_pred)
        else:
            raise ValueError(f"Unknown family: {model.family}")
    else:
        # Default to logit
        me_i = param_k * _logit_pdf(linear_pred)

    return me_i.mean()


def _ame_discrete(result, X: np.ndarray, var_idx: int, params: np.ndarray) -> float:
    """
    AME for binary variable via discrete difference.

    AME = (1/N) Σᵢ [P(yᵢ=1|Xᵢ, d=1) - P(yᵢ=1|Xᵢ, d=0)]

    Parameters
    ----------
    result : model result
    X : np.ndarray
        Covariate matrix (N x K)
    var_idx : int
        Index of binary variable
    params : np.ndarray
        Parameter estimates

    Returns
    -------
    float
        Average marginal effect (discrete difference)
    """
    model = result.model

    # Create copies with d=1 and d=0
    X1 = X.copy()
    X1[:, var_idx] = 1
    X0 = X.copy()
    X0[:, var_idx] = 0

    # Predict probabilities
    linear_pred1 = X1 @ params
    linear_pred0 = X0 @ params

    if hasattr(model, "family"):
        if model.family == "probit":
            prob1 = norm.cdf(linear_pred1)
            prob0 = norm.cdf(linear_pred0)
        elif model.family == "logit":
            prob1 = _logit_cdf(linear_pred1)
            prob0 = _logit_cdf(linear_pred0)
        else:
            raise ValueError(f"Unknown family: {model.family}")
    else:
        # Default to logit
        prob1 = _logit_cdf(linear_pred1)
        prob0 = _logit_cdf(linear_pred0)

    # Average difference
    return (prob1 - prob0).mean()


def _ame_discrete_gradient(result, X: np.ndarray, var_idx: int, params: np.ndarray) -> np.ndarray:
    """Compute gradient of discrete AME w.r.t. parameters."""
    n_params = len(params)
    n_obs = X.shape[0]
    model = result.model

    # Function to compute AME for given parameters
    def ame_func(p):
        return _ame_discrete(result, X, var_idx, p)

    # Numerical gradient
    return numerical_gradient(ame_func, params)


def compute_mem(result, varlist: Optional[List[str]] = None) -> MarginalEffectsResult:
    """
    Compute Marginal Effects at Means (MEM).

    MEM evaluates marginal effects at the mean of all covariates:
    ∂P/∂x_k |_{X=X̄}

    Parameters
    ----------
    result : NonlinearPanelResult
        Fitted model result
    varlist : list of str, optional
        Variables to compute MEM for (default: all exogenous)

    Returns
    -------
    MarginalEffectsResult
        Container with marginal effects and standard errors
    """
    model = result.model

    # Get exogenous variables
    if hasattr(model, "exog_df"):
        X = model.exog_df.values
        exog_names = model.exog_df.columns.tolist()
    else:
        X = model.exog
        exog_names = (
            model.exog_names
            if hasattr(model, "exog_names")
            else [f"x{i}" for i in range(X.shape[1])]
        )

    params = result.params
    if isinstance(params, pd.Series):
        params = params.values

    if varlist is None:
        varlist = exog_names

    # Compute mean of X
    X_mean = X.mean(axis=0)
    linear_pred_mean = X_mean @ params

    # Compute scaling factor based on model family
    if hasattr(model, "family"):
        if model.family == "probit":
            factor = norm.pdf(linear_pred_mean)
        elif model.family == "logit":
            factor = _logit_pdf(linear_pred_mean)
        else:
            raise ValueError(f"Unknown family: {model.family}")
    else:
        # Default to logit
        factor = _logit_pdf(linear_pred_mean)

    # MEM for each variable
    mem = {}
    for var in varlist:
        if var not in exog_names:
            continue
        var_idx = exog_names.index(var)

        # Check if binary for discrete difference
        x_var = X[:, var_idx]
        if _is_binary(x_var):
            # Discrete difference at means
            X_mean1 = X_mean.copy()
            X_mean1[var_idx] = 1
            X_mean0 = X_mean.copy()
            X_mean0[var_idx] = 0

            if hasattr(model, "family") and model.family == "probit":
                prob1 = norm.cdf(X_mean1 @ params)
                prob0 = norm.cdf(X_mean0 @ params)
            else:
                prob1 = _logit_cdf(X_mean1 @ params)
                prob0 = _logit_cdf(X_mean0 @ params)

            mem[var] = prob1 - prob0
        else:
            # Continuous: β_k * factor
            mem[var] = params[var_idx] * factor

    # Compute standard errors via delta method
    std_errors = {}
    cov_matrix = result.cov_params if hasattr(result, "cov_params") else result.cov

    for var in mem.keys():
        var_idx = exog_names.index(var)
        gradient = compute_me_gradient(model, params, var_idx, X, "mem")
        se_results = delta_method_se(gradient, cov_matrix)
        std_errors[var] = (
            se_results["std_error"] if "std_error" in se_results else se_results["std_errors"]
        )

    # Store mean values for reference
    at_values = {name: X_mean[i] for i, name in enumerate(exog_names)}

    return MarginalEffectsResult(mem, std_errors, result, me_type="mem", at_values=at_values)


def compute_mer(result, at: dict, varlist: Optional[List[str]] = None) -> MarginalEffectsResult:
    """
    Compute Marginal Effects at Representative values (MER).

    MER evaluates marginal effects at user-specified values.
    Variables not specified in 'at' are set to their means.

    Parameters
    ----------
    result : NonlinearPanelResult
        Fitted model result
    at : dict
        Representative values {'var1': value1, 'var2': value2}
    varlist : list of str, optional
        Variables to compute MER for (default: all exogenous)

    Returns
    -------
    MarginalEffectsResult
        Container with marginal effects and standard errors
    """
    model = result.model

    # Get exogenous variables
    if hasattr(model, "exog_df"):
        X = model.exog_df.values
        exog_names = model.exog_df.columns.tolist()
    else:
        X = model.exog
        exog_names = (
            model.exog_names
            if hasattr(model, "exog_names")
            else [f"x{i}" for i in range(X.shape[1])]
        )

    params = result.params
    if isinstance(params, pd.Series):
        params = params.values

    if varlist is None:
        varlist = exog_names

    # Start with mean values
    X_rep = X.mean(axis=0).copy()

    # Override with user-specified values
    for var, value in at.items():
        if var in exog_names:
            var_idx = exog_names.index(var)
            X_rep[var_idx] = value

    linear_pred_rep = X_rep @ params

    # Compute scaling factor based on model family
    if hasattr(model, "family"):
        if model.family == "probit":
            factor = norm.pdf(linear_pred_rep)
        elif model.family == "logit":
            factor = _logit_pdf(linear_pred_rep)
        else:
            raise ValueError(f"Unknown family: {model.family}")
    else:
        # Default to logit
        factor = _logit_pdf(linear_pred_rep)

    # MER for each variable
    mer = {}
    for var in varlist:
        if var not in exog_names:
            continue
        var_idx = exog_names.index(var)

        # Check if binary for discrete difference
        x_var = X[:, var_idx]
        if _is_binary(x_var):
            # Discrete difference at representative values
            X_rep1 = X_rep.copy()
            X_rep1[var_idx] = 1
            X_rep0 = X_rep.copy()
            X_rep0[var_idx] = 0

            if hasattr(model, "family") and model.family == "probit":
                prob1 = norm.cdf(X_rep1 @ params)
                prob0 = norm.cdf(X_rep0 @ params)
            else:
                prob1 = _logit_cdf(X_rep1 @ params)
                prob0 = _logit_cdf(X_rep0 @ params)

            mer[var] = prob1 - prob0
        else:
            # Continuous: β_k * factor
            mer[var] = params[var_idx] * factor

    # Compute standard errors via delta method
    std_errors = {}
    cov_matrix = result.cov_params if hasattr(result, "cov_params") else result.cov

    # Function to compute MER for a given parameter vector
    def mer_func(p):
        linear_pred = X_rep @ p
        if hasattr(model, "family") and model.family == "probit":
            return p * norm.pdf(linear_pred)
        else:
            return p * _logit_pdf(linear_pred)

    for var in mer.keys():
        var_idx = exog_names.index(var)

        # Numerical gradient
        def var_mer_func(p):
            linear_pred = X_rep @ p
            if _is_binary(X[:, var_idx]):
                X_rep1 = X_rep.copy()
                X_rep1[var_idx] = 1
                X_rep0 = X_rep.copy()
                X_rep0[var_idx] = 0
                if hasattr(model, "family") and model.family == "probit":
                    prob1 = norm.cdf(X_rep1 @ p)
                    prob0 = norm.cdf(X_rep0 @ p)
                else:
                    prob1 = _logit_cdf(X_rep1 @ p)
                    prob0 = _logit_cdf(X_rep0 @ p)
                return prob1 - prob0
            else:
                if hasattr(model, "family") and model.family == "probit":
                    return p[var_idx] * norm.pdf(linear_pred)
                else:
                    return p[var_idx] * _logit_pdf(linear_pred)

        gradient = numerical_gradient(var_mer_func, params)
        se_results = delta_method_se(gradient, cov_matrix)
        std_errors[var] = (
            se_results["std_error"] if "std_error" in se_results else se_results["std_errors"]
        )

    # Store representative values for reference
    at_values = {name: X_rep[i] for i, name in enumerate(exog_names)}

    return MarginalEffectsResult(mer, std_errors, result, me_type="mer", at_values=at_values)


class OrderedMarginalEffectsResult:
    """
    Container for ordered model marginal effects results.

    Attributes
    ----------
    marginal_effects : pd.DataFrame
        Estimated marginal effects for each category (categories x variables)
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

    @property
    def z_stats(self) -> pd.DataFrame:
        """Z-statistics for marginal effects."""
        return self.marginal_effects / self.std_errors

    @property
    def pvalues(self) -> pd.DataFrame:
        """P-values from two-sided z-tests."""
        z_stats = self.z_stats
        return 2 * norm.cdf(-np.abs(z_stats))

    def verify_sum_to_zero(self, tol: float = 1e-10) -> bool:
        """
        Verify that marginal effects sum to zero across categories.

        This is a property of ordered models: for each variable,
        the sum of marginal effects across all categories should be zero.

        Parameters
        ----------
        tol : float, default=1e-10
            Tolerance for checking sum equals zero

        Returns
        -------
        bool
            True if all variables sum to zero (within tolerance)
        """
        sums = self.marginal_effects.sum(axis=0)
        return np.allclose(sums, 0, atol=tol)

    def plot(
        self, var: str, figsize: tuple = (10, 6), include_ci: bool = True, alpha: float = 0.05
    ):
        """
        Plot marginal effects across categories for a variable.

        Parameters
        ----------
        var : str
            Variable name to plot
        figsize : tuple, default=(10, 6)
            Figure size
        include_ci : bool, default=True
            Include confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
        """
        import matplotlib.pyplot as plt

        categories = self.marginal_effects.index
        effects = self.marginal_effects[var].values
        ses = self.std_errors[var].values

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(categories))
        ax.bar(x, effects, alpha=0.8)

        if include_ci:
            z_crit = norm.ppf(1 - alpha / 2)
            ci_lower = effects - z_crit * ses
            ci_upper = effects + z_crit * ses
            ax.errorbar(
                x, effects, yerr=z_crit * ses, fmt="none", color="black", capsize=5, alpha=0.6
            )

        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Category")
        ax.set_ylabel("Marginal Effect")
        ax.set_title(f"{self.me_type} Marginal Effects for {var}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Cat {i}" for i in categories])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def summary(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Formatted summary table of marginal effects.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals

        Returns
        -------
        pd.DataFrame
            Summary table with effects and standard errors for all categories
        """
        print(f"\n{self.me_type} Marginal Effects for Ordered Model")
        print("=" * 70)

        if self.at_values is not None:
            print("\nEvaluated at:")
            for var_name, val in self.at_values.items():
                print(f"  {var_name} = {val:.4f}")

        # Check sum-to-zero property
        if self.verify_sum_to_zero():
            print("\n✓ Marginal effects sum to zero across categories (verified)")
        else:
            print("\n⚠ Warning: Marginal effects do not sum to zero (numerical issue)")

        # Format display
        pd.options.display.float_format = "{:.4f}".format

        # Create combined display with effects and SEs
        combined = pd.DataFrame()
        for var in self.marginal_effects.columns:
            combined[f"{var}_ME"] = self.marginal_effects[var]
            combined[f"{var}_SE"] = self.std_errors[var]

        return combined

    def __repr__(self) -> str:
        """String representation."""
        n_cats = len(self.marginal_effects.index)
        n_vars = len(self.marginal_effects.columns)
        return f"OrderedMarginalEffectsResult(type='{self.me_type}', categories={n_cats}, variables={n_vars})"


def compute_ordered_ame(model, varlist: Optional[List[str]] = None) -> OrderedMarginalEffectsResult:
    """
    Compute Average Marginal Effects for ordered choice models.

    For ordered models, we compute marginal effects for each category j:
    ∂P(y=j|X)/∂x_k = β_k × [λ(κ_{j-1} - X'β) - λ(κ_j - X'β)]

    where λ is the PDF of the error distribution (logistic or normal).

    Key property: Σ_j ∂P(y=j|X)/∂x_k = 0 (effects sum to zero across categories)

    Parameters
    ----------
    model : OrderedLogit or OrderedProbit
        Fitted ordered choice model
    varlist : list of str, optional
        Variables to compute AME for (default: all exogenous)

    Returns
    -------
    OrderedMarginalEffectsResult
        Container with marginal effects and standard errors for each category
    """
    if not hasattr(model, "beta"):
        raise ValueError("Model must be fitted before computing marginal effects")

    X = model.exog
    n_obs, n_features = X.shape
    n_categories = model.n_categories

    # Get parameter names
    if hasattr(model, "exog_names"):
        exog_names = model.exog_names
    else:
        exog_names = [f"beta_{i}" for i in range(n_features)]

    if varlist is None:
        varlist = exog_names

    beta = model.beta
    cutpoints = model.cutpoints

    # Extended cutpoints with boundaries
    cutpoints_extended = np.concatenate([[-np.inf], cutpoints, [np.inf]])

    # Initialize storage for marginal effects (categories x variables)
    ame_matrix = np.zeros((n_categories, len(varlist)))
    ame_se_matrix = np.zeros((n_categories, len(varlist)))

    for v_idx, var in enumerate(varlist):
        if var not in exog_names:
            continue

        var_idx = exog_names.index(var)
        beta_k = beta[var_idx]

        # Marginal effects for each category
        for j in range(n_categories):
            me_j_values = []

            for i in range(n_obs):
                X_i = X[i]
                linear_pred = X_i @ beta

                # ∂P(y=j|X)/∂x_k = β_k × [λ(κ_{j-1} - X'β) - λ(κ_j - X'β)]
                if type(model).__name__ == "OrderedLogit":
                    # Logistic PDF: λ(z) = Λ(z)[1 - Λ(z)]
                    lower_z = cutpoints_extended[j] - linear_pred
                    upper_z = cutpoints_extended[j + 1] - linear_pred

                    if np.isfinite(lower_z):
                        lower_pdf = _logit_pdf(lower_z)
                    else:
                        lower_pdf = 0

                    if np.isfinite(upper_z):
                        upper_pdf = _logit_pdf(upper_z)
                    else:
                        upper_pdf = 0

                    me_ij = beta_k * (lower_pdf - upper_pdf)

                elif type(model).__name__ == "OrderedProbit":
                    # Normal PDF
                    lower_z = cutpoints_extended[j] - linear_pred
                    upper_z = cutpoints_extended[j + 1] - linear_pred

                    if np.isfinite(lower_z):
                        lower_pdf = norm.pdf(lower_z)
                    else:
                        lower_pdf = 0

                    if np.isfinite(upper_z):
                        upper_pdf = norm.pdf(upper_z)
                    else:
                        upper_pdf = 0

                    me_ij = beta_k * (lower_pdf - upper_pdf)

                else:
                    raise ValueError(f"Unknown model type: {type(model)}")

                me_j_values.append(me_ij)

            # Average marginal effect for category j, variable k
            ame_matrix[j, v_idx] = np.mean(me_j_values)

            # Compute standard error via delta method (simplified)
            # This is a placeholder - full implementation would need gradients
            if hasattr(model, "bse"):
                # Approximate SE using parameter SE
                ame_se_matrix[j, v_idx] = (
                    model.bse[var_idx] * np.abs(np.mean(me_j_values) / beta_k)
                    if beta_k != 0
                    else model.bse[var_idx]
                )
            else:
                ame_se_matrix[j, v_idx] = np.nan

    # Create DataFrames
    category_names = [f"Category_{j}" for j in range(n_categories)]
    ame_df = pd.DataFrame(ame_matrix, index=category_names, columns=varlist)
    se_df = pd.DataFrame(ame_se_matrix, index=category_names, columns=varlist)

    return OrderedMarginalEffectsResult(ame_df, se_df, model, me_type="ame")


def compute_ordered_mem(model, varlist: Optional[List[str]] = None) -> OrderedMarginalEffectsResult:
    """
    Compute Marginal Effects at Means for ordered choice models.

    Evaluates marginal effects at the mean of all covariates.

    Parameters
    ----------
    model : OrderedLogit or OrderedProbit
        Fitted ordered choice model
    varlist : list of str, optional
        Variables to compute MEM for (default: all exogenous)

    Returns
    -------
    OrderedMarginalEffectsResult
        Container with marginal effects and standard errors for each category
    """
    if not hasattr(model, "beta"):
        raise ValueError("Model must be fitted before computing marginal effects")

    X = model.exog
    n_features = X.shape[1]
    n_categories = model.n_categories

    # Get parameter names
    if hasattr(model, "exog_names"):
        exog_names = model.exog_names
    else:
        exog_names = [f"beta_{i}" for i in range(n_features)]

    if varlist is None:
        varlist = exog_names

    beta = model.beta
    cutpoints = model.cutpoints

    # Extended cutpoints with boundaries
    cutpoints_extended = np.concatenate([[-np.inf], cutpoints, [np.inf]])

    # Compute at means
    X_mean = X.mean(axis=0)
    linear_pred_mean = X_mean @ beta

    # Initialize storage
    mem_matrix = np.zeros((n_categories, len(varlist)))
    mem_se_matrix = np.zeros((n_categories, len(varlist)))

    for v_idx, var in enumerate(varlist):
        if var not in exog_names:
            continue

        var_idx = exog_names.index(var)
        beta_k = beta[var_idx]

        # Marginal effects for each category at means
        for j in range(n_categories):
            if type(model).__name__ == "OrderedLogit":
                lower_z = cutpoints_extended[j] - linear_pred_mean
                upper_z = cutpoints_extended[j + 1] - linear_pred_mean

                if np.isfinite(lower_z):
                    lower_pdf = _logit_pdf(lower_z)
                else:
                    lower_pdf = 0

                if np.isfinite(upper_z):
                    upper_pdf = _logit_pdf(upper_z)
                else:
                    upper_pdf = 0

            elif type(model).__name__ == "OrderedProbit":
                lower_z = cutpoints_extended[j] - linear_pred_mean
                upper_z = cutpoints_extended[j + 1] - linear_pred_mean

                if np.isfinite(lower_z):
                    lower_pdf = norm.pdf(lower_z)
                else:
                    lower_pdf = 0

                if np.isfinite(upper_z):
                    upper_pdf = norm.pdf(upper_z)
                else:
                    upper_pdf = 0

            mem_matrix[j, v_idx] = beta_k * (lower_pdf - upper_pdf)

            # Compute standard error (simplified)
            if hasattr(model, "bse"):
                mem_se_matrix[j, v_idx] = model.bse[var_idx] * np.abs((lower_pdf - upper_pdf))
            else:
                mem_se_matrix[j, v_idx] = np.nan

    # Create DataFrames
    category_names = [f"Category_{j}" for j in range(n_categories)]
    mem_df = pd.DataFrame(mem_matrix, index=category_names, columns=varlist)
    se_df = pd.DataFrame(mem_se_matrix, index=category_names, columns=varlist)

    # Store mean values for reference
    at_values = {name: X_mean[i] for i, name in enumerate(exog_names)}

    return OrderedMarginalEffectsResult(mem_df, se_df, model, me_type="mem", at_values=at_values)
