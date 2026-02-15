"""
Poisson Pseudo-Maximum Likelihood (PPML) for Panel Data.

PPML is widely used in trade economics for estimating gravity models.
It handles zeros naturally and is robust to heteroskedasticity.

References
----------
Santos Silva, J.M.C., & Tenreyro, S. (2006). "The Log of Gravity."
    Review of Economics and Statistics, 88(4), 641-658.
Santos Silva, J.M.C., & Tenreyro, S. (2010). "On the Existence of the
    Maximum Likelihood Estimates in Poisson Regression."
    Economics Letters, 107(2), 310-312.
"""

import warnings
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from ..base import PanelModelResults
from .poisson import PoissonFixedEffects, PooledPoisson


class PPMLResult(PanelModelResults):
    """
    Results class for PPML estimation.

    Extends PanelModelResults with PPML-specific methods for
    interpreting elasticities and comparing with OLS.

    Attributes
    ----------
    model : PPML
        The fitted model instance
    params : array
        Parameter estimates
    cov : array
        Variance-covariance matrix
    fixed_effects : bool
        Whether fixed effects were used
    """

    def __init__(self, poisson_result, fixed_effects: bool = False):
        """
        Initialize from Poisson result.

        Parameters
        ----------
        poisson_result : PanelModelResults
            Results from underlying Poisson model
        fixed_effects : bool
            Whether fixed effects were used
        """
        # Copy attributes from poisson result
        super().__init__(
            model=poisson_result.model, params=poisson_result.params, vcov=poisson_result.vcov
        )

        # Copy additional attributes
        for attr in dir(poisson_result):
            if not attr.startswith("_") and not hasattr(self, attr):
                try:
                    setattr(self, attr, getattr(poisson_result, attr))
                except AttributeError:
                    pass

        self.fixed_effects = fixed_effects
        self._original_result = poisson_result

    def elasticity(self, variable: str) -> Dict[str, float]:
        """
        Compute elasticity for a variable.

        For PPML, if X is in logs, β is the elasticity directly.
        If X is in levels, the semi-elasticity needs to be converted.

        Parameters
        ----------
        variable : str
            Name of the variable

        Returns
        -------
        dict
            Dictionary with 'coefficient', 'se', 'elasticity' keys

        Notes
        -----
        For a log-transformed variable:
            ∂log(y)/∂log(x) = β

        For a level variable:
            ∂log(y)/∂x = β
            ∂y/∂x / (y/x) = β * x (evaluated at mean)

        Examples
        --------
        >>> result = model.fit()
        >>> # For log(distance)
        >>> result.elasticity('log_distance')
        {'coefficient': -1.2, 'se': 0.15, 'elasticity': -1.2}
        """
        if not hasattr(self.model, "exog_names"):
            raise AttributeError("Model must have exog_names attribute to identify variables.")

        exog_names = self.model.exog_names
        if variable not in exog_names:
            raise ValueError(
                f"Variable '{variable}' not found in model. " f"Available: {exog_names}"
            )

        var_idx = exog_names.index(variable)
        coef = self.params[var_idx]
        se = np.sqrt(self.vcov[var_idx, var_idx])

        # Check if variable is log-transformed
        is_log = variable.startswith("log_") or variable.startswith("ln_")

        if is_log:
            # Direct elasticity
            elasticity = coef
            elasticity_se = se
        else:
            # Semi-elasticity - convert to elasticity at mean
            X_mean = np.mean(self.model.exog[:, var_idx])
            elasticity = coef * X_mean
            # Delta method for SE (approximate)
            elasticity_se = se * X_mean

        return {
            "coefficient": coef,
            "se": se,
            "elasticity": elasticity,
            "elasticity_se": elasticity_se,
            "is_log_transformed": is_log,
        }

    def elasticities(self) -> pd.DataFrame:
        """
        Compute elasticities for all variables.

        Returns
        -------
        pd.DataFrame
            DataFrame with elasticities for each variable
        """
        if not hasattr(self.model, "exog_names"):
            warnings.warn("Model does not have exog_names. Using generic names.", UserWarning)
            exog_names = [f"x{i}" for i in range(len(self.params))]
        else:
            exog_names = self.model.exog_names

        results = []
        for var in exog_names:
            try:
                elast = self.elasticity(var)
                results.append(
                    {
                        "variable": var,
                        "coefficient": elast["coefficient"],
                        "std_error": elast["se"],
                        "elasticity": elast["elasticity"],
                        "elasticity_se": elast["elasticity_se"],
                        "log_transformed": elast["is_log_transformed"],
                    }
                )
            except (ValueError, IndexError):
                continue

        return pd.DataFrame(results)

    def compare_with_ols(self, ols_result) -> pd.DataFrame:
        """
        Compare PPML results with OLS on log-transformed outcome.

        Parameters
        ----------
        ols_result : PanelModelResults
            Results from OLS regression on log(y)

        Returns
        -------
        pd.DataFrame
            Comparison table

        Notes
        -----
        PPML is preferred over OLS(log(y)) when:
        1. Zeros are present in the data
        2. Heteroskedasticity is suspected
        3. E[y|X] = exp(X'β) is the correct specification

        OLS on log(y) suffers from:
        - Cannot handle zeros
        - Jensen's inequality bias: E[log(y)|X] ≠ log(E[y|X])
        - Inconsistent under heteroskedasticity
        """
        if not hasattr(self.model, "exog_names"):
            raise AttributeError("Model must have exog_names for comparison.")

        exog_names = self.model.exog_names
        n_params = min(len(self.params), len(ols_result.params))

        comparison = []
        for i in range(n_params):
            var_name = exog_names[i] if i < len(exog_names) else f"x{i}"

            ppml_coef = self.params[i]
            ppml_se = np.sqrt(self.cov[i, i])

            ols_coef = ols_result.params[i]
            ols_se = np.sqrt(ols_result.cov[i, i])

            # Difference
            diff = ppml_coef - ols_coef
            # Approximate SE of difference
            diff_se = np.sqrt(ppml_se**2 + ols_se**2)

            comparison.append(
                {
                    "variable": var_name,
                    "PPML_coef": ppml_coef,
                    "PPML_se": ppml_se,
                    "OLS_coef": ols_coef,
                    "OLS_se": ols_se,
                    "difference": diff,
                    "diff_se": diff_se,
                    "t_stat": diff / diff_se if diff_se > 0 else np.nan,
                }
            )

        return pd.DataFrame(comparison)

    def summary(self, **kwargs) -> str:
        """
        Generate summary table with PPML-specific information.

        Returns
        -------
        str
            Formatted summary
        """
        # Get base summary
        base_summary = super().summary(**kwargs)

        # Add PPML-specific info
        ppml_info = "\n" + "=" * 78 + "\n"
        ppml_info += "PPML (Poisson Pseudo-Maximum Likelihood)\n"
        ppml_info += "=" * 78 + "\n"

        if self.fixed_effects:
            ppml_info += "Specification: Fixed Effects PPML\n"
            if hasattr(self._original_result, "n_dropped"):
                ppml_info += f"Entities dropped (no variation): {self._original_result.n_dropped}\n"
        else:
            ppml_info += "Specification: Pooled PPML\n"

        ppml_info += "\nNotes:\n"
        ppml_info += "- Standard errors are cluster-robust (required for PPML)\n"
        ppml_info += "- For log-transformed variables, coefficients are elasticities\n"
        ppml_info += "- PPML handles zeros naturally and is robust to heteroskedasticity\n"
        ppml_info += "- Preferred over OLS(log y) for gravity models\n"

        # Check for zeros
        if hasattr(self.model, "endog"):
            n_zeros = np.sum(self.model.endog == 0)
            if n_zeros > 0:
                pct_zeros = 100 * n_zeros / len(self.model.endog)
                ppml_info += f"\nDependent variable: {n_zeros} zeros ({pct_zeros:.1f}%)\n"
                ppml_info += "  → PPML recommended (OLS would drop these observations)\n"

        return base_summary + ppml_info


class PPML:
    """
    Poisson Pseudo-Maximum Likelihood for Panel Data.

    PPML is the standard estimator for gravity models in international
    trade. It naturally handles zeros and is robust to heteroskedasticity.

    This is a convenient wrapper around Poisson models with:
    - Mandatory cluster-robust standard errors
    - Specialized interpretation methods for elasticities
    - Comparison tools vs OLS

    Parameters
    ----------
    endog : array-like
        Dependent variable (can include zeros)
    exog : array-like
        Independent variables (typically in logs)
    entity_id : array-like, optional
        Entity identifiers for fixed effects and clustering
    time_id : array-like, optional
        Time identifiers
    fixed_effects : bool, default True
        Whether to include entity fixed effects

    Attributes
    ----------
    model : PooledPoisson or PoissonFixedEffects
        Underlying Poisson model

    Methods
    -------
    fit(**kwargs)
        Estimate model with cluster-robust SEs
    predict(**kwargs)
        Generate predictions

    Examples
    --------
    >>> # Gravity model for bilateral trade
    >>> import pandas as pd
    >>> # df has: trade_flow, log_gdp_i, log_gdp_j, log_distance
    >>> model = PPML(
    ...     endog=df['trade_flow'],
    ...     exog=df[['log_gdp_i', 'log_gdp_j', 'log_distance']],
    ...     entity_id=df['pair_id'],
    ...     fixed_effects=True
    ... )
    >>> result = model.fit()
    >>> print(result.elasticities())

    Notes
    -----
    **Why PPML instead of OLS(log y)?**

    OLS on log(y) has several problems:
    1. Cannot handle zeros: log(0) undefined
    2. Jensen's inequality: E[log y|X] ≠ log E[y|X]
    3. Heteroskedasticity causes bias
    4. Retransformation problem

    PPML solves all these:
    - Handles zeros naturally
    - E[y|X] = exp(X'β) correctly specified
    - Consistent under heteroskedasticity (QML property)
    - No retransformation needed

    **When to use PPML:**
    - Gravity models (trade, FDI, migration)
    - Any count or non-negative continuous outcome
    - Zeros present in dependent variable
    - Heteroskedasticity suspected
    - Multiplicative model: E[y|X] = exp(X'β)

    **Interpretation:**
    For log-transformed variables:
        ∂log E[y|X] / ∂log x = β

    So β is directly the elasticity.

    References
    ----------
    Santos Silva, J.M.C., & Tenreyro, S. (2006). "The Log of Gravity."
        Review of Economics and Statistics, 88(4), 641-658.

    Head, K., & Mayer, T. (2014). "Gravity Equations: Workhorse,
        Toolkit, and Cookbook." Handbook of International Economics, 4, 131-195.
    """

    def __init__(
        self,
        endog,
        exog,
        entity_id=None,
        time_id=None,
        fixed_effects: bool = True,
        exog_names: Optional[list] = None,
    ):
        """
        Initialize PPML model.

        Parameters
        ----------
        endog : array-like
            Dependent variable
        exog : array-like
            Independent variables
        entity_id : array-like, optional
            Entity identifiers
        time_id : array-like, optional
            Time identifiers
        fixed_effects : bool, default True
            Use fixed effects
        exog_names : list, optional
            Names of exogenous variables
        """
        self.fixed_effects = fixed_effects
        self.exog_names = exog_names

        # Check for negative values
        if np.any(np.array(endog) < 0):
            raise ValueError(
                "PPML requires non-negative dependent variable. " "Found negative values."
            )

        # Create underlying Poisson model
        if fixed_effects:
            if entity_id is None:
                raise ValueError("entity_id required for fixed effects PPML")
            self.model = PoissonFixedEffects(
                endog=endog, exog=exog, entity_id=entity_id, time_id=time_id
            )
        else:
            self.model = PooledPoisson(endog=endog, exog=exog, entity_id=entity_id, time_id=time_id)

        # Add exog_names to model
        if exog_names is not None:
            self.model.exog_names = exog_names

    def fit(self, se_type: str = "cluster", **kwargs) -> PPMLResult:
        """
        Estimate PPML model.

        Cluster-robust standard errors are mandatory for PPML.

        Parameters
        ----------
        se_type : str, default 'cluster'
            Type of standard errors (forced to 'cluster' for PPML)
        **kwargs
            Additional arguments passed to underlying Poisson fit

        Returns
        -------
        PPMLResult
            Fitted model results with PPML-specific methods

        Notes
        -----
        Standard errors are cluster-robust by default. This is essential
        for PPML as it is a quasi-ML estimator.
        """
        # Force cluster-robust SEs
        if se_type != "cluster":
            warnings.warn(
                f"PPML requires cluster-robust SEs. Ignoring se_type='{se_type}' "
                f"and using 'cluster'.",
                UserWarning,
            )

        # Fit underlying model
        result = self.model.fit(se_type="cluster", **kwargs)

        # Wrap in PPMLResult
        return PPMLResult(result, fixed_effects=self.fixed_effects)

    def predict(self, X=None, **kwargs):
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like, optional
            New data for predictions. If None, uses training data.
        **kwargs
            Additional arguments passed to underlying model

        Returns
        -------
        array
            Predicted values
        """
        return self.model.predict(X=X, **kwargs)
