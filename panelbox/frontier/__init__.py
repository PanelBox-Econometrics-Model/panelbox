"""
Stochastic Frontier Analysis (SFA) module.

This module provides tools for estimating production and cost frontiers
using maximum likelihood estimation with various distributional assumptions
for the inefficiency term.

Main Classes:
    StochasticFrontier: Main model class for SFA estimation
    SFResult: Results container with efficiency estimates

Enumerations:
    FrontierType: Production or cost frontier
    DistributionType: Distribution for inefficiency (half-normal, exponential, etc.)
    ModelType: Type of SFA model (cross-section, panel variants)

True Models (Greene 2005):
    loglik_true_fixed_effects: TFE model separating heterogeneity from inefficiency
    loglik_true_random_effects: TRE model with three-component error structure
    bias_correct_tfe_analytical: Analytical bias correction for TFE
    bias_correct_tfe_jackknife: Jackknife bias correction for TFE
    variance_decomposition_tre: Decompose TRE variance into components

Statistical Tests:
    hausman_test_tfe_tre: Hausman test for TFE vs TRE
    lr_test: Likelihood ratio test for nested models
    wald_test: Wald test for parameter restrictions
    heterogeneity_significance_test: Test if σ²_w significantly differs from zero

Example:
    >>> from panelbox.frontier import StochasticFrontier
    >>>
    >>> sf = StochasticFrontier(
    ...     data=df,
    ...     depvar='log_output',
    ...     exog=['log_labor', 'log_capital'],
    ...     frontier='production',
    ...     dist='half_normal'
    ... )
    >>> result = sf.fit()
    >>> print(result.summary())
    >>> eff = result.efficiency(estimator='bc')
"""

from .data import (
    DistributionType,
    FrontierType,
    ModelType,
    add_translog,
    prepare_panel_index,
    validate_frontier_data,
)
from .model import StochasticFrontier
from .result import SFResult
from .tests import (
    compare_nested_distributions,
    hausman_test_tfe_tre,
    heterogeneity_significance_test,
    inefficiency_presence_test,
    lr_test,
    skewness_test,
    summary_model_comparison,
    vuong_test,
    wald_test,
)
from .true_models import (
    bias_correct_tfe_analytical,
    bias_correct_tfe_jackknife,
    loglik_tfe_bc95,
    loglik_tre_bc95,
    loglik_true_fixed_effects,
    loglik_true_random_effects,
    variance_decomposition_tre,
)

__all__ = [
    "StochasticFrontier",
    "SFResult",
    "FrontierType",
    "DistributionType",
    "ModelType",
    "validate_frontier_data",
    "prepare_panel_index",
    "add_translog",
    # True models
    "loglik_true_fixed_effects",
    "loglik_true_random_effects",
    "loglik_tfe_bc95",
    "loglik_tre_bc95",
    "bias_correct_tfe_analytical",
    "bias_correct_tfe_jackknife",
    "variance_decomposition_tre",
    # Tests
    "hausman_test_tfe_tre",
    "lr_test",
    "wald_test",
    "heterogeneity_significance_test",
    "summary_model_comparison",
    "inefficiency_presence_test",
    "skewness_test",
    "vuong_test",
    "compare_nested_distributions",
]
