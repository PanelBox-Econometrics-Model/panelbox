"""Model comparison tools for four-component SFA.

This module provides tools to compare the four-component model with
simpler alternatives (Pitt-Lee, True Fixed/Random Effects) to demonstrate
the value of separating persistent and transient inefficiency.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .four_component import FourComponentResult


@dataclass
class ModelComparisonResult:
    """Results from model comparison.

    Attributes:
        four_component: Results from four-component model
        model_names: List of model names
        log_likelihoods: Log-likelihood for each model
        aics: AIC for each model
        bics: BIC for each model
        variance_shares: Variance decomposition for each model
        efficiency_correlation: Correlation between efficiency estimates
    """

    four_component: FourComponentResult
    model_names: List[str]
    log_likelihoods: Dict[str, float]
    aics: Dict[str, float]
    bics: Dict[str, float]
    variance_shares: Dict[str, Dict[str, float]]
    efficiency_correlation: pd.DataFrame

    def print_summary(self):
        """Print comparison summary."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON RESULTS")
        print("=" * 70)

        print("\nModel Fit Statistics:")
        print(f"{'Model':<30} {'LogLik':>12} {'AIC':>12} {'BIC':>12}")
        print("-" * 70)
        for model in self.model_names:
            ll = self.log_likelihoods.get(model, np.nan)
            aic = self.aics.get(model, np.nan)
            bic = self.bics.get(model, np.nan)
            print(f"{model:<30} {ll:>12.2f} {aic:>12.2f} {bic:>12.2f}")

        print("\n\nVariance Decomposition (% of total variance):")
        print("-" * 70)

        # Get all unique components across models
        all_components = set()
        for shares in self.variance_shares.values():
            all_components.update(shares.keys())

        # Print header
        header = f"{'Component':<25}"
        for model in self.model_names:
            header += f"{model[:20]:>20}"
        print(header)
        print("-" * 70)

        # Print each component
        for component in sorted(all_components):
            row = f"{component:<25}"
            for model in self.model_names:
                share = self.variance_shares.get(model, {}).get(component, 0.0)
                row += f"{share:>19.2f}%"
            print(row)

        print("\n\nEfficiency Correlation Matrix:")
        print("-" * 70)
        print(self.efficiency_correlation.to_string())

        print("\n" + "=" * 70)


def compare_with_pitt_lee(
    four_comp_result: FourComponentResult,
) -> ModelComparisonResult:
    """Compare four-component model with Pitt-Lee (1981) model.

    The Pitt-Lee model does not separate persistent and transient inefficiency.
    It treats all inefficiency as time-invariant.

    Parameters:
        four_comp_result: Results from four-component model

    Returns:
        ModelComparisonResult with comparison statistics

    Notes:
        The Pitt-Lee model is nested within the four-component model.
        We can approximate it by combining persistent and transient inefficiency.
    """
    model = four_comp_result.model
    n = model.n_obs
    N = model.n_entities
    k = len(model.exog_names)

    # Four-component model statistics
    fc_variance_total = (
        four_comp_result.sigma_v**2
        + four_comp_result.sigma_u**2
        + four_comp_result.sigma_mu**2
        + four_comp_result.sigma_eta**2
    )

    fc_variance_shares = {
        "Noise (v_it)": 100 * four_comp_result.sigma_v**2 / fc_variance_total,
        "Transient Ineff (u_it)": 100 * four_comp_result.sigma_u**2 / fc_variance_total,
        "Heterogeneity (μ_i)": 100 * four_comp_result.sigma_mu**2 / fc_variance_total,
        "Persistent Ineff (η_i)": 100 * four_comp_result.sigma_eta**2 / fc_variance_total,
    }

    # Approximate Pitt-Lee model by combining components
    # Pitt-Lee: y_it = x_it'β + α_i + v_it
    # where α_i contains both heterogeneity and persistent inefficiency
    # (no separation, no transient inefficiency)

    sigma_v_pl = four_comp_result.sigma_v
    sigma_alpha_pl = np.sqrt(four_comp_result.sigma_mu**2 + four_comp_result.sigma_eta**2)

    pl_variance_total = sigma_v_pl**2 + sigma_alpha_pl**2
    pl_variance_shares = {
        "Noise (v_it)": 100 * sigma_v_pl**2 / pl_variance_total,
        "Fixed Effects (α_i)": 100 * sigma_alpha_pl**2 / pl_variance_total,
    }

    # Compute approximate log-likelihoods (based on variance decomposition)
    # Four-component has 4 variance parameters
    fc_n_params = k + 4
    # Pitt-Lee has 2 variance parameters
    pl_n_params = k + 2

    # Approximate LL based on residual variance
    # (This is a rough approximation for comparison purposes)
    residual_var_fc = four_comp_result.sigma_v**2 + four_comp_result.sigma_u**2
    residual_var_pl = sigma_v_pl**2

    ll_fc = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_fc) + 1)
    ll_pl = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_pl) + 1)

    # Compute AIC and BIC
    aic_fc = -2 * ll_fc + 2 * fc_n_params
    aic_pl = -2 * ll_pl + 2 * pl_n_params

    bic_fc = -2 * ll_fc + np.log(n) * fc_n_params
    bic_pl = -2 * ll_pl + np.log(n) * pl_n_params

    # Efficiency correlation
    # For Pitt-Lee, efficiency is time-invariant
    te_persistent = np.exp(-four_comp_result.eta_i)
    te_transient = np.exp(-four_comp_result.u_it)
    te_overall = te_persistent[model.entity_id] * te_transient

    # Pitt-Lee efficiency (only persistent component)
    te_pl = te_persistent[model.entity_id]

    corr_matrix = pd.DataFrame(
        {
            "Overall (4C)": [
                1.0,
                np.corrcoef(te_overall, te_persistent[model.entity_id])[0, 1],
                np.corrcoef(te_overall, te_transient)[0, 1],
                np.corrcoef(te_overall, te_pl)[0, 1],
            ],
            "Persistent (4C)": [
                np.corrcoef(te_overall, te_persistent[model.entity_id])[0, 1],
                1.0,
                np.corrcoef(te_persistent[model.entity_id], te_transient)[0, 1],
                np.corrcoef(te_persistent[model.entity_id], te_pl)[0, 1],
            ],
            "Transient (4C)": [
                np.corrcoef(te_overall, te_transient)[0, 1],
                np.corrcoef(te_persistent[model.entity_id], te_transient)[0, 1],
                1.0,
                np.corrcoef(te_transient, te_pl)[0, 1],
            ],
            "Pitt-Lee": [
                np.corrcoef(te_overall, te_pl)[0, 1],
                np.corrcoef(te_persistent[model.entity_id], te_pl)[0, 1],
                np.corrcoef(te_transient, te_pl)[0, 1],
                1.0,
            ],
        },
        index=["Overall (4C)", "Persistent (4C)", "Transient (4C)", "Pitt-Lee"],
    )

    return ModelComparisonResult(
        four_component=four_comp_result,
        model_names=["Four-Component", "Pitt-Lee"],
        log_likelihoods={
            "Four-Component": ll_fc,
            "Pitt-Lee": ll_pl,
        },
        aics={
            "Four-Component": aic_fc,
            "Pitt-Lee": aic_pl,
        },
        bics={
            "Four-Component": bic_fc,
            "Pitt-Lee": bic_pl,
        },
        variance_shares={
            "Four-Component": fc_variance_shares,
            "Pitt-Lee": pl_variance_shares,
        },
        efficiency_correlation=corr_matrix,
    )


def compare_with_true_effects(
    four_comp_result: FourComponentResult,
) -> ModelComparisonResult:
    """Compare four-component model with True Fixed/Random Effects.

    True Fixed Effects and True Random Effects models do not account for
    inefficiency at all. They treat all variation as either fixed effects
    or random heterogeneity.

    Parameters:
        four_comp_result: Results from four-component model

    Returns:
        ModelComparisonResult with comparison statistics

    Notes:
        - True FE: All α_i variation is fixed effects (no inefficiency)
        - True RE: All α_i variation is random heterogeneity (no inefficiency)
        - Four-Component: Separates α_i into μ_i (heterogeneity) and η_i (inefficiency)
    """
    model = four_comp_result.model
    n = model.n_obs
    N = model.n_entities
    k = len(model.exog_names)

    # Four-component model
    fc_variance_total = (
        four_comp_result.sigma_v**2
        + four_comp_result.sigma_u**2
        + four_comp_result.sigma_mu**2
        + four_comp_result.sigma_eta**2
    )

    fc_variance_shares = {
        "Noise (v_it)": 100 * four_comp_result.sigma_v**2 / fc_variance_total,
        "Transient Ineff (u_it)": 100 * four_comp_result.sigma_u**2 / fc_variance_total,
        "Heterogeneity (μ_i)": 100 * four_comp_result.sigma_mu**2 / fc_variance_total,
        "Persistent Ineff (η_i)": 100 * four_comp_result.sigma_eta**2 / fc_variance_total,
    }

    # True Fixed Effects: α_i = fixed effects (no inefficiency separation)
    sigma_v_fe = four_comp_result.sigma_v
    sigma_alpha_fe = np.sqrt(four_comp_result.sigma_mu**2 + four_comp_result.sigma_eta**2)

    fe_variance_total = sigma_v_fe**2 + sigma_alpha_fe**2
    fe_variance_shares = {
        "Noise (v_it)": 100 * sigma_v_fe**2 / fe_variance_total,
        "Fixed Effects (α_i)": 100 * sigma_alpha_fe**2 / fe_variance_total,
    }

    # True Random Effects: α_i ~ N(0, σ²_α), all random heterogeneity
    sigma_v_re = four_comp_result.sigma_v
    sigma_alpha_re = sigma_alpha_fe

    re_variance_total = sigma_v_re**2 + sigma_alpha_re**2
    re_variance_shares = {
        "Noise (v_it)": 100 * sigma_v_re**2 / re_variance_total,
        "Random Effects (α_i)": 100 * sigma_alpha_re**2 / re_variance_total,
    }

    # Approximate log-likelihoods
    fc_n_params = k + 4
    fe_n_params = k + 2
    re_n_params = k + 2

    residual_var_fc = four_comp_result.sigma_v**2 + four_comp_result.sigma_u**2
    residual_var_fe = sigma_v_fe**2
    residual_var_re = sigma_v_re**2

    ll_fc = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_fc) + 1)
    ll_fe = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_fe) + 1)
    ll_re = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_re) + 1)

    aic_fc = -2 * ll_fc + 2 * fc_n_params
    aic_fe = -2 * ll_fe + 2 * fe_n_params
    aic_re = -2 * ll_re + 2 * re_n_params

    bic_fc = -2 * ll_fc + np.log(n) * fc_n_params
    bic_fe = -2 * ll_fe + np.log(n) * fe_n_params
    bic_re = -2 * ll_re + np.log(n) * re_n_params

    # Efficiency comparisons
    te_persistent = np.exp(-four_comp_result.eta_i)
    te_transient = np.exp(-four_comp_result.u_it)
    te_overall = te_persistent[model.entity_id] * te_transient

    # True FE/RE assume perfect efficiency (TE = 1.0)
    te_fe = np.ones(n)
    te_re = np.ones(n)

    corr_matrix = pd.DataFrame(
        {
            "Overall (4C)": [
                1.0,
                np.corrcoef(te_overall, te_persistent[model.entity_id])[0, 1],
                np.corrcoef(te_overall, te_transient)[0, 1],
                0.0,
                0.0,
            ],  # No correlation with constant
            "Persistent (4C)": [
                np.corrcoef(te_overall, te_persistent[model.entity_id])[0, 1],
                1.0,
                np.corrcoef(te_persistent[model.entity_id], te_transient)[0, 1],
                0.0,
                0.0,
            ],
            "Transient (4C)": [
                np.corrcoef(te_overall, te_transient)[0, 1],
                np.corrcoef(te_persistent[model.entity_id], te_transient)[0, 1],
                1.0,
                0.0,
                0.0,
            ],
            "True FE": [0.0, 0.0, 0.0, 1.0, 1.0],
            "True RE": [0.0, 0.0, 0.0, 1.0, 1.0],
        },
        index=["Overall (4C)", "Persistent (4C)", "Transient (4C)", "True FE", "True RE"],
    )

    return ModelComparisonResult(
        four_component=four_comp_result,
        model_names=["Four-Component", "True FE", "True RE"],
        log_likelihoods={
            "Four-Component": ll_fc,
            "True FE": ll_fe,
            "True RE": ll_re,
        },
        aics={
            "Four-Component": aic_fc,
            "True FE": aic_fe,
            "True RE": aic_re,
        },
        bics={
            "Four-Component": bic_fc,
            "True FE": bic_fe,
            "True RE": bic_re,
        },
        variance_shares={
            "Four-Component": fc_variance_shares,
            "True FE": fe_variance_shares,
            "True RE": re_variance_shares,
        },
        efficiency_correlation=corr_matrix,
    )


def compare_all_models(
    four_comp_result: FourComponentResult,
) -> ModelComparisonResult:
    """Compare four-component model with all alternatives.

    This function compares the four-component model with:
    - Pitt-Lee (1981): Time-invariant inefficiency
    - True Fixed Effects: No inefficiency
    - True Random Effects: No inefficiency

    Parameters:
        four_comp_result: Results from four-component model

    Returns:
        ModelComparisonResult with comprehensive comparison

    Notes:
        This demonstrates the value of the four-component decomposition
        in separating persistent vs transient inefficiency and heterogeneity
        vs inefficiency.
    """
    model = four_comp_result.model
    n = model.n_obs
    N = model.n_entities
    k = len(model.exog_names)

    # Four-component variance decomposition
    fc_variance_total = (
        four_comp_result.sigma_v**2
        + four_comp_result.sigma_u**2
        + four_comp_result.sigma_mu**2
        + four_comp_result.sigma_eta**2
    )

    fc_variance_shares = {
        "Noise": 100 * four_comp_result.sigma_v**2 / fc_variance_total,
        "Transient Ineff": 100 * four_comp_result.sigma_u**2 / fc_variance_total,
        "Heterogeneity": 100 * four_comp_result.sigma_mu**2 / fc_variance_total,
        "Persistent Ineff": 100 * four_comp_result.sigma_eta**2 / fc_variance_total,
    }

    # Pitt-Lee approximation
    sigma_v_pl = four_comp_result.sigma_v
    sigma_u_pl = np.sqrt(four_comp_result.sigma_eta**2)  # Only persistent inefficiency
    sigma_alpha_pl = four_comp_result.sigma_mu  # Only heterogeneity

    pl_variance_total = sigma_v_pl**2 + sigma_u_pl**2 + sigma_alpha_pl**2
    pl_variance_shares = {
        "Noise": 100 * sigma_v_pl**2 / pl_variance_total,
        "Inefficiency": 100 * sigma_u_pl**2 / pl_variance_total,
        "Fixed Effects": 100 * sigma_alpha_pl**2 / pl_variance_total,
    }

    # True FE approximation
    sigma_v_fe = four_comp_result.sigma_v
    sigma_alpha_fe = np.sqrt(four_comp_result.sigma_mu**2 + four_comp_result.sigma_eta**2)

    fe_variance_total = sigma_v_fe**2 + sigma_alpha_fe**2
    fe_variance_shares = {
        "Noise": 100 * sigma_v_fe**2 / fe_variance_total,
        "Fixed Effects": 100 * sigma_alpha_fe**2 / fe_variance_total,
    }

    # True RE approximation
    re_variance_total = fe_variance_total
    re_variance_shares = {
        "Noise": 100 * sigma_v_fe**2 / re_variance_total,
        "Random Effects": 100 * sigma_alpha_fe**2 / re_variance_total,
    }

    # Compute information criteria
    fc_n_params = k + 4
    pl_n_params = k + 3
    fe_n_params = k + 2
    re_n_params = k + 2

    residual_var_fc = four_comp_result.sigma_v**2 + four_comp_result.sigma_u**2
    residual_var_pl = sigma_v_pl**2
    residual_var_fe = sigma_v_fe**2
    residual_var_re = sigma_v_fe**2

    ll_fc = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_fc) + 1)
    ll_pl = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_pl) + 1)
    ll_fe = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_fe) + 1)
    ll_re = -0.5 * n * (np.log(2 * np.pi) + np.log(residual_var_re) + 1)

    aics = {
        "Four-Component": -2 * ll_fc + 2 * fc_n_params,
        "Pitt-Lee": -2 * ll_pl + 2 * pl_n_params,
        "True FE": -2 * ll_fe + 2 * fe_n_params,
        "True RE": -2 * ll_re + 2 * re_n_params,
    }

    bics = {
        "Four-Component": -2 * ll_fc + np.log(n) * fc_n_params,
        "Pitt-Lee": -2 * ll_pl + np.log(n) * pl_n_params,
        "True FE": -2 * ll_fe + np.log(n) * fe_n_params,
        "True RE": -2 * ll_re + np.log(n) * re_n_params,
    }

    lls = {
        "Four-Component": ll_fc,
        "Pitt-Lee": ll_pl,
        "True FE": ll_fe,
        "True RE": ll_re,
    }

    # Efficiency correlations
    te_persistent = np.exp(-four_comp_result.eta_i)
    te_transient = np.exp(-four_comp_result.u_it)
    te_overall = te_persistent[model.entity_id] * te_transient
    te_pl = te_persistent[model.entity_id]  # Only persistent
    te_fe = np.ones(n)
    te_re = np.ones(n)

    corr_data = {
        "Overall (4C)": te_overall,
        "Persistent (4C)": te_persistent[model.entity_id],
        "Transient (4C)": te_transient,
        "Pitt-Lee": te_pl,
    }

    # Compute correlation matrix
    corr_matrix = pd.DataFrame(
        np.corrcoef(list(corr_data.values())), index=corr_data.keys(), columns=corr_data.keys()
    )

    return ModelComparisonResult(
        four_component=four_comp_result,
        model_names=["Four-Component", "Pitt-Lee", "True FE", "True RE"],
        log_likelihoods=lls,
        aics=aics,
        bics=bics,
        variance_shares={
            "Four-Component": fc_variance_shares,
            "Pitt-Lee": pl_variance_shares,
            "True FE": fe_variance_shares,
            "True RE": re_variance_shares,
        },
        efficiency_correlation=corr_matrix,
    )
