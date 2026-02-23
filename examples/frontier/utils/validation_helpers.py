"""
Validation and diagnostic tools for SFA models.

Functions:
- validate_frontier_assumptions: Check key SFA assumptions
- efficiency_ranking_stability: Spearman rank correlations across models
- bootstrap_efficiency_ci: Wrapper around result.bootstrap_efficiency()
- model_selection_workflow: Automated model selection workflow
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def validate_frontier_assumptions(result) -> dict[str, dict]:
    """
    Check key SFA assumptions.

    Tests:
    - Residual skewness (correct sign for production/cost frontier)
    - Distributional tests on composed error
    - Heteroskedasticity in v component

    Parameters
    ----------
    result : SFResult or PanelSFResult
        Fitted SFA result object.

    Returns
    -------
    dict
        Dictionary with test results and pass/fail flags.
    """
    diagnostics = {}

    # 1. Residual skewness test
    try:
        residuals = result.residuals
        skew = stats.skew(residuals)
        skew_stat, skew_pval = stats.skewtest(residuals)

        # For production frontier: residuals should be negatively skewed
        # For cost frontier: residuals should be positively skewed
        frontier_type = getattr(result.model, "frontier_type", "production")
        if hasattr(frontier_type, "value"):
            frontier_type = frontier_type.value

        if frontier_type == "production":
            correct_sign = skew < 0
            expected = "negative"
        else:
            correct_sign = skew > 0
            expected = "positive"

        diagnostics["skewness"] = {
            "skewness": float(skew),
            "test_statistic": float(skew_stat),
            "p_value": float(skew_pval),
            "expected_sign": expected,
            "correct_sign": bool(correct_sign),
            "pass": bool(correct_sign),
            "interpretation": (
                f"Residuals are {'correctly' if correct_sign else 'incorrectly'} "
                f"skewed ({skew:.4f}). Expected {expected} skewness for "
                f"{frontier_type} frontier."
            ),
        }
    except Exception as e:
        diagnostics["skewness"] = {"pass": False, "error": str(e)}

    # 2. Normality test on composed error
    try:
        residuals = result.residuals
        jb_stat, jb_pval = stats.jarque_bera(residuals)
        diagnostics["normality"] = {
            "jarque_bera_stat": float(jb_stat),
            "p_value": float(jb_pval),
            "pass": bool(jb_pval > 0.05),
            "interpretation": (
                f"Jarque-Bera test: statistic={jb_stat:.4f}, p={jb_pval:.4f}. "
                f"{'Cannot reject' if jb_pval > 0.05 else 'Reject'} normality of "
                f"composed error at 5% level."
            ),
        }
    except Exception as e:
        diagnostics["normality"] = {"pass": False, "error": str(e)}

    # 3. Inefficiency significance (gamma parameter)
    try:
        gamma = result.gamma
        lambda_param = result.lambda_param

        diagnostics["inefficiency_significance"] = {
            "gamma": float(gamma),
            "lambda": float(lambda_param),
            "sigma_u": float(result.sigma_u),
            "sigma_v": float(result.sigma_v),
            "pass": bool(gamma > 0.05),
            "interpretation": (
                f"γ = {gamma:.4f} ({gamma * 100:.1f}% of variance from inefficiency). "
                f"λ = σ_u/σ_v = {lambda_param:.4f}. "
                f"{'Substantial' if gamma > 0.3 else 'Moderate' if gamma > 0.1 else 'Low'} "
                f"inefficiency component."
            ),
        }
    except Exception as e:
        diagnostics["inefficiency_significance"] = {"pass": False, "error": str(e)}

    # 4. Heteroskedasticity (Breusch-Pagan on squared residuals)
    try:
        residuals = result.residuals
        n = len(residuals)
        sq_resid = residuals**2
        # Simple proxy: regress squared residuals on fitted values
        fitted = result.model.data[result.model.depvar].values - residuals
        X = np.column_stack([np.ones(n), fitted])
        beta_bp = np.linalg.lstsq(X, sq_resid, rcond=None)[0]
        fitted_sq = X @ beta_bp
        ssr = np.sum((sq_resid - fitted_sq) ** 2)
        sst = np.sum((sq_resid - sq_resid.mean()) ** 2)
        r2 = 1 - ssr / sst if sst > 0 else 0
        bp_stat = n * r2
        bp_pval = 1 - stats.chi2.cdf(bp_stat, 1)

        diagnostics["heteroskedasticity"] = {
            "bp_statistic": float(bp_stat),
            "p_value": float(bp_pval),
            "pass": bool(bp_pval > 0.05),
            "interpretation": (
                f"Breusch-Pagan test: statistic={bp_stat:.4f}, p={bp_pval:.4f}. "
                f"{'No evidence' if bp_pval > 0.05 else 'Evidence'} of "
                f"heteroskedasticity at 5% level."
            ),
        }
    except Exception as e:
        diagnostics["heteroskedasticity"] = {"pass": False, "error": str(e)}

    # Overall assessment
    n_pass = sum(1 for d in diagnostics.values() if d.get("pass", False))
    n_total = len(diagnostics)
    diagnostics["overall"] = {
        "tests_passed": n_pass,
        "tests_total": n_total,
        "pass": n_pass >= n_total - 1,  # Allow one failure
        "interpretation": f"{n_pass}/{n_total} diagnostic tests passed.",
    }

    return diagnostics


def efficiency_ranking_stability(
    results_list: list,
    model_names: list,
    efficiency_col: str = "efficiency",
    entity_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute Spearman rank correlations of efficiency rankings across models.

    Parameters
    ----------
    results_list : list
        List of fitted SFA result objects.
    model_names : list
        List of model names corresponding to results_list.
    efficiency_col : str
        Column name for efficiency in the efficiency DataFrames.
    entity_col : str, optional
        Column name for entity identifiers.

    Returns
    -------
    pd.DataFrame
        Correlation matrix (Spearman) of efficiency rankings.
    """
    efficiency_series = {}

    for name, result in zip(model_names, results_list):
        try:
            eff_df = result.efficiency(estimator="bc")
            if entity_col and entity_col in eff_df.columns:
                eff = eff_df.groupby(entity_col)[efficiency_col].mean()
            elif efficiency_col in eff_df.columns:
                eff = eff_df[efficiency_col]
            else:
                # Try to find the right column
                eff_cols = [c for c in eff_df.columns if "effic" in c.lower()]
                eff = eff_df[eff_cols[0]] if eff_cols else eff_df.iloc[:, -1]
            efficiency_series[name] = eff.values
        except Exception as e:
            print(f"Warning: Could not extract efficiency from {name}: {e}")

    if len(efficiency_series) < 2:
        return pd.DataFrame()

    # Ensure same length (take minimum)
    min_len = min(len(v) for v in efficiency_series.values())
    for key in efficiency_series:
        efficiency_series[key] = efficiency_series[key][:min_len]

    # Compute Spearman correlations
    names = list(efficiency_series.keys())
    n_models = len(names)
    corr_matrix = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            rho, _pval = stats.spearmanr(efficiency_series[names[i]], efficiency_series[names[j]])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    return pd.DataFrame(corr_matrix, index=names, columns=names)


def bootstrap_efficiency_ci(
    result,
    n_boot: int = 999,
    ci_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Wrapper around result.bootstrap_efficiency() with progress tracking.

    Parameters
    ----------
    result : SFResult or PanelSFResult
        Fitted SFA result object.
    n_boot : int
        Number of bootstrap replications.
    ci_level : float
        Confidence level for intervals.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with efficiency estimates and confidence intervals.
    """
    print(f"Starting bootstrap with {n_boot} replications...")

    try:
        boot_eff = result.bootstrap_efficiency(
            estimator="bc",
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed,
        )
        print(f"Bootstrap completed. Shape: {boot_eff.shape}")
        return boot_eff
    except Exception as e:
        print(f"Bootstrap failed: {e}")
        print("Falling back to point estimates without CIs.")
        eff = result.efficiency(estimator="bc")
        return eff


def model_selection_workflow(
    data: pd.DataFrame,
    depvar: str,
    exog: list,
    entity: Optional[str] = None,
    time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Automated model selection workflow.

    Tests distributions, panel types, and reports comparison metrics.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    depvar : str
        Dependent variable name.
    exog : list
        List of exogenous variable names.
    entity : str, optional
        Entity identifier column (for panel models).
    time : str, optional
        Time identifier column (for panel models).

    Returns
    -------
    pd.DataFrame
        Comprehensive comparison table with recommendations.
    """
    from panelbox.frontier import StochasticFrontier

    results_summary = []

    # Define distributions to test
    distributions = ["half_normal", "exponential", "truncated_normal"]

    # Define model types based on whether panel structure is available
    if entity and time:
        model_types = [None, "pitt_lee", "bc92"]
    else:
        model_types = [None]

    for dist in distributions:
        for model_type in model_types:
            model_name = f"{dist}"
            if model_type:
                model_name += f"_{model_type}"

            try:
                kwargs = {
                    "data": data,
                    "depvar": depvar,
                    "exog": exog,
                    "dist": dist,
                }
                if entity:
                    kwargs["entity"] = entity
                if time:
                    kwargs["time"] = time
                if model_type:
                    kwargs["model_type"] = model_type

                model = StochasticFrontier(**kwargs)
                result = model.fit()

                row = {
                    "Model": model_name,
                    "Distribution": dist,
                    "Panel Type": model_type or "cross-section",
                    "Log-Lik": result.loglik,
                    "AIC": result.aic,
                    "BIC": result.bic,
                    "sigma_v": result.sigma_v,
                    "sigma_u": result.sigma_u,
                    "gamma": result.gamma,
                    "Mean TE": result.mean_efficiency,
                    "Converged": result.converged,
                    "N params": result.nparams,
                }
                results_summary.append(row)
                print(f"  {model_name}: AIC={result.aic:.2f}, Mean TE={result.mean_efficiency:.4f}")

            except Exception as e:
                print(f"  {model_name}: FAILED - {e}")
                results_summary.append(
                    {
                        "Model": model_name,
                        "Distribution": dist,
                        "Panel Type": model_type or "cross-section",
                        "Log-Lik": None,
                        "AIC": None,
                        "BIC": None,
                        "Converged": False,
                    }
                )

    df = pd.DataFrame(results_summary)

    # Add recommendation
    if not df.empty and "AIC" in df.columns:
        valid = df.dropna(subset=["AIC"])
        if not valid.empty:
            best_aic = valid.loc[valid["AIC"].idxmin(), "Model"]
            best_bic = valid.loc[valid["BIC"].idxmin(), "Model"]
            print("\nRecommendation:")
            print(f"  Best by AIC: {best_aic}")
            print(f"  Best by BIC: {best_bic}")

    return df
