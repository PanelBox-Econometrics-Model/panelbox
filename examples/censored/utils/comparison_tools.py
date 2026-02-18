"""
Comparison and sensitivity analysis tools for censored and selection models.

Provides utilities for comparing Tobit vs OLS, Heckman two-step vs MLE,
and running sensitivity analyses across model specifications.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compare_tobit_ols(
    tobit_result: Any,
    ols_result: Any,
    variable_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare Tobit and OLS estimation results side by side.

    Creates a comparison table showing coefficients, standard errors,
    and significance levels for both models, highlighting the attenuation
    bias in OLS when censoring is present.

    Parameters
    ----------
    tobit_result : PooledTobit or RandomEffectsTobit fit result
        Fitted Tobit model result with params, bse attributes.
    ols_result : regression result
        Fitted OLS result with params, bse attributes.
    variable_names : list of str, optional
        Names for the variables. If None, uses generic names.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns for each model's coefficients,
        standard errors, and the ratio of Tobit to OLS coefficients.

    Examples
    --------
    >>> from panelbox.models.censored import PooledTobit
    >>> comparison = compare_tobit_ols(tobit_result, ols_result)
    >>> print(comparison)
    """
    tobit_params = np.asarray(tobit_result.params)
    ols_params = np.asarray(ols_result.params)

    tobit_se = np.asarray(tobit_result.bse)
    ols_se = np.asarray(ols_result.bse)

    # Align lengths (Tobit may have sigma parameter)
    n_ols = len(ols_params)
    n_tobit = len(tobit_params)

    if variable_names is None:
        variable_names = [f"x{i}" for i in range(n_ols)]

    rows = []
    for i in range(min(n_ols, n_tobit)):
        ratio = tobit_params[i] / ols_params[i] if ols_params[i] != 0 else np.nan
        rows.append(
            {
                "Variable": variable_names[i] if i < len(variable_names) else f"x{i}",
                "OLS_Coef": ols_params[i],
                "OLS_SE": ols_se[i],
                "Tobit_Coef": tobit_params[i],
                "Tobit_SE": tobit_se[i],
                "Ratio_Tobit_OLS": ratio,
            }
        )

    # Add sigma if present in Tobit
    if hasattr(tobit_result, "sigma"):
        rows.append(
            {
                "Variable": "sigma",
                "OLS_Coef": np.nan,
                "OLS_SE": np.nan,
                "Tobit_Coef": tobit_result.sigma,
                "Tobit_SE": np.nan,
                "Ratio_Tobit_OLS": np.nan,
            }
        )

    return pd.DataFrame(rows).set_index("Variable")


def compare_heckman_methods(
    twostep_result: Any,
    mle_result: Any,
    variable_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare Heckman two-step and MLE estimation results.

    Provides a side-by-side comparison of the two estimation methods,
    including outcome equation coefficients, selection parameters (rho, sigma),
    and model diagnostics.

    Parameters
    ----------
    twostep_result : PanelHeckmanResult
        Result from Heckman two-step estimation.
    mle_result : PanelHeckmanResult
        Result from Heckman MLE estimation.
    variable_names : list of str, optional
        Names for the outcome equation variables. If None, uses generic names.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns for each method's estimates.

    Examples
    --------
    >>> from panelbox.models.selection import PanelHeckman
    >>> model_2s = PanelHeckman(y, X, sel, Z, method="two_step")
    >>> model_ml = PanelHeckman(y, X, sel, Z, method="mle")
    >>> result_2s = model_2s.fit()
    >>> result_ml = model_ml.fit()
    >>> comparison = compare_heckman_methods(result_2s, result_ml)
    """
    ts_params = np.asarray(twostep_result.outcome_params)
    ml_params = np.asarray(mle_result.outcome_params)

    n_params = min(len(ts_params), len(ml_params))

    if variable_names is None:
        variable_names = [f"x{i}" for i in range(n_params)]

    rows = []
    for i in range(n_params):
        name = variable_names[i] if i < len(variable_names) else f"x{i}"
        rows.append(
            {
                "Variable": name,
                "TwoStep_Coef": ts_params[i],
                "MLE_Coef": ml_params[i],
                "Difference": ts_params[i] - ml_params[i],
            }
        )

    # Selection parameters
    rows.append(
        {
            "Variable": "rho",
            "TwoStep_Coef": twostep_result.rho,
            "MLE_Coef": mle_result.rho,
            "Difference": twostep_result.rho - mle_result.rho,
        }
    )
    rows.append(
        {
            "Variable": "sigma",
            "TwoStep_Coef": twostep_result.sigma,
            "MLE_Coef": mle_result.sigma,
            "Difference": twostep_result.sigma - mle_result.sigma,
        }
    )
    rows.append(
        {
            "Variable": "lambda (rho*sigma)",
            "TwoStep_Coef": twostep_result.rho * twostep_result.sigma,
            "MLE_Coef": mle_result.rho * mle_result.sigma,
            "Difference": (
                twostep_result.rho * twostep_result.sigma - mle_result.rho * mle_result.sigma
            ),
        }
    )

    return pd.DataFrame(rows).set_index("Variable")


def sensitivity_analysis(
    model_class: type,
    base_endog: np.ndarray,
    base_exog: np.ndarray,
    specifications: List[Dict[str, Any]],
    variable_names: Optional[List[str]] = None,
    **model_kwargs: Any,
) -> pd.DataFrame:
    """
    Run sensitivity analysis across multiple model specifications.

    Fits the same model class with different configurations (e.g., different
    censoring points, subsamples, or additional covariates) and compares
    the resulting parameter estimates.

    Parameters
    ----------
    model_class : type
        PanelBox model class (e.g., PooledTobit, RandomEffectsTobit).
    base_endog : np.ndarray
        Dependent variable array.
    base_exog : np.ndarray
        Base independent variables array.
    specifications : list of dict
        Each dict defines a specification with keys:
        - 'name' (str): Label for this specification.
        - 'endog' (np.ndarray, optional): Override endog.
        - 'exog' (np.ndarray, optional): Override exog.
        - Any additional keyword arguments passed to the model constructor.
    variable_names : list of str, optional
        Names for the variables.
    **model_kwargs
        Additional keyword arguments passed to all model constructors.

    Returns
    -------
    pd.DataFrame
        Table with specifications as columns and variables as rows.

    Examples
    --------
    >>> specs = [
    ...     {"name": "Baseline", "censoring_point": 0},
    ...     {"name": "Higher Censor", "censoring_point": 5},
    ... ]
    >>> results = sensitivity_analysis(PooledTobit, y, X, specs)
    """
    results = {}

    for spec in specifications:
        spec_copy = spec.copy()
        name = spec_copy.pop("name", f"Spec_{len(results)}")
        endog = spec_copy.pop("endog", base_endog)
        exog = spec_copy.pop("exog", base_exog)

        kwargs = {**model_kwargs, **spec_copy}

        try:
            model = model_class(endog=endog, exog=exog, **kwargs)
            result = model.fit()
            params = np.asarray(result.params)
            se = np.asarray(result.bse)

            col_data = {}
            for i in range(len(params)):
                var = variable_names[i] if variable_names and i < len(variable_names) else f"x{i}"
                col_data[f"{var}_coef"] = params[i]
                col_data[f"{var}_se"] = se[i]
            col_data["converged"] = getattr(result, "converged", True)
            col_data["n_obs"] = len(endog)

            results[name] = col_data

        except Exception as e:
            results[name] = {"error": str(e)}

    return pd.DataFrame(results).T
