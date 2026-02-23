"""
Model comparison utilities for Quantile Regression tutorials.

Functions for comparing QR with OLS, comparing FE methods,
and summarizing results across quantiles.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd


def compare_qr_ols(
    qr_results: dict[float, object],
    ols_result: object,
    variables: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Create comparison table of QR coefficients vs OLS.

    Parameters
    ----------
    qr_results : dict
        Mapping from tau -> QR result object with .params attribute.
    ols_result : object
        OLS result with .params attribute.
    variables : list of str, optional
        Variable names. If None, uses indices.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns: variable, OLS, tau=0.10, tau=0.25, ..., tau=0.90.
    """
    taus = sorted(qr_results.keys())

    # Get OLS params
    ols_params = np.asarray(ols_result.params).flatten()
    n_vars = len(ols_params)

    if variables is None:
        variables = [f"x{i}" for i in range(n_vars)]

    rows = []
    for i, var in enumerate(variables):
        row = {"Variable": var, "OLS": ols_params[i]}
        for tau in taus:
            res = qr_results[tau]
            params = np.asarray(res.params).flatten()
            if i < len(params):
                row[f"τ={tau:.2f}"] = params[i]
            else:
                row[f"τ={tau:.2f}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def compare_fe_methods(
    data: pd.DataFrame,
    formula: str,
    tau_list: list[float],
    entity_col: str,
    time_col: str,
    methods: list[str] = None,
) -> pd.DataFrame:
    """
    Run pooled, Canay, and penalty QR and compare coefficients.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    formula : str
        Model formula (e.g., 'y ~ x1 + x2').
    tau_list : list of float
        Quantile levels.
    entity_col : str
        Entity identifier column.
    time_col : str
        Time identifier column.
    methods : list of str
        Methods to compare.

    Returns
    -------
    pd.DataFrame
        DataFrame with method, tau, variable, coefficient, se columns.
    """
    if methods is None:
        methods = ["pooled", "canay", "penalty"]
    results_rows = []

    for method in methods:
        for tau in tau_list:
            try:
                if method == "pooled":
                    from panelbox.models.quantile import PooledQuantile

                    # Parse formula
                    dep, indep = formula.split("~")
                    dep = dep.strip()
                    indep_vars = [v.strip() for v in indep.split("+")]

                    y = data[dep].values
                    X = np.column_stack([np.ones(len(data))] + [data[v].values for v in indep_vars])
                    entity_id = data[entity_col].values

                    model = PooledQuantile(y, X, entity_id=entity_id, quantiles=tau)
                    res = model.fit()

                    var_names = ["const", *indep_vars]
                    params = res.params.flatten()
                    ses = (
                        res.std_errors.flatten()
                        if hasattr(res, "std_errors")
                        else np.zeros(len(params))
                    )

                    for j, vname in enumerate(var_names):
                        results_rows.append(
                            {
                                "method": method,
                                "tau": tau,
                                "variable": vname,
                                "coefficient": params[j],
                                "se": ses[j],
                            }
                        )

                elif method == "canay":
                    from panelbox.models.quantile import CanayTwoStep

                    model = CanayTwoStep(data, formula=formula, tau=tau)
                    res = model.fit()

                    if hasattr(res, "params"):
                        params = np.asarray(res.params).flatten()
                        ses = (
                            np.asarray(res.std_errors).flatten()
                            if hasattr(res, "std_errors")
                            else np.zeros(len(params))
                        )
                        dep, indep = formula.split("~")
                        indep_vars = [v.strip() for v in indep.split("+")]
                        var_names = indep_vars[: len(params)]

                        for j, vname in enumerate(var_names):
                            results_rows.append(
                                {
                                    "method": method,
                                    "tau": tau,
                                    "variable": vname,
                                    "coefficient": params[j],
                                    "se": ses[j],
                                }
                            )

                elif method == "penalty":
                    from panelbox.models.quantile import FixedEffectsQuantile

                    model = FixedEffectsQuantile(data, formula=formula, tau=tau)
                    res = model.fit()

                    if hasattr(res, "params"):
                        params = np.asarray(res.params).flatten()
                        ses = (
                            np.asarray(res.bse).flatten()
                            if hasattr(res, "bse")
                            else np.zeros(len(params))
                        )
                        dep, indep = formula.split("~")
                        indep_vars = [v.strip() for v in indep.split("+")]
                        var_names = indep_vars[: len(params)]

                        for j, vname in enumerate(var_names):
                            results_rows.append(
                                {
                                    "method": method,
                                    "tau": tau,
                                    "variable": vname,
                                    "coefficient": params[j],
                                    "se": ses[j],
                                }
                            )

            except Exception as e:
                print(f"Warning: {method} failed at τ={tau}: {e}")

    return pd.DataFrame(results_rows)


def inter_quantile_test(
    result_low: object,
    result_high: object,
    variable: int,
    tau_low: float,
    tau_high: float,
) -> dict:
    """
    Test H0: beta(tau_low) = beta(tau_high) using Wald test.

    Parameters
    ----------
    result_low : object
        QR result at lower quantile.
    result_high : object
        QR result at higher quantile.
    variable : int
        Variable index to test.
    tau_low : float
        Lower quantile.
    tau_high : float
        Higher quantile.

    Returns
    -------
    dict
        With keys: diff, se, t_stat, p_value, significant.
    """
    from scipy import stats as sp_stats

    # Extract params
    p_low = np.asarray(result_low.params).flatten()
    p_high = np.asarray(result_high.params).flatten()

    diff = p_high[variable] - p_low[variable]

    # Get standard errors
    if hasattr(result_low, "std_errors"):
        se_low = np.asarray(result_low.std_errors).flatten()[variable]
    elif hasattr(result_low, "bse"):
        se_low = result_low.bse[variable]
    else:
        se_low = 0.01

    if hasattr(result_high, "std_errors"):
        se_high = np.asarray(result_high.std_errors).flatten()[variable]
    elif hasattr(result_high, "bse"):
        se_high = result_high.bse[variable]
    else:
        se_high = 0.01

    # SE of difference (assuming independence between quantiles — conservative)
    se_diff = np.sqrt(se_low**2 + se_high**2)

    t_stat = diff / (se_diff + 1e-10)
    p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(t_stat)))

    return {
        "diff": diff,
        "se": se_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "tau_low": tau_low,
        "tau_high": tau_high,
    }


def pseudo_r2_table(results_dict: dict[float, object]) -> pd.DataFrame:
    """
    Create Pseudo-R2 comparison table across quantiles.

    Uses the Koenker-Machado (1999) pseudo R-squared:
    R1(tau) = 1 - V_hat(tau) / V_tilde(tau)
    where V_hat is the minimized check loss and V_tilde is the
    check loss from the intercept-only model.

    Parameters
    ----------
    results_dict : dict
        Mapping from tau -> QR result object.

    Returns
    -------
    pd.DataFrame
        Table with tau, pseudo_r2 columns.
    """
    rows = []
    for tau, res in sorted(results_dict.items()):
        if hasattr(res, "pseudo_r2"):
            r2 = res.pseudo_r2
        elif hasattr(res, "model") and hasattr(res.model, "endog"):
            # Compute manually
            y = res.model.endog
            if hasattr(res, "predict"):
                fitted = res.predict()
            else:
                fitted = res.model.exog @ (res.params if res.params.ndim == 1 else res.params[:, 0])

            resid_full = y - fitted
            check_full = np.sum(resid_full * (tau - (resid_full < 0).astype(float)))

            # Intercept-only model
            q_tau = np.quantile(y, tau)
            resid_null = y - q_tau
            check_null = np.sum(resid_null * (tau - (resid_null < 0).astype(float)))

            r2 = 1 - check_full / (check_null + 1e-10)
        else:
            r2 = np.nan

        rows.append({"tau": tau, "pseudo_r2": r2})

    return pd.DataFrame(rows)


def create_summary_table(
    results_dict: dict[float, object],
    variables: Optional[list[str]] = None,
    format: str = "wide",
) -> pd.DataFrame:
    """
    Generate formatted summary table with coefficients and standard errors.

    Parameters
    ----------
    results_dict : dict
        Mapping from tau -> QR result object.
    variables : list of str, optional
        Variable names.
    format : str
        'wide': rows=variables, columns=tau.
        'long': rows=(variable, tau), columns=(coef, se, pvalue).

    Returns
    -------
    pd.DataFrame
    """
    taus = sorted(results_dict.keys())

    # Determine n_vars from first result
    first_res = results_dict[taus[0]]
    params = np.asarray(first_res.params).flatten()
    n_vars = len(params)

    if variables is None:
        if hasattr(first_res, "param_names") and first_res.param_names:
            variables = first_res.param_names
        elif (
            hasattr(first_res, "model")
            and hasattr(first_res.model, "param_names")
            and first_res.model.param_names
        ):
            variables = first_res.model.param_names
        else:
            variables = [f"x{i}" for i in range(n_vars)]

    if format == "long":
        rows = []
        for tau in taus:
            res = results_dict[tau]
            params = np.asarray(res.params).flatten()
            if hasattr(res, "std_errors"):
                ses = np.asarray(res.std_errors).flatten()
            elif hasattr(res, "bse"):
                ses = np.asarray(res.bse).flatten()
            else:
                ses = np.zeros(n_vars)

            for i, var in enumerate(variables[:n_vars]):
                coef = params[i]
                se = ses[i] if i < len(ses) else np.nan
                t_stat = coef / (se + 1e-10)
                from scipy import stats as sp_stats

                p_val = 2 * (1 - sp_stats.norm.cdf(np.abs(t_stat)))
                rows.append(
                    {
                        "variable": var,
                        "tau": tau,
                        "coefficient": coef,
                        "se": se,
                        "t_stat": t_stat,
                        "p_value": p_val,
                    }
                )
        return pd.DataFrame(rows)

    else:  # wide
        data = {}
        for tau in taus:
            res = results_dict[tau]
            params = np.asarray(res.params).flatten()
            if hasattr(res, "std_errors"):
                ses = np.asarray(res.std_errors).flatten()
            elif hasattr(res, "bse"):
                ses = np.asarray(res.bse).flatten()
            else:
                ses = np.zeros(n_vars)

            coef_col = []
            for i in range(n_vars):
                coef_col.append(f"{params[i]:.4f}")
                if i < len(ses):
                    coef_col.append(f"({ses[i]:.4f})")
                else:
                    coef_col.append("")

            data[f"τ={tau:.2f}"] = coef_col

        # Row labels
        row_labels = []
        for var in variables[:n_vars]:
            row_labels.append(var)
            row_labels.append("")

        df = pd.DataFrame(data, index=row_labels)
        return df


def timing_benchmark(
    data: pd.DataFrame,
    formula: str,
    tau: float = 0.5,
    methods: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Benchmark computation time for different QR methods.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    formula : str
        Model formula.
    tau : float
        Quantile level.
    methods : list of str, optional
        Methods to benchmark. Defaults to ['pooled', 'canay', 'penalty'].

    Returns
    -------
    pd.DataFrame
        DataFrame with method, time_seconds, n_iterations columns.
    """
    if methods is None:
        methods = ["pooled", "canay", "penalty"]

    results = []

    for method in methods:
        try:
            start = time.time()
            n_iter = None

            if method == "pooled":
                from panelbox.models.quantile import PooledQuantile

                dep, indep = formula.split("~")
                dep = dep.strip()
                indep_vars = [v.strip() for v in indep.split("+")]

                y = data[dep].values
                X = np.column_stack([np.ones(len(data))] + [data[v].values for v in indep_vars])
                model = PooledQuantile(y, X, quantiles=tau)
                res = model.fit()
                n_iter = getattr(res, "n_iter", None)

            elif method == "canay":
                from panelbox.models.quantile import CanayTwoStep

                model = CanayTwoStep(data, formula=formula, tau=tau)
                res = model.fit()

            elif method == "penalty":
                from panelbox.models.quantile import FixedEffectsQuantile

                model = FixedEffectsQuantile(data, formula=formula, tau=tau)
                res = model.fit()

            elapsed = time.time() - start
            results.append(
                {
                    "method": method,
                    "time_seconds": round(elapsed, 4),
                    "n_iterations": n_iter if n_iter else "N/A",
                }
            )

        except Exception as e:
            results.append(
                {
                    "method": method,
                    "time_seconds": np.nan,
                    "n_iterations": f"Error: {e}",
                }
            )

    return pd.DataFrame(results)
