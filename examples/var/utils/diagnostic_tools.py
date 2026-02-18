"""
Diagnostic tools for VAR model validation.

Functions:
- residual_diagnostics: Comprehensive residual tests (Ljung-Box, Jarque-Bera, ARCH-LM)
- model_comparison_table: Compare multiple VAR specifications (AIC, BIC, etc.)
- forecast_evaluation: Evaluate forecasts (RMSE, MAE, MAPE, Theil's U)
- granger_causality_summary: Summary table of all pairwise Granger tests
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


def residual_diagnostics(results, save_dir=None):
    """
    Comprehensive residual diagnostics for a fitted PanelVAR.

    Produces:
    - Portmanteau test (Ljung-Box) for serial correlation
    - Jarque-Bera normality test
    - ARCH-LM test for conditional heteroskedasticity
    - Residual ACF/PACF plots (if save_dir provided)
    - Residual histogram with normal overlay (if save_dir provided)

    Parameters
    ----------
    results : PanelVARResult
        Fitted VAR result. Has resid_by_eq (list of np.ndarray, one per equation),
        endog_names (list of str), K (int).
    save_dir : str, optional
        Directory to save diagnostic plots.

    Returns
    -------
    dict
        Keys are variable names, values are dicts with:
        - 'ljung_box_stat', 'ljung_box_pvalue'
        - 'jarque_bera_stat', 'jarque_bera_pvalue'
        - 'arch_lm_stat', 'arch_lm_pvalue'
        - 'mean', 'std', 'skewness', 'kurtosis'
    """
    diagnostics = {}

    for k in range(results.K):
        var_name = results.endog_names[k]
        resid = np.asarray(results.resid_by_eq[k], dtype=float).ravel()
        T = len(resid)

        entry = {}

        # ---- Descriptive statistics ----
        entry["mean"] = float(np.mean(resid))
        entry["std"] = float(np.std(resid, ddof=1)) if T > 1 else 0.0
        entry["skewness"] = float(stats.skew(resid, bias=False)) if T > 2 else np.nan
        entry["kurtosis"] = float(stats.kurtosis(resid, bias=False)) if T > 3 else np.nan

        # ---- Ljung-Box portmanteau test ----
        try:
            max_lag = min(10, T - 1)
            if max_lag < 1:
                raise ValueError("Insufficient observations for Ljung-Box test")

            # Compute sample autocorrelations
            resid_demean = resid - np.mean(resid)
            gamma_0 = np.dot(resid_demean, resid_demean) / T

            if gamma_0 < 1e-15:
                raise ValueError("Residual variance is essentially zero")

            Q_stat = 0.0
            for j in range(1, max_lag + 1):
                gamma_j = np.dot(resid_demean[j:], resid_demean[:-j]) / T
                rho_j = gamma_j / gamma_0
                Q_stat += (rho_j**2) / (T - j)

            Q_stat *= T * (T + 2)

            lb_pvalue = 1.0 - stats.chi2.cdf(Q_stat, df=max_lag)
            entry["ljung_box_stat"] = float(Q_stat)
            entry["ljung_box_pvalue"] = float(lb_pvalue)
        except Exception:
            entry["ljung_box_stat"] = np.nan
            entry["ljung_box_pvalue"] = np.nan

        # ---- Jarque-Bera normality test ----
        try:
            if T < 8:
                raise ValueError("Too few observations for Jarque-Bera test")
            jb_stat, jb_pvalue = stats.jarque_bera(resid)
            entry["jarque_bera_stat"] = float(jb_stat)
            entry["jarque_bera_pvalue"] = float(jb_pvalue)
        except Exception:
            entry["jarque_bera_stat"] = np.nan
            entry["jarque_bera_pvalue"] = np.nan

        # ---- ARCH-LM test ----
        try:
            arch_lags = min(5, T // 3)
            if arch_lags < 1:
                raise ValueError("Insufficient observations for ARCH-LM test")

            resid_sq = resid**2
            n_arch = len(resid_sq)

            # Build the lag matrix for squared residuals
            y_arch = resid_sq[arch_lags:]
            X_arch = np.ones((n_arch - arch_lags, arch_lags + 1))
            for lag in range(1, arch_lags + 1):
                X_arch[:, lag] = resid_sq[arch_lags - lag : n_arch - lag]

            # OLS regression of squared residuals on their lags
            beta_arch = np.linalg.lstsq(X_arch, y_arch, rcond=None)[0]
            fitted_arch = X_arch @ beta_arch
            ss_res = np.sum((y_arch - fitted_arch) ** 2)
            ss_tot = np.sum((y_arch - np.mean(y_arch)) ** 2)

            if ss_tot < 1e-15:
                raise ValueError("Total sum of squares for ARCH test is zero")

            R_sq = 1.0 - ss_res / ss_tot
            R_sq = max(0.0, min(R_sq, 1.0))  # Clamp to [0, 1]

            n_eff = len(y_arch)
            LM_stat = n_eff * R_sq
            lm_pvalue = 1.0 - stats.chi2.cdf(LM_stat, df=arch_lags)

            entry["arch_lm_stat"] = float(LM_stat)
            entry["arch_lm_pvalue"] = float(lm_pvalue)
        except Exception:
            entry["arch_lm_stat"] = np.nan
            entry["arch_lm_pvalue"] = np.nan

        diagnostics[var_name] = entry

        # ---- Optional diagnostic plots ----
        if save_dir is not None:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from matplotlib.gridspec import GridSpec

                os.makedirs(save_dir, exist_ok=True)

                fig = plt.figure(figsize=(14, 10))
                gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
                fig.suptitle(
                    f"Residual Diagnostics: {var_name}",
                    fontsize=14,
                    fontweight="bold",
                )

                # -- Panel 1: ACF plot --
                ax_acf = fig.add_subplot(gs[0, 0])
                max_acf_lags = min(20, T - 1)
                if max_acf_lags >= 1 and gamma_0 > 1e-15:
                    acf_vals = []
                    for lag in range(1, max_acf_lags + 1):
                        g = np.dot(resid_demean[lag:], resid_demean[:-lag]) / T
                        acf_vals.append(g / gamma_0)
                    lags_arr = np.arange(1, max_acf_lags + 1)
                    ax_acf.bar(lags_arr, acf_vals, color="steelblue", width=0.6)
                    ci_bound = 1.96 / np.sqrt(T)
                    ax_acf.axhline(ci_bound, linestyle="--", color="red", alpha=0.7)
                    ax_acf.axhline(-ci_bound, linestyle="--", color="red", alpha=0.7)
                    ax_acf.axhline(0, color="black", linewidth=0.5)
                ax_acf.set_xlabel("Lag")
                ax_acf.set_ylabel("ACF")
                ax_acf.set_title("Autocorrelation Function")

                # -- Panel 2: PACF plot (Durbin-Levinson) --
                ax_pacf = fig.add_subplot(gs[0, 1])
                if max_acf_lags >= 1 and gamma_0 > 1e-15:
                    # Compute PACF using successive OLS regressions
                    pacf_vals = []
                    for order in range(1, max_acf_lags + 1):
                        y_p = resid_demean[order:]
                        X_p = np.column_stack(
                            [resid_demean[order - lag : T - lag] for lag in range(1, order + 1)]
                        )
                        try:
                            coef = np.linalg.lstsq(X_p, y_p, rcond=None)[0]
                            pacf_vals.append(coef[-1])
                        except Exception:
                            pacf_vals.append(0.0)
                    ax_pacf.bar(
                        np.arange(1, max_acf_lags + 1),
                        pacf_vals,
                        color="darkorange",
                        width=0.6,
                    )
                    ax_pacf.axhline(ci_bound, linestyle="--", color="red", alpha=0.7)
                    ax_pacf.axhline(-ci_bound, linestyle="--", color="red", alpha=0.7)
                    ax_pacf.axhline(0, color="black", linewidth=0.5)
                ax_pacf.set_xlabel("Lag")
                ax_pacf.set_ylabel("PACF")
                ax_pacf.set_title("Partial Autocorrelation Function")

                # -- Panel 3: Histogram with normal overlay --
                ax_hist = fig.add_subplot(gs[1, 0])
                n_bins = min(50, max(10, T // 5))
                ax_hist.hist(
                    resid,
                    bins=n_bins,
                    density=True,
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="white",
                )
                x_range = np.linspace(
                    np.min(resid) - 0.5 * entry["std"],
                    np.max(resid) + 0.5 * entry["std"],
                    200,
                )
                if entry["std"] > 0:
                    normal_pdf = stats.norm.pdf(x_range, loc=entry["mean"], scale=entry["std"])
                    ax_hist.plot(x_range, normal_pdf, "r-", linewidth=2, label="Normal")
                    ax_hist.legend()
                ax_hist.set_xlabel("Residual Value")
                ax_hist.set_ylabel("Density")
                ax_hist.set_title("Residual Distribution")

                # -- Panel 4: Q-Q plot --
                ax_qq = fig.add_subplot(gs[1, 1])
                sorted_resid = np.sort(resid)
                theoretical_quantiles = stats.norm.ppf((np.arange(1, T + 1) - 0.5) / T)
                ax_qq.scatter(
                    theoretical_quantiles,
                    sorted_resid,
                    alpha=0.6,
                    s=15,
                    color="steelblue",
                )
                # 45-degree reference line
                q_min = min(theoretical_quantiles.min(), sorted_resid.min())
                q_max = max(theoretical_quantiles.max(), sorted_resid.max())
                ax_qq.plot(
                    [q_min, q_max],
                    [q_min, q_max],
                    "r--",
                    linewidth=1.5,
                )
                ax_qq.set_xlabel("Theoretical Quantiles")
                ax_qq.set_ylabel("Sample Quantiles")
                ax_qq.set_title("Normal Q-Q Plot")

                fig.savefig(
                    os.path.join(save_dir, f"residual_diagnostics_{var_name}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

            except ImportError:
                pass  # matplotlib not available; skip plots silently
            except Exception:
                pass  # any plot failure should not break diagnostics

    return diagnostics


def model_comparison_table(results_list, model_names, criteria=None):
    """
    Create comparison table across multiple VAR specifications.

    Parameters
    ----------
    results_list : list of PanelVARResult
        List of fitted VAR results. Each has: n_obs, K, p, aic, bic, hqic,
        loglik, is_stable().
    model_names : list of str
        Labels for each model.
    criteria : list of str, optional
        Subset of ['aic', 'bic', 'hqic', 'loglik'] to include.
        Default: all four.

    Returns
    -------
    pd.DataFrame
        One row per model with columns: Model, N_obs, K, Lags, AIC, BIC,
        HQIC, LogLik, Stable.  Values corresponding to the best (minimum)
        AIC/BIC/HQIC are marked with an asterisk in the display.
    """
    if len(results_list) != len(model_names):
        raise ValueError(
            f"Length mismatch: {len(results_list)} results vs " f"{len(model_names)} model names."
        )

    all_criteria = ["aic", "bic", "hqic", "loglik"]
    if criteria is None:
        criteria = all_criteria
    else:
        invalid = [c for c in criteria if c not in all_criteria]
        if invalid:
            raise ValueError(f"Unknown criteria: {invalid}. " f"Valid options: {all_criteria}")

    # Column-name mapping (lowercase attr -> display label)
    label_map = {"aic": "AIC", "bic": "BIC", "hqic": "HQIC", "loglik": "LogLik"}

    rows = []
    for res, name in zip(results_list, model_names):
        row = {
            "Model": name,
            "N_obs": int(res.n_obs),
            "K": int(res.K),
            "Lags": int(res.p),
        }

        for crit in criteria:
            try:
                row[label_map[crit]] = float(getattr(res, crit))
            except AttributeError:
                row[label_map[crit]] = np.nan

        try:
            row["Stable"] = res.is_stable()
        except Exception:
            row["Stable"] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Identify best (minimum) for AIC, BIC, HQIC and mark with asterisk.
    # For LogLik, best is the maximum value.
    minimise = {"AIC", "BIC", "HQIC"}
    maximise = {"LogLik"}

    for col in df.columns:
        if col in minimise and col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            if numeric_vals.notna().any():
                best_idx = numeric_vals.idxmin()
                df.loc[best_idx, col] = f"{df.loc[best_idx, col]}*"
        elif col in maximise and col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            if numeric_vals.notna().any():
                best_idx = numeric_vals.idxmax()
                df.loc[best_idx, col] = f"{df.loc[best_idx, col]}*"

    return df


def forecast_evaluation(actual, forecasts, variable):
    """
    Evaluate multiple forecast models against actuals.

    Parameters
    ----------
    actual : pd.DataFrame, pd.Series, or np.ndarray
        Actual values.  If a 2-D array/DataFrame, the column named
        ``variable`` (or column 0 for a plain array) is used.
        If 1-D, used directly.
    forecasts : dict
        ``{model_name: np.ndarray}`` of predicted values.  Each array
        should be 1-D with the same length as ``actual`` (after column
        selection).
    variable : str
        Which variable to evaluate (used for labelling and column lookup).

    Returns
    -------
    pd.DataFrame
        One row per model with columns: RMSE, MAE, MAPE, Theil_U,
        Direction_Accuracy.
    """
    # Resolve actual to a 1-D numpy array
    if isinstance(actual, pd.DataFrame):
        if variable in actual.columns:
            y = actual[variable].values.astype(float)
        else:
            # Fall back to first column
            y = actual.iloc[:, 0].values.astype(float)
    elif isinstance(actual, pd.Series):
        y = actual.values.astype(float)
    else:
        y = np.asarray(actual, dtype=float).ravel()

    T = len(y)

    rows = []
    for model_name, pred in forecasts.items():
        f = np.asarray(pred, dtype=float).ravel()

        # Align lengths (use shorter of the two)
        n = min(T, len(f))
        y_n = y[:n]
        f_n = f[:n]

        errors = y_n - f_n

        # RMSE
        rmse = float(np.sqrt(np.mean(errors**2)))

        # MAE
        mae = float(np.mean(np.abs(errors)))

        # MAPE -- guard against zeros in actual
        abs_y = np.abs(y_n)
        nonzero_mask = abs_y > 1e-15
        if nonzero_mask.any():
            mape = float(np.mean(np.abs(errors[nonzero_mask]) / abs_y[nonzero_mask]) * 100)
        else:
            mape = np.nan

        # Theil's U statistic (Theil U1)
        denom = np.sqrt(np.mean(y_n**2))
        if denom > 1e-15:
            theil_u = rmse / denom
        else:
            theil_u = np.nan

        # Direction accuracy (percentage of correct directional predictions)
        if n > 1:
            actual_diff = np.diff(y_n)
            forecast_diff = np.diff(f_n)
            # Only count periods where actual actually changed
            valid_mask = actual_diff != 0
            if valid_mask.any():
                direction_correct = np.sign(actual_diff[valid_mask]) == np.sign(
                    forecast_diff[valid_mask]
                )
                direction_accuracy = float(np.mean(direction_correct) * 100)
            else:
                # Actual was flat; direction accuracy is not meaningful
                direction_accuracy = np.nan
        else:
            direction_accuracy = np.nan

        rows.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "Theil_U": theil_u,
                "Direction_Accuracy": direction_accuracy,
            }
        )

    df = pd.DataFrame(rows).set_index("Model")
    return df


def granger_causality_summary(results, significance=0.05):
    """
    Produce a summary table of all pairwise Granger causality tests.

    Parameters
    ----------
    results : PanelVARResult
        Fitted VAR. Has ``granger_causality(cause, effect)`` returning a
        ``GrangerCausalityResult`` (with ``wald_stat`` and ``p_value``
        attributes) and ``endog_names``.
    significance : float, default 0.05
        Significance level for marking results.

    Returns
    -------
    pd.DataFrame
        Columns: cause, effect, wald_stat, p_value, significant.
        Sorted by p_value ascending.
    """
    names = results.endog_names
    rows = []

    for i, cause_name in enumerate(names):
        for j, effect_name in enumerate(names):
            if i == j:
                continue  # skip self-pairs

            try:
                gc = results.granger_causality(cause=cause_name, effect=effect_name)
                wald_stat = float(gc.wald_stat)
                p_value = float(gc.p_value)
            except Exception as exc:
                # Record the failure but do not crash
                wald_stat = np.nan
                p_value = np.nan

            rows.append(
                {
                    "cause": cause_name,
                    "effect": effect_name,
                    "wald_stat": wald_stat,
                    "p_value": p_value,
                    "significant": p_value < significance if np.isfinite(p_value) else False,
                }
            )

    df = pd.DataFrame(rows)

    # Sort by p_value ascending (NaN last)
    df = df.sort_values("p_value", ascending=True, na_position="last").reset_index(drop=True)

    return df
