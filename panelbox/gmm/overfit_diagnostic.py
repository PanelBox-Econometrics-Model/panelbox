"""
GMM Overfit Diagnostic
======================

Diagnostics for detecting instrument proliferation and overfitting in GMM
estimation, especially critical for panels with small N (few groups).

When the number of instruments approaches or exceeds the number of groups,
standard specification tests (Hansen J, AR(2)) lose power, and coefficients
become biased toward OLS/Within estimates. This module provides multiple
diagnostic checks to detect these problems.

Classes
-------
GMMOverfitDiagnostic : Comprehensive overfitting diagnostics for GMM models

References
----------
.. [1] Roodman, D. (2009). "How to do xtabond2: An Introduction to Difference
       and System GMM in Stata." Stata Journal, 9(1), 86-136.

.. [2] Nickell, S. (1981). "Biases in Dynamic Models with Fixed Effects."
       Econometrica, 49(6), 1417-1426.

.. [3] Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of
       Linear Efficient Two-Step GMM Estimators." Journal of Econometrics,
       126(1), 25-51.
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from panelbox.gmm.results import GMMResults


class GMMOverfitDiagnostic:
    """
    Diagnostics for detecting GMM overfitting due to instrument proliferation.

    Provides five diagnostic checks:

    1. **assess_feasibility**: Instrument count vs group count (Roodman rule)
    2. **instrument_sensitivity**: Re-estimate with varying max_lag
    3. **coefficient_bounds_test**: Nickell (1981) OLS/FE bounds check
    4. **jackknife_groups**: Leave-one-group-out stability
    5. **step_comparison**: One-step vs two-step sensitivity

    Parameters
    ----------
    model : DifferenceGMM or SystemGMM
        A fitted GMM model instance
    results : GMMResults
        Results from the fitted model

    Examples
    --------
    >>> from panelbox.gmm import DifferenceGMM, GMMOverfitDiagnostic
    >>> model = DifferenceGMM(data=df, dep_var='y', lags=1, exog_vars=['x'])
    >>> results = model.fit()
    >>> diag = GMMOverfitDiagnostic(model, results)
    >>> print(diag.summary())
    """

    # Traffic light constants
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    def __init__(self, model, results: GMMResults):
        self.model = model
        self.results = results

        # Extract key dimensions
        self.n_groups = results.n_groups
        self.n_instruments = results.n_instruments
        self.n_params = results.n_params

        # Identify the AR coefficient name (first lag of dep var)
        self._ar_param_name = None
        for name in results.params.index:
            if name.startswith("L1.") or name.startswith("L2."):
                self._ar_param_name = name
                break

    def _clone_model(self, **overrides):
        """
        Create a new model instance with the same configuration, applying overrides.

        Parameters
        ----------
        **overrides
            Keyword arguments to override in the new model constructor.

        Returns
        -------
        model
            A new unfitted model instance.
        """
        from panelbox.gmm.difference_gmm import DifferenceGMM
        from panelbox.gmm.system_gmm import SystemGMM

        # Common params shared by both DifferenceGMM and SystemGMM
        base_params = {
            "data": self.model.data,
            "dep_var": self.model.dep_var,
            "lags": self.model.lags,
            "id_var": self.model.id_var,
            "time_var": self.model.time_var,
            "exog_vars": self.model.exog_vars,
            "endogenous_vars": self.model.endogenous_vars,
            "predetermined_vars": self.model.predetermined_vars,
            "time_dummies": self.model.time_dummies,
            "collapse": self.model.collapse,
            "two_step": self.model.two_step,
            "robust": self.model.robust,
            "gmm_type": self.model.gmm_type,
            "gmm_max_lag": self.model.gmm_max_lag,
            "iv_max_lag": self.model.iv_max_lag,
        }

        # Add SystemGMM-specific params
        is_system = isinstance(self.model, SystemGMM)
        if is_system:
            base_params["level_instruments"] = self.model.level_instruments

        # Apply overrides
        base_params.update(overrides)

        # Create new model of same type
        cls = SystemGMM if is_system else DifferenceGMM
        return cls(**base_params)

    def assess_feasibility(self) -> dict:
        """
        Assess instrument count feasibility using the Roodman (2009) rule.

        The rule of thumb is that the number of instruments should not exceed
        the number of groups (N). Excess instruments weaken the Hansen J test
        and bias coefficients toward OLS/Within estimates.

        Returns
        -------
        dict
            Keys: n_groups, n_instruments, instrument_ratio, signal, recommendation
        """
        ratio = self.n_instruments / self.n_groups

        if ratio <= 0.75:
            signal = self.GREEN
            recommendation = (
                f"Instrument ratio ({ratio:.2f}) is well below 1.0. "
                "Instrument proliferation is unlikely."
            )
        elif ratio <= 1.0:
            signal = self.YELLOW
            recommendation = (
                f"Instrument ratio ({ratio:.2f}) is approaching 1.0. "
                "Consider using collapse=True or reducing gmm_max_lag "
                "to limit instrument count."
            )
        else:
            signal = self.RED
            recommendation = (
                f"Instrument ratio ({ratio:.2f}) exceeds 1.0 (Roodman rule violated). "
                "Hansen J test is unreliable. Use collapse=True and/or reduce "
                "gmm_max_lag. Consider Difference GMM with fewer instruments."
            )

        return {
            "n_groups": self.n_groups,
            "n_instruments": self.n_instruments,
            "instrument_ratio": ratio,
            "signal": signal,
            "recommendation": recommendation,
        }

    def instrument_sensitivity(self, max_lag_range: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Re-estimate with varying gmm_max_lag to assess sensitivity.

        If the AR coefficient changes substantially with different instrument
        sets, this signals that results are driven by instrument count rather
        than genuine identification.

        Parameters
        ----------
        max_lag_range : list of int, optional
            Values of gmm_max_lag to try. Default: [2, 3, 4, 5, 6].

        Returns
        -------
        pd.DataFrame
            Columns: gmm_max_lag, n_instruments, ar_coef, hansen_j_pval,
            ar2_pval, signal
        """
        if max_lag_range is None:
            max_lag_range = [2, 3, 4, 5, 6]

        rows = []
        for max_lag in max_lag_range:
            try:
                new_model = self._clone_model(gmm_max_lag=max_lag)
                new_results = new_model.fit()

                ar_coef = None
                if self._ar_param_name and self._ar_param_name in new_results.params.index:
                    ar_coef = float(new_results.params[self._ar_param_name])

                rows.append(
                    {
                        "gmm_max_lag": max_lag,
                        "n_instruments": new_results.n_instruments,
                        "ar_coef": ar_coef,
                        "hansen_j_pval": new_results.hansen_j.pvalue,
                        "ar2_pval": new_results.ar2_test.pvalue,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "gmm_max_lag": max_lag,
                        "n_instruments": None,
                        "ar_coef": None,
                        "hansen_j_pval": None,
                        "ar2_pval": None,
                    }
                )

        df = pd.DataFrame(rows)

        # Compute signal based on coefficient stability
        valid_coefs = df["ar_coef"].dropna()
        if len(valid_coefs) >= 2:
            coef_range = valid_coefs.max() - valid_coefs.min()
            mean_coef = valid_coefs.mean()
            if mean_coef != 0:
                relative_range = abs(coef_range / mean_coef)
            else:
                relative_range = float("inf")

            if relative_range <= 0.10:
                df.attrs["signal"] = self.GREEN
            elif relative_range <= 0.20:
                df.attrs["signal"] = self.YELLOW
            else:
                df.attrs["signal"] = self.RED
            df.attrs["relative_range"] = relative_range
        else:
            df.attrs["signal"] = self.YELLOW
            df.attrs["relative_range"] = None

        return df

    def coefficient_bounds_test(self) -> dict:
        """
        Nickell (1981) bounds test: GMM AR coefficient should lie between
        OLS (upward biased) and FE/Within (downward biased) estimates.

        If the GMM estimate falls outside these bounds, it suggests overfitting
        or model misspecification.

        Returns
        -------
        dict
            Keys: ols_coef, fe_coef, gmm_coef, within_bounds, signal, details
        """
        if self._ar_param_name is None:
            return {
                "ols_coef": None,
                "fe_coef": None,
                "gmm_coef": None,
                "within_bounds": None,
                "signal": self.YELLOW,
                "details": "No AR coefficient found in model.",
            }

        gmm_coef = float(self.results.params[self._ar_param_name])

        # Build y and X from the original data
        data = self.model.data.copy()
        dep_var = self.model.dep_var
        id_var = self.model.id_var

        # Create lag of dependent variable
        lag_name = f"{dep_var}_L1"
        data[lag_name] = data.groupby(id_var)[dep_var].shift(1)
        data = data.dropna(subset=[lag_name])

        # Regressors: lag + exogenous vars
        regressor_names = [lag_name] + list(self.model.exog_vars)
        y = data[dep_var].values
        X = data[regressor_names].values

        # Add constant for OLS
        X_ols = np.column_stack([np.ones(len(y)), X])

        # --- Pooled OLS ---
        try:
            beta_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]
            ols_coef = float(beta_ols[1])  # coefficient on lagged dep var
        except Exception:
            ols_coef = None

        # --- Fixed Effects (Within estimator) ---
        try:
            groups = data[id_var].values
            unique_groups = np.unique(groups)

            # Demean within groups
            y_dm = y.copy().astype(float)
            X_dm = X.copy().astype(float)

            for g in unique_groups:
                mask = groups == g
                y_dm[mask] -= y_dm[mask].mean()
                X_dm[mask] -= X_dm[mask].mean(axis=0)

            beta_fe = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
            fe_coef = float(beta_fe[0])  # coefficient on lagged dep var
        except Exception:
            fe_coef = None

        # --- Bounds check ---
        if ols_coef is not None and fe_coef is not None:
            lower = min(fe_coef, ols_coef)
            upper = max(fe_coef, ols_coef)

            within_bounds = lower <= gmm_coef <= upper

            # Margin for near-boundary
            margin = 0.05 * (upper - lower) if upper != lower else 0.05

            if within_bounds:
                # Check if near boundary
                if gmm_coef - lower < margin or upper - gmm_coef < margin:
                    signal = self.YELLOW
                    details = (
                        f"GMM ({gmm_coef:.4f}) is within bounds "
                        f"[FE={fe_coef:.4f}, OLS={ols_coef:.4f}] "
                        "but near a boundary."
                    )
                else:
                    signal = self.GREEN
                    details = (
                        f"GMM ({gmm_coef:.4f}) is within bounds "
                        f"[FE={fe_coef:.4f}, OLS={ols_coef:.4f}]. "
                        "Consistent with valid identification."
                    )
            else:
                signal = self.RED
                details = (
                    f"GMM ({gmm_coef:.4f}) is OUTSIDE bounds "
                    f"[FE={fe_coef:.4f}, OLS={ols_coef:.4f}]. "
                    "This suggests overfitting or model misspecification."
                )
        else:
            within_bounds = None
            signal = self.YELLOW
            details = "Could not compute OLS and/or FE estimates."

        return {
            "ols_coef": ols_coef,
            "fe_coef": fe_coef,
            "gmm_coef": gmm_coef,
            "within_bounds": within_bounds,
            "signal": signal,
            "details": details,
        }

    def jackknife_groups(self, max_groups: int = 30) -> dict:
        """
        Leave-one-group-out re-estimation to assess model fragility.

        For each group, re-estimate the model excluding that group and compare
        the AR coefficient. Large variation indicates that results are driven
        by individual groups rather than robust identification.

        Parameters
        ----------
        max_groups : int
            Maximum number of groups to run jackknife on (default 30).
            If N > max_groups, returns a warning instead.

        Returns
        -------
        dict
            Keys: full_sample_coef, jackknife_coefs, mean, std,
            max_deviation, signal, details
        """
        if self._ar_param_name is None:
            return {
                "full_sample_coef": None,
                "jackknife_coefs": {},
                "mean": None,
                "std": None,
                "max_deviation": None,
                "signal": self.YELLOW,
                "details": "No AR coefficient found in model.",
            }

        if self.n_groups > max_groups:
            return {
                "full_sample_coef": float(self.results.params[self._ar_param_name]),
                "jackknife_coefs": {},
                "mean": None,
                "std": None,
                "max_deviation": None,
                "signal": self.YELLOW,
                "details": (
                    f"Skipped: N={self.n_groups} exceeds max_groups={max_groups}. "
                    "Jackknife is most useful for small N."
                ),
            }

        full_coef = float(self.results.params[self._ar_param_name])
        id_var = self.model.id_var
        unique_groups = self.model.data[id_var].unique()

        jackknife_coefs = {}
        for group in unique_groups:
            try:
                subset_data = self.model.data[self.model.data[id_var] != group]
                new_model = self._clone_model(data=subset_data)
                new_results = new_model.fit()

                if self._ar_param_name in new_results.params.index:
                    jackknife_coefs[group] = float(new_results.params[self._ar_param_name])
            except Exception:
                jackknife_coefs[group] = None

        valid_coefs = [v for v in jackknife_coefs.values() if v is not None]

        if len(valid_coefs) < 2:
            return {
                "full_sample_coef": full_coef,
                "jackknife_coefs": jackknife_coefs,
                "mean": None,
                "std": None,
                "max_deviation": None,
                "signal": self.RED,
                "details": "Too few successful jackknife estimations.",
            }

        jk_mean = np.mean(valid_coefs)
        jk_std = np.std(valid_coefs, ddof=1)
        max_dev = max(abs(c - full_coef) for c in valid_coefs)

        # Relative max deviation
        if abs(full_coef) > 1e-10:
            rel_dev = max_dev / abs(full_coef)
        else:
            rel_dev = float("inf")

        if rel_dev <= 0.15:
            signal = self.GREEN
            details = (
                f"Jackknife stable: max deviation {rel_dev:.1%} of full-sample "
                f"coefficient ({full_coef:.4f})."
            )
        elif rel_dev <= 0.30:
            signal = self.YELLOW
            details = (
                f"Moderate jackknife variation: max deviation {rel_dev:.1%} of "
                f"full-sample coefficient ({full_coef:.4f}). Some group sensitivity."
            )
        else:
            signal = self.RED
            details = (
                f"High jackknife variation: max deviation {rel_dev:.1%} of "
                f"full-sample coefficient ({full_coef:.4f}). Results are fragile."
            )

        return {
            "full_sample_coef": full_coef,
            "jackknife_coefs": jackknife_coefs,
            "mean": jk_mean,
            "std": jk_std,
            "max_deviation": max_dev,
            "signal": signal,
            "details": details,
        }

    def step_comparison(self) -> dict:
        """
        Compare one-step and two-step GMM estimates.

        Large differences between one-step and two-step estimates indicate
        sensitivity to the weighting matrix, which correlates with instrument
        proliferation problems.

        Returns
        -------
        dict
            Keys: one_step_coef, two_step_coef, abs_diff, rel_diff,
            se_ratio, signal, details
        """
        if self._ar_param_name is None:
            return {
                "one_step_coef": None,
                "two_step_coef": None,
                "abs_diff": None,
                "rel_diff": None,
                "se_ratio": None,
                "signal": self.YELLOW,
                "details": "No AR coefficient found in model.",
            }

        # Determine which step the original model used
        original_is_two_step = self.model.two_step

        # Get original coefficient and SE
        original_coef = float(self.results.params[self._ar_param_name])
        original_se = float(self.results.std_errors[self._ar_param_name])

        # Re-estimate with the other step
        alt_type = "one_step" if original_is_two_step else "two_step"
        try:
            alt_model = self._clone_model(gmm_type=alt_type)
            alt_results = alt_model.fit()

            if self._ar_param_name not in alt_results.params.index:
                raise ValueError("AR param not found in alternative estimation")

            alt_coef = float(alt_results.params[self._ar_param_name])
            alt_se = float(alt_results.std_errors[self._ar_param_name])
        except Exception as e:
            return {
                "one_step_coef": None if original_is_two_step else original_coef,
                "two_step_coef": original_coef if original_is_two_step else None,
                "abs_diff": None,
                "rel_diff": None,
                "se_ratio": None,
                "signal": self.YELLOW,
                "details": f"Could not estimate alternative step: {e}",
            }

        # Assign to one-step and two-step
        if original_is_two_step:
            two_step_coef = original_coef
            two_step_se = original_se
            one_step_coef = alt_coef
            one_step_se = alt_se
        else:
            one_step_coef = original_coef
            one_step_se = original_se
            two_step_coef = alt_coef
            two_step_se = alt_se

        abs_diff = abs(two_step_coef - one_step_coef)
        mean_coef = (abs(one_step_coef) + abs(two_step_coef)) / 2
        rel_diff = abs_diff / mean_coef if mean_coef > 1e-10 else float("inf")

        # SE ratio: two-step SE should not be drastically smaller
        se_ratio = two_step_se / one_step_se if one_step_se > 1e-10 else float("inf")

        # Signal
        if rel_diff <= 0.10 and 0.5 <= se_ratio <= 1.5:
            signal = self.GREEN
            details = (
                f"One-step ({one_step_coef:.4f}) and two-step ({two_step_coef:.4f}) "
                f"are consistent (diff={rel_diff:.1%}). SE ratio={se_ratio:.2f}."
            )
        elif rel_diff <= 0.20:
            signal = self.YELLOW
            details = (
                f"Moderate difference between one-step ({one_step_coef:.4f}) "
                f"and two-step ({two_step_coef:.4f}): {rel_diff:.1%}. "
                f"SE ratio={se_ratio:.2f}."
            )
        else:
            signal = self.RED
            details = (
                f"Large difference between one-step ({one_step_coef:.4f}) "
                f"and two-step ({two_step_coef:.4f}): {rel_diff:.1%}. "
                f"SE ratio={se_ratio:.2f}. "
                "Sensitive to weighting matrix - possible instrument proliferation."
            )

        return {
            "one_step_coef": one_step_coef,
            "two_step_coef": two_step_coef,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "se_ratio": se_ratio,
            "signal": signal,
            "details": details,
        }

    def summary(self, run_jackknife: bool = True) -> str:
        """
        Generate comprehensive overfitting diagnostic report.

        Runs all diagnostic checks and produces a traffic-light summary.

        Parameters
        ----------
        run_jackknife : bool
            Whether to run jackknife_groups (can be slow). Default True.

        Returns
        -------
        str
            Formatted diagnostic report.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("GMM Overfit Diagnostic Report")
        lines.append("=" * 70)
        lines.append("")

        signals = []

        # 1. Feasibility
        feas = self.assess_feasibility()
        signals.append(feas["signal"])
        lines.append(f"1. Instrument Feasibility (Roodman Rule)  [{feas['signal']}]")
        lines.append("-" * 70)
        lines.append(f"   Groups (N):          {feas['n_groups']}")
        lines.append(f"   Instruments:         {feas['n_instruments']}")
        lines.append(f"   Ratio (instr/N):     {feas['instrument_ratio']:.3f}")
        lines.append(f"   {feas['recommendation']}")
        lines.append("")

        # 2. Instrument Sensitivity
        try:
            sens_df = self.instrument_sensitivity()
            sens_signal = sens_df.attrs.get("signal", self.YELLOW)
            rel_range = sens_df.attrs.get("relative_range")
            signals.append(sens_signal)
            lines.append(f"2. Instrument Sensitivity (varying max_lag)  [{sens_signal}]")
            lines.append("-" * 70)
            for _, row in sens_df.iterrows():
                ml = int(row["gmm_max_lag"]) if row["gmm_max_lag"] is not None else "?"
                ni = int(row["n_instruments"]) if row["n_instruments"] is not None else "?"
                ac = f"{row['ar_coef']:.4f}" if row["ar_coef"] is not None else "N/A"
                hp = f"{row['hansen_j_pval']:.4f}" if row["hansen_j_pval"] is not None else "N/A"
                lines.append(f"   max_lag={ml}  instruments={ni}  " f"AR_coef={ac}  Hansen_p={hp}")
            if rel_range is not None:
                lines.append(f"   Coefficient range: {rel_range:.1%} of mean")
            lines.append("")
        except Exception as e:
            signals.append(self.YELLOW)
            lines.append(f"2. Instrument Sensitivity  [YELLOW]")
            lines.append(f"   Could not run: {e}")
            lines.append("")

        # 3. Coefficient Bounds
        bounds = self.coefficient_bounds_test()
        signals.append(bounds["signal"])
        lines.append(f"3. Coefficient Bounds (Nickell 1981)  [{bounds['signal']}]")
        lines.append("-" * 70)
        if bounds["ols_coef"] is not None:
            lines.append(f"   OLS (upper bound):   {bounds['ols_coef']:.4f}")
        if bounds["fe_coef"] is not None:
            lines.append(f"   FE  (lower bound):   {bounds['fe_coef']:.4f}")
        if bounds["gmm_coef"] is not None:
            lines.append(f"   GMM estimate:        {bounds['gmm_coef']:.4f}")
        lines.append(f"   {bounds['details']}")
        lines.append("")

        # 4. Jackknife
        if run_jackknife:
            jk = self.jackknife_groups()
            signals.append(jk["signal"])
            lines.append(f"4. Jackknife Group Sensitivity  [{jk['signal']}]")
            lines.append("-" * 70)
            if jk["mean"] is not None:
                lines.append(f"   Full-sample coef:    {jk['full_sample_coef']:.4f}")
                lines.append(f"   Jackknife mean:      {jk['mean']:.4f}")
                lines.append(f"   Jackknife std:       {jk['std']:.4f}")
                lines.append(f"   Max deviation:       {jk['max_deviation']:.4f}")
            lines.append(f"   {jk['details']}")
            lines.append("")
        else:
            lines.append("4. Jackknife Group Sensitivity  [SKIPPED]")
            lines.append("-" * 70)
            lines.append("   Skipped by user request.")
            lines.append("")

        # 5. Step Comparison
        step = self.step_comparison()
        signals.append(step["signal"])
        lines.append(f"5. One-Step vs Two-Step Comparison  [{step['signal']}]")
        lines.append("-" * 70)
        if step["one_step_coef"] is not None:
            lines.append(f"   One-step coef:       {step['one_step_coef']:.4f}")
        if step["two_step_coef"] is not None:
            lines.append(f"   Two-step coef:       {step['two_step_coef']:.4f}")
        if step["rel_diff"] is not None:
            lines.append(f"   Relative difference: {step['rel_diff']:.1%}")
        if step["se_ratio"] is not None:
            lines.append(f"   SE ratio (2s/1s):    {step['se_ratio']:.3f}")
        lines.append(f"   {step['details']}")
        lines.append("")

        # Overall verdict
        lines.append("=" * 70)
        if self.RED in signals:
            overall = self.RED
        elif self.YELLOW in signals:
            overall = self.YELLOW
        else:
            overall = self.GREEN

        lines.append(f"OVERALL VERDICT: [{overall}]")
        lines.append("-" * 70)

        if overall == self.GREEN:
            lines.append("All diagnostics pass. No evidence of instrument proliferation.")
        elif overall == self.YELLOW:
            lines.append("Some diagnostics raise caution. Review individual checks above.")
        else:
            lines.append(
                "One or more diagnostics indicate overfitting or instrument "
                "proliferation. Results may be unreliable. Consider:\n"
                "  - Using collapse=True\n"
                "  - Reducing gmm_max_lag\n"
                "  - Switching to Difference GMM\n"
                "  - Increasing sample size (more groups)"
            )

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GMMOverfitDiagnostic("
            f"n_groups={self.n_groups}, "
            f"n_instruments={self.n_instruments}, "
            f"ratio={self.n_instruments / self.n_groups:.2f})"
        )
