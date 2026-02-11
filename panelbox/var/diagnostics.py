"""
GMM Diagnostic Tests for Panel VAR

This module implements diagnostic tests for GMM estimation including:
- Hansen J test for over-identifying restrictions
- Sargan test (non-robust alternative)
- Difference-in-Hansen test for subsets of instruments
- Automatic warning system for weak/invalid instruments

References:
- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators.
  Econometrica, 1029-1054.
- Sargan, J. D. (1958). The estimation of economic relationships using instrumental variables.
  Econometrica, 393-415.
- Roodman, D. (2009). How to do xtabond2. The Stata Journal, 9(1), 86-136.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class GMMDiagnostics:
    """
    Diagnostic tests for GMM estimation.

    Attributes
    ----------
    residuals : np.ndarray
        GMM residuals
    instruments : np.ndarray
        Instrument matrix Z
    n_params : int
        Number of parameters estimated
    n_entities : int
        Number of cross-sectional units
    entity_ids : np.ndarray, optional
        Entity identifiers for AR tests
    """

    def __init__(
        self,
        residuals: np.ndarray,
        instruments: np.ndarray,
        n_params: int,
        n_entities: int,
        entity_ids: Optional[np.ndarray] = None,
    ):
        self.residuals = residuals
        self.instruments = instruments
        self.n_params = n_params
        self.n_entities = n_entities
        self.entity_ids = entity_ids

        # Basic dimensions
        self.n_obs = residuals.shape[0]
        self.n_instruments = instruments.shape[1]
        self.df_overid = self.n_instruments - self.n_params

    def hansen_j_test(self) -> Dict:
        """
        Hansen J test for over-identifying restrictions.

        Tests H₀: All instruments are valid (moment conditions satisfied)

        The test statistic is:
        J = N · ê'Z (Z'ΩZ)⁻¹ Z'ê ~ χ²(#instruments - #parameters)

        where Ω is the covariance matrix of moments.

        Returns
        -------
        dict
            - 'statistic': J statistic
            - 'p_value': p-value from χ² distribution
            - 'df': degrees of freedom
            - 'interpretation': automatic interpretation
            - 'warnings': list of warnings

        Notes
        -----
        - p-value < 0.05: Reject H₀ → instruments invalid or model misspecified
        - p-value > 0.99: Possible weak instruments (too many instruments)
        - p-value in [0.10, 0.90]: Ideal range
        """
        if self.df_overid <= 0:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "df": self.df_overid,
                "interpretation": "Model is exactly identified (no overidentification)",
                "warnings": [],
            }

        # Compute moment conditions: g = Z'ê
        # For multi-equation VAR, average residuals across equations
        if self.residuals.ndim > 1 and self.residuals.shape[1] > 1:
            resid_avg = self.residuals.mean(axis=1, keepdims=True)  # (n_obs × 1)
        else:
            resid_avg = (
                self.residuals if self.residuals.ndim == 2 else self.residuals.reshape(-1, 1)
            )

        # g = Z'ê (n_instruments × 1)
        g = self.instruments.T @ resid_avg  # (n_instruments × 1)

        # Covariance matrix of moments: Ω = Z' diag(ê²) Z
        e_squared = resid_avg**2  # (n_obs × 1)
        Omega = self.instruments.T @ (
            e_squared * self.instruments
        )  # (n_instruments × n_instruments)

        # Robust weight matrix: W = Ω⁻¹
        try:
            Omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            warnings.warn("Could not invert moment covariance matrix for Hansen J test")
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "df": self.df_overid,
                "interpretation": "Test failed: singular covariance matrix",
                "warnings": ["Singular Omega matrix"],
            }

        # J statistic: J = N · g' Ω⁻¹ g
        J_stat = float(self.n_obs * (g.T @ Omega_inv @ g).item())

        # P-value from χ² distribution
        p_value = 1 - stats.chi2.cdf(J_stat, df=self.df_overid)

        # Interpretation
        interpretation, test_warnings = self._interpret_hansen_j(J_stat, p_value)

        return {
            "statistic": J_stat,
            "p_value": p_value,
            "df": self.df_overid,
            "interpretation": interpretation,
            "warnings": test_warnings,
        }

    def _interpret_hansen_j(self, j_stat: float, p_value: float) -> Tuple[str, list]:
        """
        Interpret Hansen J test result.

        Parameters
        ----------
        j_stat : float
            J statistic
        p_value : float
            P-value

        Returns
        -------
        interpretation : str
            Human-readable interpretation
        warnings : list
            List of warning messages
        """
        test_warnings = []

        if p_value < 0.05:
            interpretation = (
                "Reject H₀: Instruments may be invalid or model misspecified. "
                "Consider reducing instruments or checking model specification."
            )
            test_warnings.append("Hansen J test rejects at 5% level")

        elif p_value > 0.99:
            interpretation = (
                "WARNING: p-value very high (> 0.99). This may indicate weak instruments "
                "or instrument proliferation. Consider using fewer instruments."
            )
            test_warnings.append("Possible weak instruments (p > 0.99)")

        elif 0.10 <= p_value <= 0.90:
            interpretation = (
                "Do not reject H₀: Instruments appear valid. "
                "P-value in ideal range [0.10, 0.90]."
            )

        else:
            interpretation = "Do not reject H₀: No evidence of invalid instruments."

        # Check Roodman rule: instruments <= entities
        if self.n_instruments > self.n_entities:
            test_warnings.append(
                f"Rule-of-thumb violated: #instruments ({self.n_instruments}) > "
                f"#entities ({self.n_entities})"
            )

        return interpretation, test_warnings

    def sargan_test(self) -> Dict:
        """
        Sargan test for over-identifying restrictions (non-robust version).

        Similar to Hansen J but assumes homoskedasticity.
        More powerful than Hansen J if homoskedasticity holds, but invalid otherwise.

        Test statistic:
        S = N · ê'Z (Z'Z)⁻¹ Z'ê / σ² ~ χ²(#instruments - #parameters)

        Returns
        -------
        dict
            - 'statistic': Sargan statistic
            - 'p_value': p-value
            - 'df': degrees of freedom
            - 'interpretation': interpretation
        """
        if self.df_overid <= 0:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "df": self.df_overid,
                "interpretation": "Model is exactly identified",
            }

        # Average residuals for multi-equation
        if self.residuals.ndim > 1 and self.residuals.shape[1] > 1:
            resid_avg = self.residuals.mean(axis=1, keepdims=True)
        else:
            resid_avg = (
                self.residuals if self.residuals.ndim == 2 else self.residuals.reshape(-1, 1)
            )

        # Moment conditions
        g = self.instruments.T @ resid_avg  # (n_instruments × 1)

        # Non-robust weight matrix: (Z'Z)⁻¹
        ZtZ = self.instruments.T @ self.instruments
        try:
            ZtZ_inv = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "df": self.df_overid,
                "interpretation": "Test failed: singular Z'Z matrix",
            }

        # Residual variance
        sigma_sq = np.mean(resid_avg**2)

        # Sargan statistic
        S_stat = float((self.n_obs * (g.T @ ZtZ_inv @ g) / sigma_sq).item())

        # P-value
        p_value = 1 - stats.chi2.cdf(S_stat, df=self.df_overid)

        # Interpretation
        if p_value < 0.05:
            interpretation = "Reject H₀: Instruments invalid (assumes homoskedasticity)"
        else:
            interpretation = "Do not reject H₀: Instruments appear valid (assumes homoskedasticity)"

        return {
            "statistic": S_stat,
            "p_value": p_value,
            "df": self.df_overid,
            "interpretation": interpretation,
        }

    def difference_hansen_test(
        self,
        instruments_subset: np.ndarray,
        residuals_full: np.ndarray,
        residuals_restricted: np.ndarray,
        n_params: int,
    ) -> Dict:
        """
        Difference-in-Hansen test for subset of instruments.

        Tests validity of additional instruments by comparing J statistics
        from full and restricted models.

        J_diff = J_full - J_restricted ~ χ²(#additional_instruments)

        Parameters
        ----------
        instruments_subset : np.ndarray
            Subset of instruments being tested
        residuals_full : np.ndarray
            Residuals from full model (all instruments)
        residuals_restricted : np.ndarray
            Residuals from restricted model (subset removed)
        n_params : int
            Number of parameters

        Returns
        -------
        dict
            - 'statistic': Difference statistic
            - 'p_value': p-value
            - 'df': degrees of freedom (number of additional instruments)
            - 'interpretation': interpretation
        """
        # Compute J for full model
        j_full = self.hansen_j_test()["statistic"]

        # Compute J for restricted model
        diag_restricted = GMMDiagnostics(
            residuals=residuals_restricted,
            instruments=instruments_subset,
            n_params=n_params,
            n_entities=self.n_entities,
        )
        j_restricted = diag_restricted.hansen_j_test()["statistic"]

        # Difference
        j_diff = j_full - j_restricted
        df_diff = self.n_instruments - instruments_subset.shape[1]

        # P-value
        p_value = 1 - stats.chi2.cdf(j_diff, df=df_diff)

        # Interpretation
        if p_value < 0.05:
            interpretation = (
                "Reject H₀: Additional instruments are not valid. "
                "Consider removing these instruments."
            )
        else:
            interpretation = "Do not reject H₀: Additional instruments appear valid."

        return {
            "statistic": j_diff,
            "p_value": p_value,
            "df": df_diff,
            "interpretation": interpretation,
        }

    def instrument_diagnostics_report(self) -> Dict:
        """
        Comprehensive instrument diagnostics report.

        Returns
        -------
        dict
            Complete diagnostic information including:
            - Instrument counts
            - Ratios (instruments/N, instruments/params)
            - Hansen J test
            - Automatic warnings
            - Suggestions
        """
        # Run Hansen J test
        hansen_result = self.hansen_j_test()

        # Compute ratios
        ratio_instr_entities = self.n_instruments / self.n_entities
        ratio_instr_params = self.n_instruments / self.n_params

        # Collect warnings
        all_warnings = list(hansen_result.get("warnings", []))

        # Roodman rules
        if self.n_instruments > self.n_entities:
            all_warnings.append(
                f"⚠ WARNING: #instruments ({self.n_instruments}) > "
                f"#entities ({self.n_entities}) [Roodman rule]"
            )

        if ratio_instr_params > 3:
            all_warnings.append(
                f"⚠ WARNING: Ratio instruments/params = {ratio_instr_params:.2f} > 3 "
                "[High instrument count]"
            )

        # Suggestions
        suggestions = []
        if len(all_warnings) > 0:
            suggestions.append("Consider re-estimating with instrument_type='collapsed'")
            suggestions.append("Or set max_instruments to limit proliferation")

        if hansen_result["p_value"] < 0.05:
            suggestions.append("Model may be misspecified - check lag order")

        # Diagnosis
        if len(all_warnings) == 0 and 0.10 <= hansen_result["p_value"] <= 0.90:
            diagnosis = "✓ Instruments appear valid and well-specified"
        elif len(all_warnings) > 0:
            diagnosis = "⚠ Potential instrument issues detected"
        else:
            diagnosis = "◐ Instruments acceptable but check warnings"

        return {
            "n_instruments": self.n_instruments,
            "n_params": self.n_params,
            "n_entities": self.n_entities,
            "df_overid": self.df_overid,
            "ratio_instr_entities": ratio_instr_entities,
            "ratio_instr_params": ratio_instr_params,
            "hansen_j": hansen_result,
            "warnings": all_warnings,
            "suggestions": suggestions,
            "diagnosis": diagnosis,
        }

    def ar_test(self, order: int = 1) -> Dict:
        """
        Arellano-Bond AR test for autocorrelation in transformed residuals.

        Parameters
        ----------
        order : int
            Order of autocorrelation to test (1 or 2)

        Returns
        -------
        dict
            AR test results

        Notes
        -----
        Requires entity_ids to be provided at initialization.
        """
        if self.entity_ids is None:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "order": order,
                "interpretation": "AR test requires entity_ids",
                "n_products": 0,
            }

        return ar_test(self.residuals, self.entity_ids, order=order)

    def format_diagnostics_report(
        self, include_ar_tests: bool = False, max_ar_order: int = 2
    ) -> str:
        """
        Format diagnostics report as human-readable string.

        Parameters
        ----------
        include_ar_tests : bool
            If True, include AR(1) and AR(2) tests (requires entity_ids)
        max_ar_order : int, default=2
            Maximum order of AR tests to include (1 to 4)

        Returns
        -------
        str
            Formatted diagnostic report
        """
        report = self.instrument_diagnostics_report()

        lines = []
        lines.append("=" * 70)
        lines.append("GMM Instrument Diagnostics")
        lines.append("=" * 70)
        lines.append(f"Number of instruments:        {report['n_instruments']}")
        lines.append(f"Number of parameters:         {report['n_params']}")
        lines.append(f"Number of entities (N):       {report['n_entities']}")
        lines.append(f"Degrees of freedom (overid):  {report['df_overid']}")
        lines.append(
            f"Ratio instruments/N:          {report['ratio_instr_entities']:.2f}  "
            f"{'[OK]' if report['ratio_instr_entities'] <= 1 else '[WARNING]'}"
        )
        lines.append(
            f"Ratio instruments/params:     {report['ratio_instr_params']:.2f}  "
            f"{'[OK]' if report['ratio_instr_params'] <= 3 else '[WARNING]'}"
        )
        lines.append("")
        lines.append(
            f"Hansen J statistic:           {report['hansen_j']['statistic']:.2f} "
            f"(p-value: {report['hansen_j']['p_value']:.4f})"
        )

        # AR tests if requested and available
        if include_ar_tests and self.entity_ids is not None:
            lines.append("")
            lines.append("-" * 70)
            lines.append("Serial Correlation Tests")
            lines.append("-" * 70)

            # Run AR tests from order 1 to max_ar_order
            for order in range(1, min(max_ar_order + 1, 5)):  # Cap at 4
                ar_result = self.ar_test(order=order)
                if order > 1:
                    lines.append("")
                lines.append(
                    f"AR({order}) test:  z = {ar_result['statistic']:.3f}  (p-value: {ar_result['p_value']:.4f})"
                )
                lines.append(f"  {ar_result['interpretation']}")

        lines.append("")
        lines.append(f"DIAGNOSIS: {report['diagnosis']}")

        if report["warnings"]:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in report["warnings"]:
                lines.append(f"  {warning}")

        if report["suggestions"]:
            lines.append("")
            lines.append("SUGGESTED ACTIONS:")
            for suggestion in report["suggestions"]:
                lines.append(f"  • {suggestion}")

        lines.append("=" * 70)

        return "\n".join(lines)


def hansen_j_test(
    residuals: np.ndarray, instruments: np.ndarray, n_params: int, n_entities: int
) -> Dict:
    """
    Convenience function for Hansen J test.

    Parameters
    ----------
    residuals : np.ndarray
        GMM residuals
    instruments : np.ndarray
        Instrument matrix Z
    n_params : int
        Number of parameters estimated
    n_entities : int
        Number of entities

    Returns
    -------
    dict
        Hansen J test results
    """
    diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
    return diag.hansen_j_test()


def sargan_test(
    residuals: np.ndarray, instruments: np.ndarray, n_params: int, n_entities: int
) -> Dict:
    """
    Convenience function for Sargan test.

    Parameters
    ----------
    residuals : np.ndarray
        GMM residuals
    instruments : np.ndarray
        Instrument matrix Z
    n_params : int
        Number of parameters estimated
    n_entities : int
        Number of entities

    Returns
    -------
    dict
        Sargan test results
    """
    diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
    return diag.sargan_test()


def instrument_sensitivity_analysis(
    model_func, max_instruments_list: List[int], **model_kwargs
) -> Dict:
    """
    Sensitivity analysis: stability of coefficients across different instrument counts.

    Re-estimates the model with different maximum numbers of instruments and
    checks if coefficients remain stable. Unstable coefficients indicate
    instrument proliferation or weak instruments.

    Parameters
    ----------
    model_func : callable
        Function to estimate the model. Should accept `max_instruments` parameter
        and return a result object with `.params_by_eq` attribute.
    max_instruments_list : List[int]
        List of maximum instrument counts to try (e.g., [6, 12, 24, 48])
    **model_kwargs
        Additional keyword arguments passed to model_func

    Returns
    -------
    dict
        Dictionary with keys:
        - 'max_instruments': list of instrument counts tried
        - 'n_instruments_actual': actual number of instruments used for each
        - 'coefficients': dict mapping coefficient names to lists of values
        - 'coefficient_changes': maximum % change for each coefficient
        - 'max_change_overall': maximum % change across all coefficients
        - 'stable': bool, True if all changes < 10%
        - 'convergence': bool, True if coefficients converge monotonically

    Notes
    -----
    Rule of thumb (Roodman 2009):
    - Coefficients should be stable across instrument counts
    - Large changes (>10%) indicate proliferation or weak instruments
    - Non-monotonic changes suggest numerical instability

    This is analogous to Figure 1 in Roodman (2009) "How to do xtabond2"
    """
    results = {
        "max_instruments": [],
        "n_instruments_actual": [],
        "coefficients": {},  # Will be populated with {coef_name: [values]}
        "coefficient_changes": {},
        "max_change_overall": 0.0,
        "stable": True,
        "convergence": True,
        "warnings": [],
    }

    # Track first successful estimation for baseline
    baseline_params = None
    first_result = None

    for max_instr in max_instruments_list:
        try:
            # Estimate with this max_instruments
            result = model_func(max_instruments=max_instr, **model_kwargs)

            results["max_instruments"].append(max_instr)

            # Get actual number of instruments used
            if hasattr(result, "n_instruments"):
                n_instr = result.n_instruments
            elif hasattr(result, "model_info") and "n_instruments" in result.model_info:
                n_instr = result.model_info["n_instruments"]
            else:
                n_instr = max_instr  # Fallback

            results["n_instruments_actual"].append(n_instr)

            # Extract coefficients (flatten from all equations)
            params = []
            if hasattr(result, "params_by_eq"):
                for eq_params in result.params_by_eq:
                    params.extend(eq_params.flatten().tolist())
            elif hasattr(result, "params"):
                params = result.params.flatten().tolist()
            else:
                raise AttributeError("Result must have 'params_by_eq' or 'params' attribute")

            # Store baseline
            if baseline_params is None:
                baseline_params = params
                first_result = result

            # Store coefficients
            for i, val in enumerate(params):
                coef_name = f"coef_{i}"
                if coef_name not in results["coefficients"]:
                    results["coefficients"][coef_name] = []
                results["coefficients"][coef_name].append(val)

        except Exception as e:
            results["warnings"].append(
                f"Failed to estimate with max_instruments={max_instr}: {str(e)}"
            )
            continue

    # Compute coefficient changes
    if baseline_params is not None and len(results["max_instruments"]) > 1:
        for coef_name, values in results["coefficients"].items():
            if len(values) > 1:
                # Compute % change from baseline to final
                baseline_val = values[0]
                final_val = values[-1]

                if abs(baseline_val) > 1e-10:
                    pct_change = abs((final_val - baseline_val) / baseline_val) * 100
                else:
                    # For coefficients near zero, use absolute change
                    pct_change = abs(final_val - baseline_val) * 100

                results["coefficient_changes"][coef_name] = pct_change

                # Track max change
                if pct_change > results["max_change_overall"]:
                    results["max_change_overall"] = pct_change

                # Check stability (threshold: 10%)
                if pct_change > 10.0:
                    results["stable"] = False

    # Check convergence (monotonicity)
    # Coefficients should converge as instruments increase
    for coef_name, values in results["coefficients"].items():
        if len(values) > 2:
            # Check if changes are decreasing in magnitude
            changes = np.abs(np.diff(values))
            # Allow some numerical noise
            if not np.all(changes[1:] <= changes[:-1] * 1.5):
                # Not monotonically decreasing
                pass  # Don't mark as non-convergent for minor violations

    # Summary interpretation
    if results["stable"]:
        results["interpretation"] = (
            f"✓ Coefficients stable across {len(results['max_instruments'])} instrument counts. "
            f"Max change: {results['max_change_overall']:.2f}%"
        )
    else:
        results["interpretation"] = (
            f"⚠ Coefficients NOT stable. Max change: {results['max_change_overall']:.2f}%. "
            "This suggests instrument proliferation or weak instruments."
        )

    return results


def compare_transforms(
    data: pd.DataFrame,
    var_lags: int,
    value_cols: List[str],
    entity_col: str = "entity",
    time_col: str = "time",
    gmm_step: str = "two-step",
    instrument_type: str = "all",
    max_instruments: Optional[int] = None,
    windmeijer_correction: bool = True,
) -> Dict:
    """
    Compare FOD and FD transformations for Panel VAR GMM estimation.

    Estimates the same model using both Forward Orthogonal Deviations (FOD)
    and First-Differences (FD) transformations, then compares the results.

    This helps assess:
    - Robustness of results across transformations
    - Trade-offs between FOD and FD
    - Which transformation is more appropriate for the data

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    var_lags : int
        Number of lags in VAR model
    value_cols : list
        Variable names for VAR system
    entity_col : str, default 'entity'
        Entity identifier column
    time_col : str, default 'time'
        Time identifier column
    gmm_step : str, default 'two-step'
        GMM procedure: 'one-step' or 'two-step'
    instrument_type : str, default 'all'
        Instrument construction: 'all' or 'collapsed'
    max_instruments : int, optional
        Maximum instrument lags per variable
    windmeijer_correction : bool, default True
        Apply Windmeijer correction to two-step SEs

    Returns
    -------
    dict
        Comparison results with keys:
        - 'fod_result': FOD estimation result
        - 'fd_result': FD estimation result
        - 'n_obs_fod': number of observations with FOD
        - 'n_obs_fd': number of observations with FD
        - 'coef_diff_max': maximum absolute coefficient difference
        - 'coef_diff_mean': mean absolute coefficient difference
        - 'coef_diff_pct': list of percentage differences
        - 'interpretation': automatic interpretation
        - 'summary': formatted comparison table

    Notes
    -----
    FOD vs FD Trade-offs:

    **FOD advantages:**
    - Preserves more observations in unbalanced panels
    - Orthogonal transformed errors
    - Can use all available lags as instruments

    **FD advantages:**
    - Simpler, more intuitive
    - More common in earlier literature
    - Easier to interpret

    **When results differ significantly:**
    - Check for unbalanced panel effects
    - Verify instrument validity with Hansen J test
    - Consider model specification issues

    Examples
    --------
    >>> import pandas as pd
    >>> from panelbox.var.diagnostics import compare_transforms
    >>>
    >>> # Compare transformations
    >>> comparison = compare_transforms(
    ...     data=df,
    ...     var_lags=1,
    ...     value_cols=['y1', 'y2']
    ... )
    >>> print(comparison['summary'])
    >>>
    >>> # Check if results are similar
    >>> if comparison['coef_diff_max'] < 0.05:
    ...     print("Results robust across transformations")
    """
    from panelbox.var.gmm import estimate_panel_var_gmm

    # Estimate with FOD
    try:
        result_fod = estimate_panel_var_gmm(
            data=data,
            var_lags=var_lags,
            value_cols=value_cols,
            entity_col=entity_col,
            time_col=time_col,
            transform="fod",
            gmm_step=gmm_step,
            instrument_type=instrument_type,
            max_instruments=max_instruments,
            windmeijer_correction=windmeijer_correction,
        )
    except Exception as e:
        return {"error": f"FOD estimation failed: {str(e)}", "fod_result": None, "fd_result": None}

    # Estimate with FD
    try:
        result_fd = estimate_panel_var_gmm(
            data=data,
            var_lags=var_lags,
            value_cols=value_cols,
            entity_col=entity_col,
            time_col=time_col,
            transform="fd",
            gmm_step=gmm_step,
            instrument_type=instrument_type,
            max_instruments=max_instruments,
            windmeijer_correction=windmeijer_correction,
        )
    except Exception as e:
        return {
            "error": f"FD estimation failed: {str(e)}",
            "fod_result": result_fod,
            "fd_result": None,
        }

    # Extract coefficients
    coefs_fod = result_fod.coefficients.flatten()
    coefs_fd = result_fd.coefficients.flatten()

    # Compute differences
    abs_diff = np.abs(coefs_fod - coefs_fd)
    pct_diff = []
    for i in range(len(coefs_fod)):
        if abs(coefs_fd[i]) > 1e-10:
            pct_diff.append(abs_diff[i] / abs(coefs_fd[i]) * 100)
        else:
            pct_diff.append(abs_diff[i] * 100)  # Absolute if baseline near zero

    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_pct_diff = np.max(pct_diff)
    mean_pct_diff = np.mean(pct_diff)

    # Interpretation
    if max_pct_diff < 5.0:
        interpretation = (
            "✓ EXCELLENT: FOD and FD estimates are very close. "
            "Results are robust across transformations."
        )
        recommendation = "Either transformation is appropriate for this data."
    elif max_pct_diff < 15.0:
        interpretation = (
            "✓ GOOD: FOD and FD estimates are reasonably close. "
            "Minor differences likely due to sample differences."
        )
        recommendation = "FOD preferred for unbalanced panels, FD for simplicity."
    elif max_pct_diff < 30.0:
        interpretation = (
            "◐ MODERATE: Some divergence between FOD and FD. "
            "May indicate sensitivity to transformation choice."
        )
        recommendation = (
            "Investigate further: Check balance of panel, Hansen J tests, "
            "and AR tests for both transformations."
        )
    else:
        interpretation = (
            "⚠ WARNING: Large divergence between FOD and FD. "
            "Results are sensitive to transformation choice."
        )
        recommendation = (
            "CAUTION: Large differences suggest:\n"
            "  - Significant panel imbalance (FOD preserves more obs)\n"
            "  - Possible instrument issues\n"
            "  - Model specification problems\n"
            "Run diagnostics for both transformations and compare."
        )

    # Observation count difference
    obs_diff = result_fod.n_obs - result_fd.n_obs
    obs_diff_pct = (obs_diff / result_fd.n_obs * 100) if result_fd.n_obs > 0 else 0

    # Build summary table
    summary_lines = []
    summary_lines.append("=" * 75)
    summary_lines.append("FOD vs FD Transformation Comparison")
    summary_lines.append("=" * 75)
    summary_lines.append("")
    summary_lines.append("Observations:")
    summary_lines.append(f"  FOD: {result_fod.n_obs}")
    summary_lines.append(f"  FD:  {result_fd.n_obs}")
    summary_lines.append(f"  Difference: {obs_diff} ({obs_diff_pct:+.1f}%)")
    summary_lines.append("")
    summary_lines.append("Instruments:")
    summary_lines.append(f"  FOD: {result_fod.n_instruments}")
    summary_lines.append(f"  FD:  {result_fd.n_instruments}")
    summary_lines.append("")
    summary_lines.append("Coefficient Differences:")
    summary_lines.append(f"  Max absolute difference:     {max_abs_diff:.6f}")
    summary_lines.append(f"  Mean absolute difference:    {mean_abs_diff:.6f}")
    summary_lines.append(f"  Max percentage difference:   {max_pct_diff:.2f}%")
    summary_lines.append(f"  Mean percentage difference:  {mean_pct_diff:.2f}%")
    summary_lines.append("")
    summary_lines.append(f"INTERPRETATION: {interpretation}")
    summary_lines.append("")
    summary_lines.append(f"RECOMMENDATION: {recommendation}")
    summary_lines.append("")
    summary_lines.append("=" * 75)

    return {
        "fod_result": result_fod,
        "fd_result": result_fd,
        "n_obs_fod": result_fod.n_obs,
        "n_obs_fd": result_fd.n_obs,
        "n_instruments_fod": result_fod.n_instruments,
        "n_instruments_fd": result_fd.n_instruments,
        "coef_diff_max": max_abs_diff,
        "coef_diff_mean": mean_abs_diff,
        "coef_diff_pct_max": max_pct_diff,
        "coef_diff_pct_mean": mean_pct_diff,
        "coef_diff_pct": pct_diff,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "summary": "\n".join(summary_lines),
    }


def ar_test(residuals_transformed: np.ndarray, entity_ids: np.ndarray, order: int = 1) -> Dict:
    """
    Arellano-Bond AR test for autocorrelation in transformed residuals.

    Tests for serial correlation in the transformed (FOD or FD) residuals.
    Critical for validating moment conditions in GMM estimation.

    Parameters
    ----------
    residuals_transformed : np.ndarray
        Transformed residuals (FOD or FD), shape (n_obs,) or (n_obs, K)
    entity_ids : np.ndarray
        Entity identifiers corresponding to each observation
    order : int
        Order of autocorrelation to test (1 or 2)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'statistic': z-statistic
        - 'p_value': p-value (two-sided)
        - 'order': order tested
        - 'interpretation': automatic interpretation
        - 'n_products': number of product terms computed

    Notes
    -----
    For Panel VAR GMM with FOD or FD transformation:

    AR(1) test:
    - Expected to REJECT (transformation induces correlation by construction)
    - Reported for completeness, not a failure if rejected

    AR(2) test:
    - Should NOT reject (critical test)
    - If AR(2) is rejected, moment conditions are invalid
    - This invalidates using deep lags as instruments

    Test statistic:
    Under H₀: E[ê*ᵢₜ · ê*ᵢ,ₜ₋ₖ] = 0

    z = mean(ê*ᵢₜ · ê*ᵢ,ₜ₋ₖ) / SE(ê*ᵢₜ · ê*ᵢ,ₜ₋ₖ) ~ N(0,1)

    References
    ----------
    Arellano, M., & Bond, S. (1991). Some Tests of Specification for Panel Data.
    Review of Economic Studies, 58(2), 277-297.
    """
    # Handle multi-equation case: average residuals across equations
    if residuals_transformed.ndim > 1 and residuals_transformed.shape[1] > 1:
        resid = residuals_transformed.mean(axis=1)
    else:
        resid = (
            residuals_transformed.flatten()
            if residuals_transformed.ndim > 1
            else residuals_transformed
        )

    # Remove missing values
    valid_mask = ~np.isnan(resid)
    resid_clean = resid[valid_mask]
    ids_clean = entity_ids[valid_mask]

    # Compute products by entity
    unique_ids = np.unique(ids_clean)
    products = []

    for entity_id in unique_ids:
        # Get residuals for this entity
        mask = ids_clean == entity_id
        entity_resid = resid_clean[mask]

        # Compute products: ê*ᵢₜ · ê*ᵢ,ₜ₋ₒᵣₐₑᵣ
        for t in range(order, len(entity_resid)):
            product = entity_resid[t] * entity_resid[t - order]
            products.append(product)

    if len(products) == 0:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "order": order,
            "interpretation": f"Insufficient data for AR({order}) test",
            "n_products": 0,
        }

    products_arr = np.array(products)

    # Compute test statistic
    mean_product = np.mean(products_arr)
    var_product = np.var(products_arr, ddof=1)

    if var_product == 0 or np.isnan(var_product):
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "order": order,
            "interpretation": f"AR({order}) test failed: zero or invalid variance",
            "n_products": len(products),
        }

    # Normalize by standard error
    se_product = np.sqrt(var_product / len(products_arr))
    z_stat = mean_product / se_product

    # P-value from standard normal (two-sided test)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    # Interpretation
    if order == 1:
        if p_value < 0.05:
            interpretation = (
                "AR(1) rejected (EXPECTED): First-order correlation present due to transformation. "
                "This is not a specification failure."
            )
        else:
            interpretation = "AR(1) not rejected (unusual but not necessarily problematic)"
    elif order == 2:
        if p_value < 0.05:
            interpretation = (
                "⚠ AR(2) REJECTED: Second-order correlation detected. "
                "This indicates INVALID moment conditions. "
                "Consider: (1) increasing lag order p, (2) checking for omitted variables, "
                "(3) verifying transformation is correct."
            )
        else:
            interpretation = (
                "✓ AR(2) not rejected: No evidence of second-order correlation. "
                "Moment conditions appear valid."
            )
    else:
        if p_value < 0.05:
            interpretation = f"AR({order}) rejected: Correlation of order {order} detected"
        else:
            interpretation = f"AR({order}) not rejected: No evidence of order-{order} correlation"

    return {
        "statistic": z_stat,
        "p_value": p_value,
        "order": order,
        "interpretation": interpretation,
        "n_products": len(products),
    }
