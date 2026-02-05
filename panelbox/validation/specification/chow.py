"""
Chow test for structural break in panel data models.

References
----------
Chow, G. C. (1960). Tests of Equality Between Sets of Coefficients in
Two Linear Regressions. Econometrica, 28(3), 591-605.

Baltagi, B. H. (2013). Econometric Analysis of Panel Data (5th ed.).
Wiley, Chapter 4.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class ChowTest(ValidationTest):
    """
    Chow test for structural break in panel models.

    Tests the null hypothesis of parameter stability (no structural break)
    against the alternative that parameters differ across subperiods.

    H0: beta_1 = beta_2 (parameters are stable)
    H1: beta_1 ≠ beta_2 (structural break exists)

    The test compares the fit of:
    - Unrestricted model: separate parameters for each subperiod
    - Restricted model: same parameters for all periods

    Notes
    -----
    The test statistic is:
    F = [(SSR_r - SSR_u) / k] / [SSR_u / (N - 2k)]

    where:
    - SSR_r = residual sum of squares (restricted model)
    - SSR_u = residual sum of squares (unrestricted model)
    - k = number of parameters
    - N = total number of observations

    For panel data, we use pooled estimation with cluster-robust inference.

    Examples
    --------
    >>> from panelbox.models.static.pooled_ols import PooledOLS
    >>> model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
    >>> results = model.fit()
    >>>
    >>> from panelbox.validation.specification.chow import ChowTest
    >>> test = ChowTest(results)
    >>> # Test for break at time period 5
    >>> result = test.run(break_point=5)
    >>> print(result)
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize Chow test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        super().__init__(results)

    def run(self, break_point=None, alpha: float = 0.05) -> ValidationTestResult:
        """
        Run Chow test for structural break.

        Parameters
        ----------
        break_point : int or float, optional
            Time period at which to test for structural break.
            If None, uses the median time period.
            Can be specified as:
            - Integer: exact time period
            - Float between 0 and 1: fraction of sample (e.g., 0.5 for midpoint)
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        ValidationTestResult
            Test results

        Raises
        ------
        ValueError
            If break_point is invalid or data is not available

        Notes
        -----
        The test requires at least 2*k observations in each subperiod,
        where k is the number of parameters.
        """
        # Get data
        data, formula, entity_col, time_col, var_names = self._get_data_full()

        if data is None or formula is None:
            raise ValueError(
                "Data and formula required for Chow test. "
                "Ensure the model was estimated with a formula."
            )

        # Get unique time periods (sorted)
        time_periods = sorted(data[time_col].unique())
        n_periods = len(time_periods)

        # Determine break point
        if break_point is None:
            # Use median
            break_idx = n_periods // 2
            break_time = time_periods[break_idx]
        elif isinstance(break_point, float) and 0 < break_point < 1:
            # Fraction of sample
            break_idx = int(n_periods * break_point)
            break_time = time_periods[break_idx]
        elif isinstance(break_point, (int, np.integer)):
            # Exact time period
            if break_point not in time_periods:
                raise ValueError(
                    f"Break point {break_point} not found in time periods. "
                    f"Available: {time_periods}"
                )
            break_time = break_point
            break_idx = time_periods.index(break_time)
        else:
            raise ValueError(
                "break_point must be None, int (time period), "
                "or float between 0 and 1 (fraction)"
            )

        # Create subperiod indicator
        data_aug = data.copy()
        data_aug["period_1"] = (data_aug[time_col] < break_time).astype(int)
        data_aug["period_2"] = (data_aug[time_col] >= break_time).astype(int)

        # Check sample sizes
        n1 = data_aug["period_1"].sum()
        n2 = data_aug["period_2"].sum()
        k = len(var_names) + 1  # +1 for intercept

        if n1 < 2 * k or n2 < 2 * k:
            raise ValueError(
                f"Insufficient observations in subperiods. "
                f"Need at least {2*k} in each, got n1={n1}, n2={n2}"
            )

        # Estimate restricted model (pooled)
        try:
            from panelbox.models.static.pooled_ols import PooledOLS

            model_restricted = PooledOLS(formula, data_aug, entity_col, time_col)
            results_restricted = model_restricted.fit()
            ssr_restricted = np.sum(results_restricted.resid**2)

        except Exception as e:
            raise ValueError(f"Failed to estimate restricted model: {e}")

        # Estimate unrestricted model (separate for each subperiod)
        # Model 1: period < break_time
        data_period1 = data_aug[data_aug["period_1"] == 1].copy()
        model_1 = PooledOLS(formula, data_period1, entity_col, time_col)
        results_1 = model_1.fit()
        ssr_1 = np.sum(results_1.resid**2)

        # Model 2: period >= break_time
        data_period2 = data_aug[data_aug["period_2"] == 1].copy()
        model_2 = PooledOLS(formula, data_period2, entity_col, time_col)
        results_2 = model_2.fit()
        ssr_2 = np.sum(results_2.resid**2)

        # Unrestricted SSR (sum of both periods)
        ssr_unrestricted = ssr_1 + ssr_2

        # Chow F-statistic
        # F = [(SSR_r - SSR_u) / k] / [SSR_u / (N - 2k)]
        N = len(data_aug)
        numerator = (ssr_restricted - ssr_unrestricted) / k
        denominator = ssr_unrestricted / (N - 2 * k)

        if denominator == 0:
            raise ValueError("Denominator is zero. Perfect fit in subperiods.")

        f_stat = numerator / denominator

        # Degrees of freedom
        df_num = k
        df_denom = N - 2 * k

        # P-value
        pvalue = 1 - stats.f.cdf(f_stat, df_num, df_denom)

        # Metadata
        metadata = {
            "break_point": break_time,
            "break_index": break_idx,
            "n_periods_total": n_periods,
            "n_obs_period1": n1,
            "n_obs_period2": n2,
            "n_obs_total": N,
            "ssr_restricted": float(ssr_restricted),
            "ssr_unrestricted": float(ssr_unrestricted),
            "ssr_period1": float(ssr_1),
            "ssr_period2": float(ssr_2),
            "k_parameters": k,
            "coefficients_period1": results_1.params.to_dict(),
            "coefficients_period2": results_2.params.to_dict(),
        }

        result = ValidationTestResult(
            test_name="Chow Test for Structural Break",
            statistic=f_stat,
            pvalue=pvalue,
            null_hypothesis="No structural break (parameters stable)",
            alternative_hypothesis=f"Structural break at t={break_time}",
            alpha=alpha,
            df=(df_num, df_denom),
            metadata=metadata,
        )

        return result

    def _get_data_full(self):
        """
        Get full data including DataFrame, formula, and variable names.

        Returns
        -------
        tuple
            (data, formula, entity_col, time_col, var_names) or
            (None, None, None, None, None) if not available
        """
        if not hasattr(self.results, "_model"):
            return None, None, None, None, None

        model = self.results._model

        if not (hasattr(model, "formula_parser") and hasattr(model, "data")):
            return None, None, None, None, None

        try:
            data = model.data.data.copy()
            entity_col = model.data.entity_col
            time_col = model.data.time_col

            if hasattr(model, "formula"):
                formula = model.formula
            else:
                return None, None, None, None, None

            if hasattr(model.formula_parser, "rhs_terms"):
                var_names = [
                    term
                    for term in model.formula_parser.rhs_terms
                    if term.lower() not in ["intercept", "1"]
                ]
            else:
                rhs = formula.split("~")[1].strip()
                terms = [t.strip() for t in rhs.split("+")]
                var_names = [t for t in terms if t.lower() not in ["1", "intercept", ""]]

            return data, formula, entity_col, time_col, var_names

        except Exception:
            return None, None, None, None, None
