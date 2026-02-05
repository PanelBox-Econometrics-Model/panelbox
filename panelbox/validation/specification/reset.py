"""
RESET test for specification errors in panel data models.

RESET = Regression Equation Specification Error Test

References
----------
Ramsey, J. B. (1969). Tests for Specification Errors in Classical Linear
Least Squares Regression Analysis. Journal of the Royal Statistical Society,
Series B, 31(2), 350-371.

Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data
(2nd ed.). MIT Press.
"""

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class RESETTest(ValidationTest):
    """
    RESET test for functional form specification.

    Tests the null hypothesis that the model is correctly specified
    (linear functional form is appropriate) against the alternative
    that nonlinear terms are needed.

    H0: E[y | X] = X*beta (linear specification is correct)
    H1: E[y | X] includes higher-order terms of fitted values

    The test augments the model with powers of fitted values (ŷ², ŷ³, ...)
    and tests if these terms are jointly significant.

    Notes
    -----
    The test is implemented by:
    1. Estimating original model: y = X*beta + e
    2. Computing fitted values: ŷ = X*beta_hat
    3. Augmenting model: y = X*beta + gamma2*ŷ² + gamma3*ŷ³ + ... + u
    4. Testing H0: gamma2 = gamma3 = ... = 0 using F-test

    For panel data, we use pooled OLS with cluster-robust standard errors
    to account for within-group correlation.

    Common practice is to include powers 2 and 3 (default).

    Examples
    --------
    >>> from panelbox.models.static.pooled_ols import PooledOLS
    >>> model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
    >>> results = model.fit()
    >>>
    >>> from panelbox.validation.specification.reset import RESETTest
    >>> test = RESETTest(results)
    >>> result = test.run(powers=[2, 3])  # Test with ŷ² and ŷ³
    >>> print(result)
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize RESET test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        super().__init__(results)

    def run(self, powers=None, alpha: float = 0.05) -> ValidationTestResult:
        """
        Run RESET test for specification errors.

        Parameters
        ----------
        powers : list of int, optional
            Powers of fitted values to include in augmented regression.
            Default is [2, 3] (quadratic and cubic terms).
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        ValidationTestResult
            Test results

        Raises
        ------
        ValueError
            If powers are invalid or data is not available

        Notes
        -----
        The test uses an F-statistic for testing joint significance of
        the polynomial terms. For panel data, we use cluster-robust
        standard errors.
        """
        if powers is None:
            powers = [2, 3]

        # Validate powers
        if not all(isinstance(p, int) and p >= 2 for p in powers):
            raise ValueError("Powers must be integers >= 2")

        # Get data
        data, formula, entity_col, time_col, var_names = self._get_data_full()

        if data is None or formula is None:
            raise ValueError(
                "Data and formula required for RESET test. "
                "Ensure the model was estimated with a formula."
            )

        # Get fitted values from original model
        fitted = self.results.fitted_values
        if fitted is None:
            raise ValueError("Fitted values not available from model results")

        # Create augmented dataset with powers of fitted values
        data_aug = data.copy()
        power_vars = []

        for power in powers:
            var_name = f"fitted_pow{power}"
            data_aug[var_name] = fitted**power
            power_vars.append(var_name)

        # Build augmented formula
        dep_var = formula.split("~")[0].strip()
        orig_vars = " + ".join(var_names)
        power_formula = " + ".join(power_vars)
        augmented_formula = f"{dep_var} ~ {orig_vars} + {power_formula}"

        # Estimate augmented model with cluster-robust SE
        try:
            from panelbox.models.static.pooled_ols import PooledOLS

            model_aug = PooledOLS(augmented_formula, data_aug, entity_col, time_col)
            results_aug = model_aug.fit(cov_type="clustered", cov_kwds={"groups": entity_col})

        except Exception as e:
            raise ValueError(f"Failed to estimate augmented model: {e}")

        # Extract coefficients on power terms
        gamma = results_aug.params[power_vars].values

        # Extract variance-covariance matrix for power terms
        vcov_gamma = results_aug.cov_params.loc[power_vars, power_vars].values

        # Wald test: gamma' * Vcov(gamma)^-1 * gamma ~ Chi2(k)
        # where k = number of power terms
        try:
            vcov_inv = np.linalg.inv(vcov_gamma)
        except np.linalg.LinAlgError:
            vcov_inv = np.linalg.pinv(vcov_gamma)

        wald_stat_array = gamma.T @ vcov_inv @ gamma
        wald_stat = float(
            wald_stat_array.item() if hasattr(wald_stat_array, "item") else wald_stat_array
        )

        # Degrees of freedom
        df_num = len(powers)
        df_denom = results_aug.nobs - results_aug.params.shape[0]

        # Convert to F-statistic
        f_stat = wald_stat / df_num

        # P-value from F distribution
        pvalue = 1 - stats.f.cdf(f_stat, df_num, df_denom)

        # Alternative: use chi-squared approximation
        pvalue_chi2 = 1 - stats.chi2.cdf(wald_stat, df_num)

        # Metadata
        gamma_dict = {power_vars[i]: float(gamma[i]) for i in range(len(gamma))}

        se_gamma = np.sqrt(np.diag(vcov_gamma))
        se_dict = {power_vars[i]: float(se_gamma[i]) for i in range(len(se_gamma))}

        metadata = {
            "powers": powers,
            "gamma_coefficients": gamma_dict,
            "standard_errors": se_dict,
            "wald_statistic": wald_stat,
            "F_statistic": f_stat,
            "df_numerator": df_num,
            "df_denominator": df_denom,
            "pvalue_chi2": pvalue_chi2,
            "augmented_formula": augmented_formula,
        }

        result = ValidationTestResult(
            test_name="RESET Test for Specification",
            statistic=f_stat,
            pvalue=pvalue,
            null_hypothesis="Model is correctly specified (linear functional form)",
            alternative_hypothesis="Nonlinear terms needed (specification error)",
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
            # Get original data
            data = model.data.data.copy()

            # Get entity and time columns
            entity_col = model.data.entity_col
            time_col = model.data.time_col

            # Get formula
            if hasattr(model, "formula"):
                formula = model.formula
            else:
                return None, None, None, None, None

            # Extract variable names from formula
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
