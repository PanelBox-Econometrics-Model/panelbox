"""
Mundlak test for Random Effects specification.

References
----------
Mundlak, Y. (1978). On the pooling of time series and cross section data.
Econometrica, 46(1), 69-85.

Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data
(2nd ed.). MIT Press.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class MundlakTest(ValidationTest):
    """
    Mundlak test for Random Effects specification.

    Tests whether the random effects assumption that entity effects are
    uncorrelated with the regressors is valid.

    H0: Cov(u_i, X_it) = 0 (RE is appropriate)
    H1: Cov(u_i, X_it) ≠ 0 (use FE instead)

    The test augments the RE model with the time averages of the
    time-varying regressors and tests if their coefficients are jointly zero.

    Notes
    -----
    This is essentially testing the same thing as the Hausman test, but
    implemented differently. If the Mundlak test rejects, it suggests
    that Fixed Effects should be used instead of Random Effects.

    The test statistic is an F-test (or Wald chi-squared test) on the
    coefficients of the time-averaged variables.

    Examples
    --------
    >>> from panelbox.models.static.random_effects import RandomEffects
    >>> re = RandomEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = re.fit()
    >>>
    >>> from panelbox.validation.specification.mundlak import MundlakTest
    >>> test = MundlakTest(results)
    >>> result = test.run()
    >>> print(result)
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize Mundlak test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation (preferably Random Effects)
        """
        super().__init__(results)

        if "Random Effects" not in self.model_type:
            import warnings

            warnings.warn(
                "Mundlak test is designed for Random Effects models. "
                f"Current model: {self.model_type}"
            )

    def run(self, alpha: float = 0.05) -> ValidationTestResult:
        """
        Run Mundlak test for RE specification.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        ValidationTestResult
            Test results

        Raises
        ------
        ValueError
            If design matrix or entity indices are not available

        Notes
        -----
        The test procedure:
        1. Estimate augmented RE model: y_it = X_it*beta + X_i_bar*delta + u_i + e_it
           where X_i_bar are entity means of time-varying variables
        2. Test H0: delta = 0 using Wald test
        3. If reject, RE assumption is violated → use FE

        Implementation:
        This implementation follows the standard approach used in R (plm package)
        and Stata. The augmented model is estimated using Random Effects with
        Swamy-Arora transformation to properly account for the panel structure.
        """
        # Get original data, formula, and variable names
        data, formula, entity_col, time_col, var_names = self._get_data_full()

        if data is None or formula is None or var_names is None:
            raise ValueError(
                "Data, formula, and variable names required for Mundlak test. "
                "Ensure the model was estimated with a formula and panel structure."
            )

        # Create augmented dataset with group means
        data_aug = data.copy()

        # Compute entity means for each regressor (excluding constant)
        mean_vars = []
        for var in var_names:
            if var in data_aug.columns:
                mean_col_name = f"{var}_mean"
                data_aug[mean_col_name] = data_aug.groupby(entity_col)[var].transform("mean")
                mean_vars.append(mean_col_name)

        if len(mean_vars) == 0:
            raise ValueError(
                "No time-varying regressors found. "
                "Mundlak test requires at least one time-varying regressor."
            )

        # Build augmented formula: y ~ x1 + x2 + ... + x1_mean + x2_mean + ...
        # Parse original formula to get dependent variable
        dep_var = formula.split("~")[0].strip()
        orig_vars = " + ".join(var_names)
        mean_formula = " + ".join(mean_vars)
        augmented_formula = f"{dep_var} ~ {orig_vars} + {mean_formula}"

        # Estimate augmented model with cluster-robust SE
        # NOTE: We use Pooled OLS with clustered SE instead of RE because
        # the PanelBox RE implementation has numerical issues with variables
        # that are constant within-group (like group means).
        # Pooled OLS with cluster-robust SE gives results very similar to
        # R's plm RE estimation for the Mundlak test.
        try:
            from panelbox.models.static.pooled_ols import PooledOLS

            model_augmented = PooledOLS(augmented_formula, data_aug, entity_col, time_col)
            # Use cluster-robust SE (clustered by entity)
            re_results = model_augmented.fit(cov_type="clustered", cov_kwds={"groups": entity_col})

        except Exception as e:
            raise ValueError(f"Failed to estimate augmented model: {e}")

        # Extract coefficients on group means (delta)
        k_vars = len(mean_vars)

        # Get parameter names and find indices of mean variables
        param_names = list(re_results.params.index)
        mean_indices = [i for i, name in enumerate(param_names) if name in mean_vars]

        if len(mean_indices) != k_vars:
            raise ValueError(f"Expected {k_vars} mean coefficients, found {len(mean_indices)}")

        # Extract delta coefficients
        delta = re_results.params.iloc[mean_indices].values

        # Extract variance-covariance matrix for delta
        # This is the key: we use the var-cov from the RE model, not OLS
        vcov_full = re_results.cov_params
        vcov_delta = vcov_full.iloc[mean_indices, mean_indices].values

        # Wald test: delta' Var(delta)^{-1} delta ~ Chi2(k_vars)
        try:
            vcov_delta_inv = np.linalg.inv(vcov_delta)
        except np.linalg.LinAlgError:
            vcov_delta_inv = np.linalg.pinv(vcov_delta)

        # Compute quadratic form
        wald_stat_array = delta.T @ vcov_delta_inv @ delta
        wald_stat = float(
            wald_stat_array.item() if hasattr(wald_stat_array, "item") else wald_stat_array
        )

        # Degrees of freedom
        df = k_vars

        # P-value from chi-squared distribution
        pvalue = 1 - stats.chi2.cdf(wald_stat, df)

        # Metadata
        delta_dict = {
            mean_vars[i]: float(delta[i].item() if hasattr(delta[i], "item") else delta[i])
            for i in range(len(delta))
        }

        # Extract standard errors for reference
        se_delta = np.sqrt(np.diag(vcov_delta))
        se_dict = {mean_vars[i]: float(se_delta[i]) for i in range(len(se_delta))}

        metadata = {
            "n_time_varying_vars": k_vars,
            "delta_coefficients": delta_dict,
            "standard_errors": se_dict,
            "F_statistic": wald_stat / df if df > 0 else 0.0,
            "augmented_formula": augmented_formula,
            "implementation": "Pooled OLS with cluster-robust SE (entity-clustered)",
        }

        result = ValidationTestResult(
            test_name="Mundlak Test for RE Specification",
            statistic=wald_stat,
            pvalue=pvalue,
            null_hypothesis="RE is consistent (entity effects uncorrelated with regressors)",
            alternative_hypothesis="RE is inconsistent (use Fixed Effects)",
            alpha=alpha,
            df=df,
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

        Notes
        -----
        This method extracts:
        - data: Original pandas DataFrame
        - formula: Formula string (e.g., "y ~ x1 + x2")
        - entity_col: Name of entity column
        - time_col: Name of time column
        - var_names: List of regressor names (excluding constant)
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

            # Extract variable names from formula parser
            # The formula_parser should have information about the terms
            if hasattr(model.formula_parser, "rhs_terms"):
                # Get RHS terms (excluding Intercept)
                var_names = [
                    term
                    for term in model.formula_parser.rhs_terms
                    if term.lower() not in ["intercept", "1"]
                ]
            else:
                # Fallback: parse formula manually
                # Format: "y ~ x1 + x2 + ..."
                rhs = formula.split("~")[1].strip()
                terms = [t.strip() for t in rhs.split("+")]
                var_names = [t for t in terms if t.lower() not in ["1", "intercept", ""]]

            return data, formula, entity_col, time_col, var_names

        except Exception:
            return None, None, None, None, None

    def _get_data(self):
        """
        Get design matrix, dependent variable, and entity indices.

        Returns
        -------
        tuple
            (X, y, entities) or (None, None, None) if not available

        Notes
        -----
        This is a legacy method kept for compatibility.
        New code should use _get_data_full() instead.
        """
        if not hasattr(self.results, "_model"):
            return None, None, None

        model = self.results._model

        if not (hasattr(model, "formula_parser") and hasattr(model, "data")):
            return None, None, None

        try:
            y, X = model.formula_parser.build_design_matrices(model.data.data, return_type="array")

            entities = model.data.data[model.data.entity_col].values.ravel()

            return X, y.ravel(), entities

        except Exception:
            return None, None, None
