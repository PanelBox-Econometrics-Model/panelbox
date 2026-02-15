"""
Davidson-MacKinnon J-Test for non-nested model comparison.

References
----------
Davidson, R., & MacKinnon, J.G. (1981). "Several Tests for Model Specification
in the Presence of Alternative Hypotheses." Econometrica, 49(3), 781-793.

The J-test compares non-nested models by testing whether fitted values from one
model have explanatory power when added to the other model.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class JTestResult:
    """
    Results from Davidson-MacKinnon J-Test.

    Attributes
    ----------
    forward : dict
        Forward test results (Model 1 vs Model 2)
        Contains: 'statistic', 'pvalue', 'alpha_coef', 'alpha_se'
    reverse : dict
        Reverse test results (Model 2 vs Model 1)
        Contains: 'statistic', 'pvalue', 'gamma_coef', 'gamma_se'
    model1_name : str
        Name/description of Model 1
    model2_name : str
        Name/description of Model 2
    direction : str
        Direction of test performed ('forward', 'reverse', 'both')
    """

    forward: Optional[Dict[str, float]]
    reverse: Optional[Dict[str, float]]
    model1_name: str
    model2_name: str
    direction: str

    def interpretation(self) -> str:
        """
        Provide automatic interpretation of J-test results.

        Returns
        -------
        str
            Interpretation based on test outcomes
        """
        if self.direction == "forward":
            return self._interpret_forward()
        elif self.direction == "reverse":
            return self._interpret_reverse()
        else:  # both
            return self._interpret_both()

    def _interpret_forward(self) -> str:
        """Interpret forward test only."""
        pval = self.forward["pvalue"]
        alpha = 0.05

        if pval < alpha:
            return (
                f"Forward test REJECTS H0 (p={pval:.4f} < {alpha}).\n"
                f"{self.model2_name} has significant explanatory power beyond {self.model1_name}.\n"
                f"Evidence suggests {self.model2_name} may be preferred."
            )
        else:
            return (
                f"Forward test FAILS TO REJECT H0 (p={pval:.4f} >= {alpha}).\n"
                f"{self.model1_name} appears adequate; {self.model2_name} adds no significant information."
            )

    def _interpret_reverse(self) -> str:
        """Interpret reverse test only."""
        pval = self.reverse["pvalue"]
        alpha = 0.05

        if pval < alpha:
            return (
                f"Reverse test REJECTS H0 (p={pval:.4f} < {alpha}).\n"
                f"{self.model1_name} has significant explanatory power beyond {self.model2_name}.\n"
                f"Evidence suggests {self.model1_name} may be preferred."
            )
        else:
            return (
                f"Reverse test FAILS TO REJECT H0 (p={pval:.4f} >= {alpha}).\n"
                f"{self.model2_name} appears adequate; {self.model1_name} adds no significant information."
            )

    def _interpret_both(self) -> str:
        """Interpret both forward and reverse tests."""
        alpha = 0.05
        forward_reject = self.forward["pvalue"] < alpha
        reverse_reject = self.reverse["pvalue"] < alpha

        interpretation = "J-Test Results (Both Directions):\n"
        interpretation += "=" * 60 + "\n\n"

        interpretation += f"Forward Test ({self.model1_name} vs {self.model2_name}):\n"
        interpretation += f"  H0: {self.model1_name} is correct\n"
        interpretation += f"  p-value = {self.forward['pvalue']:.4f}\n"
        interpretation += f"  Result: {'REJECT H0' if forward_reject else 'FAIL TO REJECT H0'}\n\n"

        interpretation += f"Reverse Test ({self.model2_name} vs {self.model1_name}):\n"
        interpretation += f"  H0: {self.model2_name} is correct\n"
        interpretation += f"  p-value = {self.reverse['pvalue']:.4f}\n"
        interpretation += f"  Result: {'REJECT H0' if reverse_reject else 'FAIL TO REJECT H0'}\n\n"

        interpretation += "Overall Interpretation:\n"
        interpretation += "-" * 60 + "\n"

        if forward_reject and not reverse_reject:
            interpretation += f"PREFER {self.model2_name}\n"
            interpretation += f"  - {self.model2_name} encompasses {self.model1_name}\n"
            interpretation += (
                f"  - {self.model2_name} has significant additional explanatory power\n"
            )
        elif reverse_reject and not forward_reject:
            interpretation += f"PREFER {self.model1_name}\n"
            interpretation += f"  - {self.model1_name} encompasses {self.model2_name}\n"
            interpretation += (
                f"  - {self.model1_name} has significant additional explanatory power\n"
            )
        elif forward_reject and reverse_reject:
            interpretation += "BOTH MODELS REJECTED\n"
            interpretation += "  - Neither model adequately encompasses the other\n"
            interpretation += "  - Both models have explanatory power the other lacks\n"
            interpretation += "  - Consider a more comprehensive specification\n"
        else:  # neither rejected
            interpretation += "BOTH MODELS ACCEPTABLE\n"
            interpretation += "  - Neither model significantly improves upon the other\n"
            interpretation += "  - Models are empirically equivalent for this data\n"
            interpretation += "  - Use economic theory or other criteria to choose\n"

        return interpretation

    def summary(self) -> pd.DataFrame:
        """
        Return formatted summary table.

        Returns
        -------
        pd.DataFrame
            Summary table with test statistics and p-values
        """
        rows = []

        if self.forward is not None:
            rows.append(
                {
                    "Test": "Forward",
                    "Null Hypothesis": f"{self.model1_name} is correct",
                    "Test Statistic": self.forward["statistic"],
                    "p-value": self.forward["pvalue"],
                    "Coefficient": self.forward.get("alpha_coef", np.nan),
                    "Std. Error": self.forward.get("alpha_se", np.nan),
                }
            )

        if self.reverse is not None:
            rows.append(
                {
                    "Test": "Reverse",
                    "Null Hypothesis": f"{self.model2_name} is correct",
                    "Test Statistic": self.reverse["statistic"],
                    "p-value": self.reverse["pvalue"],
                    "Coefficient": self.reverse.get("gamma_coef", np.nan),
                    "Std. Error": self.reverse.get("gamma_se", np.nan),
                }
            )

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """String representation."""
        return f"JTestResult(direction='{self.direction}')\n\n{self.interpretation()}"


def j_test(
    result1,
    result2,
    direction: Literal["forward", "reverse", "both"] = "both",
    model1_name: Optional[str] = None,
    model2_name: Optional[str] = None,
) -> JTestResult:
    """
    Perform Davidson-MacKinnon J-test for non-nested model comparison.

    The J-test determines whether one model's fitted values have explanatory
    power when added to the other model. For panel data, cluster-robust
    standard errors are used.

    Parameters
    ----------
    result1 : fitted model result
        First model's fitted result object. Must have:
        - .fittedvalues: fitted values
        - .model.endog: dependent variable
        - .model.exog: exogenous variables
        - .model.data.frame: original DataFrame (for panel models)
    result2 : fitted model result
        Second model's fitted result object (same requirements as result1)
    direction : {'forward', 'reverse', 'both'}, default 'both'
        - 'forward': Test if Model 2's fitted values improve Model 1
        - 'reverse': Test if Model 1's fitted values improve Model 2
        - 'both': Perform both tests
    model1_name : str, optional
        Descriptive name for Model 1 (default: 'Model 1')
    model2_name : str, optional
        Descriptive name for Model 2 (default: 'Model 2')

    Returns
    -------
    JTestResult
        Object containing test results and interpretation

    Notes
    -----
    Test Procedure:

    Forward Test (H0: Model 1 is correct):
        1. Get fitted values ŷ₂ from Model 2
        2. Augment Model 1: y = X₁'β + α·ŷ₂ + ε
        3. Test H0: α = 0
        4. If reject: Model 2 has additional explanatory power

    Reverse Test (H0: Model 2 is correct):
        1. Get fitted values ŷ₁ from Model 1
        2. Augment Model 2: y = X₂'β + γ·ŷ₁ + ε
        3. Test H0: γ = 0
        4. If reject: Model 1 has additional explanatory power

    For panel data, cluster-robust standard errors are automatically used.

    Examples
    --------
    >>> # Compare two production function specifications
    >>> model1 = PooledOLS(df, 'output', ['labor', 'capital'])
    >>> result1 = model1.fit()
    >>>
    >>> model2 = PooledOLS(df, 'output', ['labor', 'capital', 'tech'])
    >>> result2 = model2.fit()
    >>>
    >>> jtest = j_test(result1, result2, model1_name='Basic', model2_name='With Tech')
    >>> print(jtest.interpretation())
    >>> print(jtest.summary())

    References
    ----------
    Davidson, R., & MacKinnon, J.G. (1981). "Several Tests for Model
    Specification in the Presence of Alternative Hypotheses."
    Econometrica, 49(3), 781-793.
    """
    # Set default names
    if model1_name is None:
        model1_name = "Model 1"
    if model2_name is None:
        model2_name = "Model 2"

    # Validate inputs
    _validate_results(result1, result2)

    # Initialize result containers
    forward_result = None
    reverse_result = None

    # Perform forward test
    if direction in ["forward", "both"]:
        forward_result = _perform_forward_test(result1, result2)

    # Perform reverse test
    if direction in ["reverse", "both"]:
        reverse_result = _perform_reverse_test(result1, result2)

    return JTestResult(
        forward=forward_result,
        reverse=reverse_result,
        model1_name=model1_name,
        model2_name=model2_name,
        direction=direction,
    )


def _validate_results(result1, result2):
    """Validate that result objects have required attributes."""
    required_attrs = ["fittedvalues", "model"]

    for i, result in enumerate([result1, result2], 1):
        for attr in required_attrs:
            if not hasattr(result, attr):
                raise AttributeError(
                    f"result{i} missing required attribute '{attr}'. "
                    f"Ensure you pass fitted model result objects."
                )

    # Check that dependent variables match
    y1 = result1.model.endog
    y2 = result2.model.endog

    if len(y1) != len(y2):
        raise ValueError(
            f"Models have different sample sizes: {len(y1)} vs {len(y2)}. "
            f"Ensure both models use the same data."
        )

    # Check that y values are the same (allowing for small numerical differences)
    if not np.allclose(y1, y2, rtol=1e-10, atol=1e-10):
        warnings.warn(
            "Models appear to have different dependent variables. "
            "J-test requires comparing models with the same dependent variable.",
            UserWarning,
        )


def _perform_forward_test(result1, result2) -> Dict[str, float]:
    """
    Perform forward J-test: Test if Model 2's fitted values improve Model 1.

    Augmented regression: y = X₁'β + α·ŷ₂ + ε
    H0: α = 0
    """
    from scipy import stats

    # Get data
    y = result1.model.endog
    X1 = result1.model.exog
    yhat2 = result2.fittedvalues

    # Ensure yhat2 is proper shape
    if yhat2.ndim == 1:
        yhat2 = yhat2.reshape(-1, 1)

    # Augment X1 with fitted values from Model 2
    X_augmented = np.column_stack([X1, yhat2])

    # Estimate augmented model with OLS
    # Use cluster-robust SEs if panel data
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    augmented_model = OLS(y, X_augmented)

    # Try to use cluster-robust SEs if this is panel data
    try:
        # Check if original model has entity information
        if hasattr(result1.model, "data") and hasattr(result1.model.data, "frame"):
            # Panel data - use cluster-robust SEs
            if hasattr(result1.model.data, "entity_id"):
                entity_id = result1.model.data.entity_id
                augmented_result = augmented_model.fit(
                    cov_type="cluster", cov_kwds={"groups": entity_id}
                )
            else:
                # Try to infer clustering from index
                augmented_result = augmented_model.fit(cov_type="HC1")
        else:
            # No panel structure - use heteroskedasticity-robust SEs
            augmented_result = augmented_model.fit(cov_type="HC1")
    except Exception:
        # Fall back to OLS SEs
        augmented_result = augmented_model.fit()

    # Get test statistic for last coefficient (α)
    alpha_idx = -1
    alpha_coef = augmented_result.params[alpha_idx]
    alpha_se = augmented_result.bse[alpha_idx]
    t_stat = augmented_result.tvalues[alpha_idx]
    p_value = augmented_result.pvalues[alpha_idx]

    return {"statistic": t_stat, "pvalue": p_value, "alpha_coef": alpha_coef, "alpha_se": alpha_se}


def _perform_reverse_test(result1, result2) -> Dict[str, float]:
    """
    Perform reverse J-test: Test if Model 1's fitted values improve Model 2.

    Augmented regression: y = X₂'β + γ·ŷ₁ + ε
    H0: γ = 0
    """
    from scipy import stats

    # Get data
    y = result2.model.endog
    X2 = result2.model.exog
    yhat1 = result1.fittedvalues

    # Ensure yhat1 is proper shape
    if yhat1.ndim == 1:
        yhat1 = yhat1.reshape(-1, 1)

    # Augment X2 with fitted values from Model 1
    X_augmented = np.column_stack([X2, yhat1])

    # Estimate augmented model with OLS
    from statsmodels.regression.linear_model import OLS

    augmented_model = OLS(y, X_augmented)

    # Try to use cluster-robust SEs if this is panel data
    try:
        # Check if original model has entity information
        if hasattr(result2.model, "data") and hasattr(result2.model.data, "frame"):
            # Panel data - use cluster-robust SEs
            if hasattr(result2.model.data, "entity_id"):
                entity_id = result2.model.data.entity_id
                augmented_result = augmented_model.fit(
                    cov_type="cluster", cov_kwds={"groups": entity_id}
                )
            else:
                # Try to infer clustering from index
                augmented_result = augmented_model.fit(cov_type="HC1")
        else:
            # No panel structure - use heteroskedasticity-robust SEs
            augmented_result = augmented_model.fit(cov_type="HC1")
    except Exception:
        # Fall back to OLS SEs
        augmented_result = augmented_model.fit()

    # Get test statistic for last coefficient (γ)
    gamma_idx = -1
    gamma_coef = augmented_result.params[gamma_idx]
    gamma_se = augmented_result.bse[gamma_idx]
    t_stat = augmented_result.tvalues[gamma_idx]
    p_value = augmented_result.pvalues[gamma_idx]

    return {"statistic": t_stat, "pvalue": p_value, "gamma_coef": gamma_coef, "gamma_se": gamma_se}
