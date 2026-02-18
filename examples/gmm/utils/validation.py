"""
Validation and diagnostic helper functions for GMM tutorials.

Provides checklist automation, test interpretation utilities,
and specification checking tools.
"""

from typing import Any

import numpy as np
import pandas as pd


def gmm_diagnostic_checklist(result: Any) -> pd.DataFrame:
    """
    Generate a diagnostic checklist for a GMM estimation result.

    Checks standard post-estimation diagnostics and provides
    pass/fail/warning status with interpretation guidance.

    Parameters
    ----------
    result : GMM result object
        The fitted model result from PanelBox GMM estimators.

    Returns
    -------
    pd.DataFrame
        Checklist with columns: test, statistic, p_value, status, interpretation.
    """
    checks = []

    # AR(1) test — should reject (p < 0.05)
    if hasattr(result, "ar1_test"):
        ar1 = result.ar1_test
        checks.append(
            {
                "test": "AR(1) in first differences",
                "statistic": ar1.get("statistic", np.nan),
                "p_value": ar1.get("p_value", np.nan),
                "status": "PASS" if ar1.get("p_value", 1) < 0.05 else "WARNING",
                "interpretation": (
                    "Significant negative AR(1) expected in first differences. "
                    "Failure to reject may indicate model issues."
                ),
            }
        )

    # AR(2) test — should NOT reject (p > 0.05)
    if hasattr(result, "ar2_test"):
        ar2 = result.ar2_test
        checks.append(
            {
                "test": "AR(2) in first differences",
                "statistic": ar2.get("statistic", np.nan),
                "p_value": ar2.get("p_value", np.nan),
                "status": "PASS" if ar2.get("p_value", 0) > 0.05 else "FAIL",
                "interpretation": (
                    "No significant AR(2) expected. Rejection suggests "
                    "deeper serial correlation — instruments may be invalid."
                ),
            }
        )

    # Hansen J-test — should NOT reject (p > 0.05)
    if hasattr(result, "hansen_test"):
        hansen = result.hansen_test
        p = hansen.get("p_value", 0)
        if p > 0.25:
            status = "PASS"
        elif p > 0.05:
            status = "WARNING"
        else:
            status = "FAIL"
        checks.append(
            {
                "test": "Hansen J overidentification",
                "statistic": hansen.get("statistic", np.nan),
                "p_value": p,
                "status": status,
                "interpretation": (
                    "Tests joint validity of instruments. Rejection indicates "
                    "misspecification. Very high p-values (>0.99) may indicate "
                    "instrument proliferation."
                ),
            }
        )

    # Instrument count
    if hasattr(result, "n_instruments") and hasattr(result, "n_groups"):
        n_inst = result.n_instruments
        n_groups = result.n_groups
        checks.append(
            {
                "test": "Instrument count rule of thumb",
                "statistic": n_inst,
                "p_value": np.nan,
                "status": "PASS" if n_inst <= n_groups else "WARNING",
                "interpretation": (
                    f"Instruments ({n_inst}) should not exceed groups ({n_groups}). "
                    "Excess instruments weaken Hansen test and can cause overfitting."
                ),
            }
        )

    if not checks:
        checks.append(
            {
                "test": "No diagnostics available",
                "statistic": np.nan,
                "p_value": np.nan,
                "status": "N/A",
                "interpretation": "Result object does not expose standard diagnostics.",
            }
        )

    return pd.DataFrame(checks)


def interpret_test(test_name: str, p_value: float, alpha: float = 0.05) -> str:
    """
    Provide a plain-language interpretation of a GMM diagnostic test.

    Parameters
    ----------
    test_name : str
        Name of the test (e.g., "ar1", "ar2", "hansen", "sargan").
    p_value : float
        The p-value of the test.
    alpha : float
        Significance level.

    Returns
    -------
    str
        Human-readable interpretation.
    """
    reject = p_value < alpha

    interpretations = {
        "ar1": (
            f"AR(1) test: p={p_value:.4f}. "
            + (
                "Significant — expected for valid GMM in first differences."
                if reject
                else "Not significant — unusual, may indicate model issues."
            )
        ),
        "ar2": (
            f"AR(2) test: p={p_value:.4f}. "
            + (
                "Significant — suggests serial correlation beyond AR(1). "
                "Consider deeper lags or re-specification."
                if reject
                else "Not significant — no evidence of AR(2). Instruments valid "
                "at this lag structure."
            )
        ),
        "hansen": (
            f"Hansen J test: p={p_value:.4f}. "
            + (
                "Rejects — instruments may be invalid or model misspecified."
                if reject
                else "Does not reject — no evidence against instrument validity."
                + (
                    " However, very high p-value may indicate instrument proliferation."
                    if p_value > 0.99
                    else ""
                )
            )
        ),
        "sargan": (
            f"Sargan test: p={p_value:.4f}. "
            + (
                "Rejects — potential instrument invalidity. Note: Sargan is "
                "not robust to heteroskedasticity (prefer Hansen)."
                if reject
                else "Does not reject. Note: only valid under homoskedasticity."
            )
        ),
    }

    key = test_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for k, v in interpretations.items():
        if k in key:
            return v

    return f"Test '{test_name}': p={p_value:.4f}. {'Reject' if reject else 'Fail to reject'} at alpha={alpha}."


def specification_checklist() -> pd.DataFrame:
    """
    Return a pre-estimation specification checklist for GMM models.

    Returns
    -------
    pd.DataFrame
        Checklist with items to verify before running GMM.
    """
    items = [
        {
            "step": 1,
            "item": "Identify endogenous variables",
            "question": "Which regressors are correlated with the error term?",
            "guidance": "At minimum, the lagged dependent variable in dynamic models.",
        },
        {
            "step": 2,
            "item": "Choose GMM variant",
            "question": "Difference GMM or System GMM?",
            "guidance": "System GMM preferred for persistent series (rho > 0.8) or short T.",
        },
        {
            "step": 3,
            "item": "Set instrument lags",
            "question": "What are the minimum and maximum lags for GMM-style instruments?",
            "guidance": "Start with lags 2+ for Difference GMM. Avoid excessive lags.",
        },
        {
            "step": 4,
            "item": "Check instrument count",
            "question": "Is the number of instruments <= number of groups?",
            "guidance": "Use collapse option if instruments proliferate.",
        },
        {
            "step": 5,
            "item": "Select estimation step",
            "question": "One-step, two-step, or CUE?",
            "guidance": "Two-step with Windmeijer correction is standard. CUE for small samples.",
        },
        {
            "step": 6,
            "item": "Plan diagnostic tests",
            "question": "Which tests will validate the specification?",
            "guidance": "AR(1), AR(2), Hansen J are minimum. Add Difference-in-Hansen for System GMM.",
        },
    ]
    return pd.DataFrame(items)


def compare_estimators(
    results: dict[str, Any], true_params: dict[str, float] | None = None
) -> pd.DataFrame:
    """
    Compare parameter estimates across different estimators.

    Parameters
    ----------
    results : dict
        Mapping of estimator names to result objects.
    true_params : dict, optional
        True parameter values (for simulated data).

    Returns
    -------
    pd.DataFrame
        Comparison table with estimates, standard errors, and bias.
    """
    rows = []
    for name, result in results.items():
        if hasattr(result, "params") and hasattr(result, "std_errors"):
            for param_name in result.params.index:
                row = {
                    "estimator": name,
                    "parameter": param_name,
                    "estimate": result.params[param_name],
                    "std_error": result.std_errors[param_name],
                }
                if true_params and param_name in true_params:
                    row["true_value"] = true_params[param_name]
                    row["bias"] = row["estimate"] - true_params[param_name]
                rows.append(row)

    return pd.DataFrame(rows)
