"""
Unified interface for panel unit root tests.

This module provides a unified function to run multiple panel unit root tests
and compare results.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

import pandas as pd

from .breitung import BreitungResult, breitung_test
from .hadri import HadriResult, hadri_test

try:
    from ...validation.unit_root.ips import IPSTestResult, ips_test

    HAS_IPS = True
except ImportError:
    HAS_IPS = False

try:
    from ...validation.unit_root.llc import LLCTestResult, llc_test

    HAS_LLC = True
except ImportError:
    HAS_LLC = False


@dataclass
class PanelUnitRootResult:
    """
    Results from multiple panel unit root tests.

    Attributes
    ----------
    results : dict
        Dictionary mapping test names to test result objects.
    variable : str
        Name of the variable tested.
    n_entities : int
        Number of cross-sectional units.
    n_time : int
        Number of time periods.
    tests_run : list
        List of tests that were executed.

    Methods
    -------
    summary_table()
        Generate a formatted comparison table.
    interpretation()
        Provide interpretation of combined results.
    """

    results: Dict[str, Any]
    variable: str
    n_entities: int
    n_time: int
    tests_run: list

    def summary_table(self) -> str:
        """
        Generate formatted comparison table of all test results.

        Returns
        -------
        str
            Formatted table comparing test statistics and p-values.
        """
        lines = [
            "=" * 80,
            "Panel Unit Root Test Summary",
            "=" * 80,
            f"Variable: {self.variable}",
            f"Number of entities (N): {self.n_entities}",
            f"Number of periods (T): {self.n_time}",
            "",
            "Test Results:",
            "-" * 80,
            f"{'Test':<20} {'H0':<25} {'Statistic':>12} {'P-value':>10} {'Decision':>10}",
            "-" * 80,
        ]

        # Add each test result
        for test_name in self.tests_run:
            result = self.results[test_name]

            # Format H0 based on test
            if test_name == "hadri":
                h0 = "Stationarity"
            else:
                h0 = "Unit root"

            # Get statistic and p-value
            statistic = result.statistic
            pvalue = result.pvalue
            decision = "REJECT" if result.reject else "FAIL TO REJECT"

            lines.append(
                f"{test_name.upper():<20} {h0:<25} {statistic:>12.4f} {pvalue:>10.4f} {decision:>10}"
            )

        lines.append("-" * 80)
        lines.append("")
        lines.append(self.interpretation())
        lines.append("=" * 80)

        return "\n".join(lines)

    def interpretation(self) -> str:
        """
        Provide interpretation of combined test results.

        Returns
        -------
        str
            Interpretation of results across all tests.
        """
        # Count rejections by type
        unit_root_tests = []  # Tests with H0: unit root
        stationarity_tests = []  # Tests with H0: stationarity

        for test_name in self.tests_run:
            result = self.results[test_name]
            if test_name == "hadri":
                stationarity_tests.append(result.reject)
            else:
                unit_root_tests.append(result.reject)

        # Analyze results
        lines = ["Interpretation:"]

        # Check unit root tests (IPS, LLC, Breitung)
        if unit_root_tests:
            n_reject_ur = sum(unit_root_tests)
            n_total_ur = len(unit_root_tests)

            if n_reject_ur == 0:
                lines.append(f"  - All {n_total_ur} unit root test(s) fail to reject H0")
                lines.append("    → Strong evidence of unit root (non-stationarity)")
            elif n_reject_ur == n_total_ur:
                lines.append(f"  - All {n_total_ur} unit root test(s) reject H0")
                lines.append("    → Strong evidence of stationarity")
            else:
                lines.append(f"  - {n_reject_ur} out of {n_total_ur} unit root test(s) reject H0")
                lines.append("    → Mixed evidence (proceed with caution)")

        # Check stationarity tests (Hadri)
        if stationarity_tests:
            n_reject_stat = sum(stationarity_tests)
            n_total_stat = len(stationarity_tests)

            if n_reject_stat == 0:
                lines.append(f"  - Stationarity test(s) fail to reject H0")
                lines.append("    → Evidence consistent with stationarity")
            else:
                lines.append(f"  - Stationarity test(s) reject H0")
                lines.append("    → Evidence of unit root")

        # Overall recommendation
        lines.append("")
        lines.append("Overall recommendation:")

        # Simple majority rule
        total_tests = len(self.tests_run)
        total_rejections = sum(
            1 for name in self.tests_run if (name != "hadri" and self.results[name].reject)
        )
        hadri_rejects = sum(
            1 for name in self.tests_run if (name == "hadri" and self.results[name].reject)
        )

        # Evidence for stationarity: unit root tests reject OR hadri doesn't reject
        evidence_stationary = total_rejections + (len(stationarity_tests) - hadri_rejects)
        evidence_unit_root = (len(unit_root_tests) - total_rejections) + hadri_rejects

        if evidence_stationary > evidence_unit_root:
            lines.append("  ✓ Majority of tests suggest the series is STATIONARY")
            lines.append("    Safe to proceed with level regressions")
        elif evidence_unit_root > evidence_stationary:
            lines.append("  ⚠ Majority of tests suggest the series has a UNIT ROOT")
            lines.append("    Consider differencing or cointegration analysis")
        else:
            lines.append("  ⚠ Tests are evenly split - results are INCONCLUSIVE")
            lines.append("    Consider additional diagnostics or economic theory")

        return "\n".join(lines)

    def __repr__(self) -> str:
        tests_str = ", ".join(self.tests_run)
        return f"PanelUnitRootResult(tests=[{tests_str}], variable='{self.variable}')"


def panel_unit_root_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = "entity",
    time_col: str = "time",
    test: Union[str, list] = "all",
    trend: Literal["c", "ct"] = "c",
    alpha: float = 0.05,
    **kwargs,
) -> PanelUnitRootResult:
    """
    Run multiple panel unit root tests and compare results.

    This function provides a unified interface to run various panel unit root
    tests and generate a comparative summary.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    variable : str
        Name of the variable to test.
    entity_col : str, default 'entity'
        Name of the entity (cross-section) identifier column.
    time_col : str, default 'time'
        Name of the time identifier column.
    test : str or list, default 'all'
        Which test(s) to run. Options:
        - 'all': Run all available tests
        - 'hadri': Hadri (2000) LM test (H0: stationarity)
        - 'breitung': Breitung (2000) test (H0: unit root)
        - 'ips': Im-Pesaran-Shin (2003) test (H0: unit root)
        - 'llc': Levin-Lin-Chu (2002) test (H0: unit root)
        - List of test names, e.g., ['hadri', 'breitung']
    trend : {'c', 'ct'}, default 'c'
        Deterministic specification:
        - 'c': constant only
        - 'ct': constant and linear trend
    alpha : float, default 0.05
        Significance level for hypothesis tests.
    **kwargs
        Additional arguments passed to individual test functions.

    Returns
    -------
    PanelUnitRootResult
        Combined results from all requested tests.

    Notes
    -----
    Different tests have different null hypotheses:

    H0: Unit root (reject → evidence of stationarity):
        - Breitung (2000): Robust to heterogeneity
        - IPS (2003): Allows heterogeneous AR coefficients
        - LLC (2002): Assumes common AR coefficient

    H0: Stationarity (reject → evidence of unit root):
        - Hadri (2000): LM test based on KPSS

    When tests disagree, consider:
    1. Power differences across tests
    2. Assumptions about heterogeneity
    3. Sample size and balanced/unbalanced panel
    4. Economic theory and context

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from panelbox.diagnostics.unit_root import panel_unit_root_test
    >>>
    >>> # Generate panel data
    >>> np.random.seed(42)
    >>> data = []
    >>> for i in range(10):
    ...     y = np.random.randn(100).cumsum()
    ...     for t, val in enumerate(y):
    ...         data.append({'entity': i, 'time': t, 'y': val})
    >>> df = pd.DataFrame(data)
    >>>
    >>> # Run all available tests
    >>> result = panel_unit_root_test(df, 'y', test='all', trend='c')
    >>> print(result.summary_table())
    >>>
    >>> # Run specific tests
    >>> result = panel_unit_root_test(df, 'y', test=['hadri', 'breitung'])
    >>> print(result.interpretation())
    """
    # Determine which tests to run
    available_tests = ["hadri", "breitung"]
    if HAS_IPS:
        available_tests.append("ips")
    if HAS_LLC:
        available_tests.append("llc")

    if test == "all":
        tests_to_run = available_tests
    elif isinstance(test, str):
        tests_to_run = [test]
    else:
        tests_to_run = list(test)

    # Validate requested tests
    for t in tests_to_run:
        if t not in available_tests:
            if t in ["ips", "llc"]:
                raise ImportError(
                    f"Test '{t}' is not available. "
                    "Make sure the validation.unit_root module is accessible."
                )
            else:
                raise ValueError(f"Unknown test '{t}'. Available tests: {available_tests}")

    # Run tests
    results = {}

    if "hadri" in tests_to_run:
        results["hadri"] = hadri_test(
            data, variable, entity_col, time_col, trend=trend, alpha=alpha, **kwargs
        )

    if "breitung" in tests_to_run:
        results["breitung"] = breitung_test(
            data, variable, entity_col, time_col, trend=trend, alpha=alpha, **kwargs
        )

    if "ips" in tests_to_run and HAS_IPS:
        results["ips"] = ips_test(
            data, variable, entity_col, time_col, trend=trend, alpha=alpha, **kwargs
        )

    if "llc" in tests_to_run and HAS_LLC:
        results["llc"] = llc_test(
            data, variable, entity_col, time_col, trend=trend, alpha=alpha, **kwargs
        )

    # Get panel dimensions from first result
    first_result = results[tests_to_run[0]]
    n_entities = first_result.n_entities
    n_time = first_result.n_time

    return PanelUnitRootResult(
        results=results,
        variable=variable,
        n_entities=n_entities,
        n_time=n_time,
        tests_run=tests_to_run,
    )
