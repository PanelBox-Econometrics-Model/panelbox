"""Tests for robustness checks module."""

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.robustness.checks import RobustnessChecker


@pytest.fixture
def simple_panel_data():
    """Create panel data."""
    np.random.seed(42)
    data = []
    for entity in range(10):
        for time in range(5):
            x1, x2, x3 = np.random.normal(0, 1, 3)
            y = 2.0 + 1.5 * x1 - 1.0 * x2 + 0.5 * x3 + np.random.normal(0, 0.5)
            data.append({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2, "x3": x3})
    return pd.DataFrame(data)


@pytest.fixture
def mock_results(simple_panel_data):
    """Create mock results."""
    from panelbox import FixedEffects

    return FixedEffects("y ~ x1 + x2", simple_panel_data, "entity", "time").fit()


def test_init(mock_results):
    """Test initialization."""
    checker = RobustnessChecker(mock_results, verbose=False)
    assert checker.results is mock_results


def test_check_alternative_specs(mock_results):
    """Test alternative specifications."""
    checker = RobustnessChecker(mock_results, verbose=False)

    formulas = ["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + x3"]

    results_list = checker.check_alternative_specs(formulas)
    assert len(results_list) == 3
    assert all(r is not None for r in results_list)


def test_generate_robustness_table(mock_results):
    """Test robustness table generation."""
    checker = RobustnessChecker(mock_results, verbose=False)

    formulas = ["y ~ x1", "y ~ x1 + x2"]
    results_list = checker.check_alternative_specs(formulas)

    table = checker.generate_robustness_table(results_list)
    assert isinstance(table, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
