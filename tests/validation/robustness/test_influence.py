"""Tests for influence diagnostics module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.robustness.influence import InfluenceDiagnostics, InfluenceResults


@pytest.fixture
def simple_panel_data():
    """Create simple panel data."""
    np.random.seed(42)
    n_entities, n_periods = 15, 6

    data = []
    for entity in range(n_entities):
        for time in range(n_periods):
            x1, x2 = np.random.normal(0, 1, 2)
            y = 2.0 + 1.5 * x1 - 1.0 * x2 + np.random.normal(0, 0.5)
            data.append({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})

    return pd.DataFrame(data)


@pytest.fixture
def mock_results(simple_panel_data):
    """Create mock PanelResults."""
    from panelbox import FixedEffects

    return FixedEffects("y ~ x1 + x2", simple_panel_data, "entity", "time").fit()


def test_init(mock_results):
    """Test initialization."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    assert infl.results is mock_results


def test_compute(mock_results):
    """Test compute method."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    results = infl.compute()

    assert isinstance(results, InfluenceResults)
    assert hasattr(results, "cooks_d")
    assert hasattr(results, "dffits")
    assert hasattr(results, "dfbetas")
    assert hasattr(results, "leverage")


def test_influential_observations_cooks_d(mock_results):
    """Test identifying influential obs with Cook's D."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    influential = infl.influential_observations(method="cooks_d")
    assert isinstance(influential, pd.DataFrame)


def test_influential_observations_dffits(mock_results):
    """Test identifying influential obs with DFFITS."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    influential = infl.influential_observations(method="dffits")
    assert isinstance(influential, pd.DataFrame)


def test_influential_observations_dfbetas(mock_results):
    """Test identifying influential obs with DFBETAS."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    influential = infl.influential_observations(method="dfbetas")
    assert isinstance(influential, pd.DataFrame)


def test_summary(mock_results):
    """Test summary method."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    summary = infl.summary()
    assert isinstance(summary, str)
    assert "Influence Diagnostics" in summary


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib required",
)
def test_plot_influence(mock_results):
    """Test plotting."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    with patch("matplotlib.pyplot.show"):
        infl.plot_influence()


def test_invalid_method(mock_results):
    """Test invalid method raises error."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    with pytest.raises(ValueError, match="Unknown method"):
        infl.influential_observations(method="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
