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


# ============================================================================
# Additional Coverage Tests for influence.py
# ============================================================================


def test_lazy_property_leverage(mock_results):
    """Test leverage property triggers compute() if not yet done."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    # Don't call compute() explicitly; property should auto-compute
    assert infl.influence_results_ is None
    leverage = infl.leverage
    assert infl.influence_results_ is not None
    assert isinstance(leverage, pd.Series)
    assert len(leverage) > 0


def test_lazy_property_cooks_d(mock_results):
    """Test cooks_d property triggers compute() if not yet done."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    assert infl.influence_results_ is None
    cooks_d = infl.cooks_d
    assert infl.influence_results_ is not None
    assert isinstance(cooks_d, pd.Series)
    assert len(cooks_d) > 0


def test_lazy_property_dffits(mock_results):
    """Test dffits property triggers compute() if not yet done."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    assert infl.influence_results_ is None
    dffits = infl.dffits
    assert infl.influence_results_ is not None
    assert isinstance(dffits, pd.Series)


def test_lazy_property_dfbetas(mock_results):
    """Test dfbetas property triggers compute() if not yet done."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    assert infl.influence_results_ is None
    dfbetas = infl.dfbetas
    assert infl.influence_results_ is not None
    assert isinstance(dfbetas, pd.DataFrame)


def test_influential_observations_auto_compute(mock_results):
    """Test that influential_observations triggers compute() if not yet done."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    assert infl.influence_results_ is None
    influential = infl.influential_observations(method="cooks_d")
    assert infl.influence_results_ is not None
    assert isinstance(influential, pd.DataFrame)


def test_summary_auto_compute(mock_results):
    """Test that summary triggers compute() if not yet done."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    assert infl.influence_results_ is None
    summary = infl.summary()
    assert infl.influence_results_ is not None
    assert isinstance(summary, str)
    assert "Influence Diagnostics" in summary


def test_influential_observations_custom_threshold_cooks_d(mock_results):
    """Test identifying influential obs with custom Cook's D threshold."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    # Very high threshold - should find fewer observations
    influential_strict = infl.influential_observations(method="cooks_d", threshold=1000.0)
    assert len(influential_strict) == 0

    # Very low threshold - should find more observations
    influential_loose = infl.influential_observations(method="cooks_d", threshold=0.0)
    assert len(influential_loose) > 0


def test_influential_observations_custom_threshold_dffits(mock_results):
    """Test identifying influential obs with custom DFFITS threshold."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    # Very high threshold
    influential = infl.influential_observations(method="dffits", threshold=1000.0)
    assert isinstance(influential, pd.DataFrame)
    assert len(influential) == 0


def test_influential_observations_custom_threshold_dfbetas(mock_results):
    """Test identifying influential obs with custom DFBETAS threshold."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    # Very high threshold - no influential obs
    influential = infl.influential_observations(method="dfbetas", threshold=1000.0)
    assert isinstance(influential, pd.DataFrame)
    assert len(influential) == 0


def test_influence_results_summary_details(mock_results):
    """Test InfluenceResults summary contains expected details."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    results = infl.compute()

    summary = results.summary(n_top=5)
    assert "Cook's Distance:" in summary
    assert "DFFITS:" in summary
    assert "Top 5 observations" in summary
    assert "influential obs" in summary


def test_compute_verbose_logging(mock_results, caplog):
    """Test compute with verbose=True triggers logging."""
    import logging

    with caplog.at_level(logging.INFO, logger="panelbox.validation.robustness.influence"):
        infl = InfluenceDiagnostics(mock_results, verbose=True)
        infl.compute()

    assert infl.influence_results_ is not None


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib required",
)
def test_plot_influence_auto_compute(mock_results):
    """Test that plot_influence triggers compute() if not yet done."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    assert infl.influence_results_ is None

    with patch("matplotlib.pyplot.show"):
        infl.plot_influence()

    assert infl.influence_results_ is not None


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib required",
)
def test_plot_influence_save(mock_results, tmp_path):
    """Test saving influence plot to file."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    save_path = tmp_path / "influence_plot.png"
    infl.plot_influence(save_path=str(save_path))

    assert save_path.exists()


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib required",
)
def test_plot_influence_save_verbose(mock_results, tmp_path, caplog):
    """Test saving influence plot with verbose logging."""
    import logging

    infl = InfluenceDiagnostics(mock_results, verbose=True)
    infl.compute()

    save_path = tmp_path / "influence_verbose_plot.png"
    with caplog.at_level(logging.INFO, logger="panelbox.validation.robustness.influence"):
        infl.plot_influence(save_path=str(save_path))

    assert save_path.exists()


def test_plot_without_matplotlib(mock_results):
    """Test that plot_influence without matplotlib gives helpful error."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    with (
        patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}),
        pytest.raises(ImportError, match="matplotlib required"),
    ):
        infl.plot_influence()


def test_leverage_values_in_range(mock_results):
    """Test that leverage values are clipped to valid range [0, 1]."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    results = infl.compute()

    assert np.all(results.leverage >= 0)
    assert np.all(results.leverage <= 1)


def test_cooks_d_nonnegative(mock_results):
    """Test that Cook's distance values are non-negative."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    results = infl.compute()

    assert np.all(results.cooks_d >= 0)


def test_dfbetas_shape(mock_results):
    """Test that DFBETAS has correct shape (n_obs x n_params)."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    results = infl.compute()

    n = len(mock_results.resid)
    k = len(mock_results.params)
    assert results.dfbetas.shape == (n, k)


def test_influential_observations_dffits_columns(mock_results):
    """Test DFFITS influential obs DataFrame has correct columns."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    influential = infl.influential_observations(method="dffits", threshold=0.0)
    assert "observation" in influential.columns
    assert "dffits" in influential.columns
    assert "threshold" in influential.columns


def test_influential_observations_dfbetas_columns(mock_results):
    """Test DFBETAS influential obs DataFrame has correct columns."""
    infl = InfluenceDiagnostics(mock_results, verbose=False)
    infl.compute()

    influential = infl.influential_observations(method="dfbetas", threshold=0.0)
    assert "observation" in influential.columns
    assert "max_dfbetas" in influential.columns
    assert "threshold" in influential.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
