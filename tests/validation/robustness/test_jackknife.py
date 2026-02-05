"""
Tests for jackknife module.

This test suite validates the PanelJackknife class for panel data models.
"""

import pytest
import numpy as np
import pandas as pd

from panelbox.validation.robustness.jackknife import PanelJackknife, JackknifeResults


# Fixtures
@pytest.fixture
def simple_panel_data():
    """Create simple panel data for testing."""
    np.random.seed(42)
    n_entities = 15
    n_periods = 6

    data = []
    for entity in range(n_entities):
        for time in range(n_periods):
            x1 = np.random.normal(0, 1)
            x2 = np.random.normal(0, 1)
            y = 2.0 + 1.5 * x1 - 1.0 * x2 + np.random.normal(0, 0.5)

            data.append({
                'entity': entity,
                'time': time,
                'y': y,
                'x1': x1,
                'x2': x2
            })

    return pd.DataFrame(data)


@pytest.fixture
def mock_results(simple_panel_data):
    """Create mock PanelResults for testing."""
    from panelbox import FixedEffects

    fe = FixedEffects("y ~ x1 + x2", simple_panel_data, "entity", "time")
    results = fe.fit()

    return results


# Test Initialization
def test_init(mock_results):
    """Test PanelJackknife initialization."""
    jk = PanelJackknife(mock_results, verbose=False)

    assert jk.results is mock_results
    assert jk.verbose is False
    assert jk.n_entities > 0
    assert len(jk.entities) == jk.n_entities


def test_init_attributes(mock_results):
    """Test that initialization sets correct attributes."""
    jk = PanelJackknife(mock_results, verbose=False)

    assert hasattr(jk, 'model')
    assert hasattr(jk, 'formula')
    assert hasattr(jk, 'entity_col')
    assert hasattr(jk, 'time_col')
    assert hasattr(jk, 'data')
    assert hasattr(jk, 'entities')


# Test Jackknife Procedure
def test_run_jackknife(mock_results):
    """Test running jackknife procedure."""
    jk = PanelJackknife(mock_results, verbose=False)

    jk_results = jk.run()

    # Check that results object is created
    assert isinstance(jk_results, JackknifeResults)
    assert jk.jackknife_results_ is not None


def test_jackknife_results_structure(mock_results):
    """Test structure of jackknife results."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk_results = jk.run()

    # Check all required attributes
    assert hasattr(jk_results, 'jackknife_estimates')
    assert hasattr(jk_results, 'original_estimates')
    assert hasattr(jk_results, 'jackknife_mean')
    assert hasattr(jk_results, 'jackknife_bias')
    assert hasattr(jk_results, 'jackknife_se')
    assert hasattr(jk_results, 'influence')
    assert hasattr(jk_results, 'n_jackknife')


def test_jackknife_sample_count(mock_results):
    """Test that jackknife creates correct number of samples."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk_results = jk.run()

    # Number of jackknife samples should equal number of entities
    assert jk_results.n_jackknife <= jk.n_entities

    # Jackknife estimates should have N rows
    assert len(jk_results.jackknife_estimates) == jk_results.n_jackknife


def test_jackknife_parameter_names(mock_results):
    """Test that jackknife preserves parameter names."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk_results = jk.run()

    # Parameter names should match original
    assert list(jk_results.original_estimates.index) == list(jk_results.jackknife_mean.index)
    assert list(jk_results.original_estimates.index) == list(jk_results.jackknife_bias.index)
    assert list(jk_results.original_estimates.index) == list(jk_results.jackknife_se.index)


# Test Bias Estimation
def test_bias_corrected_estimates(mock_results):
    """Test bias-corrected estimates."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk.run()

    bias_corrected = jk.bias_corrected_estimates()

    # Should be a Series
    assert isinstance(bias_corrected, pd.Series)

    # Should have same parameters as original
    assert list(bias_corrected.index) == list(jk.results.params.index)

    # Bias correction formula: original - bias
    expected = jk.jackknife_results_.original_estimates - jk.jackknife_results_.jackknife_bias
    pd.testing.assert_series_equal(bias_corrected, expected)


def test_bias_corrected_before_run(mock_results):
    """Test that bias_corrected_estimates raises error before run()."""
    jk = PanelJackknife(mock_results, verbose=False)

    with pytest.raises(RuntimeError, match="Must call run"):
        jk.bias_corrected_estimates()


# Test Confidence Intervals
def test_confidence_intervals_normal(mock_results):
    """Test confidence intervals using normal approximation."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk.run()

    ci = jk.confidence_intervals(alpha=0.05, method='normal')

    # Should be DataFrame with lower and upper
    assert isinstance(ci, pd.DataFrame)
    assert 'lower' in ci.columns
    assert 'upper' in ci.columns

    # Upper should be greater than lower
    assert (ci['upper'] > ci['lower']).all()


def test_confidence_intervals_percentile(mock_results):
    """Test confidence intervals using percentile method."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk.run()

    ci = jk.confidence_intervals(alpha=0.05, method='percentile')

    assert isinstance(ci, pd.DataFrame)
    assert 'lower' in ci.columns
    assert 'upper' in ci.columns
    assert (ci['upper'] >= ci['lower']).all()


def test_confidence_intervals_invalid_method(mock_results):
    """Test that invalid method raises error."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk.run()

    with pytest.raises(ValueError, match="Unknown method"):
        jk.confidence_intervals(method='invalid')


def test_confidence_intervals_before_run(mock_results):
    """Test that confidence_intervals raises error before run()."""
    jk = PanelJackknife(mock_results, verbose=False)

    with pytest.raises(RuntimeError, match="Must call run"):
        jk.confidence_intervals()


# Test Influential Entities
def test_influential_entities(mock_results):
    """Test identification of influential entities."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk.run()

    influential = jk.influential_entities(threshold=2.0, metric='max')

    # Should be DataFrame
    assert isinstance(influential, pd.DataFrame)

    # Should have required columns
    assert 'entity' in influential.columns
    assert 'influence' in influential.columns
    assert 'threshold' in influential.columns


def test_influential_entities_metrics(mock_results):
    """Test different metrics for influential entities."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk.run()

    # Test all metrics
    infl_max = jk.influential_entities(metric='max')
    infl_mean = jk.influential_entities(metric='mean')
    infl_sum = jk.influential_entities(metric='sum')

    # All should be DataFrames
    assert isinstance(infl_max, pd.DataFrame)
    assert isinstance(infl_mean, pd.DataFrame)
    assert isinstance(infl_sum, pd.DataFrame)


def test_influential_entities_invalid_metric(mock_results):
    """Test that invalid metric raises error."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk.run()

    with pytest.raises(ValueError, match="Unknown metric"):
        jk.influential_entities(metric='invalid')


def test_influential_entities_before_run(mock_results):
    """Test that influential_entities raises error before run()."""
    jk = PanelJackknife(mock_results, verbose=False)

    with pytest.raises(RuntimeError, match="Must call run"):
        jk.influential_entities()


# Test Summary
def test_summary(mock_results):
    """Test summary generation."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk.run()

    summary = jk.summary()

    assert isinstance(summary, str)
    assert 'Jackknife Results' in summary
    assert 'Parameter Estimates' in summary
    assert 'Influential Entities' in summary


def test_summary_before_run(mock_results):
    """Test that summary raises error before run()."""
    jk = PanelJackknife(mock_results, verbose=False)

    with pytest.raises(RuntimeError, match="Must call run"):
        jk.summary()


def test_jackknife_results_summary(mock_results):
    """Test JackknifeResults summary method."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk_results = jk.run()

    summary = jk_results.summary()

    assert isinstance(summary, str)
    assert 'Jackknife Results' in summary


# Test Statistical Properties
def test_jackknife_bias_formula(mock_results):
    """Test that jackknife bias is computed correctly."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk_results = jk.run()

    N = jk_results.n_jackknife

    # Bias = (N-1) * (mean_jackknife - original)
    expected_bias = (N - 1) * (jk_results.jackknife_mean - jk_results.original_estimates)

    pd.testing.assert_series_equal(
        jk_results.jackknife_bias,
        expected_bias,
        check_names=False
    )


def test_jackknife_se_positive(mock_results):
    """Test that jackknife standard errors are positive."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk_results = jk.run()

    # All SE should be positive
    assert (jk_results.jackknife_se > 0).all()


def test_jackknife_influence_shape(mock_results):
    """Test shape of influence matrix."""
    jk = PanelJackknife(mock_results, verbose=False)
    jk_results = jk.run()

    # Influence should have N rows (entities) and k columns (parameters)
    assert jk_results.influence.shape[0] == jk_results.n_jackknife
    assert jk_results.influence.shape[1] == len(jk_results.original_estimates)


# Test Edge Cases
def test_jackknife_small_sample():
    """Test jackknife with very small sample."""
    np.random.seed(42)

    # Create small panel (5 entities, 3 periods)
    data = []
    for entity in range(5):
        for time in range(3):
            x1 = np.random.normal(0, 1)
            y = 2.0 + 1.5 * x1 + np.random.normal(0, 0.5)
            data.append({'entity': entity, 'time': time, 'y': y, 'x1': x1})

    df = pd.DataFrame(data)

    from panelbox import FixedEffects
    fe = FixedEffects("y ~ x1", df, "entity", "time")
    results = fe.fit()

    jk = PanelJackknife(results, verbose=False)
    jk_results = jk.run()

    # Should still complete
    assert jk_results.n_jackknife > 0


def test_jackknife_reproducibility(mock_results):
    """Test that jackknife is reproducible."""
    jk1 = PanelJackknife(mock_results, verbose=False)
    results1 = jk1.run()

    jk2 = PanelJackknife(mock_results, verbose=False)
    results2 = jk2.run()

    # Results should be identical
    pd.testing.assert_series_equal(results1.jackknife_mean, results2.jackknife_mean)
    pd.testing.assert_series_equal(results1.jackknife_bias, results2.jackknife_bias)
    pd.testing.assert_series_equal(results1.jackknife_se, results2.jackknife_se)


# Performance Tests
def test_jackknife_performance(mock_results):
    """Test that jackknife completes in reasonable time."""
    import time

    jk = PanelJackknife(mock_results, verbose=False)

    start = time.time()
    jk.run()
    elapsed = time.time() - start

    # Should complete in less than 10 seconds for small dataset
    assert elapsed < 10.0


# Integration Tests
def test_jackknife_integration_full_workflow(mock_results):
    """Test complete jackknife workflow."""
    # Create jackknife object
    jk = PanelJackknife(mock_results, verbose=False)

    # Run jackknife
    jk_results = jk.run()

    # Check results
    assert jk_results.n_jackknife > 0
    assert len(jk_results.jackknife_estimates) > 0

    # Get bias-corrected estimates
    bias_corrected = jk.bias_corrected_estimates()
    assert len(bias_corrected) == len(mock_results.params)

    # Get confidence intervals
    ci_normal = jk.confidence_intervals(method='normal')
    ci_percentile = jk.confidence_intervals(method='percentile')

    assert len(ci_normal) == len(mock_results.params)
    assert len(ci_percentile) == len(mock_results.params)

    # Get influential entities
    influential = jk.influential_entities()

    # Generate summary
    summary = jk.summary()
    assert 'Jackknife Results' in summary


def test_jackknife_different_models():
    """Test jackknife with different model types."""
    np.random.seed(42)
    n_entities = 10
    n_periods = 5

    data = []
    for entity in range(n_entities):
        for time in range(n_periods):
            x1 = np.random.normal(0, 1)
            y = 2.0 + 1.5 * x1 + np.random.normal(0, 0.5)
            data.append({'entity': entity, 'time': time, 'y': y, 'x1': x1})

    df = pd.DataFrame(data)

    # Test with Pooled OLS
    from panelbox import PooledOLS
    pooled = PooledOLS("y ~ x1", df, "entity", "time")
    pooled_results = pooled.fit()

    jk_pooled = PanelJackknife(pooled_results, verbose=False)
    jk_results_pooled = jk_pooled.run()

    assert jk_results_pooled.n_jackknife > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
