"""
Tests for time-series cross-validation module.

This test suite validates the TimeSeriesCV class for panel data models.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from panelbox.validation.robustness.cross_validation import TimeSeriesCV, CVResults


# Fixtures
@pytest.fixture
def simple_panel_data():
    """Create simple panel data for testing."""
    np.random.seed(42)
    n_entities = 10
    n_periods = 8

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

    # Fit a simple model
    fe = FixedEffects("y ~ x1 + x2", simple_panel_data, "entity", "time")
    results = fe.fit()

    return results


# Test Initialization
def test_init_expanding(mock_results):
    """Test TimeSeriesCV initialization with expanding window."""
    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)

    assert cv.method == 'expanding'
    assert cv.window_size is None
    assert cv.min_train_periods == 3
    assert cv.verbose is False


def test_init_rolling(mock_results):
    """Test TimeSeriesCV initialization with rolling window."""
    cv = TimeSeriesCV(
        mock_results,
        method='rolling',
        window_size=4,
        verbose=False
    )

    assert cv.method == 'rolling'
    assert cv.window_size == 4


def test_init_invalid_method(mock_results):
    """Test that invalid method raises error."""
    with pytest.raises(ValueError, match="method must be"):
        TimeSeriesCV(mock_results, method='invalid')


def test_init_rolling_without_window(mock_results):
    """Test that rolling without window_size raises error."""
    with pytest.raises(ValueError, match="window_size must be specified"):
        TimeSeriesCV(mock_results, method='rolling', verbose=False)


def test_init_invalid_min_periods(mock_results):
    """Test that invalid min_train_periods raises error."""
    with pytest.raises(ValueError, match="min_train_periods must be at least 2"):
        TimeSeriesCV(mock_results, method='expanding', min_train_periods=1)


# Test CV Fold Generation
def test_get_cv_folds_expanding(mock_results):
    """Test expanding window fold generation."""
    cv = TimeSeriesCV(mock_results, method='expanding', min_train_periods=3, verbose=False)
    folds = cv._get_cv_folds()

    # Should have n_periods - min_train_periods folds
    assert len(folds) == 8 - 3  # 5 folds

    # First fold: train on [0, 1, 2], test on 3
    train_periods, test_period = folds[0]
    assert len(train_periods) == 3
    assert test_period == 3

    # Last fold: train on [0, 1, 2, 3, 4, 5, 6], test on 7
    train_periods, test_period = folds[-1]
    assert len(train_periods) == 7
    assert test_period == 7


def test_get_cv_folds_rolling(mock_results):
    """Test rolling window fold generation."""
    cv = TimeSeriesCV(
        mock_results,
        method='rolling',
        window_size=3,
        min_train_periods=3,
        verbose=False
    )
    folds = cv._get_cv_folds()

    # Should have multiple folds
    assert len(folds) > 0

    # Check that window size is respected
    for train_periods, test_period in folds:
        assert len(train_periods) <= 3


# Test Metrics Computation
def test_compute_metrics():
    """Test metrics computation."""
    cv = TimeSeriesCV.__new__(TimeSeriesCV)

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    metrics = cv._compute_metrics(y_true, y_pred)

    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2_oos' in metrics

    # Check that RMSE is sqrt of MSE
    assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']))

    # Check that R² is reasonable
    assert -1 <= metrics['r2_oos'] <= 1


def test_compute_metrics_perfect():
    """Test metrics with perfect predictions."""
    cv = TimeSeriesCV.__new__(TimeSeriesCV)

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    metrics = cv._compute_metrics(y_true, y_pred)

    assert np.isclose(metrics['mse'], 0.0)
    assert np.isclose(metrics['rmse'], 0.0)
    assert np.isclose(metrics['mae'], 0.0)
    assert np.isclose(metrics['r2_oos'], 1.0)


# Test Cross-Validation
def test_cross_validate_expanding(mock_results):
    """Test expanding window cross-validation."""
    cv = TimeSeriesCV(mock_results, method='expanding', min_train_periods=3, verbose=False)

    cv_results = cv.cross_validate()

    # Check that results object is created
    assert isinstance(cv_results, CVResults)
    assert cv_results.method == 'expanding'
    assert cv_results.n_folds > 0

    # Check predictions
    assert len(cv_results.predictions) > 0
    assert 'actual' in cv_results.predictions.columns
    assert 'predicted' in cv_results.predictions.columns
    assert 'fold' in cv_results.predictions.columns

    # Check metrics
    assert 'mse' in cv_results.metrics
    assert 'rmse' in cv_results.metrics
    assert 'mae' in cv_results.metrics
    assert 'r2_oos' in cv_results.metrics

    # Check fold metrics
    assert len(cv_results.fold_metrics) == cv_results.n_folds


def test_cross_validate_rolling(mock_results):
    """Test rolling window cross-validation."""
    cv = TimeSeriesCV(
        mock_results,
        method='rolling',
        window_size=4,
        min_train_periods=3,
        verbose=False
    )

    cv_results = cv.cross_validate()

    assert isinstance(cv_results, CVResults)
    assert cv_results.method == 'rolling'
    assert cv_results.window_size == 4
    assert cv_results.n_folds > 0


def test_cross_validate_stores_attributes(mock_results):
    """Test that cross-validation stores results in attributes."""
    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)

    cv_results = cv.cross_validate()

    assert cv.cv_results_ is not None
    assert cv.predictions_ is not None
    assert cv.metrics_ is not None

    # Check that stored results match returned results
    assert cv.cv_results_ is cv_results
    pd.testing.assert_frame_equal(cv.predictions_, cv_results.predictions)
    assert cv.metrics_ == cv_results.metrics


# Test CVResults
def test_cv_results_summary():
    """Test CVResults summary generation."""
    predictions = pd.DataFrame({
        'actual': [1.0, 2.0, 3.0],
        'predicted': [1.1, 1.9, 3.1],
        'fold': [1, 1, 2]
    })

    fold_metrics = pd.DataFrame({
        'fold': [1, 2],
        'mse': [0.01, 0.01],
        'rmse': [0.1, 0.1],
        'mae': [0.1, 0.1],
        'r2_oos': [0.99, 0.99]
    })

    metrics = {
        'mse': 0.01,
        'rmse': 0.1,
        'mae': 0.1,
        'r2_oos': 0.99
    }

    cv_results = CVResults(
        predictions=predictions,
        metrics=metrics,
        fold_metrics=fold_metrics,
        method='expanding',
        n_folds=2
    )

    summary = cv_results.summary()

    assert 'Cross-Validation Results' in summary
    assert 'expanding' in summary.lower()
    assert 'Overall Metrics:' in summary
    assert 'Per-Fold Metrics:' in summary


# Test Summary Method
def test_summary_before_cv(mock_results):
    """Test that summary() raises error before cross_validate()."""
    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)

    with pytest.raises(RuntimeError, match="Must call cross_validate"):
        cv.summary()


def test_summary_after_cv(mock_results):
    """Test summary generation after cross-validation."""
    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)
    cv.cross_validate()

    summary = cv.summary()

    assert isinstance(summary, str)
    assert 'Cross-Validation Results' in summary
    assert 'Overall Metrics:' in summary


# Test Plotting
def test_plot_before_cv(mock_results):
    """Test that plot_predictions() raises error before cross_validate()."""
    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)

    with pytest.raises(RuntimeError, match="Must call cross_validate"):
        cv.plot_predictions()


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib required for plotting tests"
)
def test_plot_predictions(mock_results):
    """Test plotting predictions."""
    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)
    cv.cross_validate()

    # Test that plotting doesn't raise error
    with patch('matplotlib.pyplot.show'):
        cv.plot_predictions()


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib required for plotting tests"
)
def test_plot_predictions_save(mock_results, tmp_path):
    """Test saving plot to file."""
    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)
    cv.cross_validate()

    save_path = tmp_path / "cv_plot.png"
    cv.plot_predictions(save_path=str(save_path))

    assert save_path.exists()


def test_plot_without_matplotlib(mock_results):
    """Test that plotting without matplotlib gives helpful error."""
    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)
    cv.cross_validate()

    with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.pyplot': None}):
        with pytest.raises(ImportError, match="matplotlib is required"):
            cv.plot_predictions()


# Test Edge Cases
def test_cv_with_few_periods(simple_panel_data):
    """Test CV with very few time periods."""
    # Keep only 4 periods
    data_few = simple_panel_data[simple_panel_data['time'] < 4].copy()

    from panelbox import FixedEffects
    fe = FixedEffects("y ~ x1 + x2", data_few, "entity", "time")
    results = fe.fit()

    cv = TimeSeriesCV(results, method='expanding', min_train_periods=2, verbose=False)
    cv_results = cv.cross_validate()

    # Should have 2 folds (4 periods - 2 min_train = 2)
    assert cv_results.n_folds == 2


def test_cv_different_min_periods(mock_results):
    """Test CV with different min_train_periods."""
    cv1 = TimeSeriesCV(mock_results, method='expanding', min_train_periods=2, verbose=False)
    cv2 = TimeSeriesCV(mock_results, method='expanding', min_train_periods=4, verbose=False)

    folds1 = cv1._get_cv_folds()
    folds2 = cv2._get_cv_folds()

    # More min_periods means fewer folds
    assert len(folds1) > len(folds2)


# Test Reproducibility
def test_cv_reproducibility(mock_results):
    """Test that CV produces consistent results."""
    cv1 = TimeSeriesCV(mock_results, method='expanding', verbose=False)
    results1 = cv1.cross_validate()

    cv2 = TimeSeriesCV(mock_results, method='expanding', verbose=False)
    results2 = cv2.cross_validate()

    # Metrics should be identical
    assert results1.metrics['mse'] == results2.metrics['mse']
    assert results1.metrics['rmse'] == results2.metrics['rmse']
    assert results1.metrics['r2_oos'] == results2.metrics['r2_oos']


# Performance Tests
def test_cv_performance_reasonable(mock_results):
    """Test that CV completes in reasonable time."""
    import time

    cv = TimeSeriesCV(mock_results, method='expanding', verbose=False)

    start = time.time()
    cv.cross_validate()
    elapsed = time.time() - start

    # Should complete in less than 10 seconds for small dataset
    assert elapsed < 10.0


# Integration Tests
def test_cv_integration_full_workflow(mock_results):
    """Test complete CV workflow."""
    # Create CV object
    cv = TimeSeriesCV(mock_results, method='expanding', min_train_periods=3, verbose=False)

    # Run CV
    cv_results = cv.cross_validate()

    # Check results
    assert cv_results.n_folds > 0
    assert len(cv_results.predictions) > 0

    # Generate summary
    summary = cv.summary()
    assert 'Cross-Validation Results' in summary

    # Metrics should be reasonable
    assert cv_results.metrics['r2_oos'] > -1
    assert cv_results.metrics['rmse'] > 0
    assert cv_results.metrics['mae'] > 0


def test_cv_different_model_types():
    """Test CV with different model types (Pooled OLS, FE, RE)."""
    np.random.seed(42)
    n_entities = 15
    n_periods = 6

    data = []
    for entity in range(n_entities):
        for time in range(n_periods):
            x1 = np.random.normal(0, 1)
            y = 2.0 + 1.5 * x1 + np.random.normal(0, 0.5)

            data.append({
                'entity': entity,
                'time': time,
                'y': y,
                'x1': x1
            })

    df = pd.DataFrame(data)

    # Test with Pooled OLS
    from panelbox import PooledOLS
    pooled = PooledOLS("y ~ x1", df, "entity", "time")
    pooled_results = pooled.fit()

    cv_pooled = TimeSeriesCV(pooled_results, method='expanding', verbose=False)
    cv_results_pooled = cv_pooled.cross_validate()

    assert cv_results_pooled.n_folds > 0
    assert 'r2_oos' in cv_results_pooled.metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
