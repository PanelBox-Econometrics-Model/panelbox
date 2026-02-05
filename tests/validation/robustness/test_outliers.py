"""
Tests for outlier detection module.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.robustness.outliers import OutlierDetector, OutlierResults


@pytest.fixture
def simple_panel_data():
    """Create simple panel data with outliers."""
    np.random.seed(42)
    n_entities = 15
    n_periods = 6

    data = []
    for entity in range(n_entities):
        for time in range(n_periods):
            x1 = np.random.normal(0, 1)
            x2 = np.random.normal(0, 1)
            y = 2.0 + 1.5 * x1 - 1.0 * x2 + np.random.normal(0, 0.5)

            # Add some outliers
            if entity == 0 and time == 0:
                y += 10  # Outlier

            data.append({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})

    return pd.DataFrame(data)


@pytest.fixture
def mock_results(simple_panel_data):
    """Create mock PanelResults."""
    from panelbox import FixedEffects

    fe = FixedEffects("y ~ x1 + x2", simple_panel_data, "entity", "time")
    return fe.fit()


# Test Initialization
def test_init(mock_results):
    """Test OutlierDetector initialization."""
    detector = OutlierDetector(mock_results, verbose=False)
    assert detector.results is mock_results
    assert detector.verbose is False


# Test Univariate Detection
def test_detect_outliers_iqr(mock_results):
    """Test IQR method."""
    detector = OutlierDetector(mock_results, verbose=False)
    results = detector.detect_outliers_univariate(method="iqr", threshold=1.5)

    assert isinstance(results, OutlierResults)
    assert "is_outlier" in results.outliers.columns
    assert results.n_outliers >= 0


def test_detect_outliers_zscore(mock_results):
    """Test Z-score method."""
    detector = OutlierDetector(mock_results, verbose=False)
    results = detector.detect_outliers_univariate(method="zscore", threshold=2.5)

    assert isinstance(results, OutlierResults)
    assert results.n_outliers >= 0


def test_detect_outliers_invalid_method(mock_results):
    """Test invalid method raises error."""
    detector = OutlierDetector(mock_results, verbose=False)

    with pytest.raises(ValueError, match="Unknown method"):
        detector.detect_outliers_univariate(method="invalid")


# Test Multivariate Detection
def test_detect_outliers_multivariate(mock_results):
    """Test Mahalanobis distance method."""
    detector = OutlierDetector(mock_results, verbose=False)
    results = detector.detect_outliers_multivariate(threshold=3.0)

    assert isinstance(results, OutlierResults)
    assert "mahalanobis_distance" in results.outliers.columns
    assert results.n_outliers >= 0


# Test Residual-based Detection
def test_detect_outliers_standardized_residuals(mock_results):
    """Test standardized residuals method."""
    detector = OutlierDetector(mock_results, verbose=False)
    results = detector.detect_outliers_residuals(method="standardized", threshold=2.5)

    assert isinstance(results, OutlierResults)
    assert "standardized_residual" in results.outliers.columns


def test_detect_outliers_studentized_residuals(mock_results):
    """Test studentized residuals method."""
    detector = OutlierDetector(mock_results, verbose=False)
    results = detector.detect_outliers_residuals(method="studentized", threshold=2.5)

    assert isinstance(results, OutlierResults)
    assert "studentized_residual" in results.outliers.columns


# Test Leverage Detection
def test_detect_leverage_points(mock_results):
    """Test leverage point detection."""
    detector = OutlierDetector(mock_results, verbose=False)
    leverage_df = detector.detect_leverage_points()

    assert isinstance(leverage_df, pd.DataFrame)
    assert "leverage" in leverage_df.columns
    assert "is_high_leverage" in leverage_df.columns
    assert len(leverage_df) == len(mock_results.resid)


def test_detect_leverage_custom_threshold(mock_results):
    """Test leverage with custom threshold."""
    detector = OutlierDetector(mock_results, verbose=False)
    leverage_df = detector.detect_leverage_points(threshold=0.1)

    assert isinstance(leverage_df, pd.DataFrame)


# Test Results Summary
def test_outlier_results_summary():
    """Test OutlierResults summary."""
    outliers_df = pd.DataFrame(
        {
            "entity": [0, 1, 2],
            "time": [0, 1, 2],
            "value": [1.0, 5.0, 2.0],
            "is_outlier": [False, True, False],
            "distance": [0.5, 2.5, 0.3],
        }
    )

    results = OutlierResults(
        outliers=outliers_df, method="Test method", threshold=2.0, n_outliers=1
    )

    summary = results.summary()
    assert isinstance(summary, str)
    assert "Outlier Detection Results" in summary


# Test Plotting
@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib required for plotting tests",
)
def test_plot_diagnostics(mock_results):
    """Test plotting diagnostics."""
    detector = OutlierDetector(mock_results, verbose=False)

    with patch("matplotlib.pyplot.show"):
        detector.plot_diagnostics()


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib required for plotting tests",
)
def test_plot_diagnostics_save(mock_results, tmp_path):
    """Test saving diagnostic plots."""
    detector = OutlierDetector(mock_results, verbose=False)

    save_path = tmp_path / "diagnostics.png"
    detector.plot_diagnostics(save_path=str(save_path))

    assert save_path.exists()


def test_plot_without_matplotlib(mock_results):
    """Test that plotting without matplotlib gives error."""
    detector = OutlierDetector(mock_results, verbose=False)

    with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
        with pytest.raises(ImportError, match="matplotlib is required"):
            detector.plot_diagnostics()


# Integration Tests
def test_full_workflow(mock_results):
    """Test complete outlier detection workflow."""
    detector = OutlierDetector(mock_results, verbose=False)

    # Univariate
    iqr_results = detector.detect_outliers_univariate(method="iqr")
    assert iqr_results.n_outliers >= 0

    # Multivariate
    mahal_results = detector.detect_outliers_multivariate()
    assert mahal_results.n_outliers >= 0

    # Residuals
    resid_results = detector.detect_outliers_residuals()
    assert resid_results.n_outliers >= 0

    # Leverage
    leverage = detector.detect_leverage_points()
    assert len(leverage) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
