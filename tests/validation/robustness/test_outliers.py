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
def test_plot_diagnostics(mock_results):
    """Test plotting diagnostics."""
    pytest.importorskip("matplotlib")

    detector = OutlierDetector(mock_results, verbose=False)

    with patch("matplotlib.pyplot.show"):
        detector.plot_diagnostics()


def test_plot_diagnostics_save(mock_results, tmp_path):
    """Test saving diagnostic plots."""
    pytest.importorskip("matplotlib")

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


class TestUnivariateOnVariable:
    """Test univariate detection on specific variables."""

    def test_iqr_on_x1(self, mock_results):
        """Test IQR on x1 variable."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_univariate(variable="x1", method="iqr")

        assert "x1" in results.method
        assert len(results.outliers) == len(mock_results.resid)

    def test_zscore_on_x2(self, mock_results):
        """Test Z-score on x2 variable."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_univariate(variable="x2", method="zscore", threshold=3.0)

        assert "x2" in results.method
        assert results.threshold == 3.0


class TestThresholdBehavior:
    """Test different threshold values."""

    def test_iqr_strict_threshold(self, mock_results):
        """Test that higher IQR threshold finds fewer outliers."""
        detector = OutlierDetector(mock_results, verbose=False)

        results_1_5 = detector.detect_outliers_univariate(method="iqr", threshold=1.5)
        results_3_0 = detector.detect_outliers_univariate(method="iqr", threshold=3.0)

        assert results_3_0.n_outliers <= results_1_5.n_outliers

    def test_zscore_strict_threshold(self, mock_results):
        """Test that higher Z-score threshold finds fewer outliers."""
        detector = OutlierDetector(mock_results, verbose=False)

        results_2 = detector.detect_outliers_univariate(method="zscore", threshold=2.0)
        results_3 = detector.detect_outliers_univariate(method="zscore", threshold=3.0)

        assert results_3.n_outliers <= results_2.n_outliers


class TestSingularCovariance:
    """Test handling of singular covariance matrices."""

    def test_multivariate_with_singular_cov_warning(self, mock_results):
        """Test that singular covariance triggers warning."""
        detector = OutlierDetector(mock_results, verbose=False)

        # Force singular covariance by patching
        with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError):
            with pytest.warns(UserWarning, match="singular"):
                detector.detect_outliers_multivariate()

    def test_leverage_with_singular_cov_warning(self, mock_results):
        """Test that singular covariance in leverage triggers warning."""
        detector = OutlierDetector(mock_results, verbose=False)

        with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError):
            with pytest.warns(UserWarning, match="pseudo-inverse"):
                detector.detect_leverage_points()


class TestSummaryFormats:
    """Test summary output formats."""

    def test_summary_with_many_outliers(self):
        """Test summary with more than 10 outliers."""
        outliers_df = pd.DataFrame(
            {
                "entity": list(range(20)),
                "time": [0] * 20,
                "value": [float(i) for i in range(20)],
                "is_outlier": [True] * 20,
                "distance": [float(i) for i in range(20)],
            }
        )

        results = OutlierResults(outliers=outliers_df, method="Test", threshold=1.0, n_outliers=20)

        summary = results.summary()
        assert "Top 10 outliers" in summary
        # Should only show 10 even though there are 20

    def test_summary_percentage_calculation(self):
        """Test that percentage is calculated correctly."""
        outliers_df = pd.DataFrame(
            {
                "entity": [0, 1, 2, 3],
                "time": [0, 0, 0, 0],
                "value": [1.0, 2.0, 3.0, 4.0],
                "is_outlier": [True, False, False, False],
                "distance": [1.0, 0.5, 0.3, 0.2],
            }
        )

        results = OutlierResults(outliers=outliers_df, method="Test", threshold=1.0, n_outliers=1)

        summary = results.summary()
        assert "25.00%" in summary  # 1 out of 4


class TestResidualMethods:
    """Test different residual-based methods."""

    def test_invalid_residual_method(self, mock_results):
        """Test that invalid residual method raises error."""
        detector = OutlierDetector(mock_results, verbose=False)

        with pytest.raises(ValueError, match="Unknown method"):
            detector.detect_outliers_residuals(method="invalid_method")

    def test_standardized_vs_studentized(self, mock_results):
        """Test that standardized and studentized give different results."""
        detector = OutlierDetector(mock_results, verbose=False)

        std_results = detector.detect_outliers_residuals(method="standardized")
        stu_results = detector.detect_outliers_residuals(method="studentized")

        # They should have different column names
        assert "standardized_residual" in std_results.outliers.columns
        assert "studentized_residual" in stu_results.outliers.columns


class TestDataFrameColumns:
    """Test that output DataFrames have correct columns."""

    def test_iqr_columns(self, mock_results):
        """Test IQR result columns."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_univariate(method="iqr")

        expected_cols = {"entity", "time", "value", "is_outlier", "distance"}
        assert expected_cols.issubset(set(results.outliers.columns))

    def test_zscore_columns(self, mock_results):
        """Test Z-score result columns."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_univariate(method="zscore")

        expected_cols = {"entity", "time", "value", "is_outlier", "distance"}
        assert expected_cols.issubset(set(results.outliers.columns))

    def test_mahalanobis_columns(self, mock_results):
        """Test Mahalanobis result columns."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_multivariate()

        expected_cols = {"entity", "time", "mahalanobis_distance", "is_outlier", "distance"}
        assert expected_cols.issubset(set(results.outliers.columns))


class TestNonNegativeDistances:
    """Test that distances are non-negative."""

    def test_iqr_distance_non_negative(self, mock_results):
        """Test IQR distance is non-negative."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_univariate(method="iqr")

        assert (results.outliers["distance"] >= 0).all()

    def test_zscore_distance_non_negative(self, mock_results):
        """Test Z-score distance is non-negative."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_univariate(method="zscore")

        assert (results.outliers["distance"] >= 0).all()

    def test_mahalanobis_distance_non_negative(self, mock_results):
        """Test Mahalanobis distance is non-negative."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_multivariate()

        assert (results.outliers["mahalanobis_distance"] >= 0).all()


class TestOutlierCountConsistency:
    """Test that n_outliers matches actual count."""

    def test_iqr_count_matches(self, mock_results):
        """Test IQR outlier count."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_univariate(method="iqr")

        assert results.n_outliers == results.outliers["is_outlier"].sum()

    def test_zscore_count_matches(self, mock_results):
        """Test Z-score outlier count."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_univariate(method="zscore")

        assert results.n_outliers == results.outliers["is_outlier"].sum()

    def test_mahalanobis_count_matches(self, mock_results):
        """Test Mahalanobis outlier count."""
        detector = OutlierDetector(mock_results, verbose=False)
        results = detector.detect_outliers_multivariate()

        assert results.n_outliers == results.outliers["is_outlier"].sum()


class TestVerboseMessages:
    """Test verbose output messages."""

    def test_verbose_iqr(self, mock_results, capsys):
        """Test verbose output for IQR."""
        detector = OutlierDetector(mock_results, verbose=True)
        detector.detect_outliers_univariate(method="iqr")

        captured = capsys.readouterr()
        assert "Detected" in captured.out
        assert "IQR" in captured.out

    def test_verbose_mahalanobis(self, mock_results, capsys):
        """Test verbose output for Mahalanobis."""
        detector = OutlierDetector(mock_results, verbose=True)
        detector.detect_outliers_multivariate()

        captured = capsys.readouterr()
        assert "Detected" in captured.out
        assert "Mahalanobis" in captured.out

    def test_verbose_leverage(self, mock_results, capsys):
        """Test verbose output for leverage."""
        detector = OutlierDetector(mock_results, verbose=True)
        detector.detect_leverage_points()

        captured = capsys.readouterr()
        assert "Detected" in captured.out
        assert "leverage" in captured.out


class TestLeverageDefaultThreshold:
    """Test default leverage threshold."""

    def test_default_threshold_calculation(self, mock_results):
        """Test that default threshold is 2*k/n."""
        detector = OutlierDetector(mock_results, verbose=False)
        leverage_df = detector.detect_leverage_points()

        n = len(mock_results.resid)
        k = len(mock_results.params)
        expected_threshold = 2 * k / n

        # Verify by checking with explicit threshold
        leverage_df2 = detector.detect_leverage_points(threshold=expected_threshold)

        assert (leverage_df["is_high_leverage"] == leverage_df2["is_high_leverage"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
