"""
Tests for Quantile Monotonicity and Non-Crossing Constraints.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.quantile.monotonicity import (
    CrossingReport,
    MonotonicityComparison,
    QuantileMonotonicity,
)
from panelbox.models.quantile.pooled import PooledQuantile
from panelbox.utils.data import PanelData


class TestCrossingDetection:
    """Tests for crossing detection functionality."""

    @pytest.fixture
    def crossing_data(self):
        """Generate data that induces crossing."""
        np.random.seed(123)
        n = 200

        # Generate data with heteroskedasticity that causes crossing
        X = np.random.randn(n, 2)

        # Create y with strong interaction that causes crossing
        y = 2 + X[:, 0] - 2 * X[:, 1] + X[:, 0] * X[:, 1] * np.random.randn(n)

        return X, y

    def test_detect_crossing(self, crossing_data):
        """Test detection of crossing quantile curves."""
        X, y = crossing_data
        tau_list = [0.25, 0.5, 0.75]

        # Estimate quantiles independently (may cross)
        results = {}
        for tau in tau_list:
            # Create simple result object for testing
            from ...optimization.quantile.interior_point import frisch_newton_qr

            beta, _ = frisch_newton_qr(X, y, tau)

            # Mock result object
            class MockResult:
                def __init__(self, params, X):
                    self.params = params
                    self.model = type("Model", (), {"X": X, "nobs": len(X)})()

            results[tau] = MockResult(beta, X)

        # Detect crossing
        report = QuantileMonotonicity.detect_crossing(results, X[:50])

        # Check report structure
        assert isinstance(report, CrossingReport)
        assert hasattr(report, "has_crossing")
        assert hasattr(report, "crossings")
        assert hasattr(report, "total_inversions")
        assert hasattr(report, "pct_affected")

    def test_no_crossing_detection(self):
        """Test when there's no crossing."""
        np.random.seed(456)
        n = 100
        X = np.ones((n, 1))

        # Create perfectly ordered quantiles
        results = {}
        for i, tau in enumerate([0.25, 0.5, 0.75]):

            class MockResult:
                def __init__(self, params, X):
                    self.params = params
                    self.model = type("Model", (), {"X": X, "nobs": len(X)})()

            # Ensure monotonic coefficients
            results[tau] = MockResult(np.array([i]), X)

        report = QuantileMonotonicity.detect_crossing(results, X)

        assert report.has_crossing == False
        assert report.total_inversions == 0
        assert len(report.crossings) == 0


class TestRearrangement:
    """Tests for rearrangement method."""

    @pytest.fixture
    def crossed_results(self):
        """Generate crossed quantile results."""
        np.random.seed(789)
        n = 100
        X = np.random.randn(n, 2)

        # Create crossed coefficients
        results = {}
        coefficients = {
            0.25: np.array([1.0, 0.5]),
            0.5: np.array([0.8, 0.3]),  # Crosses with 0.25
            0.75: np.array([1.2, 0.6]),
        }

        for tau, beta in coefficients.items():

            class MockResult:
                def __init__(self, params, X):
                    self.params = params
                    self.model = type("Model", (), {"X": X})()

            results[tau] = MockResult(beta, X)

        return results, X

    def test_rearrangement(self, crossed_results):
        """Test rearrangement to fix crossing."""
        results, X = crossed_results

        # Apply rearrangement
        rearranged = QuantileMonotonicity.rearrangement(results, X)

        # Check non-crossing after rearrangement
        report = QuantileMonotonicity.detect_crossing(rearranged, X)
        assert report.has_crossing == False

        # Check that results are modified
        assert len(rearranged) == len(results)
        for tau in rearranged:
            assert hasattr(rearranged[tau], "rearranged")
            assert rearranged[tau].rearranged == True


class TestIsotonicRegression:
    """Tests for isotonic regression on coefficients."""

    def test_isotonic_coefficients(self):
        """Test isotonic regression on coefficient paths."""
        # Non-monotonic coefficient path
        tau_list = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        coef_matrix = np.array(
            [
                [1.0, 2.0],
                [1.5, 1.8],  # Decrease in second coef
                [2.0, 2.2],
                [2.3, 2.0],  # Decrease in second coef
                [2.5, 2.5],
            ]
        )

        # Apply isotonic regression
        monotonized = QuantileMonotonicity.isotonic_regression(coef_matrix, tau_list)

        # Check monotonicity
        for j in range(coef_matrix.shape[1]):
            diffs = np.diff(monotonized[:, j])
            assert np.all(diffs >= -1e-10), f"Non-monotonic coefficient {j}"

        # Check shape preserved
        assert monotonized.shape == coef_matrix.shape


class TestConstrainedQR:
    """Tests for constrained quantile regression."""

    @pytest.fixture
    def simple_data(self):
        """Simple data for testing."""
        np.random.seed(111)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2 + X[:, 1] + np.random.randn(n)
        return X, y

    def test_constrained_estimation(self, simple_data):
        """Test constrained QR estimation."""
        X, y = simple_data
        tau_list = np.array([0.25, 0.5, 0.75])

        # Estimate with constraints
        results = QuantileMonotonicity.constrained_qr(X, y, tau_list, max_iter=100)

        # Check results structure
        assert len(results) == 3
        for tau in tau_list:
            assert tau in results
            assert len(results[tau]) == X.shape[1]

        # Check non-crossing
        predictions = np.column_stack([X @ results[tau] for tau in tau_list])

        for i in range(len(X)):
            diffs = np.diff(predictions[i])
            assert np.all(diffs >= -1e-6), f"Crossing at observation {i}"

    def test_simultaneous_qr(self, simple_data):
        """Test simultaneous QR with soft penalty."""
        X, y = simple_data
        tau_list = np.array([0.25, 0.5, 0.75])

        # Estimate with soft penalty
        results = QuantileMonotonicity.simultaneous_qr(X, y, tau_list, lambda_nc=1.0, max_iter=50)

        # Check results
        assert len(results) == 3
        for tau in tau_list:
            assert tau in results


class TestProjection:
    """Tests for projection to monotone space."""

    def test_averaging_projection(self):
        """Test averaging method for projection."""
        # Create crossed predictions
        predictions = np.array(
            [
                [1.0, 0.8, 1.2],  # Crossing at position 1
                [2.0, 2.5, 3.0],  # No crossing
                [1.5, 1.3, 1.8],  # Crossing at position 1
            ]
        )

        # Project to monotone
        projected = QuantileMonotonicity.project_to_monotone(predictions, method="averaging")

        # Check monotonicity
        for i in range(len(predictions)):
            diffs = np.diff(projected[i])
            assert np.all(diffs >= -1e-10)

    def test_isotonic_projection(self):
        """Test isotonic regression projection."""
        # Create crossed predictions
        predictions = np.array([[1.0, 0.8, 1.2, 1.1, 1.5], [2.0, 2.5, 2.3, 3.0, 3.2]])

        # Project to monotone
        projected = QuantileMonotonicity.project_to_monotone(predictions, method="isotonic")

        # Check monotonicity
        for i in range(len(predictions)):
            diffs = np.diff(projected[i])
            assert np.all(diffs >= -1e-10)


class TestMonotonicityComparison:
    """Tests for comparing monotonicity methods."""

    @pytest.fixture
    def comparison_data(self):
        """Data for comparison."""
        np.random.seed(222)
        n = 150
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        y = 1 + 2 * X[:, 1] - X[:, 2] + np.random.randn(n) * (1 + 0.5 * np.abs(X[:, 1]))
        return X, y

    def test_method_comparison(self, comparison_data):
        """Test comparison of different methods."""
        X, y = comparison_data
        tau_list = np.array([0.2, 0.4, 0.6, 0.8])

        # Create comparison object
        comp = MonotonicityComparison(X, y, tau_list)

        # Compare methods (use subset for speed)
        df_results = comp.compare_methods(methods=["unconstrained", "rearrangement"])

        # Check results structure
        assert isinstance(df_results, pd.DataFrame)
        assert "method" in df_results.columns
        assert "has_crossing" in df_results.columns
        assert "total_inversions" in df_results.columns
        assert "total_loss" in df_results.columns

        # Rearrangement should have no crossing
        rearr_row = df_results[df_results["method"] == "rearrangement"]
        if not rearr_row.empty:
            assert rearr_row["has_crossing"].values[0] == False


class TestCrossingReport:
    """Tests for CrossingReport functionality."""

    def test_report_summary(self):
        """Test report summary generation."""
        # Create a report with crossings
        crossings = [
            {
                "tau_pair": (0.25, 0.5),
                "n_inversions": 10,
                "pct_inversions": 20.0,
                "max_violation": 0.5,
                "mean_violation": 0.3,
            }
        ]

        report = CrossingReport(
            has_crossing=True, crossings=crossings, total_inversions=10, pct_affected=20.0
        )

        # Test summary (should not raise)
        report.summary()

        # Test DataFrame conversion
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "tau1" in df.columns
        assert "tau2" in df.columns

    def test_no_crossing_report(self):
        """Test report when no crossing."""
        report = CrossingReport(
            has_crossing=False, crossings=[], total_inversions=0, pct_affected=0.0
        )

        # Summary should indicate no crossing
        report.summary()

        # DataFrame should be empty
        df = report.to_dataframe()
        assert df.empty
