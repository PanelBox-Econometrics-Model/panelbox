"""
Test suite for Moran's I spatial autocorrelation tests.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from panelbox.validation.spatial.moran_i import MoranIByPeriodResult, MoranIPanelTest


def _create_rook_weights(nrow, ncol):
    """Create a rook contiguity weight matrix for an nrow x ncol grid."""
    N = nrow * ncol
    W = np.zeros((N, N))
    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            if j + 1 < ncol:
                W[idx, idx + 1] = 1.0
                W[idx + 1, idx] = 1.0
            if i + 1 < nrow:
                W[idx, idx + ncol] = 1.0
                W[idx + ncol, idx] = 1.0
    return W


class TestMoranIPanelTest:
    """Test suite for global Moran's I panel test."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.N = 25  # 5x5 grid
        self.T = 10
        self.NT = self.N * self.T

        self.W = _create_rook_weights(5, 5)

        self.entity_index = np.repeat(np.arange(self.N), self.T)
        self.time_index = np.tile(np.arange(self.T), self.N)

    def test_init_validation(self):
        """Test initialization validates dimensions."""
        residuals = np.random.randn(self.NT)
        test = MoranIPanelTest(residuals, self.W, self.entity_index, self.time_index)
        assert test.N == self.N
        assert test.T == self.T

    def test_init_invalid_residuals_length(self):
        """Test that mismatched residuals length raises error."""
        residuals = np.random.randn(100)  # Wrong length
        with pytest.raises(ValueError, match="Residuals length"):
            MoranIPanelTest(residuals, self.W, self.entity_index, self.time_index)

    def test_init_invalid_W_shape(self):
        """Test that mismatched W shape raises error."""
        residuals = np.random.randn(self.NT)
        W_wrong = _create_rook_weights(2, 5)  # 10x10 instead of 25x25
        with pytest.raises(ValueError, match="W shape"):
            MoranIPanelTest(residuals, W_wrong, self.entity_index, self.time_index)

    def test_no_spatial_autocorrelation(self):
        """Test Moran's I with no spatial autocorrelation."""
        np.random.seed(42)
        residuals = np.random.randn(self.NT)
        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        result = test.run(alpha=0.05)

        # With purely random data, statistic should be close to expected value
        assert abs(result.statistic - result.metadata["expected_value"]) < 0.5
        # p-value should generally not be extremely small
        assert result.pvalue > 0.001

    def test_positive_spatial_autocorrelation(self):
        """Test Moran's I with strong positive spatial autocorrelation."""
        rho = 0.7
        epsilon = np.random.randn(self.N)
        I_rhoW_inv = linalg.inv(np.eye(self.N) - rho * self.W)
        u_spatial = I_rhoW_inv @ epsilon

        # Build in entity-major order to match index layout
        residuals = np.empty(self.NT)
        for i in range(self.N):
            for t in range(self.T):
                noise = np.random.randn() * 0.1
                residuals[i * self.T + t] = u_spatial[i] + noise

        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        result = test.run(alpha=0.05)

        assert result.pvalue < 0.05
        assert result.statistic > result.metadata["expected_value"]

    def test_negative_spatial_autocorrelation(self):
        """Test Moran's I with negative spatial autocorrelation."""
        np.random.seed(123)
        pattern = np.zeros(self.N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                pattern[idx] = 1 if (i + j) % 2 == 0 else -1

        # Build residuals in entity-major order to match index layout
        residuals = np.empty(self.NT)
        for i in range(self.N):
            for t in range(self.T):
                noise = np.random.randn() * 0.05
                residuals[i * self.T + t] = pattern[i] + noise

        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        result = test.run(alpha=0.05)
        assert result.statistic < result.metadata["expected_value"]

    def test_by_period_method(self):
        """Test Moran's I computed by period."""
        np.random.seed(10)
        expected_rhos = np.linspace(0.0, 0.9, self.T)

        # Build per-period spatial vectors with increasing autocorrelation
        u_by_t = []
        for _t, rho in enumerate(expected_rhos):
            epsilon = np.random.randn(self.N)
            if rho > 0:
                I_rhoW_inv = linalg.inv(np.eye(self.N) - rho * self.W)
                u_t = I_rhoW_inv @ epsilon
            else:
                u_t = epsilon
            u_by_t.append(u_t)

        # Build residuals in entity-major order to match index layout
        residuals = np.empty(self.NT)
        for i in range(self.N):
            for t in range(self.T):
                residuals[i * self.T + t] = u_by_t[t][i]

        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="period"
        )
        results = test.run(alpha=0.05)

        assert isinstance(results, MoranIByPeriodResult)
        assert len(results.results_by_period) == self.T

        # Check that we have increasing Moran's I (allow some noise)
        period_stats = [results.results_by_period[t]["statistic"] for t in range(self.T)]
        # Last period should have higher Moran's I than first
        assert period_stats[-1] > period_stats[0]

    def test_permutation_inference(self):
        """Test permutation-based inference."""
        np.random.seed(42)
        rho = 0.4
        epsilon = np.random.randn(self.N)
        I_rhoW_inv = linalg.inv(np.eye(self.N) - rho * self.W)
        u_spatial = I_rhoW_inv @ epsilon

        # Build in entity-major order: each entity repeats same value across T
        residuals = np.repeat(u_spatial, self.T)

        test_normal = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        result_normal = test_normal.run(alpha=0.05, inference="normal")

        np.random.seed(77)
        test_perm = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        result_perm = test_perm.run(alpha=0.05, n_permutations=49, inference="permutation")

        assert result_normal.pvalue < 0.05
        assert result_perm.pvalue < 0.2  # lenient due to few permutations
        assert abs(result_normal.statistic - result_perm.statistic) < 1e-10

    def test_pooled_result_metadata(self):
        """Test that pooled result contains proper metadata."""
        residuals = np.random.randn(self.NT)
        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        result = test.run()

        assert result.test_name == "Moran's I (Pooled)"
        assert "expected_value" in result.metadata
        assert "variance" in result.metadata
        assert "z_statistic" in result.metadata
        assert "N" in result.metadata
        assert "T" in result.metadata
        assert "conclusion" in result.metadata

    def test_pooled_positive_conclusion(self):
        """Test conclusion string for positive autocorrelation."""
        np.random.seed(42)
        rho = 0.7
        epsilon = np.random.randn(self.N)
        I_rhoW_inv = linalg.inv(np.eye(self.N) - rho * self.W)
        u_spatial = I_rhoW_inv @ epsilon
        # Entity-major: each entity repeats same value across T
        residuals = np.repeat(u_spatial, self.T)

        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        result = test.run(alpha=0.05)

        assert "Positive spatial autocorrelation" in result.metadata["conclusion"]

    def test_pooled_no_autocorrelation_conclusion(self):
        """Test conclusion string for no autocorrelation."""
        np.random.seed(99)
        residuals = np.random.randn(self.NT) * 0.01
        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        result = test.run(alpha=0.05)

        if result.pvalue >= 0.05:
            assert "Fail to reject" in result.metadata["conclusion"]


class TestMoranIByPeriodResult:
    """Test MoranIByPeriodResult dataclass."""

    def test_summary(self):
        """Test summary method."""
        results_dict = {
            0: {
                "statistic": 0.1,
                "expected_value": -0.04,
                "variance": 0.01,
                "z_statistic": 1.4,
                "pvalue": 0.16,
            },
            1: {
                "statistic": 0.3,
                "expected_value": -0.04,
                "variance": 0.01,
                "z_statistic": 3.4,
                "pvalue": 0.001,
            },
        }
        result = MoranIByPeriodResult(results_dict)
        summary = result.summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "statistic" in summary.columns

    def test_str(self):
        """Test string representation."""
        results_dict = {
            0: {
                "statistic": 0.1,
                "expected_value": -0.04,
                "variance": 0.01,
                "z_statistic": 1.4,
                "pvalue": 0.16,
            },
        }
        result = MoranIByPeriodResult(results_dict)
        result_str = str(result)
        assert "Moran's I by Period" in result_str


class TestMoranIComputeVariance:
    """Test the _compute_variance method edge cases."""

    def test_variance_with_small_n(self):
        """Test variance computation with n <= 3."""
        np.random.seed(42)
        N = 3
        T = 1
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        residuals = np.random.randn(N * T)
        entity_index = np.arange(N)
        time_index = np.zeros(N)

        test = MoranIPanelTest(residuals, W, entity_index, time_index, method="pooled")
        VI = test._compute_variance(3, 1.0, 1.0, 1.0, 2.0)
        assert VI == 1.0  # Returns 1.0 for n <= 3

    def test_variance_with_zero_S0(self):
        """Test variance computation with S0 = 0."""
        np.random.seed(42)
        N = 5
        W = _create_rook_weights(1, 5)
        residuals = np.random.randn(N)
        entity_index = np.arange(N)
        time_index = np.zeros(N)

        test = MoranIPanelTest(residuals, W, entity_index, time_index, method="pooled")
        VI = test._compute_variance(5, 0.0, 1.0, 1.0, 2.0)
        assert VI == 1.0


class TestMoranIPlotting:
    """Test plotting methods."""

    def setup_method(self):
        np.random.seed(42)
        self.N = 25
        self.T = 5
        self.NT = self.N * self.T
        self.W = _create_rook_weights(5, 5)
        self.entity_index = np.repeat(np.arange(self.N), self.T)
        self.time_index = np.tile(np.arange(self.T), self.N)

    def test_plot_by_period_matplotlib(self):
        """Test matplotlib plotting by period."""
        import matplotlib.pyplot as plt

        residuals = np.random.randn(self.NT)
        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="period"
        )
        fig = test.plot(backend="matplotlib")

        assert fig is not None
        plt.close(fig)

    def test_plot_moran_scatterplot_matplotlib(self):
        """Test matplotlib Moran scatterplot."""
        import matplotlib.pyplot as plt

        residuals = np.random.randn(self.NT)
        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        fig = test.plot(backend="matplotlib")

        assert fig is not None
        plt.close(fig)

    def test_plot_by_period_plotly(self):
        """Test plotly plotting by period."""
        pytest.importorskip("plotly")

        residuals = np.random.randn(self.NT)
        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="period"
        )
        fig = test.plot(backend="plotly")
        assert fig is not None

    def test_plot_moran_scatterplot_plotly(self):
        """Test plotly Moran scatterplot."""
        pytest.importorskip("plotly")

        residuals = np.random.randn(self.NT)
        test = MoranIPanelTest(
            residuals, self.W, self.entity_index, self.time_index, method="pooled"
        )
        fig = test.plot(backend="plotly")
        assert fig is not None


class TestMoranIntegration:
    """Integration tests for Moran's I functionality."""

    def test_panel_to_cross_section_consistency(self):
        """Test that panel Moran's I reduces to cross-section for T=1."""
        np.random.seed(42)
        N = 25
        W = _create_rook_weights(5, 5)

        rho = 0.5
        epsilon = np.random.randn(N)
        I_rhoW_inv = linalg.inv(np.eye(N) - rho * W)
        residuals = I_rhoW_inv @ epsilon

        entity_index = np.arange(N)
        time_index = np.zeros(N)

        test = MoranIPanelTest(residuals, W, entity_index, time_index, method="pooled")
        result = test.run()

        assert result.pvalue < 0.05
        assert result.statistic > 0

    def test_by_period_summary(self):
        """Test by-period results summary."""
        np.random.seed(42)
        N = 25
        T = 5
        W = _create_rook_weights(5, 5)
        residuals = np.random.randn(N * T)
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)

        test = MoranIPanelTest(residuals, W, entity_index, time_index, method="period")
        results = test.run()

        # Test each period result
        for t in range(T):
            assert t in results.results_by_period
            period_result = results.results_by_period[t]
            assert "statistic" in period_result
            assert "pvalue" in period_result
            assert 0 <= period_result["pvalue"] <= 1
