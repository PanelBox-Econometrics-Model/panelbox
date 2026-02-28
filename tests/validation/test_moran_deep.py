"""Deep coverage tests for panelbox.validation.spatial.moran_i.

Targets remaining uncovered lines after test_moran_coverage.py:
- Lines 92, 95: constructor validation (ValueError for bad dimensions)
- Branch 527->540: matplotlib significant periods scatter
- Dead code documentation: lines 185-186, 239-244, 352, 383, 572

Note: Lines 185-186, 239-244, 572 handle unbalanced panels where some
time periods have fewer entities than N. These are unreachable because
the constructor enforces len(residuals) == N*T, which guarantees every
period has exactly N observations in balanced data.

Lines 352 and 383 are also unreachable: 352 requires n_valid <= 2 but
line 308 returns early for n_valid < 3; 383 requires denom == 0 but
earlier checks at lines 362/370 return early for n <= 3 or S0 <= 0.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from panelbox.validation.spatial.moran_i import MoranIPanelTest


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _make_contiguity_W(N):
    """Create an NxN row-standardized linear chain weight matrix."""
    W = np.zeros((N, N))
    for i in range(N - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


class TestConstructorValidation:
    """Cover lines 92 and 95: constructor raises ValueError on bad dimensions."""

    def test_residuals_length_too_short(self):
        """Line 92: residuals length < N * T should raise ValueError."""
        N, T = 5, 4
        W = _make_contiguity_W(N)
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.random.randn(N * T - 1)

        with pytest.raises(ValueError, match="Residuals length"):
            MoranIPanelTest(residuals, W, entity_index, time_index)

    def test_residuals_length_too_long(self):
        """Line 92: residuals length > N * T should also raise."""
        N, T = 5, 4
        W = _make_contiguity_W(N)
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.random.randn(N * T + 3)

        with pytest.raises(ValueError, match="Residuals length"):
            MoranIPanelTest(residuals, W, entity_index, time_index)

    def test_W_too_large(self):
        """Line 95: W larger than N should raise ValueError."""
        N, T = 5, 4
        W = _make_contiguity_W(N + 1)  # 6x6 instead of 5x5
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.random.randn(N * T)

        with pytest.raises(ValueError, match=r"W shape .* incompatible"):
            MoranIPanelTest(residuals, W, entity_index, time_index)

    def test_W_too_small(self):
        """Line 95: W smaller than N should raise ValueError."""
        N, T = 5, 4
        W = _make_contiguity_W(N - 1)  # 4x4 instead of 5x5
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.random.randn(N * T)

        with pytest.raises(ValueError, match=r"W shape .* incompatible"):
            MoranIPanelTest(residuals, W, entity_index, time_index)


class TestMatplotlibSignificantPeriods:
    """Cover branch 527->540: matplotlib significant period scatter markers.

    Uses a large grid (7x7=49) with strong spatial autocorrelation to ensure
    at least one period has p < 0.05.
    """

    def test_matplotlib_significant_periods_large_grid(self):
        """Branch 527->540: matplotlib scatter for significant periods."""
        from scipy import linalg

        np.random.seed(42)
        grid_size = 7
        N = grid_size * grid_size  # 49
        T = 4

        # Build grid-based weight matrix
        W_grid = np.zeros((N, N))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if j + 1 < grid_size:
                    W_grid[idx, idx + 1] = 1.0
                    W_grid[idx + 1, idx] = 1.0
                if i + 1 < grid_size:
                    W_grid[idx, idx + grid_size] = 1.0
                    W_grid[idx + grid_size, idx] = 1.0

        rs = W_grid.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W_std = W_grid / rs

        rho = 0.8
        I_rhoW_inv = linalg.inv(np.eye(N) - rho * W_std)

        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)

        residuals = np.empty(N * T)
        for t in range(T):
            np.random.seed(42 + t)
            eps = np.random.randn(N)
            u_t = I_rhoW_inv @ eps
            for i in range(N):
                residuals[i * T + t] = u_t[i]

        test = MoranIPanelTest(residuals, W_grid, entity_index, time_index, method="period")

        # Verify at least one period is significant
        results = test.run(alpha=0.05)
        pvalues = [results.results_by_period[t]["pvalue"] for t in results.results_by_period]
        assert any(p < 0.05 for p in pvalues), (
            f"Expected at least one significant period, got pvalues={pvalues}"
        )

        # Plot with matplotlib to trigger the significant scatter branch
        fig = test.plot(backend="matplotlib")
        assert fig is not None


class TestNoSignificantPeriods:
    """Cover FALSE branches 490->501 and 527->540.

    When no periods are significant (all pvalues >= 0.05), the
    'if any(significant)' block is skipped in both plotly and matplotlib.
    """

    def _make_nonsignificant_data(self):
        """Create data with no spatially significant periods (seed=0)."""
        np.random.seed(0)
        N, T = 5, 4
        W = _make_contiguity_W(N)
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.random.randn(N * T)
        return residuals, W, entity_index, time_index

    def test_no_significant_periods_matplotlib(self):
        """Branch 527->540 FALSE: skip scatter for no significant periods."""
        residuals, W, entity_index, time_index = self._make_nonsignificant_data()

        test = MoranIPanelTest(residuals, W, entity_index, time_index, method="period")

        # Verify no periods are significant
        results = test.run(alpha=0.05)
        pvalues = [results.results_by_period[t]["pvalue"] for t in results.results_by_period]
        assert all(p >= 0.05 for p in pvalues), (
            f"Expected no significant periods with seed=0, got pvalues={pvalues}"
        )

        fig = test.plot(backend="matplotlib")
        assert fig is not None

    def test_no_significant_periods_plotly(self):
        """Branch 490->501 FALSE: skip scatter for no significant periods."""
        pytest.importorskip("plotly")

        residuals, W, entity_index, time_index = self._make_nonsignificant_data()

        test = MoranIPanelTest(residuals, W, entity_index, time_index, method="period")

        # Verify no periods are significant
        results = test.run(alpha=0.05)
        pvalues = [results.results_by_period[t]["pvalue"] for t in results.results_by_period]
        assert all(p >= 0.05 for p in pvalues)

        fig = test.plot(backend="plotly")
        assert fig is not None
        # Should have only 2 traces (Moran's I + expected), no significant markers
        assert len(fig.data) == 2


class TestComputeVarianceEdgeCases:
    """Additional tests for _compute_variance edge cases."""

    def setup_method(self):
        np.random.seed(42)
        N, T = 5, 4
        W = _make_contiguity_W(N)
        residuals = np.random.randn(N * T)
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        self.test = MoranIPanelTest(residuals, W, entity_index, time_index, method="pooled")

    def test_variance_n_equals_0(self):
        """n=0 triggers n <= 1 early return (line 362-363)."""
        VI = self.test._compute_variance(0, 1.0, 1.0, 1.0, 2.0)
        assert VI == 1.0

    def test_variance_n_equals_4_normal(self):
        """n=4 is the minimum for normal computation (n > 3)."""
        VI = self.test._compute_variance(4, 2.0, 1.0, 4.0, 3.0)
        assert np.isfinite(VI)
        assert VI > 0

    def test_variance_negative_result_clamped(self):
        """When formula yields negative VI, max(VI, 1e-10) ensures positive."""
        VI = self.test._compute_variance(4, 100.0, 0.01, 0.01, 100.0)
        assert VI >= 1e-10
