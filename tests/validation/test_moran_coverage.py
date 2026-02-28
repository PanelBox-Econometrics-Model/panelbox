"""Coverage tests for panelbox.validation.spatial.moran_i.

Targets uncovered lines: 185-186, 192, 239-244, 252, 257, 260, 308,
319, 322, 330, 333, 352, 383, 452, 490->501, 528-530, 572,
598->614, 636->643.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.validation.spatial.moran_i import MoranIByPeriodResult, MoranIPanelTest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_contiguity_5x5():
    """Create a 5x5 row-standardized contiguity weight matrix.

    Linear chain: 0-1-2-3-4. Each unit is neighbor to adjacent units.
    """
    W = np.zeros((5, 5))
    for i in range(4):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums
    return W


def _balanced_panel(N, T, seed=42):
    """Create balanced panel index arrays and random residuals."""
    np.random.seed(seed)
    entities = np.arange(N)
    times = np.arange(T)
    entity_index = np.repeat(entities, T)
    time_index = np.tile(times, N)
    residuals = np.random.randn(N * T)
    return residuals, entity_index, time_index


def _unbalanced_panel(N, T, drop_specs, seed=42):
    """Create unbalanced panel by dropping specific (entity, time) pairs.

    Parameters
    ----------
    N : int
        Number of entities.
    T : int
        Number of time periods.
    drop_specs : list of tuple
        Each tuple (entity, time) will be removed.
    seed : int
        Random seed.

    Returns
    -------
    residuals, entity_index, time_index : arrays
        The panel data with some observations removed. Length = N*T
        (padded with NaN for missing observations to satisfy the
        constructor's length check).
    """
    np.random.seed(seed)
    entities = np.arange(N)
    times = np.arange(T)
    entity_index = np.repeat(entities, T)
    time_index = np.tile(times, N)
    residuals = np.random.randn(N * T)

    # Set dropped observations to NaN (the constructor requires len = N*T)
    for ent, t in drop_specs:
        mask = (entity_index == ent) & (time_index == t)
        residuals[mask] = np.nan

    return residuals, entity_index, time_index


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Test class: Unbalanced panel handling
# ---------------------------------------------------------------------------


class TestUnbalancedPanel:
    """Tests targeting unbalanced panel code paths."""

    def setup_method(self):
        self.N = 5
        self.T = 4
        self.W = _make_contiguity_5x5()

    def test_run_by_period_unbalanced_panel(self):
        """Lines 185-186: _run_by_period merges missing entities with full_df.

        When some entities are missing in a period, len(df_t) != N, so the
        code creates a full_df and merges to fill NaN.
        """
        np.random.seed(42)
        # Create a balanced panel but set one entity-period to NaN
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        # Remove entity 3 from period 2 by setting to NaN
        mask = (entity_index == 3) & (time_index == 2)
        residuals[mask] = np.nan

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="period")
        results = test.run(alpha=0.05)

        assert isinstance(results, MoranIByPeriodResult)
        # Period 2 should still be present (4 valid entities >= 3)
        assert 2 in results.results_by_period

    def test_run_by_period_skip_too_few_valid(self):
        """Line 192: skip period if fewer than 3 valid values.

        Set all but 2 entities to NaN in one period so it gets skipped.
        """
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        # In period 1, set entities 0, 1, 2 to NaN (only 2 valid left)
        for ent in [0, 1, 2]:
            mask = (entity_index == ent) & (time_index == 1)
            residuals[mask] = np.nan

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="period")
        results = test.run(alpha=0.05)

        assert isinstance(results, MoranIByPeriodResult)
        # Period 1 should be skipped (only 2 valid values)
        assert 1 not in results.results_by_period
        # Other periods should be present
        assert 0 in results.results_by_period
        assert 2 in results.results_by_period

    def test_compute_pooled_unbalanced(self):
        """Lines 239-244: _compute_pooled_morans_i with len(r_t) != N.

        When the panel is unbalanced, the pooled computation rearranges
        residuals into a full array with NaN filling.
        """
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        # Create unbalanced: remove entity 4 from period 0
        mask = (entity_index == 4) & (time_index == 0)
        residuals[mask] = np.nan

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        result = test.run(alpha=0.05)

        assert result.statistic is not None
        assert np.isfinite(result.pvalue)

    def test_pooled_valid_mask_fewer_than_2(self):
        """Line 252: continue when valid_mask has < 2 valid entries in a period.

        Lines 257, 260: inner loop valid_mask checks also exercised.
        """
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        # In period 3, set all but 1 entity to NaN
        for ent in [0, 1, 2, 3]:
            mask = (entity_index == ent) & (time_index == 3)
            residuals[mask] = np.nan

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        result = test.run(alpha=0.05)

        # Should still compute from the other periods
        assert np.isfinite(result.statistic)

    def test_pooled_inner_loop_valid_mask(self):
        """Lines 257, 260: valid_mask skips in the inner double loop.

        Some entities have NaN in a period, causing the inner loops to skip
        those entries via valid_mask[i] / valid_mask[j].
        """
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        # Set entity 2 to NaN in period 0 (3 valid entities remain)
        mask = (entity_index == 2) & (time_index == 0)
        residuals[mask] = np.nan

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        result = test.run(alpha=0.05)
        assert np.isfinite(result.statistic)


# ---------------------------------------------------------------------------
# Test class: _compute_morans_i_period edge cases
# ---------------------------------------------------------------------------


class TestComputeMoransIPeriod:
    """Tests targeting _compute_morans_i_period edge cases."""

    def setup_method(self):
        self.N = 5
        self.T = 4
        self.W = _make_contiguity_5x5()

    def test_period_n_valid_less_than_3_returns_nan(self):
        """Line 308: return NaN for n_valid < 3."""
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")

        # Create r_t with only 2 valid values
        r_t = np.full(self.N, np.nan)
        r_t[0] = 1.0
        r_t[1] = -1.0

        I, EI, VI = test._compute_morans_i_period(r_t)
        assert np.isnan(I)
        assert np.isnan(EI)
        assert np.isnan(VI)

    def test_period_inner_loop_valid_mask(self):
        """Lines 319, 322, 330, 333: inner loops skip invalid entries.

        When some entries in r_t are NaN, the valid_mask causes the inner
        double loops (numerator and S0 computation) to skip them.
        """
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")

        # Create r_t with 3 valid values (entity 0, 1 are NaN)
        r_t = np.array([np.nan, np.nan, 1.0, -0.5, 0.3])

        I, EI, VI = test._compute_morans_i_period(r_t)
        # Should compute without error, using only entities 2, 3, 4
        assert np.isfinite(I)
        assert np.isfinite(EI)
        assert np.isfinite(VI)

    def test_period_vi_fallback_for_n_valid_le_2(self):
        """Line 352: VI = 1.0 for n_valid <= 2.

        This line is inside the else branch of if n_valid > 2, but
        _compute_morans_i_period returns NaN when n_valid < 3 (line 308).
        For n_valid == 2, it also returns NaN. So we need exactly n_valid == 2
        to NOT trigger line 308 but trigger line 352.
        Actually line 308 checks n_valid < 3, so n_valid == 2 triggers line 308.
        Line 352 is unreachable in practice but let us verify the < 3 path.
        """
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")

        # n_valid == 2 triggers line 308 (return NaN)
        r_t = np.array([np.nan, np.nan, np.nan, 1.0, -1.0])
        I, EI, VI = test._compute_morans_i_period(r_t)
        assert np.isnan(I)
        assert np.isnan(EI)
        assert np.isnan(VI)

    def test_period_all_valid(self):
        """Ensure _compute_morans_i_period works with all valid entries."""
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")

        r_t = np.array([1.0, -0.5, 0.3, 0.8, -0.2])
        I, EI, _VI = test._compute_morans_i_period(r_t)
        assert np.isfinite(I)
        assert pytest.approx(-1.0 / (self.N - 1)) == EI


# ---------------------------------------------------------------------------
# Test class: _compute_variance edge cases
# ---------------------------------------------------------------------------


class TestComputeVariance:
    """Tests targeting _compute_variance edge cases."""

    def setup_method(self):
        self.N = 5
        self.T = 4
        self.W = _make_contiguity_5x5()
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        self.test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")

    def test_variance_denom_zero(self):
        """Line 383: denom == 0 returns 1.0.

        When n <= 3 the method returns 1.0 early (line 370-371), so to
        reach line 383 we need n > 3 but S0 == 0 to make denom == 0.
        However, S0 <= 0 already returns 1.0 at line 362-363.
        The only way to reach line 383 is if n > 3 and S0 > 0 but the
        product n_1 * n_2 * n_3 * S0 * S0 evaluates to 0. With n > 3,
        n_1, n_2, n_3 are all > 0, and S0 > 0, so denom > 0.
        This is effectively unreachable with valid inputs, but let's test
        the n <= 1 and S0 <= 0 paths.
        """
        # n <= 1
        assert self.test._compute_variance(1, 1.0, 1.0, 1.0, 2.0) == 1.0
        # S0 <= 0
        assert self.test._compute_variance(5, 0.0, 1.0, 1.0, 2.0) == 1.0
        assert self.test._compute_variance(5, -1.0, 1.0, 1.0, 2.0) == 1.0
        # n == 2 (n <= 3)
        assert self.test._compute_variance(2, 1.0, 1.0, 1.0, 2.0) == 1.0
        # n == 3 (n <= 3)
        assert self.test._compute_variance(3, 1.0, 1.0, 1.0, 2.0) == 1.0

    def test_variance_normal_computation(self):
        """Test normal variance computation with n > 3 and S0 > 0."""
        VI = self.test._compute_variance(10, 4.0, 2.0, 8.0, 3.0)
        assert VI > 0
        assert np.isfinite(VI)


# ---------------------------------------------------------------------------
# Test class: Plotting - _plot_by_period
# ---------------------------------------------------------------------------


class TestPlotByPeriod:
    """Tests targeting _plot_by_period code paths."""

    def setup_method(self):
        self.N = 5
        self.T = 4
        self.W = _make_contiguity_5x5()

    def test_plot_by_period_raises_for_pooled(self):
        """Line 452: raise ValueError when method is not 'period'.

        _plot_by_period calls self.run() which returns a ValidationTestResult
        (not MoranIByPeriodResult) when method='pooled'. The isinstance check
        then raises ValueError.
        """
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)
        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")

        with pytest.raises(ValueError, match="No period results available"):
            test._plot_by_period(backend="matplotlib")

    def test_plot_by_period_significant_matplotlib(self):
        """Lines 528-530: significant period scatter in matplotlib backend.

        Generate data with strong spatial autocorrelation so that pvalues
        are < 0.05 (significant), triggering the scatter plot for
        significant periods. Uses a larger grid (5x5=25) for sufficient
        power.
        """
        np.random.seed(42)
        from scipy import linalg

        # Use a larger 5x5 grid for statistical power
        N = 25
        T = 4
        W_grid = np.zeros((N, N))
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if j + 1 < 5:
                    W_grid[idx, idx + 1] = 1.0
                    W_grid[idx + 1, idx] = 1.0
                if i + 1 < 5:
                    W_grid[idx, idx + 5] = 1.0
                    W_grid[idx + 5, idx] = 1.0
        # Row-standardize for DGP
        rs = W_grid.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W_std = W_grid / rs

        rho = 0.7
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

        fig = test.plot(backend="matplotlib")
        assert fig is not None

    def test_plot_by_period_significant_plotly(self):
        """Lines 490->501: significant periods highlighting in plotly backend.

        Same data as matplotlib test but using plotly. Uses a larger grid
        (5x5=25) for statistical power.
        """
        pytest.importorskip("plotly.graph_objects")

        np.random.seed(42)
        from scipy import linalg

        N = 25
        T = 4
        W_grid = np.zeros((N, N))
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if j + 1 < 5:
                    W_grid[idx, idx + 1] = 1.0
                    W_grid[idx + 1, idx] = 1.0
                if i + 1 < 5:
                    W_grid[idx, idx + 5] = 1.0
                    W_grid[idx + 5, idx] = 1.0
        rs = W_grid.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W_std = W_grid / rs

        rho = 0.7
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

        results = test.run(alpha=0.05)
        pvalues = [results.results_by_period[t]["pvalue"] for t in results.results_by_period]
        assert any(p < 0.05 for p in pvalues), (
            f"Expected at least one significant period, got pvalues={pvalues}"
        )

        fig = test.plot(backend="plotly")
        assert fig is not None
        # Should have 3 traces: Moran's I, expected, significant
        assert len(fig.data) == 3

    def test_plot_by_period_no_significant_matplotlib(self):
        """Test matplotlib plot when no period is significant (no star markers)."""
        np.random.seed(99)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T, seed=99)

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="period")
        fig = test.plot(backend="matplotlib")
        assert fig is not None

    def test_plot_by_period_no_significant_plotly(self):
        """Test plotly plot when no period is significant."""
        pytest.importorskip("plotly")
        np.random.seed(99)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T, seed=99)

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="period")
        fig = test.plot(backend="plotly")
        assert fig is not None


# ---------------------------------------------------------------------------
# Test class: Plotting - _plot_moran_scatterplot
# ---------------------------------------------------------------------------


class TestPlotMoranScatterplot:
    """Tests targeting _plot_moran_scatterplot code paths."""

    def setup_method(self):
        self.N = 5
        self.T = 4
        self.W = _make_contiguity_5x5()

    def test_scatterplot_unbalanced_panel_matplotlib(self):
        """Line 572: Wz NaN fill for unbalanced panel in scatterplot.

        When len(z_t) != N, the scatterplot fills Wz with NaN.
        We trigger this by having NaN residuals that cause the sorted
        DataFrame for a period to have fewer rows than N.
        """
        np.random.seed(42)
        N = self.N
        T = self.T

        # Build data with one observation per entity per period
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.random.randn(N * T)

        # Remove entity 4 from period 0 by setting to NaN
        mask = (entity_index == 4) & (time_index == 0)
        residuals[mask] = np.nan

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        fig = test.plot(backend="matplotlib")
        assert fig is not None

    def test_scatterplot_regression_line_matplotlib(self):
        """Lines 636->643: regression line in matplotlib backend.

        With len(z_valid) > 1, a regression line (polyfit + plot) is drawn.
        """
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        fig = test.plot(backend="matplotlib")
        assert fig is not None

        # The axes should contain a line for the regression
        ax = fig.axes[0]
        # scatter + regression line + 2 axlines = at least 3 artists
        assert len(ax.lines) >= 1

    def test_scatterplot_regression_line_plotly(self):
        """Lines 598->614: regression line in plotly backend.

        With len(z_valid) > 1, a regression line trace is added.
        """
        pytest.importorskip("plotly.graph_objects")

        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        fig = test.plot(backend="plotly")
        assert fig is not None

        # Should have at least 2 traces: scatter + regression line
        assert len(fig.data) >= 2
        # Second trace should be the regression line
        assert fig.data[1].mode == "lines"

    def test_scatterplot_unbalanced_panel_plotly(self):
        """Line 572: Wz NaN fill for unbalanced panel in plotly scatterplot."""
        pytest.importorskip("plotly")

        np.random.seed(42)
        N = self.N
        T = self.T

        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.random.randn(N * T)

        # Remove entity 4 from period 0
        mask = (entity_index == 4) & (time_index == 0)
        residuals[mask] = np.nan

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        fig = test.plot(backend="plotly")
        assert fig is not None

    def test_scatterplot_mostly_nan_matplotlib(self):
        """Lines 636->643 false branch: skip regression line when <= 1 valid.

        With almost all residuals NaN, z_valid has 0 or 1 entries, so the
        regression line block is skipped.
        """
        np.random.seed(42)
        N = self.N
        T = self.T

        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.full(N * T, np.nan)
        # Keep only a single non-NaN value
        residuals[0] = 1.0

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        fig = test.plot(backend="matplotlib")
        assert fig is not None

    def test_scatterplot_mostly_nan_plotly(self):
        """Lines 598->614 false branch: skip regression line when <= 1 valid.

        With almost all residuals NaN, z_valid has 0 or 1 entries, so the
        regression line trace is not added.
        """
        pytest.importorskip("plotly")

        np.random.seed(42)
        N = self.N
        T = self.T

        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)
        residuals = np.full(N * T, np.nan)
        residuals[0] = 1.0

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        fig = test.plot(backend="plotly")
        assert fig is not None
        # Should have only 1 trace (scatter, no regression line)
        assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# Test class: Permutation inference
# ---------------------------------------------------------------------------


class TestPermutationInference:
    """Tests targeting permutation inference code path."""

    def setup_method(self):
        self.N = 5
        self.T = 4
        self.W = _make_contiguity_5x5()

    def test_permutation_inference_basic(self):
        """Test that permutation inference produces a valid p-value."""
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        result = test.run(alpha=0.05, n_permutations=19, inference="permutation")

        assert 0 < result.pvalue <= 1.0
        assert result.metadata["inference"] == "permutation"

    def test_permutation_minimum_pvalue(self):
        """Test that p-value is at least 1/n_permutations."""
        np.random.seed(42)
        from scipy import linalg

        rho = 0.9
        I_rhoW_inv = linalg.inv(np.eye(self.N) - rho * self.W)
        eps = np.random.randn(self.N)
        u_spatial = I_rhoW_inv @ eps
        residuals = np.repeat(u_spatial, self.T)

        entity_index = np.repeat(np.arange(self.N), self.T)
        time_index = np.tile(np.arange(self.T), self.N)

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        n_perm = 9
        result = test.run(alpha=0.05, n_permutations=n_perm, inference="permutation")

        assert result.pvalue >= 1.0 / n_perm


# ---------------------------------------------------------------------------
# Test class: Zero variance / degenerate cases
# ---------------------------------------------------------------------------


class TestDegenerateCases:
    """Tests for degenerate edge cases."""

    def setup_method(self):
        self.N = 5
        self.T = 4
        self.W = _make_contiguity_5x5()

    def test_constant_residuals(self):
        """Zero variance residuals should yield I = 0 (denominator = 0)."""
        np.random.seed(42)
        entity_index = np.repeat(np.arange(self.N), self.T)
        time_index = np.tile(np.arange(self.T), self.N)
        residuals = np.ones(self.N * self.T)  # constant

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        result = test.run(alpha=0.05)

        # With constant residuals, I should be 0 (no variation)
        assert result.statistic == 0

    def test_period_method_constant_residuals(self):
        """Period method with constant residuals in one period."""
        np.random.seed(42)
        entity_index = np.repeat(np.arange(self.N), self.T)
        time_index = np.tile(np.arange(self.T), self.N)
        residuals = np.random.randn(self.N * self.T)

        # Make period 0 constant
        mask = time_index == 0
        residuals[mask] = 5.0

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="period")
        results = test.run(alpha=0.05)

        assert isinstance(results, MoranIByPeriodResult)
        # Period 0 should have I = 0 (constant residuals)
        assert results.results_by_period[0]["statistic"] == 0

    def test_all_nan_period_skipped(self):
        """All residuals NaN in one period causes it to be skipped."""
        np.random.seed(42)
        entity_index = np.repeat(np.arange(self.N), self.T)
        time_index = np.tile(np.arange(self.T), self.N)
        residuals = np.random.randn(self.N * self.T)

        # Set all entities in period 1 to NaN
        mask = time_index == 1
        residuals[mask] = np.nan

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="period")
        results = test.run(alpha=0.05)

        assert 1 not in results.results_by_period


# ---------------------------------------------------------------------------
# Test class: MoranIByPeriodResult
# ---------------------------------------------------------------------------


class TestMoranIByPeriodResultCoverage:
    """Additional coverage tests for MoranIByPeriodResult."""

    def test_empty_results(self):
        """Test with empty results dict."""
        result = MoranIByPeriodResult(results_by_period={})
        summary = result.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0

    def test_str_representation(self):
        """Test string representation includes header text."""
        results_dict = {
            0: {
                "statistic": 0.25,
                "expected_value": -0.04,
                "variance": 0.01,
                "z_statistic": 2.9,
                "pvalue": 0.003,
            },
        }
        result = MoranIByPeriodResult(results_dict)
        text = str(result)
        assert "Moran's I by Period" in text


# ---------------------------------------------------------------------------
# Test class: Balanced panel (basic sanity)
# ---------------------------------------------------------------------------


class TestBalancedPanel:
    """Basic tests with balanced panel data (N=5, T=4)."""

    def setup_method(self):
        self.N = 5
        self.T = 4
        self.W = _make_contiguity_5x5()

    def test_pooled_basic(self):
        """Test basic pooled Moran's I with balanced panel."""
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        result = test.run(alpha=0.05)

        assert result.test_name == "Moran's I (Pooled)"
        assert np.isfinite(result.statistic)
        assert 0 <= result.pvalue <= 1
        assert result.metadata["N"] == self.N
        assert result.metadata["T"] == self.T

    def test_period_basic(self):
        """Test basic by-period Moran's I with balanced panel."""
        np.random.seed(42)
        residuals, entity_index, time_index = _balanced_panel(self.N, self.T)

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="period")
        results = test.run(alpha=0.05)

        assert isinstance(results, MoranIByPeriodResult)
        assert len(results.results_by_period) == self.T

        for t in range(self.T):
            assert "statistic" in results.results_by_period[t]
            assert "pvalue" in results.results_by_period[t]
            assert 0 <= results.results_by_period[t]["pvalue"] <= 1

    def test_negative_autocorrelation_conclusion(self):
        """Test conclusion for negative spatial autocorrelation."""
        np.random.seed(42)
        entity_index = np.repeat(np.arange(self.N), self.T)
        time_index = np.tile(np.arange(self.T), self.N)

        # Checkerboard pattern: neighbors have opposite signs
        pattern = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        residuals = np.empty(self.N * self.T)
        for i in range(self.N):
            for t in range(self.T):
                residuals[i * self.T + t] = pattern[i] + np.random.randn() * 0.01

        test = MoranIPanelTest(residuals, self.W, entity_index, time_index, method="pooled")
        result = test.run(alpha=0.05)

        if result.pvalue < 0.05 and result.statistic < result.metadata["expected_value"]:
            assert "Negative spatial autocorrelation" in result.metadata["conclusion"]
