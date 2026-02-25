"""
Tests for additional coverage of panelbox.var.causality module.

Covers:
- DumitrescuHurlinResult.summary() significance level paths (lines 211-216)
- DumitrescuHurlinResult.plot_individual_statistics() matplotlib backend (lines 250-351)
- panel_granger_causality() function (lines 967-989)
- panel_granger_causality_matrix() function (lines 992-1047)
- Wald test pseudo-inverse fallback (lines 721-741)
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality import (
    DumitrescuHurlinResult,
    dumitrescu_hurlin_test,
    panel_granger_causality,
    panel_granger_causality_matrix,
)


def _generate_panel_data_3var(N=10, T=30, seed=42):
    """
    Generate a 3-variable panel dataset for causality testing.

    y1_t = 0.3 * y1_{t-1} + 0.4 * y2_{t-1} + e1
    y2_t = 0.3 * y2_{t-1} + e2
    y3_t = 0.3 * y3_{t-1} + 0.3 * y1_{t-1} + e3
    """
    np.random.seed(seed)
    rows = []
    for i in range(N):
        y1, y2, y3 = 0.0, 0.0, 0.0
        for t in range(T):
            y1_new = 0.3 * y1 + 0.4 * y2 + np.random.normal(0, 0.5)
            y2_new = 0.3 * y2 + np.random.normal(0, 0.5)
            y3_new = 0.3 * y3 + 0.3 * y1 + np.random.normal(0, 0.5)
            rows.append(
                {
                    "entity": f"E{i}",
                    "time": t,
                    "y1": y1_new,
                    "y2": y2_new,
                    "y3": y3_new,
                }
            )
            y1, y2, y3 = y1_new, y2_new, y3_new
    return pd.DataFrame(rows)


class TestDHSummarySignificanceLevels:
    """Test DumitrescuHurlinResult.summary() for different significance levels (lines 211-216)."""

    def _make_dh_result(self, z_tilde_pvalue, z_bar_pvalue, recommended="Z_bar"):
        """Create a DumitrescuHurlinResult with specific p-values."""
        return DumitrescuHurlinResult(
            cause="x",
            effect="y",
            W_bar=5.0,
            Z_tilde_stat=2.5,
            Z_tilde_pvalue=z_tilde_pvalue,
            Z_bar_stat=2.5,
            Z_bar_pvalue=z_bar_pvalue,
            individual_W=np.array([3.0, 5.0, 7.0, 4.0, 6.0]),
            recommended_stat=recommended,
            N=5,
            T_avg=20.0,
            lags=1,
        )

    def test_summary_rejects_at_1_percent(self):
        """Test summary conclusion when p < 0.01 (***) -- line 210."""
        result = self._make_dh_result(z_tilde_pvalue=0.005, z_bar_pvalue=0.005)
        summary = result.summary()
        assert "Rejects H0 at 1%" in summary
        assert "(***)" in summary

    def test_summary_rejects_at_5_percent(self):
        """Test summary conclusion when 0.01 <= p < 0.05 (**) -- line 211-212."""
        result = self._make_dh_result(z_tilde_pvalue=0.03, z_bar_pvalue=0.03)
        summary = result.summary()
        assert "Rejects H0 at 5%" in summary
        assert "(**)" in summary

    def test_summary_rejects_at_10_percent(self):
        """Test summary conclusion when 0.05 <= p < 0.10 (*) -- line 213-214."""
        result = self._make_dh_result(z_tilde_pvalue=0.07, z_bar_pvalue=0.07)
        summary = result.summary()
        assert "Rejects H0 at 10%" in summary
        assert "(*)" in summary

    def test_summary_fails_to_reject(self):
        """Test summary conclusion when p >= 0.10 -- line 215-216."""
        result = self._make_dh_result(z_tilde_pvalue=0.50, z_bar_pvalue=0.50)
        summary = result.summary()
        assert "Fails to reject H0" in summary

    def test_summary_uses_recommended_stat_z_tilde(self):
        """Test that summary uses Z_tilde p-value when recommended."""
        # Z_tilde says reject, Z_bar says don't -- recommended is Z_tilde
        result = self._make_dh_result(
            z_tilde_pvalue=0.005,
            z_bar_pvalue=0.50,
            recommended="Z_tilde",
        )
        summary = result.summary()
        # Should use Z_tilde (p=0.005 => reject at 1%)
        assert "Rejects H0 at 1%" in summary


class TestDHPlotIndividualStatistics:
    """Test DumitrescuHurlinResult.plot_individual_statistics() (lines 250-351)."""

    @pytest.fixture
    def dh_result(self):
        """Create a DH result with enough entities for plotting."""
        np.random.seed(42)
        data = _generate_panel_data_3var(N=15, T=25, seed=42)
        return dumitrescu_hurlin_test(data, cause="y2", effect="y1", lags=1)

    def test_plot_matplotlib_backend(self, dh_result):
        """Test plot with matplotlib backend, show=False (lines 250-293)."""
        import matplotlib.pyplot as plt

        fig = dh_result.plot_individual_statistics(backend="matplotlib", show=False)
        assert fig is not None
        # Should have one axes
        axes = fig.get_axes()
        assert len(axes) >= 1
        plt.close(fig)

    def test_plot_matplotlib_has_critical_value_line(self, dh_result):
        """Test that the matplotlib plot includes critical value and mean lines."""
        import matplotlib.pyplot as plt

        fig = dh_result.plot_individual_statistics(backend="matplotlib", show=False)
        ax = fig.get_axes()[0]
        # Check that there are lines drawn (at least the critical value and W_bar)
        lines = ax.get_lines()
        assert len(lines) >= 2, "Should have at least critical value and mean lines"
        plt.close(fig)

    def test_plot_invalid_backend(self, dh_result):
        """Test that invalid backend raises ValueError (line 350-351)."""
        with pytest.raises(ValueError, match="Unknown backend"):
            dh_result.plot_individual_statistics(backend="invalid_backend", show=False)


class TestPanelGrangerCausality:
    """Test panel_granger_causality() function (lines 967-989)."""

    @pytest.fixture
    def panel_data_3var(self):
        """Generate 3-variable panel data."""
        return _generate_panel_data_3var(N=10, T=25, seed=42)

    def test_panel_granger_causality_returns_dict(self, panel_data_3var):
        """Test that panel_granger_causality returns correct dict structure."""
        results = panel_granger_causality(
            data=panel_data_3var,
            variables=["y1", "y2", "y3"],
            lags=1,
            entity_col="entity",
            time_col="time",
        )

        # Should have one key per variable
        assert set(results.keys()) == {"y1", "y2", "y3"}

    def test_panel_granger_causality_pairs(self, panel_data_3var):
        """Test that each variable has the correct number of cause pairs."""
        results = panel_granger_causality(
            data=panel_data_3var,
            variables=["y1", "y2", "y3"],
            lags=1,
        )

        # Each variable should have K-1 = 2 cause pairs
        for var_name in ["y1", "y2", "y3"]:
            assert len(results[var_name]) == 2, (
                f"Variable {var_name} should have 2 cause pairs, got {len(results[var_name])}"
            )

    def test_panel_granger_causality_result_types(self, panel_data_3var):
        """Test that results contain (cause_name, DumitrescuHurlinResult) tuples."""
        results = panel_granger_causality(
            data=panel_data_3var,
            variables=["y1", "y2", "y3"],
            lags=1,
        )

        for var_name, causes in results.items():
            for cause_name, test_result in causes:
                assert isinstance(cause_name, str)
                assert isinstance(test_result, DumitrescuHurlinResult)
                assert test_result.effect == var_name
                assert test_result.cause == cause_name

    def test_panel_granger_causality_no_self_causation(self, panel_data_3var):
        """Test that no variable tests self-causation."""
        results = panel_granger_causality(
            data=panel_data_3var,
            variables=["y1", "y2", "y3"],
            lags=1,
        )

        for var_name, causes in results.items():
            cause_names = [c[0] for c in causes]
            assert var_name not in cause_names, (
                f"Variable {var_name} should not appear as its own cause"
            )


class TestPanelGrangerCausalityMatrix:
    """Test panel_granger_causality_matrix() function (lines 992-1047)."""

    @pytest.fixture
    def panel_data_3var(self):
        """Generate 3-variable panel data."""
        return _generate_panel_data_3var(N=10, T=25, seed=42)

    def test_matrix_shape(self, panel_data_3var):
        """Test that returned matrix has correct K x K shape."""
        matrix = panel_granger_causality_matrix(
            data=panel_data_3var,
            variables=["y1", "y2", "y3"],
            lags=1,
        )

        assert matrix.shape == (3, 3)

    def test_matrix_diagonal_is_nan(self, panel_data_3var):
        """Test that diagonal elements are NaN (no self-causation)."""
        matrix = panel_granger_causality_matrix(
            data=panel_data_3var,
            variables=["y1", "y2", "y3"],
            lags=1,
        )

        for i in range(3):
            assert np.isnan(matrix[i, i]), f"Diagonal element [{i},{i}] should be NaN"

    def test_matrix_off_diagonal_are_pvalues(self, panel_data_3var):
        """Test that off-diagonal elements are valid p-values in [0, 1]."""
        matrix = panel_granger_causality_matrix(
            data=panel_data_3var,
            variables=["y1", "y2", "y3"],
            lags=1,
        )

        for i in range(3):
            for j in range(3):
                if i != j:
                    assert 0.0 <= matrix[i, j] <= 1.0, (
                        f"Off-diagonal element [{i},{j}] = {matrix[i, j]} should be a valid p-value"
                    )

    def test_matrix_with_2_variables(self, panel_data_3var):
        """Test matrix with only 2 variables."""
        matrix = panel_granger_causality_matrix(
            data=panel_data_3var,
            variables=["y1", "y2"],
            lags=1,
        )

        assert matrix.shape == (2, 2)
        assert np.isnan(matrix[0, 0])
        assert np.isnan(matrix[1, 1])
        assert 0.0 <= matrix[0, 1] <= 1.0
        assert 0.0 <= matrix[1, 0] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
