"""Tests for causality module coverage gaps."""

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


@pytest.fixture
def panel_df():
    """Create a small panel DataFrame for causality tests."""
    np.random.seed(42)
    n_entities, n_periods = 10, 30
    data_rows = []
    for i in range(n_entities):
        y1 = np.zeros(n_periods)
        y2 = np.zeros(n_periods)
        y3 = np.zeros(n_periods)
        for t in range(1, n_periods):
            # y1 causes y2 (strong relationship)
            y1[t] = 0.5 * y1[t - 1] + np.random.randn()
            y2[t] = 0.3 * y2[t - 1] + 0.6 * y1[t - 1] + np.random.randn() * 0.5
            y3[t] = 0.4 * y3[t - 1] + np.random.randn()
        for t in range(n_periods):
            data_rows.append({"entity": i, "time": t, "y1": y1[t], "y2": y2[t], "y3": y3[t]})
    return pd.DataFrame(data_rows)


class TestDHResultConclusionStrings:
    """Test DH result conclusion strings at different significance levels (lines 211-216)."""

    def test_conclusion_rejects_at_1_percent(self):
        """Test DH summary with p < 0.01."""
        result = DumitrescuHurlinResult(
            cause="x",
            effect="y",
            W_bar=50.0,
            Z_tilde_stat=5.0,
            Z_tilde_pvalue=0.001,
            Z_bar_stat=5.0,
            Z_bar_pvalue=0.001,
            individual_W=np.array([40.0, 50.0, 60.0]),
            recommended_stat="Z_bar",
            N=3,
            T_avg=30.0,
            lags=1,
        )
        summary = result.summary()
        assert "Rejects H0 at 1%: Granger causality detected (***)" in summary

    def test_conclusion_rejects_at_5_percent(self):
        """Test DH summary with 0.01 < p < 0.05."""
        result = DumitrescuHurlinResult(
            cause="x",
            effect="y",
            W_bar=10.0,
            Z_tilde_stat=2.5,
            Z_tilde_pvalue=0.03,
            Z_bar_stat=2.5,
            Z_bar_pvalue=0.03,
            individual_W=np.array([8.0, 10.0, 12.0]),
            recommended_stat="Z_bar",
            N=3,
            T_avg=30.0,
            lags=1,
        )
        summary = result.summary()
        assert "Rejects H0 at 5%: Granger causality detected (**)" in summary

    def test_conclusion_rejects_at_10_percent(self):
        """Test DH summary with 0.05 < p < 0.10."""
        result = DumitrescuHurlinResult(
            cause="x",
            effect="y",
            W_bar=5.0,
            Z_tilde_stat=1.8,
            Z_tilde_pvalue=0.07,
            Z_bar_stat=1.8,
            Z_bar_pvalue=0.07,
            individual_W=np.array([4.0, 5.0, 6.0]),
            recommended_stat="Z_bar",
            N=3,
            T_avg=30.0,
            lags=1,
        )
        summary = result.summary()
        assert "Rejects H0 at 10%: Granger causality detected (*)" in summary

    def test_conclusion_fails_to_reject(self):
        """Test DH summary with p > 0.10."""
        result = DumitrescuHurlinResult(
            cause="x",
            effect="y",
            W_bar=1.0,
            Z_tilde_stat=0.5,
            Z_tilde_pvalue=0.60,
            Z_bar_stat=0.5,
            Z_bar_pvalue=0.60,
            individual_W=np.array([0.5, 1.0, 1.5]),
            recommended_stat="Z_bar",
            N=3,
            T_avg=30.0,
            lags=1,
        )
        summary = result.summary()
        assert "Fails to reject H0: No evidence of Granger causality" in summary

    def test_conclusion_uses_z_tilde_when_recommended(self):
        """Test that conclusion uses Z_tilde stat when recommended."""
        # Z_tilde p-value < 0.01 but Z_bar p-value > 0.10
        result = DumitrescuHurlinResult(
            cause="x",
            effect="y",
            W_bar=50.0,
            Z_tilde_stat=5.0,
            Z_tilde_pvalue=0.001,
            Z_bar_stat=0.5,
            Z_bar_pvalue=0.60,
            individual_W=np.array([40.0, 50.0, 60.0]),
            recommended_stat="Z_tilde",
            N=3,
            T_avg=5.0,
            lags=1,
        )
        summary = result.summary()
        # Should use Z_tilde (p=0.001) not Z_bar (p=0.60)
        assert "Rejects H0 at 1%: Granger causality detected (***)" in summary


class TestDHPlotMethods:
    """Test DumitrescuHurlinResult.plot_individual_statistics() (lines 250-351)."""

    @pytest.fixture
    def dh_result(self, panel_df):
        """Run a DH test and return the result."""
        return dumitrescu_hurlin_test(panel_df, cause="y1", effect="y2", lags=1)

    def test_plot_matplotlib(self, dh_result):
        """Test matplotlib backend for individual statistics plot."""
        import matplotlib.pyplot as plt

        fig = dh_result.plot_individual_statistics(backend="matplotlib", show=False)
        assert fig is not None
        # Should have axes with content
        axes = fig.get_axes()
        assert len(axes) >= 1
        plt.close(fig)

    def test_plot_plotly(self, dh_result):
        """Test plotly backend for individual statistics plot."""
        pytest.importorskip("plotly")
        fig = dh_result.plot_individual_statistics(backend="plotly", show=False)
        assert fig is not None
        # Plotly figure should have data traces
        assert len(fig.data) >= 1

    def test_plot_invalid_backend(self, dh_result):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            dh_result.plot_individual_statistics(backend="invalid", show=False)


class TestEntityWaldSingularFallback:
    """Test entity-level Wald test singular covariance fallback (lines 737-741)."""

    def test_singular_R_cov_R_uses_pinv(self):
        """Test that singular R_cov_R matrix falls back to pinv.

        We mock np.linalg.inv to raise LinAlgError on the R_cov_R call,
        while allowing other inv calls to proceed normally.
        """
        from unittest.mock import patch

        np.random.seed(42)
        n_entities, n_periods = 5, 30
        data_rows = []
        for i in range(n_entities):
            y1 = np.random.randn(n_periods)
            y2 = np.random.randn(n_periods)
            for t in range(n_periods):
                data_rows.append({"entity": i, "time": t, "y1": y1[t], "y2": y2[t]})
        df = pd.DataFrame(data_rows)

        original_inv = np.linalg.inv
        call_count = [0]

        def mock_inv(a):
            # In dumitrescu_hurlin_test, inv is called twice per entity:
            # 1) X_reg.T @ X_reg -> allow
            # 2) R_cov_R -> raise
            call_count[0] += 1
            if a.shape[0] == 1:
                # R_cov_R for lags=1 is 1x1 -> simulate singular
                raise np.linalg.LinAlgError("Singular matrix (mock)")
            return original_inv(a)

        with patch("panelbox.var.causality.np.linalg.inv", side_effect=mock_inv):
            result = dumitrescu_hurlin_test(df, cause="y1", effect="y2", lags=1)

        assert isinstance(result, DumitrescuHurlinResult)
        assert len(result.individual_W) == n_entities
        assert np.all(np.isfinite(result.individual_W))


class TestPanelGrangerCausality:
    """Test panel_granger_causality() function (lines 967-989)."""

    def test_panel_granger_causality_basic(self, panel_df):
        """Test pairwise Granger causality for multiple variables."""
        results = panel_granger_causality(
            data=panel_df,
            variables=["y1", "y2", "y3"],
            lags=1,
            entity_col="entity",
            time_col="time",
        )

        # Should have results for each effect variable
        assert "y1" in results
        assert "y2" in results
        assert "y3" in results

        # Each effect should have (K-1) cause entries (excluding self)
        for effect_var, causes in results.items():
            assert len(causes) == 2  # 3 variables - 1 self
            for cause_name, test_result in causes:
                assert cause_name != effect_var
                assert isinstance(test_result, DumitrescuHurlinResult)
                assert test_result.effect == effect_var
                assert test_result.cause == cause_name

    def test_panel_granger_causality_two_vars(self, panel_df):
        """Test with two variables."""
        results = panel_granger_causality(
            data=panel_df,
            variables=["y1", "y2"],
            lags=1,
        )
        assert len(results) == 2
        assert len(results["y1"]) == 1
        assert len(results["y2"]) == 1


class TestPanelGrangerCausalityMatrix:
    """Test panel_granger_causality_matrix() function (lines 992-1047)."""

    def test_matrix_shape_and_diagonal(self, panel_df):
        """Test matrix shape is (n_vars, n_vars), diagonal is NaN."""
        matrix = panel_granger_causality_matrix(
            data=panel_df,
            variables=["y1", "y2", "y3"],
            lags=1,
            entity_col="entity",
            time_col="time",
        )

        assert matrix.shape == (3, 3)
        # Diagonal should be NaN
        for i in range(3):
            assert np.isnan(matrix[i, i])
        # Off-diagonal should be p-values in [0, 1]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert 0.0 <= matrix[i, j] <= 1.0

    def test_matrix_two_variables(self, panel_df):
        """Test matrix with two variables."""
        matrix = panel_granger_causality_matrix(
            data=panel_df,
            variables=["y1", "y2"],
            lags=1,
        )
        assert matrix.shape == (2, 2)
        assert np.isnan(matrix[0, 0])
        assert np.isnan(matrix[1, 1])
        # Off-diagonal should be valid p-values
        assert 0.0 <= matrix[0, 1] <= 1.0
        assert 0.0 <= matrix[1, 0] <= 1.0


class TestDHSummaryContent:
    """Additional tests for DH summary completeness."""

    def test_dh_summary_from_real_test(self, panel_df):
        """Test summary from a real DH test run."""
        result = dumitrescu_hurlin_test(panel_df, cause="y1", effect="y2", lags=1)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Dumitrescu-Hurlin" in summary
        assert "Individual Entity Statistics" in summary
        assert "Min W_i:" in summary
        assert "Max W_i:" in summary
        assert "Recommended:" in summary
