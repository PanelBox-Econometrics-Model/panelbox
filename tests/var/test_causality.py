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


class TestCausalityCoverage:
    """
    Additional tests targeting uncovered lines in panelbox/var/causality.py.

    Covers:
    - instantaneous_causality() function (lines 817-836)
    - InstantaneousCausalityResult.summary() (lines 394-423)
    - DH test plotly backend (lines 296-348)
    - DH test pinv fallback (lines 739-741)
    - GrangerCausalityResult significance paths and summary (lines 71-113)
    - DH test validation errors (lines 655, 658, 665, 684)
    - granger_causality_matrix() (lines 862-881) via mock
    - instantaneous_causality_matrix() (lines 905-928) via mock
    """

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _panel_data(N=10, T=25, seed=42):
        return _generate_panel_data_3var(N=N, T=T, seed=seed)

    # ------------------------------------------------------------------ #
    # instantaneous_causality  (lines 817-836)
    # ------------------------------------------------------------------ #

    def test_instantaneous_causality_correlated(self):
        """Correlated residuals should produce a small p-value."""
        from panelbox.var.causality import instantaneous_causality

        np.random.seed(42)
        resid1 = np.random.randn(200)
        resid2 = 0.6 * resid1 + 0.4 * np.random.randn(200)

        result = instantaneous_causality(resid1, resid2, "y1", "y2")

        assert result.var1 == "y1"
        assert result.var2 == "y2"
        assert result.n_obs == 200
        assert 0.0 < abs(result.correlation) < 1.0
        assert result.lr_stat > 0
        assert result.p_value < 0.05  # should reject

    def test_instantaneous_causality_uncorrelated(self):
        """Independent residuals should yield a large p-value."""
        from panelbox.var.causality import instantaneous_causality

        np.random.seed(123)
        resid1 = np.random.randn(200)
        resid2 = np.random.randn(200)

        result = instantaneous_causality(resid1, resid2, "a", "b")

        assert result.n_obs == 200
        assert result.lr_stat >= 0
        # With large sample of truly independent data, p-value should be > 0.05
        # (not guaranteed but highly likely)
        assert result.p_value > 0.01

    def test_instantaneous_causality_perfect_correlation(self):
        """Perfect correlation gives lr_stat = inf, p_value = 0."""
        from panelbox.var.causality import instantaneous_causality

        resid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = instantaneous_causality(resid, resid, "v1", "v2")

        assert result.lr_stat == np.inf
        assert result.p_value == 0.0

    def test_instantaneous_causality_length_mismatch(self):
        """Different length residuals should raise ValueError."""
        from panelbox.var.causality import instantaneous_causality

        with pytest.raises(ValueError, match="same length"):
            instantaneous_causality(np.array([1, 2, 3]), np.array([1, 2]), "a", "b")

    # ------------------------------------------------------------------ #
    # InstantaneousCausalityResult.summary()  (lines 394-423)
    # ------------------------------------------------------------------ #

    def _make_ic_result(self, p_value, correlation=0.5):
        from panelbox.var.causality import InstantaneousCausalityResult

        return InstantaneousCausalityResult(
            var1="y1",
            var2="y2",
            correlation=correlation,
            lr_stat=10.0,
            p_value=p_value,
            n_obs=100,
        )

    def test_ic_summary_rejects_1pct(self):
        """InstantaneousCausalityResult summary at 1% level."""
        result = self._make_ic_result(p_value=0.005)
        summary = result.summary()
        assert "Instantaneous Causality Test" in summary
        assert "Rejects H0 at 1%" in summary
        assert "(***)" in summary
        assert "y1" in summary
        assert "y2" in summary

    def test_ic_summary_rejects_5pct(self):
        """InstantaneousCausalityResult summary at 5% level."""
        result = self._make_ic_result(p_value=0.03)
        summary = result.summary()
        assert "Rejects H0 at 5%" in summary
        assert "(**)" in summary

    def test_ic_summary_rejects_10pct(self):
        """InstantaneousCausalityResult summary at 10% level."""
        result = self._make_ic_result(p_value=0.07)
        summary = result.summary()
        assert "Rejects H0 at 10%" in summary
        assert "(*)" in summary

    def test_ic_summary_fails_to_reject(self):
        """InstantaneousCausalityResult summary when not significant."""
        result = self._make_ic_result(p_value=0.50)
        summary = result.summary()
        assert "Fails to reject H0" in summary

    def test_ic_repr(self):
        """InstantaneousCausalityResult __repr__."""
        result = self._make_ic_result(p_value=0.03, correlation=0.45)
        r = repr(result)
        assert "InstantaneousCausalityResult" in r
        assert "y1" in r
        assert "y2" in r

    # ------------------------------------------------------------------ #
    # DH plot: plotly backend  (lines 296-348)
    # ------------------------------------------------------------------ #

    def test_plot_plotly_backend(self):
        """Test DH result plot with plotly backend."""
        import plotly.graph_objects as go

        data = self._panel_data(N=15, T=25)
        dh = dumitrescu_hurlin_test(data, cause="y2", effect="y1", lags=1)
        fig = dh.plot_individual_statistics(backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        # Should have at least 3 traces: histogram + critical value line + mean line
        assert len(fig.data) >= 3

    # ------------------------------------------------------------------ #
    # DH test: pinv fallback  (lines 739-741)
    # ------------------------------------------------------------------ #

    def test_dh_test_pinv_fallback(self):
        """
        Force the pinv fallback by monkeypatching np.linalg.inv to raise
        LinAlgError on the R_cov_R inversion inside dumitrescu_hurlin_test.
        """
        data = self._panel_data(N=5, T=15)

        # We intercept np.linalg.inv so that after the first successful call
        # (for computing cov_beta_i = sigma2 * inv(X'X)), the *second* call
        # within the same entity loop iteration (inv(R_cov_R)) raises
        # LinAlgError, triggering the pinv fallback on line 741.
        original_inv = np.linalg.inv
        call_count = {"n": 0}

        def patched_inv(a):
            call_count["n"] += 1
            # Every second call is for R_cov_R (within each entity iteration:
            # first call = inv(X'X) for cov_beta, second call = inv(R_cov_R))
            if call_count["n"] % 2 == 0:
                raise np.linalg.LinAlgError("Singular matrix (patched)")
            return original_inv(a)

        import unittest.mock as mock

        with mock.patch("numpy.linalg.inv", side_effect=patched_inv):
            result = dumitrescu_hurlin_test(data, cause="y1", effect="y2", lags=1)

        # The test should complete and produce valid results
        assert isinstance(result, DumitrescuHurlinResult)
        assert result.N == 5
        assert np.isfinite(result.W_bar)
        assert np.isfinite(result.Z_tilde_pvalue)

    # ------------------------------------------------------------------ #
    # GrangerCausalityResult (lines 71-113)
    # ------------------------------------------------------------------ #

    def test_granger_causality_result_post_init_1pct(self):
        """GrangerCausalityResult auto-conclusion at 1% level."""
        from panelbox.var.causality import GrangerCausalityResult

        r = GrangerCausalityResult(
            cause="x",
            effect="y",
            wald_stat=20.0,
            f_stat=20.0,
            df=1,
            p_value=0.005,
            lags_tested=1,
        )
        assert "Rejects H0 at 1%" in r.conclusion
        assert "(***)" in r.conclusion

    def test_granger_causality_result_post_init_5pct(self):
        """GrangerCausalityResult auto-conclusion at 5% level."""
        from panelbox.var.causality import GrangerCausalityResult

        r = GrangerCausalityResult(
            cause="x",
            effect="y",
            wald_stat=5.0,
            f_stat=5.0,
            df=1,
            p_value=0.03,
            lags_tested=1,
        )
        assert "Rejects H0 at 5%" in r.conclusion
        assert "(**)" in r.conclusion

    def test_granger_causality_result_post_init_10pct(self):
        """GrangerCausalityResult auto-conclusion at 10% level."""
        from panelbox.var.causality import GrangerCausalityResult

        r = GrangerCausalityResult(
            cause="x",
            effect="y",
            wald_stat=3.0,
            f_stat=3.0,
            df=1,
            p_value=0.07,
            lags_tested=1,
        )
        assert "Rejects H0 at 10%" in r.conclusion
        assert "(*)" in r.conclusion

    def test_granger_causality_result_post_init_not_reject(self):
        """GrangerCausalityResult auto-conclusion when not significant."""
        from panelbox.var.causality import GrangerCausalityResult

        r = GrangerCausalityResult(
            cause="x",
            effect="y",
            wald_stat=0.5,
            f_stat=0.5,
            df=1,
            p_value=0.50,
            lags_tested=1,
        )
        assert "Fails to reject H0" in r.conclusion

    def test_granger_causality_result_summary(self):
        """GrangerCausalityResult.summary() formatted output."""
        from panelbox.var.causality import GrangerCausalityResult

        r = GrangerCausalityResult(
            cause="x",
            effect="y",
            wald_stat=10.0,
            f_stat=10.0,
            df=1,
            p_value=0.001,
            p_value_f=0.002,
            lags_tested=2,
        )
        summary = r.summary()
        assert "Granger Causality Test" in summary
        assert "Wald statistic" in summary
        assert "F-statistic" in summary
        assert "P-value (F)" in summary  # p_value_f is not None
        assert "Conclusion" in summary

    def test_granger_causality_result_summary_no_f_pvalue(self):
        """GrangerCausalityResult.summary() without F p-value."""
        from panelbox.var.causality import GrangerCausalityResult

        r = GrangerCausalityResult(
            cause="x",
            effect="y",
            wald_stat=10.0,
            f_stat=10.0,
            df=1,
            p_value=0.001,
            p_value_f=None,
            lags_tested=2,
        )
        summary = r.summary()
        assert "P-value (F)" not in summary

    def test_granger_causality_result_repr(self):
        """GrangerCausalityResult __repr__."""
        from panelbox.var.causality import GrangerCausalityResult

        r = GrangerCausalityResult(
            cause="x",
            effect="y",
            wald_stat=10.0,
            f_stat=10.0,
            df=1,
            p_value=0.001,
            lags_tested=2,
        )
        text = repr(r)
        assert "GrangerCausalityResult" in text
        assert "x" in text
        assert "y" in text

    # ------------------------------------------------------------------ #
    # DH test validation errors  (lines 654-684)
    # ------------------------------------------------------------------ #

    def test_dh_test_missing_variable(self):
        """DH test raises ValueError if cause/effect not in data."""
        data = self._panel_data(N=5, T=15)
        with pytest.raises(ValueError, match="not found in data"):
            dumitrescu_hurlin_test(data, cause="nonexistent", effect="y1", lags=1)

    def test_dh_test_missing_entity_col(self):
        """DH test raises ValueError if entity column not in data."""
        data = self._panel_data(N=5, T=15)
        with pytest.raises(ValueError, match="Entity or time column not found"):
            dumitrescu_hurlin_test(data, cause="y1", effect="y2", lags=1, entity_col="group")

    def test_dh_test_single_entity(self):
        """DH test raises ValueError if fewer than 2 entities."""
        data = self._panel_data(N=1, T=15)
        with pytest.raises(ValueError, match="at least 2 entities"):
            dumitrescu_hurlin_test(data, cause="y1", effect="y2", lags=1)

    def test_dh_test_insufficient_observations(self):
        """DH test raises ValueError if entity has insufficient time periods."""
        # Create data with only 2 time periods per entity but lags=2
        rows = []
        for i in range(5):
            for t in range(3):
                rows.append({"entity": f"E{i}", "time": t, "y1": float(t), "y2": float(t)})
        data = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="insufficient observations"):
            dumitrescu_hurlin_test(data, cause="y1", effect="y2", lags=2)

    # ------------------------------------------------------------------ #
    # DH __repr__  (lines 353-361)
    # ------------------------------------------------------------------ #

    def test_dh_result_repr_z_bar(self):
        """DumitrescuHurlinResult __repr__ with Z_bar recommended."""
        result = DumitrescuHurlinResult(
            cause="x",
            effect="y",
            W_bar=5.0,
            Z_tilde_stat=2.5,
            Z_tilde_pvalue=0.01,
            Z_bar_stat=3.0,
            Z_bar_pvalue=0.003,
            individual_W=np.array([3.0, 5.0, 7.0]),
            recommended_stat="Z_bar",
            N=3,
            T_avg=20.0,
            lags=1,
        )
        text = repr(result)
        assert "DumitrescuHurlinResult" in text
        assert "x" in text
        assert "y" in text

    def test_dh_result_repr_z_tilde(self):
        """DumitrescuHurlinResult __repr__ with Z_tilde recommended."""
        result = DumitrescuHurlinResult(
            cause="x",
            effect="y",
            W_bar=5.0,
            Z_tilde_stat=2.5,
            Z_tilde_pvalue=0.01,
            Z_bar_stat=3.0,
            Z_bar_pvalue=0.003,
            individual_W=np.array([3.0, 5.0, 7.0]),
            recommended_stat="Z_tilde",
            N=3,
            T_avg=20.0,
            lags=1,
        )
        text = repr(result)
        assert "DumitrescuHurlinResult" in text

    # ------------------------------------------------------------------ #
    # construct_granger_restriction_matrix  (lines 461-475)
    # ------------------------------------------------------------------ #

    def test_construct_restriction_matrix(self):
        """Test restriction matrix construction."""
        from panelbox.var.causality import construct_granger_restriction_matrix

        exog_names = ["const", "L1.y", "L2.y", "L1.x", "L2.x"]
        R = construct_granger_restriction_matrix(exog_names, "x", lags=2)
        assert R.shape == (2, 5)
        # L1.x is index 3, L2.x is index 4
        assert R[0, 3] == 1.0
        assert R[1, 4] == 1.0
        # Everything else is zero
        assert R[0, 0] == 0.0
        assert R[0, 1] == 0.0

    def test_construct_restriction_matrix_missing_lag(self):
        """Test restriction matrix raises for missing lag variable."""
        from panelbox.var.causality import construct_granger_restriction_matrix

        exog_names = ["const", "L1.y", "L1.x"]
        with pytest.raises(ValueError, match="not found in regressors"):
            construct_granger_restriction_matrix(exog_names, "x", lags=2)

    # ------------------------------------------------------------------ #
    # granger_causality_wald  (lines 521-536)
    # ------------------------------------------------------------------ #

    def test_granger_causality_wald(self):
        """Test Wald-based Granger causality with known data."""
        from panelbox.var.causality import granger_causality_wald

        exog_names = ["const", "L1.y", "L1.x"]
        params = np.array([0.5, 0.3, 0.8])  # x has large coefficient
        cov = np.diag([0.01, 0.01, 0.01])

        result = granger_causality_wald(
            params=params,
            cov_params=cov,
            exog_names=exog_names,
            causing_var="x",
            caused_var="y",
            lags=1,
            n_obs=100,
        )

        assert result.cause == "x"
        assert result.effect == "y"
        assert result.wald_stat > 0
        assert result.f_stat > 0
        assert 0 <= result.p_value <= 1
        assert result.p_value_f is not None

    def test_granger_causality_wald_no_nobs(self):
        """Test Wald Granger without n_obs -- no F p-value."""
        from panelbox.var.causality import granger_causality_wald

        exog_names = ["const", "L1.y", "L1.x"]
        params = np.array([0.5, 0.3, 0.8])
        cov = np.diag([0.01, 0.01, 0.01])

        result = granger_causality_wald(
            params=params,
            cov_params=cov,
            exog_names=exog_names,
            causing_var="x",
            caused_var="y",
            lags=1,
            n_obs=None,
        )

        assert result.p_value_f is None

    # ------------------------------------------------------------------ #
    # dumitrescu_hurlin_moments edge cases  (lines 585, 602)
    # ------------------------------------------------------------------ #

    def test_dh_moments_insufficient_df(self):
        """dumitrescu_hurlin_moments with T too small raises ValueError."""
        from panelbox.var.causality import dumitrescu_hurlin_moments

        # T=5, p=2, K=2 => df = 5 - 2*2 - 1 = 0 => should raise
        with pytest.raises(ValueError, match="Insufficient degrees of freedom"):
            dumitrescu_hurlin_moments(T=5, p=2, K=2)

    def test_dh_moments_small_df_fallback(self):
        """dumitrescu_hurlin_moments with df <= 2 uses asymptotic Var."""
        from panelbox.var.causality import dumitrescu_hurlin_moments

        # T=6, p=2, K=1 => df = 6 - 1*2 - 1 = 3 > 2 (will use finite sample)
        # T=4, p=1, K=1 => df = 4 - 1*1 - 1 = 2 => uses asymptotic fallback
        E_W, Var_W = dumitrescu_hurlin_moments(T=4, p=1, K=1)
        assert E_W == 1  # p=1
        assert Var_W == 2 * 1  # 2p asymptotic

    # ------------------------------------------------------------------ #
    # granger_causality_matrix (lines 862-881) and
    # instantaneous_causality_matrix (lines 905-928)
    # via lightweight mock of PanelVARResult
    # ------------------------------------------------------------------ #

    def test_granger_causality_matrix_via_mock(self):
        """Test granger_causality_matrix using a mock PanelVARResult."""
        from unittest.mock import MagicMock

        from panelbox.var.causality import (
            GrangerCausalityResult,
            granger_causality_matrix,
        )

        mock_result = MagicMock()
        mock_result.K = 2
        mock_result.endog_names = ["y1", "y2"]

        def mock_granger(cause, effect):
            return GrangerCausalityResult(
                cause=cause,
                effect=effect,
                wald_stat=5.0,
                f_stat=5.0,
                df=1,
                p_value=0.025,
                lags_tested=1,
            )

        mock_result.granger_causality = mock_granger

        df = granger_causality_matrix(mock_result)

        assert df.shape == (2, 2)
        assert np.isnan(df.loc["y1", "y1"])
        assert np.isnan(df.loc["y2", "y2"])
        assert df.loc["y1", "y2"] == pytest.approx(0.025)
        assert df.loc["y2", "y1"] == pytest.approx(0.025)

    def test_granger_causality_matrix_exception_handling(self):
        """Test granger_causality_matrix when granger_causality raises."""
        from unittest.mock import MagicMock

        from panelbox.var.causality import granger_causality_matrix

        mock_result = MagicMock()
        mock_result.K = 2
        mock_result.endog_names = ["y1", "y2"]
        mock_result.granger_causality.side_effect = RuntimeError("test error")

        df = granger_causality_matrix(mock_result)

        assert df.shape == (2, 2)
        # Off-diagonal should be NaN due to exception
        assert np.isnan(df.loc["y1", "y2"])
        assert np.isnan(df.loc["y2", "y1"])

    def test_instantaneous_causality_matrix_via_mock(self):
        """Test instantaneous_causality_matrix using a mock PanelVARResult."""
        from unittest.mock import MagicMock

        from panelbox.var.causality import instantaneous_causality_matrix

        np.random.seed(42)
        resid1 = np.random.randn(100)
        resid2 = 0.5 * resid1 + np.random.randn(100)

        mock_result = MagicMock()
        mock_result.K = 2
        mock_result.endog_names = ["y1", "y2"]
        mock_result.resid_by_eq = [resid1, resid2]

        corr_df, pvalue_df = instantaneous_causality_matrix(mock_result)

        assert corr_df.shape == (2, 2)
        assert pvalue_df.shape == (2, 2)
        # Diagonal of correlation should be 1.0
        assert corr_df.loc["y1", "y1"] == pytest.approx(1.0)
        assert corr_df.loc["y2", "y2"] == pytest.approx(1.0)
        # Off-diagonal correlation should be non-zero
        assert abs(corr_df.loc["y1", "y2"]) > 0.1
        # P-values should be symmetric
        assert pvalue_df.loc["y1", "y2"] == pytest.approx(pvalue_df.loc["y2", "y1"])
        # Diagonal of p-values should be NaN
        assert np.isnan(pvalue_df.loc["y1", "y1"])

    # ------------------------------------------------------------------ #
    # DH test with T_avg < 10 (recommends Z_tilde)  (line 769)
    # ------------------------------------------------------------------ #

    def test_dh_recommends_z_tilde_for_short_T(self):
        """DH test recommends Z_tilde when average T < 10."""
        # N=5, T=8 (< 10) => recommended_stat == "Z_tilde"
        data = self._panel_data(N=5, T=8)
        result = dumitrescu_hurlin_test(data, cause="y1", effect="y2", lags=1)
        assert result.recommended_stat == "Z_tilde"
        assert result.T_avg < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
