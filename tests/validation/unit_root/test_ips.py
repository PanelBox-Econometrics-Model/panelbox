"""
Pytest tests for IPS panel unit root test.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.unit_root.ips import IPSTest, IPSTestResult


def _generate_stationary_panel(n_entities=10, n_time=50, seed=42):
    """Generate stationary panel data."""
    np.random.seed(seed)
    data_list = []
    for i in range(n_entities):
        rho = 0.3 + 0.3 * (i / n_entities)
        y = np.zeros(n_time)
        y[0] = np.random.randn()
        for t in range(1, n_time):
            y[t] = rho * y[t - 1] + np.random.randn()
        data_list.append(pd.DataFrame({"entity": i, "time": range(n_time), "y": y}))
    return pd.concat(data_list, ignore_index=True)


def _generate_unit_root_panel(n_entities=10, n_time=50, seed=123):
    """Generate unit root (random walk) panel data."""
    np.random.seed(seed)
    data_list = []
    for i in range(n_entities):
        y = np.cumsum(np.random.randn(n_time))
        data_list.append(pd.DataFrame({"entity": i, "time": range(n_time), "y": y}))
    return pd.concat(data_list, ignore_index=True)


class TestIPSTestInit:
    """Test IPSTest initialization."""

    def test_init_valid(self):
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        assert test.n_entities == 10
        assert test.trend == "c"

    def test_init_invalid_variable(self):
        data = _generate_stationary_panel()
        with pytest.raises(ValueError, match="not found"):
            IPSTest(data, "invalid", "entity", "time")

    def test_init_invalid_entity_col(self):
        data = _generate_stationary_panel()
        with pytest.raises(ValueError, match="not found"):
            IPSTest(data, "y", "invalid", "time")

    def test_init_invalid_trend(self):
        data = _generate_stationary_panel()
        with pytest.raises(ValueError, match="trend must be"):
            IPSTest(data, "y", "entity", "time", trend="invalid")


class TestIPSTestRun:
    """Test IPSTest run method."""

    def test_run_stationary_data(self):
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        result = test.run()

        assert isinstance(result, IPSTestResult)
        assert result.test_type == "IPS"
        assert 0 <= result.pvalue <= 1
        assert result.n_entities == 10
        assert result.deterministics == "Constant"

    def test_run_unit_root_data(self):
        data = _generate_unit_root_panel()
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        result = test.run()

        assert isinstance(result, IPSTestResult)
        assert 0 <= result.pvalue <= 1

    def test_run_trend_none(self):
        """Test with trend='n' (no deterministic terms)."""
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="n")
        result = test.run()

        assert result.deterministics == "None"
        assert isinstance(result.statistic, float)

    def test_run_trend_constant_trend(self):
        """Test with trend='ct' (constant and trend)."""
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="ct")
        result = test.run()

        assert result.deterministics == "Constant and Trend"
        assert isinstance(result.statistic, float)

    def test_run_auto_lags(self):
        """Test with automatic lag selection (lags=None)."""
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
        result = test.run()

        assert isinstance(result, IPSTestResult)
        # Lags may be int or list depending on heterogeneity
        if isinstance(result.lags, list):
            assert all(l >= 0 for l in result.lags)
        else:
            assert result.lags >= 0

    def test_run_dict_lags(self):
        """Test with per-entity lag specification (dict)."""
        data = _generate_stationary_panel(n_entities=5)
        entities = data["entity"].unique()
        lags_dict = dict.fromkeys(entities, 1)

        test = IPSTest(data, "y", "entity", "time", lags=lags_dict, trend="c")
        result = test.run()

        assert isinstance(result, IPSTestResult)

    def test_run_individual_stats(self):
        """Test that individual entity stats are available."""
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        result = test.run()

        assert len(result.individual_stats) == result.n_entities
        for _entity, t_stat in result.individual_stats.items():
            assert np.isfinite(t_stat)


class TestIPSTestResult:
    """Test IPSTestResult."""

    def test_conclusion_reject(self):
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        result = test.run()

        if result.pvalue < 0.05:
            assert "Reject" in result.conclusion
        else:
            assert "Fail to reject" in result.conclusion

    def test_str_representation(self):
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        result = test.run()

        result_str = str(result)
        assert "Im-Pesaran-Shin" in result_str
        assert "W-statistic" in result_str
        assert "P-value" in result_str
        assert "Cross-sections" in result_str

    def test_str_with_variable_lags(self):
        """Test string repr with variable lags (list)."""
        data = _generate_stationary_panel()
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
        result = test.run()

        result_str = str(result)
        assert "Lags" in result_str


class TestIPSCriticalValues:
    """Test critical value lookups for different T and trends."""

    def test_critical_values_trend_n_short_T(self):
        data = _generate_stationary_panel(n_time=20)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="n")
        result = test.run()
        assert isinstance(result.statistic, float)

    def test_critical_values_trend_n_medium_T(self):
        data = _generate_stationary_panel(n_time=40)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="n")
        result = test.run()
        assert isinstance(result.statistic, float)

    def test_critical_values_trend_n_long_T(self):
        data = _generate_stationary_panel(n_time=60)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="n")
        result = test.run()
        assert isinstance(result.statistic, float)

    def test_critical_values_trend_ct_short_T(self):
        data = _generate_stationary_panel(n_time=20)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="ct")
        result = test.run()
        assert isinstance(result.statistic, float)

    def test_critical_values_trend_ct_medium_T(self):
        data = _generate_stationary_panel(n_time=40)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="ct")
        result = test.run()
        assert isinstance(result.statistic, float)

    def test_critical_values_trend_ct_long_T(self):
        data = _generate_stationary_panel(n_time=60)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="ct")
        result = test.run()
        assert isinstance(result.statistic, float)

    def test_critical_values_trend_c_long_T(self):
        """Test critical values for trend='c' with long T (>50) - covers lines 390-391."""
        data = _generate_stationary_panel(n_time=80)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        result = test.run()
        assert isinstance(result.statistic, float)

    def test_critical_values_trend_c_medium_T(self):
        """Test critical values for trend='c' with T in (25,50] - covers lines 384-385."""
        data = _generate_stationary_panel(n_time=40)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        result = test.run()
        assert isinstance(result.statistic, float)


class TestIPSTestInitTimecol:
    """Test invalid time column - covers line 185."""

    def test_init_invalid_time_col(self):
        data = _generate_stationary_panel()
        with pytest.raises(ValueError, match="Time column"):
            IPSTest(data, "y", "entity", "invalid_time")


class TestIPSSelectLagsShortData:
    """Test _select_lags_for_entity with very short data - covers line 218."""

    def test_select_lags_very_short_data(self):
        """When T is very short, max_lags < 1, so return 0."""
        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
        # Entity data of length 3 → max_lags = min(12, 3//4) = 0 → returns 0
        entity_data = np.array([1.0, 2.0, 3.0])
        result = test._select_lags_for_entity(entity_data, max_lags=12)
        assert result == 0

    def test_select_lags_max_lags_zero(self):
        """When max_lags is explicitly 0, should return 0."""
        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
        entity_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = test._select_lags_for_entity(entity_data, max_lags=0)
        assert result == 0


class TestIPSComputeAICEntity:
    """Test _compute_aic_entity edge cases - covers lines 237, 248-250, 256-258, 269-270."""

    def test_compute_aic_too_short_data(self):
        """When entity data is too short for given lags - covers line 237."""
        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        # data length 3, lags 5 → len < lags + 3 → return np.inf
        entity_data = np.array([1.0, 2.0, 3.0])
        aic = test._compute_aic_entity(entity_data, lags=5)
        assert aic == np.inf

    def test_compute_aic_with_lags(self):
        """Test AIC computation with multiple lags - covers lines 248-250."""
        np.random.seed(42)
        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
        entity_data = np.random.randn(50).cumsum()
        aic = test._compute_aic_entity(entity_data, lags=3)
        assert np.isfinite(aic)

    def test_compute_aic_with_trend_ct(self):
        """Test AIC computation with constant and trend - covers lines 256-258."""
        np.random.seed(42)
        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="ct")
        entity_data = np.random.randn(50).cumsum()
        aic = test._compute_aic_entity(entity_data, lags=1)
        assert np.isfinite(aic)

    def test_compute_aic_singular_matrix(self):
        """Test AIC when matrix causes exception - covers lines 269-270."""
        from unittest.mock import patch

        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
        entity_data = np.random.randn(20)
        # Force an exception in lstsq to trigger the except block
        with patch("numpy.linalg.lstsq", side_effect=np.linalg.LinAlgError("singular")):
            aic = test._compute_aic_entity(entity_data, lags=1)
        assert aic == np.inf


class TestIPSSelectLagsException:
    """Test _select_lags_for_entity when AIC computation fails - covers lines 229-230."""

    def test_select_lags_with_aic_failures(self):
        """When some AIC computations fail, they should be skipped."""
        data = _generate_stationary_panel(n_entities=3, n_time=20)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
        # Use data that's just barely enough for lag 0 but will fail for higher lags
        entity_data = np.array([1.0, 2.0, 3.5, 4.0, 5.5, 6.0])
        result = test._select_lags_for_entity(entity_data, max_lags=3)
        assert isinstance(result, int)
        assert result >= 0


class TestIPSADFTestEntity:
    """Test _adf_test_entity edge cases - covers lines 289, 300-302, 333-349."""

    def test_adf_test_entity_too_short(self):
        """Entity data too short for given lags - covers line 289."""
        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        entity_data = np.array([1.0, 2.0, 3.0])
        t_stat, n_obs = test._adf_test_entity(entity_data, lags=5)
        assert np.isnan(t_stat)
        assert n_obs == 0

    def test_adf_test_entity_with_lags(self):
        """Test ADF with multiple lags - covers lines 300-302."""
        np.random.seed(42)
        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=3, trend="c")
        entity_data = np.random.randn(50).cumsum()
        t_stat, n_obs = test._adf_test_entity(entity_data, lags=3)
        assert np.isfinite(t_stat)
        assert n_obs > 0

    def test_adf_test_entity_singular_matrix(self):
        """Test ADF when matrix inversion fails - covers lines 333-334."""
        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        # Constant data: X'X will be singular → exception in inv()
        entity_data = np.ones(20)
        t_stat, n_obs = test._adf_test_entity(entity_data, lags=1)
        assert np.isnan(t_stat)
        assert n_obs == 0

    def test_adf_test_entity_no_lags_no_trend(self):
        """Test ADF with no lags and no trend (simple ADF, else branch) - covers lines 335-349."""
        np.random.seed(42)
        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=0, trend="n")
        entity_data = np.random.randn(50).cumsum()
        t_stat, n_obs = test._adf_test_entity(entity_data, lags=0)
        assert np.isfinite(t_stat)
        assert n_obs > 0

    def test_adf_test_entity_no_lags_no_trend_singular(self):
        """Test simple ADF (no lags, no trend) with singular data - covers lines 348-349."""
        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=0, trend="n")
        # Constant data → y_lag is all same → division by zero in se_rho
        entity_data = np.ones(20)
        t_stat, _n_obs = test._adf_test_entity(entity_data, lags=0)
        # Should either return nan (exception) or a valid result
        # Constant data: dy = all zeros, y_lag = all ones
        # lstsq will give params[0] = 0, sigma2 = 0, se_rho = 0 → 0/0 → nan or exception
        assert np.isnan(t_stat) or np.isfinite(t_stat)

    def test_adf_test_entity_no_lags_constant_trend(self):
        """Test ADF with no lags but with constant - X list will have constant, not empty."""
        np.random.seed(42)
        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=0, trend="c")
        entity_data = np.random.randn(50).cumsum()
        t_stat, n_obs = test._adf_test_entity(entity_data, lags=0)
        assert np.isfinite(t_stat)
        assert n_obs > 0


class TestIPSRunEdgeCases:
    """Test run() edge cases - covers lines 452-459."""

    def test_run_all_entities_insufficient_data(self):
        """All entities have insufficient data → ValueError - covers line 459."""
        # Create panel with very short time series and high lags
        data_list = []
        for i in range(5):
            data_list.append(pd.DataFrame({"entity": i, "time": range(3), "y": [1.0, 2.0, 3.0]}))
        data = pd.concat(data_list, ignore_index=True)
        test = IPSTest(data, "y", "entity", "time", lags=10, trend="c")
        with pytest.raises(ValueError, match="Insufficient data"):
            test.run()

    def test_run_some_entities_nan_stats(self):
        """Some entities have valid data, some don't - covers lines 452-456."""
        np.random.seed(42)
        data_list = []
        # Good entities with enough data
        for i in range(5):
            y = np.random.randn(50).cumsum()
            data_list.append(pd.DataFrame({"entity": i, "time": range(50), "y": y}))
        # Bad entities with very short data
        for i in range(5, 8):
            data_list.append(pd.DataFrame({"entity": i, "time": range(3), "y": [1.0, 2.0, 3.0]}))
        data = pd.concat(data_list, ignore_index=True)
        test = IPSTest(data, "y", "entity", "time", lags=5, trend="c")
        result = test.run()
        # Only the good entities should be included
        assert result.n_entities == 5
        assert len(result.individual_stats) == 5

    def test_run_entity_with_constant_data_skipped(self):
        """Entity with constant data produces NaN t-stat, gets skipped - covers 452-456."""
        np.random.seed(42)
        data_list = []
        # Good entities
        for i in range(5):
            y = np.random.randn(50).cumsum()
            data_list.append(pd.DataFrame({"entity": i, "time": range(50), "y": y}))
        # Entity with constant data → NaN t-stat
        data_list.append(pd.DataFrame({"entity": 99, "time": range(50), "y": np.ones(50)}))
        data = pd.concat(data_list, ignore_index=True)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        result = test.run()
        # Entity 99 should be excluded
        assert 99 not in result.individual_stats
        assert result.n_entities == 5


class TestIPSResultStr:
    """Test IPSTestResult __str__ with variable lags."""

    def test_str_with_list_lags(self):
        """Test string representation when lags is a list - covers line 77 and 91."""
        result = IPSTestResult(
            statistic=-2.5,
            t_bar=-1.8,
            pvalue=0.006,
            lags=[1, 2, 3, 1, 2],
            n_obs=200,
            n_entities=5,
            individual_stats={0: -2.0, 1: -1.5, 2: -2.5, 3: -1.0, 4: -1.8},
            test_type="IPS",
            deterministics="Constant",
        )
        result_str = str(result)
        assert "Variable" in result_str
        assert "mean=" in result_str

    def test_conclusion_fail_to_reject(self):
        """Test conclusion when pvalue >= 0.05 - covers line 77."""
        result = IPSTestResult(
            statistic=0.5,
            t_bar=-0.5,
            pvalue=0.70,
            lags=1,
            n_obs=200,
            n_entities=5,
            individual_stats={0: -0.5},
            test_type="IPS",
            deterministics="Constant",
        )
        assert "Fail to reject" in result.conclusion


class TestIPSAutoLagWithTrend:
    """Test auto lag selection with different trends."""

    def test_auto_lags_trend_ct(self):
        """Test auto lag selection with trend='ct' - covers lines 256-258 via auto lag."""
        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="ct")
        result = test.run()
        assert isinstance(result, IPSTestResult)

    def test_auto_lags_trend_n(self):
        """Test auto lag selection with trend='n'."""
        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="n")
        result = test.run()
        assert isinstance(result, IPSTestResult)


class TestIPSRunNoLagsNoTrend:
    """Test the simple ADF branch (no lags, no trend) via run()."""

    def test_run_no_lags_no_trend(self):
        """Test full run with lags=0 and trend='n' - exercises simple ADF branch (lines 335-349)."""
        np.random.seed(42)
        data = _generate_stationary_panel(n_entities=5, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=0, trend="n")
        result = test.run()
        assert isinstance(result, IPSTestResult)
        assert result.n_entities == 5
        assert result.deterministics == "None"


class TestIPSSimpleADFException:
    """Test exception in simple ADF branch (lines 348-349) and lag selection exception (229-230)."""

    def test_adf_simple_branch_exception(self):
        """Force exception in the simple ADF (else) branch - covers lines 348-349."""
        from unittest.mock import patch

        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=0, trend="n")
        entity_data = np.random.randn(50).cumsum()

        # Patch lstsq to raise an exception inside the simple ADF branch
        with patch("numpy.linalg.lstsq", side_effect=np.linalg.LinAlgError("singular")):
            t_stat, n_obs = test._adf_test_entity(entity_data, lags=0)
        assert np.isnan(t_stat)
        assert n_obs == 0

    def test_select_lags_aic_exception(self):
        """Force _compute_aic_entity to raise exception during lag selection - covers lines 229-230."""
        from unittest.mock import patch

        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
        entity_data = np.random.randn(50).cumsum()

        # Make _compute_aic_entity raise for some calls and return values for others
        call_count = [0]
        original_compute_aic = test._compute_aic_entity

        def mock_compute_aic(data, lags):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise ValueError("simulated AIC failure")
            return original_compute_aic(data, lags)

        with patch.object(test, "_compute_aic_entity", side_effect=mock_compute_aic):
            result = test._select_lags_for_entity(entity_data, max_lags=4)
        assert isinstance(result, int)
        assert result >= 0

    def test_adf_entity_exception_in_main_branch(self):
        """Force exception in the main (non-simple) ADF branch - covers lines 333-334."""
        from unittest.mock import patch

        data = _generate_stationary_panel(n_entities=3, n_time=50)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        entity_data = np.random.randn(50).cumsum()

        # Patch np.linalg.inv to raise (used in the main ADF branch for X_cov)

        def mock_inv(*args, **kwargs):
            raise np.linalg.LinAlgError("singular matrix")

        with patch("numpy.linalg.inv", side_effect=mock_inv):
            t_stat, n_obs = test._adf_test_entity(entity_data, lags=1)
        assert np.isnan(t_stat)
        assert n_obs == 0


class TestIPSCriticalValuesBranches:
    """Additional critical value tests for specific T ranges."""

    def test_get_critical_values_c_medium(self):
        """Directly test _get_critical_values for trend='c', T in (25,50] - covers lines 384-385."""
        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        crit = test._get_critical_values(35)
        assert crit["mean"] == -1.66
        assert crit["std"] == 0.96

    def test_get_critical_values_c_long(self):
        """Directly test _get_critical_values for trend='c', T > 50 - covers lines 390-391."""
        data = _generate_stationary_panel(n_entities=3, n_time=10)
        test = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
        crit = test._get_critical_values(60)
        assert crit["mean"] == -1.73
        assert crit["std"] == 1.00
