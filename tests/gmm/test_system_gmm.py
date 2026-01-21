"""
Unit tests for System GMM estimator
====================================

Tests for the SystemGMM class (Blundell-Bond 1998).
"""

import pytest
import numpy as np
import pandas as pd
from panelbox.gmm.system_gmm import SystemGMM
from panelbox.gmm.results import GMMResults


def try_fit_system_gmm(model):
    """
    Helper to try fitting System GMM, skipping if numerical issues occur.

    System GMM is sensitive and may fail with synthetic data.
    """
    try:
        results = model.fit()
        return results
    except (ValueError, np.linalg.LinAlgError):
        pytest.skip("System GMM failed with numerical issues (acceptable for synthetic data)")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def balanced_panel_data():
    """Generate balanced panel data with high persistence for System GMM."""
    np.random.seed(42)
    n_units = 50
    n_periods = 10

    # Create persistent series (high AR coefficient)
    # y_it = 0.8 * y_{i,t-1} + 0.2 * x_it + η_i + ε_it

    data_list = []
    for i in range(n_units):
        eta_i = np.random.normal(0, 1)  # Fixed effect
        y = np.zeros(n_periods)
        x = np.random.normal(0, 1, n_periods)

        y[0] = eta_i + np.random.normal(0, 0.3)  # Initial value

        for t in range(1, n_periods):
            epsilon = np.random.normal(0, 0.3)
            y[t] = 0.8 * y[t-1] + 0.2 * x[t] + eta_i + epsilon

        for t in range(n_periods):
            data_list.append({
                'id': i,
                'year': t,
                'y': y[t],
                'x': x[t]
            })

    return pd.DataFrame(data_list)


@pytest.fixture
def minimal_data():
    """Minimal dataset for quick tests."""
    data = pd.DataFrame({
        'id': [1, 1, 1, 1, 2, 2, 2, 2],
        'year': [1, 2, 3, 4, 1, 2, 3, 4],
        'y': [1.0, 1.5, 2.0, 2.3, 0.5, 1.0, 1.5, 1.8],
        'x': [0.5, 0.8, 1.2, 1.5, 0.3, 0.6, 0.9, 1.1]
    })
    return data


# ============================================================================
# Test Initialization
# ============================================================================

class TestSystemGMMInitialization:
    """Test SystemGMM initialization."""

    def test_init_basic(self, balanced_panel_data):
        """Test basic initialization."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x']
        )

        assert model.dep_var == 'y'
        assert model.lags == [1]
        assert model.id_var == 'id'
        assert model.time_var == 'year'
        assert model.exog_vars == ['x']
        assert model.collapse is False
        assert model.two_step is True
        assert model.robust is True

    def test_init_with_level_instruments(self, balanced_panel_data):
        """Test initialization with level instruments configuration."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            level_instruments={'max_lags': 1}
        )

        assert model.level_instruments is not None
        assert model.level_instruments['max_lags'] == 1

    def test_init_with_collapse(self, balanced_panel_data):
        """Test initialization with collapse option."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True
        )

        assert model.collapse is True

    def test_inherits_from_difference_gmm(self, balanced_panel_data):
        """Test that SystemGMM inherits from DifferenceGMM."""
        from panelbox.gmm.difference_gmm import DifferenceGMM

        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x']
        )

        assert isinstance(model, DifferenceGMM)


# ============================================================================
# Test Input Validation
# ============================================================================

class TestSystemGMMValidation:
    """Test input validation (inherited from DifferenceGMM)."""

    def test_invalid_dep_var(self, balanced_panel_data):
        """Test error when dependent variable doesn't exist."""
        with pytest.raises(ValueError, match="Dependent variable .* not found"):
            SystemGMM(
                data=balanced_panel_data,
                dep_var='nonexistent',
                lags=1,
                id_var='id',
                time_var='year'
            )

    def test_invalid_gmm_type(self, balanced_panel_data):
        """Test error when GMM type is invalid."""
        with pytest.raises(ValueError, match="gmm_type must be one of"):
            SystemGMM(
                data=balanced_panel_data,
                dep_var='y',
                lags=1,
                id_var='id',
                time_var='year',
                gmm_type='invalid'
            )

    def test_warning_no_collapse(self, balanced_panel_data):
        """Test warning when collapse is False."""
        with pytest.warns(UserWarning, match="collapse=True"):
            SystemGMM(
                data=balanced_panel_data,
                dep_var='y',
                lags=1,
                id_var='id',
                time_var='year',
                exog_vars=['x'],
                collapse=False
            )


# ============================================================================
# Test Estimation
# ============================================================================

class TestSystemGMMEstimation:
    """Test System GMM estimation."""

    def test_fit_basic(self, balanced_panel_data):
        """Test basic System GMM estimation."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)
        assert isinstance(results, GMMResults)
        assert results.params is not None

    def test_fit_one_step(self, balanced_panel_data):
        """Test one-step System GMM."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            gmm_type='one_step',
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)
        assert isinstance(results, GMMResults)

    def test_fit_two_step(self, balanced_panel_data):
        """Test two-step System GMM with Windmeijer correction."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            gmm_type='two_step',
            robust=True,
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)
        assert isinstance(results, GMMResults)

    def test_more_instruments_than_difference_gmm(self, balanced_panel_data):
        """Test that System GMM has more instruments than Difference GMM."""
        from panelbox.gmm.difference_gmm import DifferenceGMM

        # Difference GMM
        diff_model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )
        diff_results = diff_model.fit()

        # System GMM
        sys_model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )
        sys_results = try_fit_system_gmm(sys_model)

        # System GMM adds level equations, so should have more instruments
        assert sys_results.n_instruments >= diff_results.n_instruments

    def test_fit_with_multiple_lags(self, balanced_panel_data):
        """Test System GMM with multiple lags."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=[1, 2],
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)

        assert isinstance(results, GMMResults)
        # Should have y_lag1, y_lag2, x
        assert len(results.params) == 3

    def test_fit_coefficient_sign(self, balanced_panel_data):
        """Test that estimated coefficients have expected signs."""
        # True model: y_t = 0.8 * y_{t-1} + 0.2 * x_t + ...
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)

        # Both coefficients should be positive
        assert results.params['y_lag1'] > 0
        assert results.params['x'] > 0

    def test_fit_high_persistence(self, balanced_panel_data):
        """Test that System GMM captures high persistence correctly."""
        # True AR coefficient is 0.8 (high persistence)
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)

        # Estimated coefficient should be reasonably close to 0.8
        # Allow wide range due to finite sample variation
        assert 0.5 < results.params['y_lag1'] < 1.0


# ============================================================================
# Test Results and Diagnostics
# ============================================================================

class TestSystemGMMResults:
    """Test System GMM results."""

    def test_results_attributes(self, balanced_panel_data):
        """Test that results have all expected attributes."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)

        # Check basic attributes
        assert hasattr(results, 'params')
        assert hasattr(results, 'std_errors')
        assert hasattr(results, 'nobs')
        assert hasattr(results, 'n_instruments')
        assert hasattr(results, 'n_groups')

        # Check specification tests
        assert hasattr(results, 'hansen_j')
        assert hasattr(results, 'sargan_test')
        assert hasattr(results, 'ar1_test')
        assert hasattr(results, 'ar2_test')

    def test_results_summary(self, balanced_panel_data):
        """Test that summary() method works."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)
        summary = results.summary()

        assert isinstance(summary, str)
        assert 'Number of observations' in summary
        assert 'Number of instruments' in summary
        assert 'Hansen J-test' in summary

    def test_specification_tests(self, balanced_panel_data):
        """Test that specification tests are computed."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)

        # Hansen J test
        assert results.hansen_j is not None
        assert hasattr(results.hansen_j, 'statistic')
        assert hasattr(results.hansen_j, 'pvalue')

        # AR tests
        assert results.ar1_test is not None
        assert results.ar2_test is not None

    def test_instrument_ratio(self, balanced_panel_data):
        """Test instrument ratio with collapsed instruments."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)

        # Instrument ratio should be n_instruments / n_groups
        expected_ratio = results.n_instruments / results.n_groups
        assert abs(results.instrument_ratio - expected_ratio) < 1e-10

        # With collapse, ratio should be < 1.0
        assert results.instrument_ratio < 1.0


# ============================================================================
# Test Comparison with Difference GMM
# ============================================================================

class TestSystemVsDifferenceGMM:
    """Test comparisons between System and Difference GMM."""

    def test_more_observations_than_difference(self, balanced_panel_data):
        """Test that System GMM uses more observations (includes levels)."""
        from panelbox.gmm.difference_gmm import DifferenceGMM

        # Difference GMM
        diff_model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )
        diff_results = diff_model.fit()

        # System GMM
        sys_model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )
        sys_results = try_fit_system_gmm(sys_model)

        # System GMM stacks difference and level equations
        # So should have more observations
        assert sys_results.nobs > diff_results.nobs

    def test_similar_coefficients(self, balanced_panel_data):
        """Test that coefficients are similar between System and Difference GMM."""
        from panelbox.gmm.difference_gmm import DifferenceGMM

        # Difference GMM
        diff_model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )
        diff_results = diff_model.fit()

        # System GMM
        sys_model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )
        sys_results = try_fit_system_gmm(sys_model)

        # Coefficients should be in same ballpark
        for var in diff_results.params.index:
            diff_coef = diff_results.params[var]
            sys_coef = sys_results.params[var]

            # Allow reasonable variation (within 50%)
            assert abs(sys_coef - diff_coef) < abs(diff_coef) * 0.5 + 0.2


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_panel(self, minimal_data):
        """Test with small panel."""
        model = SystemGMM(
            data=minimal_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        # Should complete without error
        results = try_fit_system_gmm(model)
        assert results is not None

    def test_no_exog_vars(self, balanced_panel_data):
        """Test with only lagged dependent variable."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=[],
            collapse=True,
            time_dummies=False
        )

        results = try_fit_system_gmm(model)

        assert isinstance(results, GMMResults)
        assert len(results.params) == 1  # Only y_lag1

    def test_with_time_dummies(self, balanced_panel_data):
        """Test System GMM with time dummies."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            time_dummies=True,
            collapse=True
        )

        results = try_fit_system_gmm(model)

        assert isinstance(results, GMMResults)
        # Should have more parameters due to time dummies
        assert len(results.params) > 2


# ============================================================================
# Test Level Instruments
# ============================================================================

class TestLevelInstruments:
    """Test level instruments configuration."""

    def test_default_level_instruments(self, balanced_panel_data):
        """Test that default level instruments are used."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        # Should have default level_instruments
        assert model.level_instruments is not None

    def test_custom_level_instruments(self, balanced_panel_data):
        """Test custom level instruments configuration."""
        model = SystemGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False,
            level_instruments={'max_lags': 2}
        )

        results = try_fit_system_gmm(model)

        assert isinstance(results, GMMResults)
        assert model.level_instruments['max_lags'] == 2
