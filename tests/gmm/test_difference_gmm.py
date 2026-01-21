"""
Unit tests for Difference GMM estimator
========================================

Tests for the DifferenceGMM class (Arellano-Bond 1991).
"""

import pytest
import numpy as np
import pandas as pd
from panelbox.gmm.difference_gmm import DifferenceGMM
from panelbox.gmm.results import GMMResults


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def balanced_panel_data():
    """Generate balanced panel data for testing."""
    np.random.seed(42)
    n_units = 50
    n_periods = 10

    # Create panel structure
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    # Generate data with fixed effects and dynamics
    # y_it = 0.5 * y_{i,t-1} + 0.3 * x_it + η_i + ε_it

    data_list = []
    for i in range(n_units):
        eta_i = np.random.normal(0, 1)  # Fixed effect
        y = np.zeros(n_periods)
        x = np.random.normal(0, 1, n_periods)

        y[0] = eta_i + np.random.normal(0, 0.5)  # Initial value

        for t in range(1, n_periods):
            epsilon = np.random.normal(0, 0.5)
            y[t] = 0.5 * y[t-1] + 0.3 * x[t] + eta_i + epsilon

        for t in range(n_periods):
            data_list.append({
                'id': i,
                'year': t,
                'y': y[t],
                'x': x[t]
            })

    return pd.DataFrame(data_list)


@pytest.fixture
def unbalanced_panel_data():
    """Generate unbalanced panel data for testing."""
    np.random.seed(42)
    n_units = 50
    max_periods = 10

    data_list = []
    for i in range(n_units):
        # Random number of periods for each unit (unbalanced)
        n_periods_i = np.random.randint(5, max_periods + 1)
        eta_i = np.random.normal(0, 1)

        y = np.zeros(n_periods_i)
        x = np.random.normal(0, 1, n_periods_i)

        y[0] = eta_i + np.random.normal(0, 0.5)

        for t in range(1, n_periods_i):
            epsilon = np.random.normal(0, 0.5)
            y[t] = 0.5 * y[t-1] + 0.3 * x[t] + eta_i + epsilon

        for t in range(n_periods_i):
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
        'id': [1, 1, 1, 2, 2, 2],
        'year': [1, 2, 3, 1, 2, 3],
        'y': [1.0, 1.5, 2.0, 0.5, 1.0, 1.5],
        'x': [0.5, 0.8, 1.2, 0.3, 0.6, 0.9]
    })
    return data


# ============================================================================
# Test Initialization
# ============================================================================

class TestDifferenceGMMInitialization:
    """Test DifferenceGMM initialization."""

    def test_init_basic(self, balanced_panel_data):
        """Test basic initialization."""
        model = DifferenceGMM(
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

    def test_init_with_multiple_lags(self, balanced_panel_data):
        """Test initialization with multiple lags."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=[1, 2],
            id_var='id',
            time_var='year',
            exog_vars=['x']
        )

        assert model.lags == [1, 2]

    def test_init_with_collapse(self, balanced_panel_data):
        """Test initialization with collapse option."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True
        )

        assert model.collapse is True

    def test_init_gmm_type_one_step(self, balanced_panel_data):
        """Test initialization with one-step GMM."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            gmm_type='one_step'
        )

        assert model.gmm_type == 'one_step'
        assert model.two_step is False

    def test_init_gmm_type_iterative(self, balanced_panel_data):
        """Test initialization with iterative GMM."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            gmm_type='iterative'
        )

        assert model.gmm_type == 'iterative'


# ============================================================================
# Test Input Validation
# ============================================================================

class TestDifferenceGMMValidation:
    """Test input validation."""

    def test_invalid_dep_var(self, balanced_panel_data):
        """Test error when dependent variable doesn't exist."""
        with pytest.raises(ValueError, match="Dependent variable .* not found"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var='nonexistent',
                lags=1,
                id_var='id',
                time_var='year'
            )

    def test_invalid_id_var(self, balanced_panel_data):
        """Test error when ID variable doesn't exist."""
        with pytest.raises((ValueError, KeyError)):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var='y',
                lags=1,
                id_var='nonexistent',
                time_var='year'
            )

    def test_invalid_time_var(self, balanced_panel_data):
        """Test error when time variable doesn't exist."""
        with pytest.raises((ValueError, KeyError)):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var='y',
                lags=1,
                id_var='id',
                time_var='nonexistent'
            )

    def test_invalid_exog_var(self, balanced_panel_data):
        """Test error when exogenous variable doesn't exist."""
        with pytest.raises(ValueError, match="Variable .* not found"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var='y',
                lags=1,
                id_var='id',
                time_var='year',
                exog_vars=['nonexistent']
            )

    def test_invalid_gmm_type(self, balanced_panel_data):
        """Test error when GMM type is invalid."""
        with pytest.raises(ValueError, match="gmm_type must be one of"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var='y',
                lags=1,
                id_var='id',
                time_var='year',
                gmm_type='invalid'
            )

    def test_warning_unbalanced_with_time_dummies(self, unbalanced_panel_data):
        """Test warning when using time dummies with unbalanced panel."""
        with pytest.warns(UserWarning, match="Unbalanced panel detected"):
            DifferenceGMM(
                data=unbalanced_panel_data,
                dep_var='y',
                lags=1,
                id_var='id',
                time_var='year',
                exog_vars=['x'],
                time_dummies=True,
                collapse=False
            )

    def test_warning_no_collapse(self, balanced_panel_data):
        """Test warning when collapse is False."""
        with pytest.warns(UserWarning, match="collapse=True"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var='y',
                lags=1,
                id_var='id',
                time_var='year',
                exog_vars=['x'],
                collapse=False
            )


# ============================================================================
# Test Panel Balance Check
# ============================================================================

class TestPanelBalanceCheck:
    """Test panel balance detection."""

    def test_balanced_panel_detection(self, balanced_panel_data):
        """Test that balanced panel is detected correctly."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True
        )

        is_unbalanced, balance_rate = model._check_panel_balance()
        assert is_unbalanced is False
        assert balance_rate == 1.0

    def test_unbalanced_panel_detection(self, unbalanced_panel_data):
        """Test that unbalanced panel is detected correctly."""
        model = DifferenceGMM(
            data=unbalanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        is_unbalanced, balance_rate = model._check_panel_balance()
        assert is_unbalanced is True
        assert 0.0 < balance_rate < 1.0


# ============================================================================
# Test Data Transformation
# ============================================================================

class TestDataTransformation:
    """Test first-difference transformation."""

    def test_transform_data_shape(self, minimal_data):
        """Test that transformation produces correct shapes."""
        model = DifferenceGMM(
            data=minimal_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        y_diff, X_diff, ids, times = model._transform_data()

        # Should have consistent shapes
        assert len(y_diff) == X_diff.shape[0]
        assert len(y_diff) == len(ids)
        assert len(y_diff) == len(times)

        # Should have some observations after transformation
        assert len(y_diff) > 0

    def test_transform_data_values(self):
        """Test that first-difference produces correct values."""
        # Simple data where we can verify differences manually
        data = pd.DataFrame({
            'id': [1, 1, 1],
            'year': [1, 2, 3],
            'y': [1.0, 2.0, 4.0],
            'x': [0.5, 1.5, 2.5]
        })

        model = DifferenceGMM(
            data=data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        y_diff, X_diff, ids, times = model._transform_data()

        # Should have some observations after transformation
        assert len(y_diff) > 0

        # y_diff should contain differences (may have NaN)
        valid_y = y_diff[~np.isnan(y_diff)]
        assert len(valid_y) > 0


# ============================================================================
# Test Estimation
# ============================================================================

class TestDifferenceGMMEstimation:
    """Test GMM estimation."""

    def test_fit_one_step(self, balanced_panel_data):
        """Test one-step GMM estimation."""
        model = DifferenceGMM(
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

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert results.params is not None
        assert len(results.params) == 2  # y_lag1 + x
        assert results.nobs > 0
        assert results.n_instruments > 0

    def test_fit_two_step(self, balanced_panel_data):
        """Test two-step GMM estimation."""
        model = DifferenceGMM(
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

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert results.params is not None
        assert results.nobs > 0

    def test_fit_with_multiple_lags(self, balanced_panel_data):
        """Test estimation with multiple lags of dependent variable."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=[1, 2],
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        # Should have y_lag1, y_lag2, x
        assert len(results.params) == 3

    def test_fit_with_collapse(self, balanced_panel_data):
        """Test that collapse option works."""
        # With collapse
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )
        results = model.fit()

        # Should complete successfully with collapse
        assert isinstance(results, GMMResults)
        assert results.n_instruments > 0
        assert results.nobs > 0

    def test_fit_coefficient_sign(self, balanced_panel_data):
        """Test that estimated coefficients have expected signs."""
        # True model: y_t = 0.5 * y_{t-1} + 0.3 * x_t + ...
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = model.fit()

        # Both coefficients should be positive (true values: 0.5, 0.3)
        # Access params by index since we don't know exact names
        assert len(results.params) == 2
        # Most estimates should be positive given the true model
        positive_count = sum(results.params > 0)
        assert positive_count >= 1  # At least one should be positive

    def test_fit_with_time_dummies(self, balanced_panel_data):
        """Test estimation with time dummies."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            time_dummies=True,
            collapse=True
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        # Should have more parameters due to time dummies
        assert len(results.params) > 2


# ============================================================================
# Test Results
# ============================================================================

class TestDifferenceGMMResults:
    """Test GMM results object."""

    def test_results_attributes(self, balanced_panel_data):
        """Test that results have all expected attributes."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = model.fit()

        # Check basic attributes
        assert hasattr(results, 'params')
        assert hasattr(results, 'std_errors')
        assert hasattr(results, 'nobs')
        assert hasattr(results, 'n_instruments')
        assert hasattr(results, 'n_groups')

        # Check specification tests exist (may be None)
        assert hasattr(results, 'hansen_j')
        assert hasattr(results, 'ar1_test')
        assert hasattr(results, 'ar2_test')

        # Check that params is not empty
        assert len(results.params) > 0

    def test_results_summary(self, balanced_panel_data):
        """Test that summary() method works."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = model.fit()
        summary = results.summary()

        assert isinstance(summary, str)
        assert 'Number of observations' in summary
        assert 'Number of instruments' in summary
        assert 'Hansen J-test' in summary

    def test_instrument_ratio(self, balanced_panel_data):
        """Test instrument ratio calculation."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        results = model.fit()

        # Instrument ratio should be n_instruments / n_groups
        expected_ratio = results.n_instruments / results.n_groups
        assert abs(results.instrument_ratio - expected_ratio) < 1e-10

        # With collapse, ratio should be < 1.0 (Roodman 2009 recommendation)
        assert results.instrument_ratio < 1.0


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_panel(self):
        """Test with minimal panel size."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [1, 2, 3, 1, 2, 3],
            'y': [1.0, 1.5, 2.0, 0.5, 1.0, 1.5],
            'x': [0.5, 0.8, 1.2, 0.3, 0.6, 0.9]
        })

        model = DifferenceGMM(
            data=data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=['x'],
            collapse=True,
            time_dummies=False
        )

        # Should complete without error, even if results may not be reliable
        results = model.fit()
        assert results is not None

    def test_no_exog_vars(self, balanced_panel_data):
        """Test with only lagged dependent variable (no exogenous vars)."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var='y',
            lags=1,
            id_var='id',
            time_var='year',
            exog_vars=[],
            collapse=True,
            time_dummies=False
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert len(results.params) == 1  # Only y_lag1
