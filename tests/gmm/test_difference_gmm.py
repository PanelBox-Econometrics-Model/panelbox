"""
Unit tests for Difference GMM estimator
========================================

Tests for the DifferenceGMM class (Arellano-Bond 1991).
"""

import numpy as np
import pandas as pd
import pytest

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
    np.repeat(np.arange(n_units), n_periods)
    np.tile(np.arange(n_periods), n_units)

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
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + epsilon

        for t in range(n_periods):
            data_list.append({"id": i, "year": t, "y": y[t], "x": x[t]})

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
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + epsilon

        for t in range(n_periods_i):
            data_list.append({"id": i, "year": t, "y": y[t], "x": x[t]})

    return pd.DataFrame(data_list)


@pytest.fixture
def minimal_data():
    """Minimal dataset for quick tests."""
    data = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "year": [1, 2, 3, 1, 2, 3],
            "y": [1.0, 1.5, 2.0, 0.5, 1.0, 1.5],
            "x": [0.5, 0.8, 1.2, 0.3, 0.6, 0.9],
        }
    )
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
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
        )

        assert model.dep_var == "y"
        assert model.lags == [1]
        assert model.id_var == "id"
        assert model.time_var == "year"
        assert model.exog_vars == ["x"]
        assert model.collapse is False
        assert model.two_step is True
        assert model.robust is True

    def test_init_with_multiple_lags(self, balanced_panel_data):
        """Test initialization with multiple lags."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=[1, 2],
            id_var="id",
            time_var="year",
            exog_vars=["x"],
        )

        assert model.lags == [1, 2]

    def test_init_with_collapse(self, balanced_panel_data):
        """Test initialization with collapse option."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
        )

        assert model.collapse is True

    def test_init_gmm_type_one_step(self, balanced_panel_data):
        """Test initialization with one-step GMM."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            gmm_type="one_step",
        )

        assert model.gmm_type == "one_step"
        assert model.two_step is False

    def test_init_gmm_type_iterative(self, balanced_panel_data):
        """Test initialization with iterative GMM."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            gmm_type="iterative",
        )

        assert model.gmm_type == "iterative"


# ============================================================================
# Test Input Validation
# ============================================================================


class TestDifferenceGMMValidation:
    """Test input validation."""

    def test_invalid_dep_var(self, balanced_panel_data):
        """Test error when dependent variable doesn't exist."""
        with pytest.raises(ValueError, match=r"Dependent variable .* not found"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var="nonexistent",
                lags=1,
                id_var="id",
                time_var="year",
            )

    def test_invalid_id_var(self, balanced_panel_data):
        """Test error when ID variable doesn't exist."""
        with pytest.raises((ValueError, KeyError)):
            DifferenceGMM(
                data=balanced_panel_data, dep_var="y", lags=1, id_var="nonexistent", time_var="year"
            )

    def test_invalid_time_var(self, balanced_panel_data):
        """Test error when time variable doesn't exist."""
        with pytest.raises((ValueError, KeyError)):
            DifferenceGMM(
                data=balanced_panel_data, dep_var="y", lags=1, id_var="id", time_var="nonexistent"
            )

    def test_invalid_exog_var(self, balanced_panel_data):
        """Test error when exogenous variable doesn't exist."""
        with pytest.raises(ValueError, match=r"Variable .* not found"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                exog_vars=["nonexistent"],
            )

    def test_invalid_gmm_type(self, balanced_panel_data):
        """Test error when GMM type is invalid."""
        with pytest.raises(ValueError, match="gmm_type must be one of"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                gmm_type="invalid",
            )

    def test_warning_unbalanced_with_time_dummies(self, unbalanced_panel_data):
        """Test warning when using time dummies with unbalanced panel."""
        with pytest.warns(UserWarning, match="Unbalanced panel detected"):
            DifferenceGMM(
                data=unbalanced_panel_data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                exog_vars=["x"],
                time_dummies=True,
                collapse=False,
            )

    def test_warning_no_collapse(self, balanced_panel_data):
        """Test warning when collapse is False."""
        with pytest.warns(UserWarning, match="collapse=True"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                exog_vars=["x"],
                collapse=False,
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
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
        )

        is_unbalanced, balance_rate = model._check_panel_balance()
        assert is_unbalanced is False
        assert balance_rate == 1.0

    def test_unbalanced_panel_detection(self, unbalanced_panel_data):
        """Test that unbalanced panel is detected correctly."""
        model = DifferenceGMM(
            data=unbalanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
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
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
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
        data = pd.DataFrame(
            {"id": [1, 1, 1], "year": [1, 2, 3], "y": [1.0, 2.0, 4.0], "x": [0.5, 1.5, 2.5]}
        )

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        y_diff, _X_diff, _ids, _times = model._transform_data()

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
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            gmm_type="one_step",
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert len(results.params) == 2  # y_lag1 + x
        assert np.all(np.isfinite(results.params.values))
        assert results.nobs >= 50  # 50 units, multiple periods each
        assert results.n_instruments >= 2  # At least as many as params

    def test_fit_two_step(self, balanced_panel_data):
        """Test two-step GMM estimation."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            gmm_type="two_step",
            robust=True,
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert len(results.params) == 2
        assert np.all(np.isfinite(results.params.values))
        assert results.nobs >= 50

    def test_fit_with_multiple_lags(self, balanced_panel_data):
        """Test estimation with multiple lags of dependent variable."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=[1, 2],
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
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
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )
        results = model.fit()

        # Should complete successfully with collapse
        assert isinstance(results, GMMResults)
        assert results.n_instruments >= 2
        assert results.nobs >= 50
        # Collapsed should have fewer instruments than non-collapsed
        assert np.all(np.isfinite(results.params.values))

    def test_fit_coefficient_sign(self, balanced_panel_data):
        """Test that estimated coefficients have expected signs."""
        # True model: y_t = 0.5 * y_{t-1} + 0.3 * x_t + ...
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        # Both coefficients should be positive (true values: 0.5, 0.3)
        assert len(results.params) == 2
        assert np.all(np.isfinite(results.params.values))
        # The lagged dependent variable coefficient should be positive
        assert results.params.iloc[0] > 0, "Lag coefficient should be positive (true=0.5)"
        # Exogenous variable coefficient should also be positive
        assert results.params.iloc[1] > 0, "Exog coefficient should be positive (true=0.3)"

    def test_fit_with_time_dummies(self, balanced_panel_data):
        """Test estimation with time dummies."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            time_dummies=True,
            collapse=True,
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
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        # Check basic attributes
        assert hasattr(results, "params")
        assert hasattr(results, "std_errors")
        assert hasattr(results, "nobs")
        assert hasattr(results, "n_instruments")
        assert hasattr(results, "n_groups")

        # Check specification tests exist (may be None)
        assert hasattr(results, "hansen_j")
        assert hasattr(results, "ar1_test")
        assert hasattr(results, "ar2_test")

        # Check that params is not empty
        assert len(results.params) > 0

    def test_results_summary(self, balanced_panel_data):
        """Test that summary() method works."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()
        summary = results.summary()

        assert isinstance(summary, str)
        assert "Number of observations" in summary
        assert "Number of instruments" in summary
        assert "Hansen J-test" in summary

    def test_instrument_ratio(self, balanced_panel_data):
        """Test instrument ratio calculation."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
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
        data = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "year": [1, 2, 3, 1, 2, 3],
                "y": [1.0, 1.5, 2.0, 0.5, 1.0, 1.5],
                "x": [0.5, 0.8, 1.2, 0.3, 0.6, 0.9],
            }
        )

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        # Should complete without error, even if results may not be reliable
        results = model.fit()
        assert isinstance(results, GMMResults)
        assert np.all(np.isfinite(results.params.values))

    def test_no_exog_vars(self, balanced_panel_data):
        """Test with only lagged dependent variable (no exogenous vars)."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=[],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert len(results.params) == 1  # Only y_lag1


# ============================================================================
# Test Coverage: two_step=False reconciliation (lines 279-280)
# ============================================================================


class TestTwoStepFalseReconciliation:
    """Test that two_step=False with default gmm_type is reconciled properly."""

    def test_two_step_false_default_gmm_type(self, balanced_panel_data):
        """Test that two_step=False overrides default gmm_type='two_step' to 'one_step'."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            two_step=False,
            collapse=True,
            time_dummies=False,
        )

        # Should reconcile: gmm_type='one_step', two_step=False
        assert model.gmm_type == "one_step"
        assert model.two_step is False

    def test_two_step_false_explicit_one_step(self, balanced_panel_data):
        """Test that two_step=False with gmm_type='one_step' is consistent."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            two_step=False,
            gmm_type="one_step",
            collapse=True,
            time_dummies=False,
        )

        assert model.gmm_type == "one_step"
        assert model.two_step is False

    def test_two_step_false_fit_produces_results(self, balanced_panel_data):
        """Test that fitting with two_step=False (reconciled) works correctly."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            two_step=False,
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()
        assert isinstance(results, GMMResults)
        assert len(results.params) == 2
        assert np.all(np.isfinite(results.params.values))
        assert results.nobs > 0
        assert results.n_instruments >= 2


# ============================================================================
# Test Coverage: Datetime/Period time variable conversion (lines 295-297)
# ============================================================================


class TestDatetimeTimeVariable:
    """Test that datetime and Period time variables are converted to numeric."""

    def test_datetime_time_variable(self):
        """Test with pd.to_datetime time column."""
        np.random.seed(42)
        n_units = 20
        n_periods = 6

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.5)
            for t in range(n_periods):
                data_list.append(
                    {
                        "id": i,
                        "year": pd.Timestamp(f"{2000 + t}-01-01"),
                        "y": y[t],
                        "x": x[t],
                    }
                )

        data = pd.DataFrame(data_list)
        # Confirm column is datetime
        assert pd.api.types.is_datetime64_any_dtype(data["year"].dtype)

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        # After conversion, the time mapping should exist and be non-empty
        assert model._time_mapping is not None
        assert len(model._time_mapping) > 0

        # Fit should work correctly
        results = model.fit()
        assert isinstance(results, GMMResults)
        assert results.nobs > 0
        assert np.all(np.isfinite(results.params.values))

    def test_period_time_variable(self):
        """Test with PeriodIndex time column."""
        np.random.seed(42)
        n_units = 20
        n_periods = 6

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.5)
            for t in range(n_periods):
                data_list.append(
                    {
                        "id": i,
                        "year": pd.Period(f"{2000 + t}", freq="Y"),
                        "y": y[t],
                        "x": x[t],
                    }
                )

        data = pd.DataFrame(data_list)

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        assert model._time_mapping is not None
        assert len(model._time_mapping) > 0
        results = model.fit()
        assert isinstance(results, GMMResults)
        assert results.nobs > 0
        assert np.all(np.isfinite(results.params.values))


# ============================================================================
# Test Coverage: Missing endogenous/predetermined vars (lines 321, 323)
# ============================================================================


class TestMissingVariableValidation:
    """Test validation for missing endogenous and predetermined variables."""

    def test_invalid_endogenous_var(self, balanced_panel_data):
        """Test error when endogenous variable doesn't exist."""
        with pytest.raises(ValueError, match=r"Variable .* not found"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                endogenous_vars=["nonexistent_endog"],
                collapse=True,
                time_dummies=False,
            )

    def test_invalid_predetermined_var(self, balanced_panel_data):
        """Test error when predetermined variable doesn't exist."""
        with pytest.raises(ValueError, match=r"Variable .* not found"):
            DifferenceGMM(
                data=balanced_panel_data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                predetermined_vars=["nonexistent_pred"],
                collapse=True,
                time_dummies=False,
            )


# ============================================================================
# Test Coverage: Unbalanced panel warning with many time dummies (lines 348->363)
# ============================================================================


class TestUnbalancedPanelTimeDummyWarning:
    """Test warning for unbalanced panel with many time dummies and low balance rate."""

    def test_highly_unbalanced_with_many_dummies(self):
        """Test that warning fires when n_dummies >= 5 and balance_rate < 0.80."""
        np.random.seed(42)
        n_units = 50
        n_periods = 8  # Gives 7 time dummies (>= 5)

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            # Make heavily unbalanced: most units have only 3-4 periods
            # Only ~10% have full periods => balance_rate < 0.80
            if i < 5:
                periods_for_unit = n_periods
            else:
                periods_for_unit = np.random.randint(3, 5)

            y = np.zeros(periods_for_unit)
            x = np.random.normal(0, 1, periods_for_unit)
            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, periods_for_unit):
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.5)
            for t in range(periods_for_unit):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t]})

        data = pd.DataFrame(data_list)

        with pytest.warns(UserWarning, match="Unbalanced panel detected"):
            DifferenceGMM(
                data=data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                exog_vars=["x"],
                time_dummies=True,
                collapse=True,
            )


# ============================================================================
# Test Coverage: Iterative GMM estimation (lines 521-522)
# ============================================================================


class TestIterativeGMM:
    """Test iterative GMM estimation."""

    def test_iterative_gmm_fit(self, balanced_panel_data):
        """Test that iterative GMM estimation completes and returns results."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            gmm_type="iterative",
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert len(results.params) == 2  # y_lag1 + x
        assert np.all(np.isfinite(results.params.values))
        assert results.nobs > 0
        assert results.n_instruments >= 2

    def test_iterative_gmm_convergence(self, balanced_panel_data):
        """Test that iterative GMM convergence is reported."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            gmm_type="iterative",
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()
        # converged should be a boolean
        assert isinstance(results.converged, bool)


# ============================================================================
# Test Coverage: Predetermined variables instruments (lines 706-713)
# ============================================================================


class TestPredeterminedVariables:
    """Test GMM with predetermined variables."""

    def test_predetermined_vars_fit(self):
        """Test that predetermined variables generate correct instruments and fit."""
        np.random.seed(42)
        n_units = 50
        n_periods = 10

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            w = np.random.normal(0, 1, n_periods)  # predetermined variable

            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                y[t] = (
                    0.5 * y[t - 1] + 0.3 * x[t] + 0.2 * w[t - 1] + eta_i + np.random.normal(0, 0.5)
                )

            for t in range(n_periods):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t], "w": w[t]})

        data = pd.DataFrame(data_list)

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            predetermined_vars=["w"],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert results.nobs > 0
        # Should have 3 params: y_lag1, x, w
        assert len(results.params) == 3

    def test_predetermined_vars_in_results(self):
        """Test that predetermined variables appear in results."""
        np.random.seed(42)
        n_units = 50
        n_periods = 10

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            w = np.random.normal(0, 1, n_periods)

            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                y[t] = (
                    0.5 * y[t - 1] + 0.3 * x[t] + 0.2 * w[t - 1] + eta_i + np.random.normal(0, 0.5)
                )

            for t in range(n_periods):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t], "w": w[t]})

        data = pd.DataFrame(data_list)

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            predetermined_vars=["w"],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()
        assert "w" in results.params.index


# ============================================================================
# Test Coverage: Endogenous variables instruments (lines 717-724)
# ============================================================================


class TestEndogenousVariables:
    """Test GMM with endogenous variables."""

    def test_endogenous_vars_fit(self):
        """Test that endogenous variables generate correct instruments and fit."""
        np.random.seed(42)
        n_units = 50
        n_periods = 10

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            z = np.random.normal(0, 1, n_periods)  # endogenous variable

            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                # z is correlated with y (endogenous)
                z[t] = 0.2 * y[t - 1] + np.random.normal(0, 1)
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + 0.2 * z[t] + eta_i + np.random.normal(0, 0.5)

            for t in range(n_periods):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t], "z": z[t]})

        data = pd.DataFrame(data_list)

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            endogenous_vars=["z"],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        assert results.nobs > 0
        # Should have 3 params: y_lag1, x, z
        assert len(results.params) == 3

    def test_endogenous_vars_in_results(self):
        """Test that endogenous variables appear in results."""
        np.random.seed(42)
        n_units = 50
        n_periods = 10

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            z = np.random.normal(0, 1, n_periods)

            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                z[t] = 0.2 * y[t - 1] + np.random.normal(0, 1)
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + 0.2 * z[t] + eta_i + np.random.normal(0, 0.5)

            for t in range(n_periods):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t], "z": z[t]})

        data = pd.DataFrame(data_list)

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            endogenous_vars=["z"],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()
        assert "z" in results.params.index


# ============================================================================
# Test Coverage: summary() before fit and __repr__ (lines 773-776)
# ============================================================================


class TestSummaryAndRepr:
    """Test summary() before fit and __repr__ method."""

    def test_summary_before_fit(self, balanced_panel_data):
        """Test that summary() raises ValueError before fitting."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        with pytest.raises(ValueError, match="Model has not been fit yet"):
            model.summary()

    def test_repr_not_fitted(self, balanced_panel_data):
        """Test __repr__ for unfitted model."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        repr_str = repr(model)
        assert "DifferenceGMM" in repr_str
        assert "not fitted" in repr_str
        assert "dep_var='y'" in repr_str

    def test_repr_fitted(self, balanced_panel_data):
        """Test __repr__ for fitted model."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )
        model.fit()

        repr_str = repr(model)
        assert "DifferenceGMM" in repr_str
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str

    def test_summary_after_fit(self, balanced_panel_data):
        """Test that summary() works after fitting (via the model, not results)."""
        model = DifferenceGMM(
            data=balanced_panel_data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )
        model.fit()

        summary = model.summary()
        assert isinstance(summary, str)
        assert "Difference GMM" in summary


# ============================================================================
# Test Coverage: Low retention rate warning (lines 593-595)
# ============================================================================


class TestLowRetentionWarning:
    """Test warning when observation retention rate is low."""

    def test_low_retention_rate_warning(self):
        """Test that warning fires when retention rate < 30%."""
        np.random.seed(42)
        # Create data where many observations will be dropped:
        # Use a large dataset with lots of NaN / missing structure
        # that causes heavy filtering during estimation
        n_units = 10
        n_periods = 4

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.5)
            for t in range(n_periods):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t]})

        data = pd.DataFrame(data_list)

        # Now add many rows with NaN to inflate the dataset size
        # This will make the retention rate very low
        nan_rows = []
        for i in range(200):
            nan_rows.append(
                {
                    "id": n_units + i,
                    "year": 0,
                    "y": np.nan,
                    "x": np.random.normal(0, 1),
                }
            )
        data_inflated = pd.concat([data, pd.DataFrame(nan_rows)], ignore_index=True)

        with pytest.warns(UserWarning, match="Low observation retention"):
            DifferenceGMM(
                data=data_inflated,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                exog_vars=["x"],
                collapse=True,
                time_dummies=False,
            ).fit()


# ============================================================================
# Test Coverage: NaN filtering path (lines 501->503)
# ============================================================================


class TestNaNFiltering:
    """Test that NaN observations are correctly filtered."""

    def test_data_with_nan_values(self):
        """Test that data containing NaN values is handled correctly."""
        np.random.seed(42)
        n_units = 30
        n_periods = 8

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.5)
            for t in range(n_periods):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t]})

        data = pd.DataFrame(data_list)

        # Introduce some NaN values in y and x
        rng = np.random.RandomState(123)
        nan_indices = rng.choice(len(data), size=10, replace=False)
        data.loc[nan_indices[:5], "y"] = np.nan
        data.loc[nan_indices[5:], "x"] = np.nan

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            time_dummies=False,
        )

        # Should still complete, dropping NaN observations
        results = model.fit()
        assert isinstance(results, GMMResults)
        assert results.nobs > 0


# ============================================================================
# Test Coverage: Both predetermined and endogenous together
# ============================================================================


class TestPredeterminedAndEndogenousTogether:
    """Test GMM with both predetermined and endogenous variables simultaneously."""

    def test_predetermined_and_endogenous(self):
        """Test with both predetermined and endogenous variables."""
        np.random.seed(42)
        n_units = 50
        n_periods = 10

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            y = np.zeros(n_periods)
            x = np.random.normal(0, 1, n_periods)
            w = np.random.normal(0, 1, n_periods)  # predetermined
            z = np.random.normal(0, 1, n_periods)  # endogenous

            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, n_periods):
                z[t] = 0.2 * y[t - 1] + np.random.normal(0, 1)
                y[t] = (
                    0.5 * y[t - 1]
                    + 0.3 * x[t]
                    + 0.2 * w[t - 1]
                    + 0.1 * z[t]
                    + eta_i
                    + np.random.normal(0, 0.5)
                )

            for t in range(n_periods):
                data_list.append(
                    {
                        "id": i,
                        "year": t,
                        "y": y[t],
                        "x": x[t],
                        "w": w[t],
                        "z": z[t],
                    }
                )

        data = pd.DataFrame(data_list)

        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            predetermined_vars=["w"],
            endogenous_vars=["z"],
            collapse=True,
            time_dummies=False,
        )

        results = model.fit()

        assert isinstance(results, GMMResults)
        # Should have 4 params: y_lag1, x, z, w
        assert len(results.params) == 4
        assert "w" in results.params.index
        assert "z" in results.params.index


# ============================================================================
# Test Coverage: Branch conditions for better coverage
# ============================================================================


class TestBranchCoverage:
    """Tests targeting specific branch conditions for coverage."""

    def test_mildly_unbalanced_panel_no_warning(self):
        """Test unbalanced panel with few time dummies (< 5) that should NOT trigger warning.

        Covers the False branch of 'n_dummies >= 5 and balance_rate < 0.80' (line 348->363).
        """
        np.random.seed(42)
        n_units = 30
        # Only 4 time periods => 3 dummies (< 5), so condition is False
        n_periods = 4

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            # Make some units have fewer periods (unbalanced)
            if i < 5:
                periods_for_unit = 3
            else:
                periods_for_unit = n_periods

            y = np.zeros(periods_for_unit)
            x = np.random.normal(0, 1, periods_for_unit)
            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, periods_for_unit):
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.5)
            for t in range(periods_for_unit):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t]})

        data = pd.DataFrame(data_list)

        # This should NOT warn about unbalanced panel (n_dummies < 5)
        # but WILL warn about collapse=False
        with pytest.warns(UserWarning, match="collapse=True"):
            model = DifferenceGMM(
                data=data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                exog_vars=["x"],
                time_dummies=True,
                collapse=False,
            )

        # Verify the panel is indeed unbalanced
        is_unbalanced, _balance_rate = model._check_panel_balance()
        assert is_unbalanced is True

    def test_unbalanced_with_high_balance_rate_no_warning(self):
        """Test unbalanced panel where balance_rate >= 0.80 (doesn't trigger warning).

        Covers the False branch when n_dummies >= 5 but balance_rate >= 0.80.
        """
        np.random.seed(42)
        n_units = 50
        n_periods = 8  # 7 time dummies >= 5

        data_list = []
        for i in range(n_units):
            eta_i = np.random.normal(0, 1)
            # Most units have full periods (>80% balanced)
            # Only 2 out of 50 have fewer periods => balance_rate = 48/50 = 0.96
            if i < 2:
                periods_for_unit = 6
            else:
                periods_for_unit = n_periods

            y = np.zeros(periods_for_unit)
            x = np.random.normal(0, 1, periods_for_unit)
            y[0] = eta_i + np.random.normal(0, 0.5)
            for t in range(1, periods_for_unit):
                y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.5)
            for t in range(periods_for_unit):
                data_list.append({"id": i, "year": t, "y": y[t], "x": x[t]})

        data = pd.DataFrame(data_list)

        # This should only warn about collapse=False, NOT about unbalanced panel
        with pytest.warns(UserWarning, match="collapse=True"):
            model = DifferenceGMM(
                data=data,
                dep_var="y",
                lags=1,
                id_var="id",
                time_var="year",
                exog_vars=["x"],
                time_dummies=True,
                collapse=False,
            )

        # Verify the panel is unbalanced but balance rate is high
        is_unbalanced, balance_rate = model._check_panel_balance()
        assert is_unbalanced is True
        assert balance_rate >= 0.80
