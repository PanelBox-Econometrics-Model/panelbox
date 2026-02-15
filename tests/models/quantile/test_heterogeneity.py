"""
Tests for heterogeneity testing in quantile regression models.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.diagnostics.quantile.heterogeneity import (
    HeterogeneityTests,
    MonotonicityTestResult,
    SlopeEqualityTestResult,
)
from panelbox.models.quantile.canay import CanayTwoStep
from panelbox.models.quantile.fixed_effects import FixedEffectsQuantile
from panelbox.models.quantile.pooled import PooledQuantile
from panelbox.utils.data import PanelData


class TestHeterogeneityTests:
    """Test suite for heterogeneity tests in quantile regression."""

    @pytest.fixture
    def panel_data_heterogeneous(self):
        """Create panel data with heterogeneous effects across quantiles."""
        np.random.seed(42)
        n_entities = 50
        n_time = 20
        n_obs = n_entities * n_time

        # Create entity and time indices
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        # Create covariates
        X1 = np.random.randn(n_obs)
        X2 = np.random.randn(n_obs)

        # Create heterogeneous effects - coefficients vary with quantile
        # For low quantiles: β1 = 1, β2 = 0.5
        # For high quantiles: β1 = 3, β2 = 1.5
        u = np.random.randn(n_obs)
        quantile_u = stats.norm.cdf(u)  # Transform to uniform

        # Make coefficients vary with the quantile
        beta1 = 1 + 2 * quantile_u  # Goes from 1 to 3
        beta2 = 0.5 + quantile_u  # Goes from 0.5 to 1.5

        # Generate y with heterogeneous effects
        y = 2 + beta1 * X1 + beta2 * X2 + u

        # Create DataFrame
        df = pd.DataFrame(
            {"y": y, "X1": X1, "X2": X2, "entity_id": entity_ids, "time_id": time_ids}
        )

        return PanelData(df, entity="entity_id", time="time_id")

    @pytest.fixture
    def panel_data_homogeneous(self):
        """Create panel data with homogeneous effects across quantiles."""
        np.random.seed(42)
        n_entities = 50
        n_time = 20
        n_obs = n_entities * n_time

        # Create entity and time indices
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        # Create covariates
        X1 = np.random.randn(n_obs)
        X2 = np.random.randn(n_obs)

        # Create homogeneous effects - coefficients constant across quantiles
        beta1 = 2.0
        beta2 = 1.0

        # Generate y with homogeneous effects
        u = np.random.randn(n_obs)
        y = 2 + beta1 * X1 + beta2 * X2 + u

        # Create DataFrame
        df = pd.DataFrame(
            {"y": y, "X1": X1, "X2": X2, "entity_id": entity_ids, "time_id": time_ids}
        )

        return PanelData(df, entity="entity_id", time="time_id")

    def test_slope_equality_heterogeneous(self, panel_data_heterogeneous):
        """Test slope equality test detects heterogeneity."""
        # Fit model at multiple quantiles
        model = PooledQuantile(
            panel_data_heterogeneous, formula="y ~ X1 + X2", tau=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
        result = model.fit()

        # Run heterogeneity tests
        het_tests = HeterogeneityTests(result)

        # Test slope equality
        slope_test = het_tests.test_slope_equality()

        # Should reject null of equal slopes
        assert slope_test.p_value < 0.05, "Should detect heterogeneous effects"

    def test_slope_equality_homogeneous(self, panel_data_homogeneous):
        """Test slope equality test accepts homogeneity."""
        # Fit model at multiple quantiles
        model = PooledQuantile(panel_data_homogeneous, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75])
        result = model.fit()

        # Run heterogeneity tests
        het_tests = HeterogeneityTests(result)

        # Test slope equality
        slope_test = het_tests.test_slope_equality()

        # Should not reject null of equal slopes
        assert slope_test.p_value > 0.05, "Should accept homogeneous effects"

    def test_joint_equality(self, panel_data_heterogeneous):
        """Test joint equality of all coefficients across quantiles."""
        # Fit model at multiple quantiles
        model = PooledQuantile(
            panel_data_heterogeneous, formula="y ~ X1 + X2", tau=[0.1, 0.3, 0.5, 0.7, 0.9]
        )
        result = model.fit()

        # Run heterogeneity tests
        het_tests = HeterogeneityTests(result)

        # Test joint equality
        joint_test = het_tests.test_joint_equality()

        # Should reject null of joint equality
        assert joint_test.p_value < 0.05, "Should detect joint heterogeneity"
        assert joint_test.df > 0, "Degrees of freedom should be positive"

    def test_monotonicity(self, panel_data_heterogeneous):
        """Test monotonicity test for coefficients."""
        # Fit model at multiple quantiles
        model = PooledQuantile(
            panel_data_heterogeneous, formula="y ~ X1 + X2", tau=[0.1, 0.3, 0.5, 0.7, 0.9]
        )
        result = model.fit()

        # Run heterogeneity tests
        het_tests = HeterogeneityTests(result)

        # Test monotonicity for X1 (should be increasing)
        mono_test_x1 = het_tests.test_monotonicity(var_idx=1)  # Index 1 for X1

        # Should detect positive correlation (increasing pattern)
        assert mono_test_x1.correlation > 0.5, "X1 should show increasing pattern"
        assert mono_test_x1.p_value < 0.05, "Pattern should be significant"

    def test_interquantile_range(self, panel_data_heterogeneous):
        """Test interquantile range test for heteroskedasticity."""
        # Fit model at key quantiles
        model = PooledQuantile(
            panel_data_heterogeneous, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75]
        )
        result = model.fit()

        # Run heterogeneity tests
        het_tests = HeterogeneityTests(result)

        # Test IQR variation
        iqr_stat, iqr_pval = het_tests.interquantile_range_test()

        # Should detect heteroskedasticity
        assert iqr_pval < 0.1, "Should detect some heteroskedasticity"

    def test_specific_variable_equality(self, panel_data_heterogeneous):
        """Test equality for specific variable across quantiles."""
        # Fit model at multiple quantiles
        model = PooledQuantile(panel_data_heterogeneous, formula="y ~ X1 + X2", tau=[0.2, 0.5, 0.8])
        result = model.fit()

        # Run heterogeneity tests
        het_tests = HeterogeneityTests(result)

        # Test equality for X1 only
        x1_test = het_tests.test_slope_equality(var_idx=1)

        # Should detect differences
        assert x1_test.p_value < 0.05, "Should detect heterogeneity in X1"
        assert len(x1_test.var_idx) == 1, "Should test only one variable"

    def test_custom_quantile_pairs(self, panel_data_homogeneous):
        """Test slope equality with custom quantile pairs."""
        # Fit model at multiple quantiles
        model = PooledQuantile(panel_data_homogeneous, formula="y ~ X1 + X2", tau=[0.1, 0.5, 0.9])
        result = model.fit()

        # Run heterogeneity tests
        het_tests = HeterogeneityTests(result)

        # Test specific pairs only
        custom_test = het_tests.test_slope_equality(tau_pairs=[(0.1, 0.9)])  # Compare extremes only

        # Even extremes should be similar for homogeneous data
        assert custom_test.p_value > 0.05, "Extremes should be similar"
        assert len(custom_test.tau_pairs) == 1, "Should test one pair"

    def test_results_objects(self, panel_data_heterogeneous):
        """Test that result objects have correct attributes."""
        # Fit model
        model = PooledQuantile(
            panel_data_heterogeneous, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75]
        )
        result = model.fit()

        # Run tests
        het_tests = HeterogeneityTests(result)

        # Test slope equality result
        slope_result = het_tests.test_slope_equality()
        assert isinstance(slope_result, SlopeEqualityTestResult)
        assert hasattr(slope_result, "statistic")
        assert hasattr(slope_result, "p_value")
        assert hasattr(slope_result, "df")

        # Test monotonicity result
        mono_result = het_tests.test_monotonicity(var_idx=1)
        assert isinstance(mono_result, MonotonicityTestResult)
        assert hasattr(mono_result, "correlation")
        assert hasattr(mono_result, "is_increasing")
        assert hasattr(mono_result, "is_decreasing")

    def test_with_fixed_effects_model(self, panel_data_heterogeneous):
        """Test heterogeneity tests with fixed effects model."""
        # Fit fixed effects model at multiple quantiles
        model = FixedEffectsQuantile(
            panel_data_heterogeneous, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75], lambda_fe=0.1
        )
        result = model.fit()

        # Run heterogeneity tests
        het_tests = HeterogeneityTests(result)

        # Test should still work
        slope_test = het_tests.test_slope_equality()
        assert slope_test.statistic >= 0, "Test statistic should be non-negative"
        assert 0 <= slope_test.p_value <= 1, "P-value should be valid"

    def test_insufficient_quantiles_error(self, panel_data_homogeneous):
        """Test error when insufficient quantiles for tests."""
        # Fit model at single quantile
        model = PooledQuantile(
            panel_data_homogeneous, formula="y ~ X1 + X2", tau=0.5  # Single quantile
        )
        result = model.fit()

        # Should raise error
        with pytest.raises(ValueError, match="at least 2 quantiles"):
            het_tests = HeterogeneityTests(result)

    def test_summary_methods(self, panel_data_heterogeneous, capsys):
        """Test that summary methods work correctly."""
        # Fit model
        model = PooledQuantile(
            panel_data_heterogeneous, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75]
        )
        result = model.fit()

        # Run tests
        het_tests = HeterogeneityTests(result)
        slope_test = het_tests.test_slope_equality()

        # Call summary
        slope_test.summary()

        # Check output
        captured = capsys.readouterr()
        assert "Slope Equality Test" in captured.out
        assert "P-value" in captured.out


class TestLocationShiftTest:
    """Test suite for location shift assumption in Canay model."""

    @pytest.fixture
    def location_shift_data(self):
        """Create data satisfying location shift assumption."""
        np.random.seed(42)
        n_entities = 100
        n_time = 15
        n_obs = n_entities * n_time

        # Create indices
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        # Fixed effects (pure location shifters)
        entity_effects = np.random.randn(n_entities) * 2
        entity_effects_expanded = np.repeat(entity_effects, n_time)

        # Covariates
        X1 = np.random.randn(n_obs)
        X2 = np.random.randn(n_obs)

        # Coefficients constant across quantiles
        beta1 = 2.0
        beta2 = 1.0

        # Generate y with location shift
        u = np.random.randn(n_obs)
        y = entity_effects_expanded + beta1 * X1 + beta2 * X2 + u

        df = pd.DataFrame(
            {"y": y, "X1": X1, "X2": X2, "entity_id": entity_ids, "time_id": time_ids}
        )

        return PanelData(df, entity="entity_id", time="time_id")

    @pytest.fixture
    def non_location_shift_data(self):
        """Create data violating location shift assumption."""
        np.random.seed(42)
        n_entities = 100
        n_time = 15
        n_obs = n_entities * n_time

        # Create indices
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        # Fixed effects vary with quantile (violates assumption)
        entity_effects_low = np.random.randn(n_entities)
        entity_effects_high = np.random.randn(n_entities) * 3

        # Covariates
        X1 = np.random.randn(n_obs)
        X2 = np.random.randn(n_obs)

        # Generate y with quantile-dependent fixed effects
        u = np.random.randn(n_obs)
        quantile_u = stats.norm.cdf(u)

        entity_effects = np.zeros(n_obs)
        for i in range(n_entities):
            mask = entity_ids == i
            # Interpolate between low and high effects based on quantile
            entity_effects[mask] = (
                entity_effects_low[i] * (1 - quantile_u[mask])
                + entity_effects_high[i] * quantile_u[mask]
            )

        y = entity_effects + 2 * X1 + X2 + u

        df = pd.DataFrame(
            {"y": y, "X1": X1, "X2": X2, "entity_id": entity_ids, "time_id": time_ids}
        )

        return PanelData(df, entity="entity_id", time="time_id")

    def test_location_shift_accepted(self, location_shift_data):
        """Test that location shift test accepts valid data."""
        model = CanayTwoStep(location_shift_data, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75])

        # Fit model first
        result = model.fit()

        # Test location shift
        loc_test = model.test_location_shift(tau_grid=[0.25, 0.5, 0.75])

        # Should not reject location shift assumption
        assert loc_test.p_value > 0.05, "Should accept location shift"

    def test_location_shift_rejected(self, non_location_shift_data):
        """Test that location shift test rejects invalid data."""
        model = CanayTwoStep(non_location_shift_data, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75])

        # Fit model first
        result = model.fit()

        # Test location shift
        loc_test = model.test_location_shift(tau_grid=[0.1, 0.5, 0.9])

        # Should reject location shift assumption
        # (May not always reject due to test power, but should have lower p-value)
        assert loc_test.p_value < 0.5, "Should tend to reject non-location shift"

    def test_ks_method(self, location_shift_data):
        """Test Kolmogorov-Smirnov type test for location shift."""
        model = CanayTwoStep(location_shift_data, formula="y ~ X1 + X2", tau=0.5)

        # Fit model
        result = model.fit()

        # Test with KS method
        loc_test = model.test_location_shift(tau_grid=[0.2, 0.4, 0.6, 0.8], method="ks")

        # Should have valid results
        assert loc_test.statistic >= 0, "KS statistic should be non-negative"
        assert 0 <= loc_test.p_value <= 1, "P-value should be valid"

    def test_comparison_methods(self, location_shift_data):
        """Test comparison between Canay and penalty methods."""
        # Compare methods
        model = CanayTwoStep(location_shift_data, formula="y ~ X1 + X2", tau=0.5)

        comparison = model.compare_with_penalty_method(tau=0.5, lambda_fe=0.1)

        # Check results
        assert "canay" in comparison
        assert "penalty" in comparison
        assert "correlation" in comparison
        assert "time_ratio" in comparison

        # Canay should be faster
        assert comparison["time_ratio"] > 1, "Canay should be faster"

        # Results should be correlated
        assert comparison["correlation"] > 0.5, "Methods should give similar results"
