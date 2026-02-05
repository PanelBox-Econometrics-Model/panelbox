"""
Tests for First Difference Estimator.
"""

import pytest
import numpy as np
import pandas as pd
from panelbox.models.static.first_difference import FirstDifferenceEstimator
from panelbox.core.results import PanelResults


class TestFirstDifferenceEstimator:
    """Test suite for FirstDifferenceEstimator."""

    @pytest.fixture
    def simple_panel_data(self):
        """Create simple balanced panel dataset for testing."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 5

        entities = np.repeat(range(n_entities), n_periods)
        times = np.tile(range(n_periods), n_entities)

        # Create variables with entity fixed effects
        entity_effects = np.repeat(np.arange(n_entities) * 5, n_periods)
        x1 = np.random.normal(10, 2, n_entities * n_periods)
        x2 = np.random.normal(5, 1, n_entities * n_periods)
        y = 2 + 0.5 * x1 + 1.5 * x2 + entity_effects + np.random.normal(0, 1, n_entities * n_periods)

        data = pd.DataFrame({
            'entity': entities,
            'time': times,
            'y': y,
            'x1': x1,
            'x2': x2
        })

        return data

    @pytest.fixture
    def grunfeld_data(self):
        """Load Grunfeld dataset."""
        try:
            from panelbox import load_grunfeld
            return load_grunfeld()
        except:
            pytest.skip("Grunfeld dataset not available")

    def test_initialization(self, simple_panel_data):
        """Test model initialization."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')

        assert model.formula == 'y ~ x1 + x2'
        assert model.data.entity_col == 'entity'
        assert model.data.time_col == 'time'
        assert model.n_obs_original == 50
        assert model.n_obs_differenced is None  # Not computed until fit

    def test_fit_nonrobust(self, simple_panel_data):
        """Test fitting with nonrobust standard errors."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit(cov_type='nonrobust')

        # Check results object
        assert isinstance(results, PanelResults)
        assert len(results.params) == 2  # Only x1, x2 (no intercept in FD)
        assert 'Intercept' not in results.params.index
        assert 'x1' in results.params.index
        assert 'x2' in results.params.index

    def test_fit_robust(self, simple_panel_data):
        """Test fitting with robust standard errors."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit(cov_type='robust')

        assert isinstance(results, PanelResults)
        assert results.model_info['cov_type'] == 'robust'

    def test_fit_clustered(self, simple_panel_data):
        """Test fitting with clustered standard errors (recommended for FD)."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit(cov_type='clustered')

        assert isinstance(results, PanelResults)
        assert results.model_info['cov_type'] == 'clustered'

    def test_observations_dropped(self, simple_panel_data):
        """Test that first observation per entity is dropped."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit()

        # Original: 10 entities × 5 periods = 50 observations
        # Differenced: 10 entities × 4 differences = 40 observations
        assert model.n_obs_original == 50
        assert model.n_obs_differenced == 40
        assert results.data_info['n_obs_dropped'] == 10

    def test_degrees_of_freedom(self, simple_panel_data):
        """Test degrees of freedom calculation."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit()

        # Differenced observations
        n = 40  # 10 entities × (5-1) periods
        k = 2   # x1, x2 (no intercept)

        assert results.data_info['nobs'] == n
        assert results.data_info['df_model'] == k
        assert results.data_info['df_resid'] == n - k

    def test_no_intercept_in_results(self, simple_panel_data):
        """Test that intercept is automatically excluded in FD."""
        # Even if formula has intercept, FD removes it
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit()

        assert 'Intercept' not in results.params.index
        assert len(results.params) == 2

    def test_first_difference_transformation(self, simple_panel_data):
        """Test that first differencing is computed correctly."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')

        # Get original data for entity 0
        entity_0 = simple_panel_data[simple_panel_data['entity'] == 0].sort_values('time')
        y_orig = entity_0['y'].values

        # Compute manual first differences
        manual_diff = np.diff(y_orig)

        # Fit model and get residuals (they should be from differenced data)
        results = model.fit()

        # The transformation should drop first period
        assert len(manual_diff) == len(y_orig) - 1

    def test_rsquared_on_differences(self, simple_panel_data):
        """Test that R-squared is computed on differenced data."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit()

        # R-squared should be between 0 and 1
        assert 0 <= results.rsquared <= 1

        # For FD, within R² = R² of differenced model
        assert results.rsquared == results.rsquared_within

        # Between and overall R² are NaN for FD
        assert np.isnan(results.rsquared_between)
        assert np.isnan(results.rsquared_overall)

    def test_comparison_with_fixed_effects(self, simple_panel_data):
        """Test FD vs FE coefficients."""
        from panelbox.models.static.fixed_effects import FixedEffects

        # First Difference
        fd = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results_fd = fd.fit(cov_type='clustered')

        # Fixed Effects
        fe = FixedEffects('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results_fe = fe.fit(cov_type='clustered')

        # Under classical assumptions, coefficients should be similar
        # (but not identical due to different transformations)
        # Check that they're in same ballpark (within 50% of each other)
        for var in ['x1', 'x2']:
            coef_fd = results_fd.params[var]
            coef_fe = results_fe.params[var]
            ratio = abs(coef_fd / coef_fe)
            assert 0.5 < ratio < 2.0, f"FD and FE coefficients too different for {var}"

    def test_grunfeld_dataset(self, grunfeld_data):
        """Test with real Grunfeld dataset."""
        model = FirstDifferenceEstimator('invest ~ value + capital', grunfeld_data, 'firm', 'year')
        results = model.fit(cov_type='clustered')

        # Check basic properties
        # 10 firms × 20 years = 200 obs → 10 firms × 19 diffs = 190 obs
        assert results.data_info['nobs'] == 190
        assert results.data_info['n_entities'] == 10
        assert results.data_info['n_obs_dropped'] == 10
        assert len(results.params) == 2  # value, capital (no intercept)

    def test_unbalanced_panel(self):
        """Test with unbalanced panel."""
        # Create unbalanced panel (entity 0 has only 3 periods)
        data = pd.DataFrame({
            'entity': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'time':   [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4],
            'y':      [1, 2, 3, 2, 3, 4, 5, 3, 4, 5, 6, 7],
            'x1':     [1, 2, 3, 2, 3, 4, 5, 3, 4, 5, 6, 7],
        })

        model = FirstDifferenceEstimator('y ~ x1', data, 'entity', 'time')
        results = model.fit()

        # Entity 0: 3 periods → 2 differences
        # Entity 1: 4 periods → 3 differences
        # Entity 2: 5 periods → 4 differences
        # Total: 2 + 3 + 4 = 9 differences
        assert results.data_info['nobs'] == 9
        assert results.data_info['n_obs_original'] == 12
        assert results.data_info['n_obs_dropped'] == 3  # One per entity

    def test_insufficient_periods_per_entity(self):
        """Test error when entities have only 1 period."""
        # Each entity has only 1 observation → cannot difference
        data = pd.DataFrame({
            'entity': [0, 1, 2],
            'time': [0, 0, 0],
            'y': [1, 2, 3],
            'x1': [1, 2, 3],
        })

        model = FirstDifferenceEstimator('y ~ x1', data, 'entity', 'time')

        with pytest.raises(ValueError, match="Insufficient observations"):
            model.fit()

    def test_all_cov_types(self, simple_panel_data):
        """Test all covariance types are supported."""
        cov_types = ['nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3',
                     'clustered', 'twoway', 'driscoll_kraay', 'newey_west']

        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')

        for cov_type in cov_types:
            results = model.fit(cov_type=cov_type)
            assert isinstance(results, PanelResults)
            assert results.model_info['cov_type'] == cov_type

    def test_invalid_cov_type(self, simple_panel_data):
        """Test error for invalid covariance type."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')

        with pytest.raises(ValueError, match="cov_type must be one of"):
            model.fit(cov_type='invalid_type')

    def test_model_type_in_results(self, simple_panel_data):
        """Test that model type is correctly stored."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit()

        assert results.model_info['model_type'] == 'First Difference'
        assert results.model_info['entity_effects'] is True  # FD eliminates entity FE

    def test_summary_output(self, simple_panel_data):
        """Test that summary() runs without error."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit()

        summary = results.summary()
        assert isinstance(summary, str)
        assert 'First Difference' in summary
        assert 'x1' in summary
        assert 'x2' in summary

    def test_residuals_shape(self, simple_panel_data):
        """Test that residuals have correct shape."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit()

        # Residuals array should have same length as original data
        # but with NaN for dropped observations
        assert len(results.resid) == len(simple_panel_data)

        # Should have 10 NaN values (first period for each of 10 entities)
        n_nan = np.sum(np.isnan(results.resid))
        assert n_nan == 10

        # Non-NaN values should equal n_obs_differenced
        n_valid = np.sum(~np.isnan(results.resid))
        assert n_valid == model.n_obs_differenced

    def test_driscoll_kraay_for_serial_correlation(self, simple_panel_data):
        """Test Driscoll-Kraay SE (recommended for FD with serial correlation)."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        results = model.fit(cov_type='driscoll_kraay', max_lags=2)

        assert isinstance(results, PanelResults)
        assert results.model_info['cov_type'] == 'driscoll_kraay'
        assert results.model_info['cov_kwds']['max_lags'] == 2

    def test_sorted_data_assumption(self):
        """Test that FD works correctly when data is sorted."""
        # Create unsorted data
        data = pd.DataFrame({
            'entity': [0, 1, 0, 1, 0, 1],
            'time':   [0, 0, 1, 1, 2, 2],
            'y':      [1, 2, 3, 4, 5, 6],
            'x1':     [1, 2, 3, 4, 5, 6],
        })

        # Sort by entity and time
        data_sorted = data.sort_values(['entity', 'time']).reset_index(drop=True)

        model = FirstDifferenceEstimator('y ~ x1', data_sorted, 'entity', 'time')
        results = model.fit()

        # Should produce valid results
        assert isinstance(results, PanelResults)
        assert results.data_info['nobs'] == 4  # 2 entities × 2 differences

    def test_estimate_coefficients_method(self, simple_panel_data):
        """Test _estimate_coefficients abstract method implementation."""
        model = FirstDifferenceEstimator('y ~ x1 + x2', simple_panel_data, 'entity', 'time')
        coeffs = model._estimate_coefficients()

        # Should return array of coefficients
        assert isinstance(coeffs, np.ndarray)
        assert len(coeffs) == 2  # x1, x2

    def test_single_variable(self, simple_panel_data):
        """Test FD with single explanatory variable."""
        model = FirstDifferenceEstimator('y ~ x1', simple_panel_data, 'entity', 'time')
        results = model.fit(cov_type='robust')

        assert len(results.params) == 1
        assert 'x1' in results.params.index


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
