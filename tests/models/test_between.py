"""
Tests for Between Estimator.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.core.results import PanelResults
from panelbox.models.static.between import BetweenEstimator


class TestBetweenEstimator:
    """Test suite for BetweenEstimator."""

    @pytest.fixture
    def simple_panel_data(self):
        """Create simple balanced panel dataset for testing."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 5

        entities = np.repeat(range(n_entities), n_periods)
        times = np.tile(range(n_periods), n_entities)

        # Create variables with clear between variation
        # Each entity has different average level
        entity_effects = np.repeat(np.arange(n_entities) * 10, n_periods)
        x1 = entity_effects + np.random.normal(0, 1, n_entities * n_periods)
        x2 = np.random.normal(0, 1, n_entities * n_periods)
        y = (
            2
            + 0.5 * x1
            + 1.5 * x2
            + entity_effects
            + np.random.normal(0, 1, n_entities * n_periods)
        )

        data = pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})

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
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")

        assert model.formula == "y ~ x1 + x2"
        assert model.data.entity_col == "entity"
        assert model.data.time_col == "time"
        assert model.entity_means is None  # Not computed until fit

    def test_fit_nonrobust(self, simple_panel_data):
        """Test fitting with nonrobust standard errors."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit(cov_type="nonrobust")

        # Check results object
        assert isinstance(results, PanelResults)
        assert len(results.params) == 3  # intercept + 2 variables
        assert "Intercept" in results.params.index
        assert "x1" in results.params.index
        assert "x2" in results.params.index

        # Check that entity means were computed
        assert model.entity_means is not None
        assert len(model.entity_means) == 10  # 10 entities
        assert "entity" in model.entity_means.columns
        assert "y" in model.entity_means.columns
        assert "x1" in model.entity_means.columns
        assert "x2" in model.entity_means.columns

    def test_fit_robust(self, simple_panel_data):
        """Test fitting with robust standard errors."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit(cov_type="robust")

        assert isinstance(results, PanelResults)
        assert results.cov_type == "robust"

    def test_fit_clustered(self, simple_panel_data):
        """Test fitting with clustered standard errors."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit(cov_type="clustered")

        assert isinstance(results, PanelResults)
        assert results.cov_type == "clustered"

    def test_rsquared_between(self, simple_panel_data):
        """Test that R-squared measures are computed correctly."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit()

        # For between estimator, rsquared = rsquared_between
        assert results.rsquared == results.rsquared_between
        assert 0 <= results.rsquared <= 1
        assert 0 <= results.rsquared_between <= 1
        assert 0 <= results.rsquared_overall <= 1

        # Within R² should be 0 or NaN for between estimator
        assert results.rsquared_within == 0.0

    def test_degrees_of_freedom(self, simple_panel_data):
        """Test degrees of freedom calculation."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit()

        # Between estimator uses N entities, not N*T observations
        assert results.data_info["nobs"] == 10  # 10 entities
        assert results.data_info["n_entities"] == 10
        assert results.data_info["df_model"] == 2  # x1, x2 (slopes only)
        assert results.data_info["df_resid"] == 10 - 3  # n - k (including intercept)

    def test_entity_means_structure(self, simple_panel_data):
        """Test structure of entity means DataFrame."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        model.fit()

        entity_means = model.entity_means

        # Check structure
        assert isinstance(entity_means, pd.DataFrame)
        assert len(entity_means) == 10  # 10 entities
        assert "entity" in entity_means.columns
        assert "y" in entity_means.columns
        assert "x1" in entity_means.columns
        assert "x2" in entity_means.columns

        # Check that means are computed correctly
        # For entity 0, compute manual mean
        entity_0_data = simple_panel_data[simple_panel_data["entity"] == 0]
        manual_mean_y = entity_0_data["y"].mean()
        manual_mean_x1 = entity_0_data["x1"].mean()

        computed_mean_y = entity_means[entity_means["entity"] == 0]["y"].values[0]
        computed_mean_x1 = entity_means[entity_means["entity"] == 0]["x1"].values[0]

        np.testing.assert_allclose(computed_mean_y, manual_mean_y, rtol=1e-10)
        np.testing.assert_allclose(computed_mean_x1, manual_mean_x1, rtol=1e-10)

    def test_no_intercept_formula(self, simple_panel_data):
        """Test between estimator without intercept."""
        model = BetweenEstimator("y ~ x1 + x2 - 1", simple_panel_data, "entity", "time")
        results = model.fit()

        # Should only have x1 and x2
        assert len(results.params) == 2
        assert "Intercept" not in results.params.index
        assert "x1" in results.params.index
        assert "x2" in results.params.index

    def test_comparison_with_fixed_effects(self, simple_panel_data):
        """Test that BE captures different variation than FE."""
        from panelbox.models.static.fixed_effects import FixedEffects

        # Between estimator
        be = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results_be = be.fit()

        # Fixed effects
        fe = FixedEffects("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results_fe = fe.fit()

        # Coefficients can differ substantially
        # BE uses between variation, FE uses within variation
        assert results_be.rsquared_between > 0  # BE should have positive between R²
        assert results_fe.rsquared_within > 0  # FE should have positive within R²

        # R² measures should be different
        assert results_be.rsquared != results_fe.rsquared

    def test_grunfeld_dataset(self, grunfeld_data):
        """Test with real Grunfeld dataset."""
        model = BetweenEstimator("invest ~ value + capital", grunfeld_data, "firm", "year")
        results = model.fit(cov_type="robust")

        # Check basic properties
        assert results.data_info["nobs"] == 10  # 10 firms
        assert results.data_info["n_entities"] == 10
        assert len(results.params) == 3  # intercept + value + capital

        # Between R² should be high (known for Grunfeld data)
        assert results.rsquared_between > 0.8

    def test_insufficient_entities_error(self):
        """Test error when too few entities."""
        # Only 2 entities with 3 parameters → df_resid would be negative
        data = pd.DataFrame(
            {
                "entity": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1, 2, 3, 4],
                "x1": [1, 2, 3, 4],
                "x2": [2, 3, 4, 5],
            }
        )

        model = BetweenEstimator("y ~ x1 + x2", data, "entity", "time")

        with pytest.raises(ValueError, match="Insufficient degrees of freedom"):
            model.fit()

    def test_all_cov_types(self, simple_panel_data):
        """Test all covariance types are supported."""
        cov_types = [
            "nonrobust",
            "robust",
            "hc0",
            "hc1",
            "hc2",
            "hc3",
            "clustered",
            "driscoll_kraay",
            "newey_west",
        ]

        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")

        for cov_type in cov_types:
            results = model.fit(cov_type=cov_type)
            assert isinstance(results, PanelResults)
            assert results.cov_type == cov_type

    def test_invalid_cov_type(self, simple_panel_data):
        """Test error for invalid covariance type."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="cov_type must be one of"):
            model.fit(cov_type="invalid_type")

    def test_model_type_in_results(self, simple_panel_data):
        """Test that model type is correctly stored."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit()

        assert results.model_type == "Between Estimator"

    def test_summary_output(self, simple_panel_data):
        """Test that summary() runs without error."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit()

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Between Estimator" in summary
        assert "x1" in summary
        assert "x2" in summary

    def test_residuals_and_fitted(self, simple_panel_data):
        """Test that residuals and fitted values have correct shape."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit()

        # Should have values for all original observations (mapped from entity means)
        assert len(results.resid) == len(simple_panel_data)
        assert len(results.fittedvalues) == len(simple_panel_data)

        # Check that residuals are computed correctly for first entity
        entity_0_indices = simple_panel_data[simple_panel_data["entity"] == 0].index
        entity_0_residuals = results.resid[entity_0_indices]

        # All residuals for same entity should be identical (entity mean)
        np.testing.assert_allclose(entity_0_residuals, entity_0_residuals[0], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
