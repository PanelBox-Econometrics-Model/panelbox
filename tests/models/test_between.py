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
        except Exception:
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

    @pytest.mark.parametrize("cov_type", ["robust", "clustered"])
    def test_fit_cov_type(self, simple_panel_data, cov_type):
        """Test fitting with different covariance types."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit(cov_type=cov_type)

        assert isinstance(results, PanelResults)
        assert results.cov_type == cov_type

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
        assert results.nobs == 10  # 10 entities
        assert results.n_entities == 10
        assert results.df_model == 2  # x1, x2 (slopes only)
        assert results.df_resid == 10 - 3  # n - k (including intercept)

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
        assert results.nobs == 10  # 10 firms
        assert results.n_entities == 10
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

    @pytest.mark.parametrize(
        "cov_type",
        [
            "nonrobust",
            "robust",
            "hc0",
            "hc1",
            "hc2",
            "hc3",
            "clustered",
            "driscoll_kraay",
            "newey_west",
        ],
    )
    def test_all_cov_types(self, simple_panel_data, cov_type):
        """Test all covariance types are supported."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
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


class TestBetweenPredictWithNewdata:
    """Tests for _between_predict closure (lines 463-482)."""

    @pytest.fixture
    def simple_panel_data(self):
        """Create simple balanced panel dataset for testing."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 5

        entities = np.repeat(range(n_entities), n_periods)
        times = np.tile(range(n_periods), n_entities)

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

    def test_predict_without_newdata(self, simple_panel_data):
        """Test predict() without newdata returns fitted values."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit()

        pred = results.predict()
        np.testing.assert_array_equal(pred, results.fittedvalues)

    def test_predict_with_newdata(self, simple_panel_data):
        """Test predict() with newdata triggers the _between_predict closure (lines 463-482)."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit()

        # Create new data with same entity structure
        new_data = pd.DataFrame(
            {
                "entity": [0, 0, 0, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2],
                "x1": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
                "x2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        pred = results.predict(newdata=new_data)

        # Should return one prediction per entity (group means)
        assert len(pred) == 2  # 2 entities


class TestBetweenCovarianceTypes:
    """Tests for various covariance type paths (lines 210-268)."""

    @pytest.fixture
    def simple_panel_data(self):
        """Create simple balanced panel dataset for testing."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 5

        entities = np.repeat(range(n_entities), n_periods)
        times = np.tile(range(n_periods), n_entities)

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

    def test_driscoll_kraay_cov(self, simple_panel_data):
        """Test Driscoll-Kraay covariance (lines 216-221)."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay", max_lags=2)

        assert results is not None
        assert results.cov_type == "driscoll_kraay"
        assert (results.std_errors > 0).all()

    def test_newey_west_cov(self, simple_panel_data):
        """Test Newey-West covariance (lines 223-226)."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit(cov_type="newey_west", max_lags=2)

        assert results is not None
        assert results.cov_type == "newey_west"
        assert (results.std_errors > 0).all()

    def test_pcse_cov(self, simple_panel_data):
        """Test PCSE covariance (lines 228-229)."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit(cov_type="pcse")

        assert results is not None
        assert results.cov_type == "pcse"

    def test_clustered_with_cluster_col(self, simple_panel_data):
        """Test clustered covariance with a custom cluster_col (lines 240-246)."""
        data = simple_panel_data.copy()
        # Create a region column that is constant within entity (required for between)
        data["region"] = data["entity"] % 3

        # Include region in the formula so it appears in entity_means
        model = BetweenEstimator("y ~ x1 + x2 + region", data, "entity", "time")
        results = model.fit(cov_type="clustered", cluster_col="region")

        assert results is not None
        assert results.cov_type == "clustered"
        assert (results.std_errors > 0).all()

    def test_clustered_without_cluster_col(self, simple_panel_data):
        """Test clustered covariance without cluster_col falls back to robust (line 242)."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")
        results = model.fit(cov_type="clustered")

        assert results is not None
        assert results.cov_type == "clustered"
        assert (results.std_errors > 0).all()

    def test_twoway_cov(self, simple_panel_data):
        """Test two-way clustered covariance (lines 248-268)."""
        data = simple_panel_data.copy()
        data["region"] = data["entity"] % 3

        # Include region in formula so it's in entity_means
        model = BetweenEstimator("y ~ x1 + x2 + region", data, "entity", "time")
        results = model.fit(cov_type="twoway", cluster_col2="region")

        assert results is not None
        assert results.cov_type == "twoway"
        assert (results.std_errors > 0).all()

    def test_twoway_cov_without_cluster_col2_raises(self, simple_panel_data):
        """Test that twoway without cluster_col2 raises ValueError."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="twoway clustering requires cluster_col2"):
            model.fit(cov_type="twoway")


class TestBetweenEstimateCoefficientsDirectly:
    """Tests for _estimate_coefficients() method (lines 492-522)."""

    @pytest.fixture
    def simple_panel_data(self):
        """Create simple balanced panel dataset for testing."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 5

        entities = np.repeat(range(n_entities), n_periods)
        times = np.tile(range(n_periods), n_entities)

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

    def test_estimate_coefficients_directly(self, simple_panel_data):
        """Test calling _estimate_coefficients() directly."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")

        beta = model._estimate_coefficients()

        assert beta is not None
        # Should have 3 coefficients: Intercept, x1, x2
        assert len(beta.ravel()) == 3

    def test_estimate_coefficients_matches_fit(self, simple_panel_data):
        """Test that _estimate_coefficients() gives same betas as fit()."""
        model = BetweenEstimator("y ~ x1 + x2", simple_panel_data, "entity", "time")

        beta_direct = model._estimate_coefficients().ravel()

        results = model.fit()
        beta_fit = results.params.values

        np.testing.assert_array_almost_equal(beta_direct, beta_fit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
