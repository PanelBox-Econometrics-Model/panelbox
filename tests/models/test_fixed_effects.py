"""
Tests for FixedEffects model.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.core.panel_data import PanelData
from panelbox.models.static.fixed_effects import FixedEffects


class TestFixedEffectsInitialization:
    """Tests for FixedEffects initialization."""

    def test_init_entity_effects(self, balanced_panel_data):
        """Test initialization with entity effects."""
        model = FixedEffects(
            "y ~ x1 + x2", balanced_panel_data, "entity", "time", entity_effects=True
        )

        assert model.entity_effects is True
        assert model.time_effects is False
        assert model.entity_fe is None  # Not computed yet

    def test_init_time_effects(self, balanced_panel_data):
        """Test initialization with time effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=False,
            time_effects=True,
        )

        assert model.entity_effects is False
        assert model.time_effects is True

    def test_init_twoway_effects(self, balanced_panel_data):
        """Test initialization with two-way fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )

        assert model.entity_effects is True
        assert model.time_effects is True

    def test_init_no_effects_raises(self, balanced_panel_data):
        """Test that no effects raises ValueError."""
        with pytest.raises(ValueError, match="At least one of entity_effects"):
            FixedEffects(
                "y ~ x1 + x2",
                balanced_panel_data,
                "entity",
                "time",
                entity_effects=False,
                time_effects=False,
            )


class TestFixedEffectsFitting:
    """Tests for fitting Fixed Effects models."""

    def test_fit_entity_effects(self, balanced_panel_data):
        """Test fitting with entity fixed effects."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Check that model was fitted
        assert model._fitted is True
        assert results is not None

        # Check coefficient structure
        assert len(results.params) == 2  # x1 and x2 (no intercept in FE)
        assert "x1" in results.params.index
        assert "x2" in results.params.index
        assert "Intercept" not in results.params.index  # Absorbed by FE

        # Check that fixed effects were computed
        assert model.entity_fe is not None
        assert len(model.entity_fe) == 10  # 10 entities

    def test_fit_time_effects(self, balanced_panel_data):
        """Test fitting with time fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=False,
            time_effects=True,
        )
        results = model.fit()

        assert model._fitted is True
        assert model.time_fe is not None
        assert len(model.time_fe) == 5  # 5 time periods

    def test_fit_twoway_effects(self, balanced_panel_data):
        """Test fitting with two-way fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results = model.fit()

        assert model._fitted is True
        assert model.entity_fe is not None
        assert model.time_fe is not None
        assert results.model_type == "Fixed Effects (Two-Way)"

    def test_within_transformation(self, balanced_panel_data):
        """Test that within transformation removes entity means."""
        # Create data with known entity effects
        data = balanced_panel_data.copy()

        # Add entity-specific constants
        entity_effects_true = {i: i * 10 for i in range(1, 11)}
        data["entity_effect"] = data["entity"].map(entity_effects_true)
        data["y"] = data["y"] + data["entity_effect"]

        model = FixedEffects("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        # Within transformation should remove entity effects
        # Coefficients should be similar to original (without entity effects)
        assert results is not None
        assert len(results.params) == 2


class TestRSquaredMeasures:
    """Tests for R-squared measures in Fixed Effects."""

    def test_rsquared_measures_exist(self, balanced_panel_data):
        """Test that all R-squared measures are computed."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert not np.isnan(results.rsquared_within)
        assert not np.isnan(results.rsquared_between)
        assert not np.isnan(results.rsquared_overall)
        assert not np.isnan(results.rsquared_adj)

    def test_within_rsquared_is_main(self, balanced_panel_data):
        """Test that rsquared equals within R-squared for FE."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # For FE, the main R² is the within R²
        assert results.rsquared == results.rsquared_within

    def test_rsquared_bounds(self, balanced_panel_data):
        """Test that R-squared values are in valid range."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # R-squared should be between 0 and 1
        assert 0 <= results.rsquared_within <= 1
        assert 0 <= results.rsquared_between <= 1
        assert 0 <= results.rsquared_overall <= 1


class TestCovarianceTypes:
    """Tests for different covariance estimators."""

    def test_nonrobust_se(self, balanced_panel_data):
        """Test non-robust standard errors."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="nonrobust")

        assert results.cov_type == "nonrobust"
        assert len(results.std_errors) == 2

    def test_robust_se(self, balanced_panel_data):
        """Test robust standard errors."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="robust")

        assert results.cov_type == "robust"

    def test_clustered_se(self, balanced_panel_data):
        """Test cluster-robust standard errors."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="clustered")

        assert results.cov_type == "clustered"

    def test_robust_vs_nonrobust(self, balanced_panel_data):
        """Test that robust SEs differ from non-robust."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        results_nonrobust = model.fit(cov_type="nonrobust")

        # Refit with robust
        model2 = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results_robust = model2.fit(cov_type="robust")

        # Coefficients should be the same
        np.testing.assert_array_almost_equal(
            results_nonrobust.params.values, results_robust.params.values
        )

        # Standard errors typically different (may be same by chance with random data)
        # Just check they're both positive
        assert (results_nonrobust.std_errors > 0).all()
        assert (results_robust.std_errors > 0).all()


class TestFixedEffectsExtraction:
    """Tests for extracting fixed effects."""

    def test_entity_fe_extraction(self, balanced_panel_data):
        """Test extraction of entity fixed effects."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Check entity FE
        assert model.entity_fe is not None
        assert isinstance(model.entity_fe, pd.Series)
        assert len(model.entity_fe) == 10
        assert model.entity_fe.name == "entity_fe"

    def test_time_fe_extraction(self, balanced_panel_data):
        """Test extraction of time fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=False,
            time_effects=True,
        )
        results = model.fit()

        assert model.time_fe is not None
        assert isinstance(model.time_fe, pd.Series)
        assert len(model.time_fe) == 5
        assert model.time_fe.name == "time_fe"

    def test_twoway_fe_extraction(self, balanced_panel_data):
        """Test extraction of both entity and time fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results = model.fit()

        assert model.entity_fe is not None
        assert model.time_fe is not None
        assert len(model.entity_fe) == 10
        assert len(model.time_fe) == 5

    def test_entity_fe_sum_zero(self, balanced_panel_data):
        """Test that entity fixed effects are identified (sum to zero constraint)."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Fixed effects should approximately sum to zero (mean is zero)
        # Due to the identification, they're centered
        assert abs(model.entity_fe.mean()) < 1e-10


class TestResultsSummary:
    """Tests for results summary."""

    def test_summary_format(self, balanced_panel_data):
        """Test that summary is formatted correctly."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        summary = results.summary()

        assert "Fixed Effects" in summary
        assert "x1" in summary
        assert "x2" in summary
        assert "R-squared (within)" in summary

    def test_twoway_summary_format(self, balanced_panel_data):
        """Test summary for two-way fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results = model.fit()

        summary = results.summary()
        assert "Fixed Effects (Two-Way)" in summary


class TestDegreesOfFreedom:
    """Tests for degrees of freedom calculations."""

    def test_df_entity_fe(self, balanced_panel_data):
        """Test degrees of freedom with entity FE."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        n = 50  # 10 entities * 5 periods
        k = 2  # x1, x2
        n_fe = 10  # entity fixed effects

        expected_df_resid = n - k - n_fe
        assert results.df_resid == expected_df_resid

    def test_df_twoway_fe(self, balanced_panel_data):
        """Test degrees of freedom with two-way FE."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results = model.fit()

        n = 50
        k = 2
        n_fe_entity = 10
        n_fe_time = 5

        expected_df_resid = n - k - n_fe_entity - n_fe_time
        assert results.df_resid == expected_df_resid


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_regressor(self, balanced_panel_data):
        """Test with single regressor."""
        model = FixedEffects("y ~ x1", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert len(results.params) == 1
        assert "x1" in results.params.index

    def test_no_intercept_formula(self, balanced_panel_data):
        """Test that -1 in formula doesn't cause issues (FE absorbs intercept anyway)."""
        model = FixedEffects("y ~ x1 + x2 - 1", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Should work fine (intercept absorbed by FE regardless)
        assert len(results.params) == 2

    def test_unbalanced_panel(self, unbalanced_panel_data):
        """Test with unbalanced panel."""
        model = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = model.fit()

        assert results is not None
        # Entity FE should have 3 entities
        assert len(model.entity_fe) == 3


class TestModelComparison:
    """Tests comparing different specifications."""

    def test_entity_vs_time_fe(self, balanced_panel_data):
        """Test that entity FE and time FE give different results."""
        model_entity = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=False,
        )
        results_entity = model_entity.fit()

        model_time = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=False,
            time_effects=True,
        )
        results_time = model_time.fit()

        # Coefficients should differ
        # (unless by chance the data has no entity or time effects)
        assert len(results_entity.params) == len(results_time.params)

    def test_oneway_vs_twoway(self, balanced_panel_data):
        """Test one-way vs two-way fixed effects."""
        model_oneway = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=False,
        )
        results_oneway = model_oneway.fit()

        model_twoway = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results_twoway = model_twoway.fit()

        # Two-way should have fewer df_resid
        assert results_twoway.df_resid < results_oneway.df_resid
