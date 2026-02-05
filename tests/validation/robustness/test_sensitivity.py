"""
Tests for SensitivityAnalysis module.

Author: PanelBox Development Team
Date: 2026-01-22
"""

import numpy as np
import pandas as pd
import pytest

# Optional matplotlib import
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.validation.robustness.sensitivity import SensitivityAnalysis, SensitivityResults

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def balanced_panel_data():
    """Generate balanced panel data for testing."""
    np.random.seed(42)

    n_entities = 20
    n_periods = 8
    n_obs = n_entities * n_periods

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    x1 = np.random.randn(n_obs) * 2 + 5
    x2 = np.random.randn(n_obs) * 3 + 10

    # Entity fixed effects
    entity_effects = np.repeat(np.random.randn(n_entities) * 2, n_periods)

    errors = np.random.randn(n_obs)

    y = 2.0 * x1 - 1.5 * x2 + entity_effects + errors

    data = pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})

    return data


@pytest.fixture
def fitted_model(balanced_panel_data):
    """Fit a Fixed Effects model for testing."""
    fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
    results = fe.fit()
    return results


@pytest.fixture
def sensitivity_analyzer(fitted_model):
    """Create SensitivityAnalysis instance."""
    return SensitivityAnalysis(fitted_model, show_progress=False)


# ============================================================================
# Test SensitivityAnalysis Initialization
# ============================================================================


class TestSensitivityAnalysisInit:
    """Tests for SensitivityAnalysis initialization."""

    def test_initialization(self, fitted_model):
        """Test basic initialization."""
        sensitivity = SensitivityAnalysis(fitted_model)

        assert sensitivity.results is fitted_model
        assert sensitivity.model is fitted_model._model
        assert len(sensitivity.params) == 2
        assert len(sensitivity.entities) == 20
        assert len(sensitivity.time_periods) == 8

    def test_data_storage(self, sensitivity_analyzer):
        """Test that original data is stored correctly."""
        assert sensitivity_analyzer.entity_col == "entity"
        assert sensitivity_analyzer.time_col == "time"
        assert len(sensitivity_analyzer.data) == 160  # 20 entities × 8 periods

    def test_entity_and_time_extraction(self, sensitivity_analyzer):
        """Test extraction of unique entities and time periods."""
        assert sensitivity_analyzer.n_entities == 20
        assert sensitivity_analyzer.n_periods == 8
        assert sensitivity_analyzer.entities == list(range(1, 21))
        assert sensitivity_analyzer.time_periods == list(range(1, 9))


# ============================================================================
# Test Leave-One-Out Entities
# ============================================================================


class TestLeaveOneOutEntities:
    """Tests for leave-one-out analysis by entities."""

    def test_loo_entities_runs(self, sensitivity_analyzer):
        """Test that LOO entities analysis runs without error."""
        results = sensitivity_analyzer.leave_one_out_entities()

        assert isinstance(results, SensitivityResults)
        assert results.method == "leave_one_out_entities"
        assert results.estimates is not None
        assert results.std_errors is not None

    def test_loo_entities_dimensions(self, sensitivity_analyzer):
        """Test dimensions of LOO entities results."""
        results = sensitivity_analyzer.leave_one_out_entities()

        # Should have 20 rows (one for each entity excluded)
        assert len(results.estimates) == 20

        # Should have 2 columns (x1 and x2)
        assert results.estimates.shape[1] == 2

        # Column names should match parameter names
        assert list(results.estimates.columns) == ["x1", "x2"]

    def test_loo_entities_statistics(self, sensitivity_analyzer):
        """Test that statistics are calculated correctly."""
        results = sensitivity_analyzer.leave_one_out_entities()

        assert isinstance(results.statistics, dict)

        # Should have statistics for both parameters
        assert "x1" in results.statistics
        assert "x2" in results.statistics

        # Check keys in statistics
        expected_keys = [
            "mean",
            "std",
            "min",
            "max",
            "range",
            "max_abs_deviation",
            "mean_abs_deviation",
            "max_std_deviation",
            "n_beyond_threshold",
            "pct_beyond_threshold",
        ]

        for key in expected_keys:
            assert key in results.statistics["x1"]

    def test_loo_entities_subsample_info(self, sensitivity_analyzer):
        """Test subsample information is recorded."""
        results = sensitivity_analyzer.leave_one_out_entities()

        assert len(results.subsample_info) == 20

        # Check that info contains expected columns
        assert "excluded" in results.subsample_info.columns
        assert "n_obs" in results.subsample_info.columns
        assert "converged" in results.subsample_info.columns

        # Each subsample should have 152 observations (19 entities × 8 periods)
        assert all(results.subsample_info["n_obs"] == 152)

    def test_loo_entities_influential_units(self, sensitivity_analyzer):
        """Test identification of influential entities."""
        results = sensitivity_analyzer.leave_one_out_entities(influence_threshold=2.0)

        assert isinstance(results.influential_units, list)

        # For well-behaved data, we shouldn't have many influential entities
        # (This is a heuristic check - may vary with random data)

    def test_loo_entities_estimates_close_to_original(self, sensitivity_analyzer):
        """Test that LOO estimates are close to original estimates."""
        results = sensitivity_analyzer.leave_one_out_entities()

        original_x1 = sensitivity_analyzer.params["x1"]
        original_x2 = sensitivity_analyzer.params["x2"]

        # Mean of LOO estimates should be close to original
        mean_x1 = results.estimates["x1"].mean()
        mean_x2 = results.estimates["x2"].mean()

        # Allow 20% deviation (heuristic)
        assert abs(mean_x1 - original_x1) / abs(original_x1) < 0.20
        assert abs(mean_x2 - original_x2) / abs(original_x2) < 0.20


# ============================================================================
# Test Leave-One-Out Periods
# ============================================================================


class TestLeaveOneOutPeriods:
    """Tests for leave-one-out analysis by time periods."""

    def test_loo_periods_runs(self, sensitivity_analyzer):
        """Test that LOO periods analysis runs without error."""
        results = sensitivity_analyzer.leave_one_out_periods()

        assert isinstance(results, SensitivityResults)
        assert results.method == "leave_one_out_periods"
        assert results.estimates is not None
        assert results.std_errors is not None

    def test_loo_periods_dimensions(self, sensitivity_analyzer):
        """Test dimensions of LOO periods results."""
        results = sensitivity_analyzer.leave_one_out_periods()

        # Should have 8 rows (one for each period excluded)
        assert len(results.estimates) == 8

        # Should have 2 columns (x1 and x2)
        assert results.estimates.shape[1] == 2

    def test_loo_periods_subsample_info(self, sensitivity_analyzer):
        """Test subsample information for periods."""
        results = sensitivity_analyzer.leave_one_out_periods()

        assert len(results.subsample_info) == 8

        # Each subsample should have 140 observations (20 entities × 7 periods)
        assert all(results.subsample_info["n_obs"] == 140)

    def test_loo_periods_statistics(self, sensitivity_analyzer):
        """Test that statistics are calculated for periods."""
        results = sensitivity_analyzer.leave_one_out_periods()

        assert isinstance(results.statistics, dict)
        assert "x1" in results.statistics
        assert "x2" in results.statistics

    def test_loo_periods_estimates_close_to_original(self, sensitivity_analyzer):
        """Test that LOO period estimates are close to original."""
        results = sensitivity_analyzer.leave_one_out_periods()

        original_x1 = sensitivity_analyzer.params["x1"]
        original_x2 = sensitivity_analyzer.params["x2"]

        mean_x1 = results.estimates["x1"].mean()
        mean_x2 = results.estimates["x2"].mean()

        # Allow 20% deviation
        assert abs(mean_x1 - original_x1) / abs(original_x1) < 0.20
        assert abs(mean_x2 - original_x2) / abs(original_x2) < 0.20


# ============================================================================
# Test Subset Sensitivity
# ============================================================================


class TestSubsetSensitivity:
    """Tests for subsample sensitivity analysis."""

    def test_subset_sensitivity_runs(self, sensitivity_analyzer):
        """Test that subset sensitivity runs without error."""
        results = sensitivity_analyzer.subset_sensitivity(
            n_subsamples=10, subsample_size=0.8, random_state=42
        )

        assert isinstance(results, SensitivityResults)
        assert results.method == "subset_sensitivity"

    def test_subset_sensitivity_dimensions(self, sensitivity_analyzer):
        """Test dimensions of subset sensitivity results."""
        results = sensitivity_analyzer.subset_sensitivity(
            n_subsamples=15, subsample_size=0.75, random_state=42
        )

        # Should have 15 rows (one for each subsample)
        assert len(results.estimates) == 15

        # Should have 2 columns
        assert results.estimates.shape[1] == 2

    def test_subset_sensitivity_subsample_info(self, sensitivity_analyzer):
        """Test subsample information."""
        results = sensitivity_analyzer.subset_sensitivity(
            n_subsamples=10, subsample_size=0.8, random_state=42
        )

        assert len(results.subsample_info) == 10

        # Check that subsample size is approximately correct
        # 80% of 20 entities = 16 entities
        assert all(results.subsample_info["n_entities"] == 16)

    def test_subset_sensitivity_reproducibility(self, sensitivity_analyzer):
        """Test that results are reproducible with same random_state."""
        results1 = sensitivity_analyzer.subset_sensitivity(
            n_subsamples=10, subsample_size=0.8, random_state=42
        )

        results2 = sensitivity_analyzer.subset_sensitivity(
            n_subsamples=10, subsample_size=0.8, random_state=42
        )

        # Estimates should be identical
        pd.testing.assert_frame_equal(results1.estimates, results2.estimates)

    def test_subset_sensitivity_different_seeds(self, sensitivity_analyzer):
        """Test that different random states give different results."""
        results1 = sensitivity_analyzer.subset_sensitivity(
            n_subsamples=10, subsample_size=0.8, random_state=42
        )

        results2 = sensitivity_analyzer.subset_sensitivity(
            n_subsamples=10, subsample_size=0.8, random_state=123
        )

        # Estimates should be different
        assert not results1.estimates.equals(results2.estimates)

    def test_subset_sensitivity_invalid_size(self, sensitivity_analyzer):
        """Test that invalid subsample_size raises error."""
        with pytest.raises(ValueError, match="subsample_size must be between 0 and 1"):
            sensitivity_analyzer.subset_sensitivity(n_subsamples=10, subsample_size=1.5)

        with pytest.raises(ValueError, match="subsample_size must be between 0 and 1"):
            sensitivity_analyzer.subset_sensitivity(n_subsamples=10, subsample_size=0.0)

    def test_subset_sensitivity_invalid_n_subsamples(self, sensitivity_analyzer):
        """Test that invalid n_subsamples raises error."""
        with pytest.raises(ValueError, match="n_subsamples must be at least 2"):
            sensitivity_analyzer.subset_sensitivity(n_subsamples=1)

    def test_subset_sensitivity_estimates_close_to_original(self, sensitivity_analyzer):
        """Test that subsample estimates are close to original."""
        results = sensitivity_analyzer.subset_sensitivity(
            n_subsamples=20, subsample_size=0.8, random_state=42
        )

        original_x1 = sensitivity_analyzer.params["x1"]
        original_x2 = sensitivity_analyzer.params["x2"]

        mean_x1 = results.estimates["x1"].mean()
        mean_x2 = results.estimates["x2"].mean()

        # Allow 20% deviation
        assert abs(mean_x1 - original_x1) / abs(original_x1) < 0.20
        assert abs(mean_x2 - original_x2) / abs(original_x2) < 0.20


# ============================================================================
# Test Plotting
# ============================================================================


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestPlotting:
    """Tests for sensitivity plotting functions."""

    def test_plot_loo_entities(self, sensitivity_analyzer):
        """Test plotting LOO entities results."""
        results = sensitivity_analyzer.leave_one_out_entities()

        fig = sensitivity_analyzer.plot_sensitivity(results)

        assert isinstance(fig, plt.Figure)

        # Should have 2 subplots (one for each parameter)
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_plot_loo_periods(self, sensitivity_analyzer):
        """Test plotting LOO periods results."""
        results = sensitivity_analyzer.leave_one_out_periods()

        fig = sensitivity_analyzer.plot_sensitivity(results)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_plot_subset_sensitivity(self, sensitivity_analyzer):
        """Test plotting subset sensitivity results."""
        results = sensitivity_analyzer.subset_sensitivity(n_subsamples=15, random_state=42)

        fig = sensitivity_analyzer.plot_sensitivity(results)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_plot_single_parameter(self, sensitivity_analyzer):
        """Test plotting single parameter."""
        results = sensitivity_analyzer.leave_one_out_entities()

        fig = sensitivity_analyzer.plot_sensitivity(results, params=["x1"])

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_custom_figsize(self, sensitivity_analyzer):
        """Test plotting with custom figure size."""
        results = sensitivity_analyzer.leave_one_out_entities()

        fig = sensitivity_analyzer.plot_sensitivity(results, figsize=(10, 5))

        assert isinstance(fig, plt.Figure)
        # Check figure size (approximately)
        assert abs(fig.get_figwidth() - 10) < 0.1
        assert abs(fig.get_figheight() - 5) < 0.1

        plt.close(fig)

    def test_plot_without_reference_line(self, sensitivity_analyzer):
        """Test plotting without reference line."""
        results = sensitivity_analyzer.leave_one_out_entities()

        fig = sensitivity_analyzer.plot_sensitivity(results, reference_line=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_without_confidence_band(self, sensitivity_analyzer):
        """Test plotting without confidence band."""
        results = sensitivity_analyzer.leave_one_out_entities()

        fig = sensitivity_analyzer.plot_sensitivity(results, confidence_band=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# Test Summary
# ============================================================================


class TestSummary:
    """Tests for summary generation."""

    def test_summary_loo_entities(self, sensitivity_analyzer):
        """Test summary for LOO entities."""
        results = sensitivity_analyzer.leave_one_out_entities()

        summary = sensitivity_analyzer.summary(results)

        assert isinstance(summary, pd.DataFrame)

        # Should have 2 rows (one for each parameter)
        assert len(summary) == 2

        # Check expected columns
        expected_cols = [
            "Parameter",
            "Original",
            "Mean",
            "Std",
            "Min",
            "Max",
            "Range",
            "Max Deviation",
            "Max Dev (SE)",
            "N Valid",
        ]

        for col in expected_cols:
            assert col in summary.columns

    def test_summary_loo_periods(self, sensitivity_analyzer):
        """Test summary for LOO periods."""
        results = sensitivity_analyzer.leave_one_out_periods()

        summary = sensitivity_analyzer.summary(results)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2

    def test_summary_subset_sensitivity(self, sensitivity_analyzer):
        """Test summary for subset sensitivity."""
        results = sensitivity_analyzer.subset_sensitivity(n_subsamples=10, random_state=42)

        summary = sensitivity_analyzer.summary(results)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2

    def test_summary_values(self, sensitivity_analyzer):
        """Test that summary values are reasonable."""
        results = sensitivity_analyzer.leave_one_out_entities()

        summary = sensitivity_analyzer.summary(results)

        # All N Valid should be 20 (no failures)
        assert all(summary["N Valid"] == 20)

        # Min should be less than Max
        assert all(summary["Min"] < summary["Max"])

        # Range should equal Max - Min
        assert all(abs(summary["Range"] - (summary["Max"] - summary["Min"])) < 1e-10)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_sample(self):
        """Test with very small sample."""
        np.random.seed(42)

        # Only 3 entities, 3 periods
        n_entities = 3
        n_periods = 3
        n_obs = n_entities * n_periods

        entities = np.repeat(range(1, n_entities + 1), n_periods)
        times = np.tile(range(1, n_periods + 1), n_entities)

        x1 = np.random.randn(n_obs)
        x2 = np.random.randn(n_obs)

        entity_effects = np.repeat(np.random.randn(n_entities), n_periods)
        errors = np.random.randn(n_obs)

        y = 2.0 * x1 - 1.5 * x2 + entity_effects + errors

        data = pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})

        fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
        results = fe.fit()

        sensitivity = SensitivityAnalysis(results, show_progress=False)

        # Should still run (though may have convergence issues)
        loo_results = sensitivity.leave_one_out_entities()

        assert len(loo_results.estimates) == 3

    def test_reproducibility_across_methods(self, sensitivity_analyzer):
        """Test that same random_state gives reproducible results."""
        # Run subset sensitivity twice with same seed
        results1 = sensitivity_analyzer.subset_sensitivity(n_subsamples=10, random_state=12345)

        results2 = sensitivity_analyzer.subset_sensitivity(n_subsamples=10, random_state=12345)

        pd.testing.assert_frame_equal(results1.estimates, results2.estimates)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple analyses."""

    def test_all_methods_together(self, sensitivity_analyzer):
        """Test running all sensitivity methods together."""
        # LOO entities
        loo_entities = sensitivity_analyzer.leave_one_out_entities()
        assert loo_entities.method == "leave_one_out_entities"

        # LOO periods
        loo_periods = sensitivity_analyzer.leave_one_out_periods()
        assert loo_periods.method == "leave_one_out_periods"

        # Subset sensitivity
        subset = sensitivity_analyzer.subset_sensitivity(n_subsamples=10, random_state=42)
        assert subset.method == "subset_sensitivity"

        # All should complete successfully
        assert loo_entities.estimates is not None
        assert loo_periods.estimates is not None
        assert subset.estimates is not None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_workflow_with_plotting(self, sensitivity_analyzer):
        """Test complete workflow with plotting."""
        # Run analysis
        results = sensitivity_analyzer.leave_one_out_entities()

        # Generate summary
        summary = sensitivity_analyzer.summary(results)
        assert len(summary) == 2

        # Plot results
        fig = sensitivity_analyzer.plot_sensitivity(results)
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_comparison_across_methods(self, sensitivity_analyzer):
        """Test comparing results across different methods."""
        loo_entities = sensitivity_analyzer.leave_one_out_entities()
        loo_periods = sensitivity_analyzer.leave_one_out_periods()
        subset = sensitivity_analyzer.subset_sensitivity(n_subsamples=20, random_state=42)

        # All should have similar mean estimates
        mean_x1_entities = loo_entities.estimates["x1"].mean()
        mean_x1_periods = loo_periods.estimates["x1"].mean()
        mean_x1_subset = subset.estimates["x1"].mean()

        # They should all be reasonably close
        # (allow 30% variation due to different sampling methods)
        original_x1 = sensitivity_analyzer.params["x1"]

        assert abs(mean_x1_entities - original_x1) / abs(original_x1) < 0.30
        assert abs(mean_x1_periods - original_x1) / abs(original_x1) < 0.30
        assert abs(mean_x1_subset - original_x1) / abs(original_x1) < 0.30
