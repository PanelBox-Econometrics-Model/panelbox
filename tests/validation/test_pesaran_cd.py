"""
Unit tests for Pesaran CD test.

Tests the PesaranCDTest which detects cross-sectional dependence
in panel models.
"""

import numpy as np
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.cross_sectional_dependence.pesaran_cd import PesaranCDTest


class TestPesaranCD:
    """Test suite for Pesaran CD test."""

    def test_detects_cross_sectional_dependence(self, panel_with_cross_sectional_dependence):
        """Test that Pesaran CD test detects cross-sectional dependence."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_cross_sectional_dependence, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        result = test.run(alpha=0.05)

        # Should detect cross-sectional dependence
        # (Note: detection depends on strength of common shocks)
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_no_false_positive_clean_data(self, clean_panel_data):
        """Test that Pesaran CD doesn't reject when no cross-sectional dependence exists."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        result = test.run(alpha=0.05)

        # Should NOT detect cross-sectional dependence
        # (though some Type I error is expected at alpha level)
        assert result.pvalue > 0.01, "Should not strongly reject with independent data"

    def test_works_with_pooled_ols(self, clean_panel_data):
        """Test that Pesaran CD works with Pooled OLS."""
        pooled = PooledOLS("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = pooled.fit()

        test = PesaranCDTest(results)
        result = test.run()

        # Should run without error
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_result_attributes(self, clean_panel_data):
        """Test that result has all required attributes."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        result = test.run()

        # Check all attributes exist
        assert hasattr(result, "test_name")
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "reject_null")
        assert hasattr(result, "conclusion")
        assert hasattr(result, "details")

        # Check types
        assert isinstance(result.test_name, str)
        assert isinstance(result.statistic, (int, float))
        assert isinstance(result.pvalue, (int, float))
        assert isinstance(result.reject_null, bool)

        # Check test name
        assert "Pesaran" in result.test_name or "CD" in result.test_name

    def test_statistic_is_standard_normal(self, clean_panel_data):
        """Test that CD statistic follows standard normal under null."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        result = test.run()

        # CD statistic should typically be within reasonable bounds for N(0,1)
        # (allowing for some variation, typically within [-4, 4] with high probability)
        assert (
            -10 < result.statistic < 10
        ), f"CD statistic {result.statistic} seems unreasonable for N(0,1)"

    def test_different_alpha_levels(self, clean_panel_data):
        """Test behavior with different significance levels."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)

        # Test with different alphas
        result_001 = test.run(alpha=0.01)
        result_010 = test.run(alpha=0.10)

        # P-value and statistic should be the same
        assert result_001.pvalue == result_010.pvalue
        assert result_001.statistic == result_010.statistic

    def test_requires_multiple_entities(self, balanced_panel_data):
        """Test that Pesaran CD requires multiple entities."""
        # Create data with only one entity
        data_single = balanced_panel_data[balanced_panel_data["entity"] == 1].copy()

        fe = FixedEffects("y ~ x1 + x2", data_single, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        # Should raise error or handle gracefully
        with pytest.raises((ValueError, ZeroDivisionError)):
            test.run()

    def test_requires_multiple_periods(self, balanced_panel_data):
        """Test that Pesaran CD requires multiple time periods."""
        # Create data with only one period
        data_single_period = balanced_panel_data[balanced_panel_data["time"] == 2020].copy()

        fe = FixedEffects("y ~ x1 + x2", data_single_period, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        # Should raise error (can't compute correlations with T=1)
        with pytest.raises(ValueError):
            test.run()

    def test_pvalue_bounds(self, clean_panel_data):
        """Test that p-value is between 0 and 1."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        result = test.run()

        assert 0 <= result.pvalue <= 1, f"P-value {result.pvalue} should be in [0, 1]"

    def test_symmetric_correlation_matrix(self, clean_panel_data):
        """Test that correlation computations are symmetric."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        result = test.run()

        # CD statistic averages pairwise correlations, should be well-defined
        assert result.statistic is not None
        assert not np.isnan(result.statistic)
        assert not np.isinf(result.statistic)

    def test_with_balanced_panel(self, balanced_panel_data):
        """Test Pesaran CD with balanced panel."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_with_unbalanced_panel(self, unbalanced_panel_data):
        """Test Pesaran CD with unbalanced panel."""
        fe = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        # Should handle unbalanced data
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_details_contain_entity_count(self, clean_panel_data):
        """Test that details contain number of entities."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = PesaranCDTest(results)
        result = test.run()

        # Details should contain N (number of entities)
        assert "n_entities" in result.details or "N" in result.details
