"""
Tests for Granger causality testing in Panel VAR.

This module tests Granger causality functionality.
"""

import numpy as np
import pytest

from panelbox.var import PanelVAR, PanelVARData


class TestGrangerCausality:
    """Tests for Granger causality testing."""

    def test_granger_causality_basic(self, simple_panel_data):
        """Test basic Granger causality test."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test if y1 Granger-causes y2
        gc_result = results.test_granger_causality("y1", "y2")

        # Check result properties
        assert hasattr(gc_result, "statistic")
        assert hasattr(gc_result, "pvalue")
        assert hasattr(gc_result, "df")
        assert hasattr(gc_result, "hypothesis")

        # P-value should be between 0 and 1
        assert 0 <= gc_result.pvalue <= 1

        # DF should equal number of lags (2 restrictions: L1.y1 and L2.y1 in y2 equation)
        assert gc_result.df == 2

        # Hypothesis string should mention both variables
        assert "y1" in gc_result.hypothesis
        assert "y2" in gc_result.hypothesis
        assert "Granger" in gc_result.hypothesis

    def test_granger_causality_reverse(self, simple_panel_data):
        """Test Granger causality in reverse direction."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test if y2 Granger-causes y1
        gc_result = results.test_granger_causality("y2", "y1")

        assert hasattr(gc_result, "statistic")
        assert hasattr(gc_result, "pvalue")
        assert 0 <= gc_result.pvalue <= 1
        assert gc_result.df == 2

    def test_granger_causality_different_lags(self, simple_panel_data):
        """Test Granger causality with different lag orders."""
        # Test with p=1
        data_1 = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )
        model_1 = PanelVAR(data_1)
        results_1 = model_1.fit()
        gc_1 = results_1.test_granger_causality("y1", "y2")

        # DF should be 1 (only L1.y1)
        assert gc_1.df == 1

        # Test with p=3
        data_3 = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=3
        )
        model_3 = PanelVAR(data_3)
        results_3 = model_3.fit()
        gc_3 = results_3.test_granger_causality("y1", "y2")

        # DF should be 3 (L1.y1, L2.y1, L3.y1)
        assert gc_3.df == 3

    def test_granger_causality_invalid_variable(self, simple_panel_data):
        """Test that invalid variable names raise errors."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test with non-existent causing variable
        with pytest.raises(ValueError, match="No lags"):
            results.test_granger_causality("nonexistent", "y2")

        # Test with non-existent caused variable
        with pytest.raises(ValueError, match="not found in endogenous"):
            results.test_granger_causality("y1", "nonexistent")

    def test_granger_causality_with_dgp(self, var_dgp_data):
        """Test Granger causality with known DGP."""
        df, true_params = var_dgp_data

        # In the DGP, both variables should Granger-cause each other
        # because A1 and A2 have non-zero off-diagonal elements
        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test both directions
        gc_12 = results.test_granger_causality("y1", "y2")
        gc_21 = results.test_granger_causality("y2", "y1")

        # Both should have valid statistics
        assert np.isfinite(gc_12.statistic)
        assert np.isfinite(gc_21.statistic)

        # Both should have p-values
        assert 0 <= gc_12.pvalue <= 1
        assert 0 <= gc_21.pvalue <= 1

    def test_granger_causality_repr(self, simple_panel_data):
        """Test string representation of Granger causality result."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        gc_result = results.test_granger_causality("y1", "y2")

        # Test __repr__
        repr_str = repr(gc_result)
        assert isinstance(repr_str, str)
        assert "Wald" in repr_str
        assert "H0" in repr_str


class TestWaldTestGeneral:
    """Tests for general Wald test functionality."""

    def test_wald_test_single_restriction(self, simple_panel_data):
        """Test Wald test with single restriction."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test that first coefficient is zero: R = [1, 0]
        k = len(results.params_by_eq[0])
        R = np.zeros((1, k))
        R[0, 0] = 1.0

        wald_result = results.wald_test(R, equation=0)

        # Check properties
        assert hasattr(wald_result, "statistic")
        assert hasattr(wald_result, "pvalue")
        assert hasattr(wald_result, "df")

        # DF should be 1 (one restriction)
        assert wald_result.df == 1

        # Statistic should be positive
        assert wald_result.statistic >= 0

        # P-value should be between 0 and 1
        assert 0 <= wald_result.pvalue <= 1

    def test_wald_test_multiple_restrictions(self, simple_panel_data):
        """Test Wald test with multiple restrictions."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test that first two coefficients are jointly zero
        k = len(results.params_by_eq[0])
        R = np.zeros((2, k))
        R[0, 0] = 1.0
        R[1, 1] = 1.0

        wald_result = results.wald_test(R, equation=0)

        # DF should be 2 (two restrictions)
        assert wald_result.df == 2

        # Statistic and p-value should be valid
        assert wald_result.statistic >= 0
        assert 0 <= wald_result.pvalue <= 1

    def test_wald_test_with_r_vector(self, simple_panel_data):
        """Test Wald test with non-zero r vector."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test that first coefficient equals 0.5: β_1 = 0.5
        k = len(results.params_by_eq[0])
        R = np.zeros((1, k))
        R[0, 0] = 1.0
        r = np.array([0.5])

        wald_result = results.wald_test(R, r=r, equation=0)

        # Check that result is valid
        assert hasattr(wald_result, "statistic")
        assert hasattr(wald_result, "pvalue")
        assert wald_result.df == 1

    def test_wald_test_invalid_equation(self, simple_panel_data):
        """Test that invalid equation index raises error."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        R = np.array([[1, 0]])

        # Test with equation index out of bounds
        with pytest.raises(ValueError, match="equation must be between"):
            results.wald_test(R, equation=5)

        with pytest.raises(ValueError, match="equation must be between"):
            results.wald_test(R, equation=-1)
