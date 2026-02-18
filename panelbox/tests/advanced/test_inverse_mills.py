"""Tests for Inverse Mills Ratio utilities."""

import numpy as np
import pytest


class TestComputeIMR:
    """Test IMR computation."""

    def test_basic(self):
        """Test basic IMR computation."""
        from panelbox.models.selection.inverse_mills import compute_imr

        linear_pred = np.array([0.0, 1.0, -1.0, 2.0])
        imr = compute_imr(linear_pred)

        assert len(imr) == 4
        assert all(imr > 0), "IMR must be positive"

    def test_known_values(self):
        """Test against known values."""
        from scipy import stats

        from panelbox.models.selection.inverse_mills import compute_imr

        # At z=0: phi(0)/Phi(0) = 0.3989/0.5 ≈ 0.798
        linear_pred = np.array([0.0])
        imr = compute_imr(linear_pred)

        expected = stats.norm.pdf(0) / stats.norm.cdf(0)
        np.testing.assert_almost_equal(imr[0], expected, decimal=5)

    def test_extreme_values(self):
        """Test with extreme values."""
        from panelbox.models.selection.inverse_mills import compute_imr

        # Very negative (high selection probability)
        linear_pred = np.array([-5.0, 5.0])
        imr = compute_imr(linear_pred)

        # Should not produce inf or nan
        assert np.all(np.isfinite(imr))


class TestSelectionEffect:
    """Test selection effect test."""

    def test_significant_positive(self):
        """Test detection of significant positive selection."""
        from panelbox.models.selection.inverse_mills import test_selection_effect

        result = test_selection_effect(imr_coefficient=0.5, imr_se=0.1)

        assert result["reject"] == True
        assert result["statistic"] > 0
        assert "selection bias detected" in result["interpretation"].lower()

    def test_significant_negative(self):
        """Test detection of significant negative selection."""
        from panelbox.models.selection.inverse_mills import test_selection_effect

        result = test_selection_effect(imr_coefficient=-0.5, imr_se=0.1)

        assert result["reject"] == True
        assert result["statistic"] < 0
        assert "selection bias detected" in result["interpretation"].lower()

    def test_not_significant(self):
        """Test non-significant selection."""
        from panelbox.models.selection.inverse_mills import test_selection_effect

        result = test_selection_effect(imr_coefficient=0.05, imr_se=0.1)

        assert result["reject"] == False
        assert "no significant" in result["interpretation"].lower()

    def test_invalid_se(self):
        """Test with invalid standard error."""
        from panelbox.models.selection.inverse_mills import test_selection_effect

        # Test that it raises ZeroDivisionError or handles it gracefully
        with pytest.raises(ZeroDivisionError):
            test_selection_effect(imr_coefficient=0.5, imr_se=0)


class TestIMRDiagnostics:
    """Test IMR diagnostics."""

    def test_basic(self):
        """Test basic diagnostics."""
        from panelbox.models.selection.inverse_mills import imr_diagnostics

        linear_pred = np.array([0.0, 1.0, -0.5, 1.5, -1.0])
        selection = np.array([1, 1, 0, 1, 0])

        diag = imr_diagnostics(linear_pred, selection)

        assert "imr_mean" in diag
        assert "imr_std" in diag
        assert "selection_rate" in diag
        assert diag["n_selected"] == 3
        assert diag["n_total"] == 5
        assert diag["selection_rate"] == 0.6

    def test_all_selected(self):
        """Test when all observations selected."""
        from panelbox.models.selection.inverse_mills import imr_diagnostics

        linear_pred = np.array([1.0, 2.0, 1.5])
        selection = np.array([1, 1, 1])

        diag = imr_diagnostics(linear_pred, selection)

        assert diag["selection_rate"] == 1.0
        assert diag["n_selected"] == 3
