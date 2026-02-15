"""
Tests for Panel Heckman diagnostic methods.

This module tests the diagnostic functionality added in FASE 2:
- selection_effect()
- imr_diagnostics()
- compare_ols_heckman()
- plot_imr()
"""

import numpy as np
import pytest
from scipy import stats

from panelbox.models.selection import PanelHeckman


class TestHeckmanDiagnostics:
    """Test diagnostic methods for Panel Heckman."""

    def setup_method(self):
        """Setup test data with selection bias."""
        np.random.seed(42)

        # Generate data with known selection bias
        self.n = 500
        self.k_outcome = 3
        self.k_selection = 4

        # Regressors
        self.X = np.random.randn(self.n, self.k_outcome)
        self.X[:, 0] = 1  # Intercept

        # Selection equation with exclusion restriction
        self.Z = np.random.randn(self.n, self.k_selection)
        self.Z[:, 0] = 1
        self.Z[:, -1] = np.random.randn(self.n)  # Exclusion restriction

        # True parameters
        self.beta_true = np.array([2.0, 0.5, -0.3])
        self.gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        self.rho_true = 0.5  # Known selection bias

        # Generate correlated errors
        mean = [0, 0]
        cov = [[1, self.rho_true], [self.rho_true, 1]]
        errors = np.random.multivariate_normal(mean, cov, self.n)
        u = errors[:, 0]
        e = errors[:, 1]

        # Selection
        s_star = self.Z @ self.gamma_true + u
        self.selection = (s_star > 0).astype(int)

        # Outcome
        y_star = self.X @ self.beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

        # Fit model
        self.model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        self.result = self.model.fit(method="two_step")

    def test_selection_effect_method_exists(self):
        """Test that selection_effect() method exists."""
        assert hasattr(self.result, "selection_effect")

    def test_selection_effect_returns_dict(self):
        """Test that selection_effect() returns dictionary with expected keys."""
        test_result = self.result.selection_effect()

        assert isinstance(test_result, dict)
        assert "statistic" in test_result
        assert "pvalue" in test_result
        assert "reject" in test_result
        assert "interpretation" in test_result

    def test_selection_effect_detects_bias(self):
        """Test that selection_effect() detects known selection bias."""
        test_result = self.result.selection_effect()

        # With true rho = 0.5, should reject H0: rho = 0
        assert test_result["reject"] == True
        assert test_result["pvalue"] < 0.05
        assert "detected" in test_result["interpretation"].lower()

    def test_selection_effect_no_bias_case(self):
        """Test selection_effect() when there is no selection bias."""
        # Generate data with rho = 0 (no selection)
        np.random.seed(123)
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]  # rho = 0
        errors = np.random.multivariate_normal(mean, cov, self.n)
        u = errors[:, 0]
        e = errors[:, 1]

        s_star = self.Z @ self.gamma_true + u
        selection = (s_star > 0).astype(int)
        y_star = self.X @ self.beta_true + e
        y = np.where(selection == 1, y_star, np.nan)

        model = PanelHeckman(y, self.X, selection, self.Z)
        result = model.fit()
        test_result = result.selection_effect()

        # Should not reject H0 (though this is stochastic)
        # At minimum, p-value should be larger
        assert test_result["pvalue"] > 0.001

    def test_imr_diagnostics_method_exists(self):
        """Test that imr_diagnostics() method exists."""
        assert hasattr(self.result, "imr_diagnostics")

    def test_imr_diagnostics_returns_dict(self):
        """Test that imr_diagnostics() returns dictionary."""
        diag = self.result.imr_diagnostics()

        assert isinstance(diag, dict)
        assert "imr_mean" in diag
        assert "imr_std" in diag
        assert "imr_min" in diag
        assert "imr_max" in diag
        assert "high_imr_count" in diag
        assert "selection_rate" in diag
        assert "n_selected" in diag
        assert "n_total" in diag

    def test_imr_diagnostics_values_valid(self):
        """Test that IMR diagnostics have valid values."""
        diag = self.result.imr_diagnostics()

        # IMR should be non-negative
        assert diag["imr_mean"] >= 0
        assert diag["imr_min"] >= 0
        assert diag["imr_max"] >= diag["imr_min"]

        # Selection rate should be in [0, 1]
        assert 0 <= diag["selection_rate"] <= 1

        # Counts should match
        assert diag["n_selected"] == np.sum(self.selection)
        assert diag["n_total"] == self.n

    def test_compare_ols_heckman_method_exists(self):
        """Test that compare_ols_heckman() method exists."""
        assert hasattr(self.result, "compare_ols_heckman")

    def test_compare_ols_heckman_returns_dict(self):
        """Test that compare_ols_heckman() returns dictionary."""
        comparison = self.result.compare_ols_heckman()

        assert isinstance(comparison, dict)
        assert "beta_ols" in comparison
        assert "beta_heckman" in comparison
        assert "difference" in comparison
        assert "pct_difference" in comparison
        assert "max_abs_difference" in comparison
        assert "interpretation" in comparison

    def test_compare_ols_heckman_dimensions(self):
        """Test that comparison arrays have correct dimensions."""
        comparison = self.result.compare_ols_heckman()

        assert len(comparison["beta_ols"]) == self.k_outcome
        assert len(comparison["beta_heckman"]) == self.k_outcome
        assert len(comparison["difference"]) == self.k_outcome

    def test_compare_ols_heckman_detects_bias(self):
        """Test that OLS vs Heckman comparison detects bias."""
        comparison = self.result.compare_ols_heckman()

        # With selection bias (rho = 0.5), OLS should differ from Heckman
        assert comparison["max_abs_difference"] > 0.05
        assert "bias" in comparison["interpretation"].lower()

    def test_plot_imr_method_exists(self):
        """Test that plot_imr() method exists."""
        assert hasattr(self.result, "plot_imr")

    def test_plot_imr_creates_figure(self):
        """Test that plot_imr() creates a matplotlib figure."""
        pytest.importorskip("matplotlib")

        fig = self.result.plot_imr()

        assert fig is not None
        # Should have 2 subplots (scatter + histogram)
        assert len(fig.axes) == 2

    def test_plot_imr_without_matplotlib(self):
        """Test that plot_imr() raises error if matplotlib not available."""
        # This test would need to mock matplotlib import failure
        # Skip for now
        pass


class TestInverseMillsRatioFunctions:
    """Test standalone IMR functions."""

    def test_compute_imr_import(self):
        """Test that compute_imr can be imported."""
        from panelbox.models.selection import compute_imr

        assert callable(compute_imr)

    def test_compute_imr_basic(self):
        """Test basic IMR computation."""
        from panelbox.models.selection import compute_imr

        z = np.array([0.0, 1.0, -1.0, 2.0])
        imr = compute_imr(z)

        # IMR should be positive
        assert np.all(imr > 0)

        # IMR(z) should decrease as z increases (selection prob increases)
        assert imr[1] < imr[0]  # z=1 < z=0
        assert imr[3] < imr[1]  # z=2 < z=1

    def test_compute_imr_with_selection(self):
        """Test IMR computation with selection indicator."""
        from panelbox.models.selection import compute_imr

        z = np.array([0.5, -0.5, 1.0, -1.0])
        selected = np.array([1, 0, 1, 0])

        imr = compute_imr(z, selected)

        # Selected observations use φ/Φ (positive)
        assert imr[0] > 0
        assert imr[2] > 0

        # Non-selected use -φ/(1-Φ) (negative)
        assert imr[1] < 0
        assert imr[3] < 0

    def test_imr_derivative(self):
        """Test IMR derivative computation."""
        from panelbox.models.selection import imr_derivative

        z = np.array([0.0, 1.0, -1.0])
        deriv = imr_derivative(z)

        # Derivative should be negative (IMR decreases with z)
        assert np.all(deriv < 0)

    def test_test_selection_effect_import(self):
        """Test that test_selection_effect can be imported."""
        from panelbox.models.selection import test_selection_effect

        assert callable(test_selection_effect)

    def test_test_selection_effect_basic(self):
        """Test basic selection effect test."""
        from panelbox.models.selection import test_selection_effect

        # Significant coefficient
        result = test_selection_effect(imr_coefficient=0.5, imr_se=0.1)

        assert result["reject"] == True
        assert result["pvalue"] < 0.05

        # Non-significant coefficient
        result_null = test_selection_effect(imr_coefficient=0.01, imr_se=0.1)

        assert result_null["reject"] == False
        assert result_null["pvalue"] > 0.05

    def test_imr_diagnostics_function(self):
        """Test IMR diagnostics function."""
        from panelbox.models.selection import imr_diagnostics

        linear_pred = np.random.randn(100)
        selected = np.random.randint(0, 2, 100)

        diag = imr_diagnostics(linear_pred, selected)

        assert "imr_mean" in diag
        assert "selection_rate" in diag
        assert diag["n_total"] == 100
        assert diag["n_selected"] == np.sum(selected)


class TestMurphyTopelCorrection:
    """Test Murphy-Topel variance correction (if implemented)."""

    def test_murphy_topel_module_exists(self):
        """Test that murphy_topel module can be imported."""
        try:
            from panelbox.models.selection import murphy_topel

            assert murphy_topel is not None
        except ImportError:
            pytest.skip("Murphy-Topel module not yet implemented")

    def test_murphy_topel_variance_function(self):
        """Test Murphy-Topel variance correction function."""
        pytest.skip("Murphy-Topel correction not yet fully integrated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
