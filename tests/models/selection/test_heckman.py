"""
Tests for Panel Heckman selection model.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.models.selection import PanelHeckman


class TestPanelHeckman:
    """Test suite for Panel Heckman selection model."""

    def setup_method(self):
        """Setup test data with selection."""
        np.random.seed(42)

        # Generate data
        self.n = 500
        self.k_outcome = 3
        self.k_selection = 4

        # Regressors
        self.X = np.random.randn(self.n, self.k_outcome)
        self.X[:, 0] = 1  # Intercept

        # Selection equation has exclusion restriction
        self.Z = np.random.randn(self.n, self.k_selection)
        self.Z[:, 0] = 1  # Intercept
        self.Z[:, -1] = np.random.randn(self.n)  # Exclusion restriction

        # True parameters
        self.beta_true = np.array([2.0, 0.5, -0.3])
        self.gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        self.sigma_true = 1.0
        self.rho_true = 0.5

        # Generate correlated errors
        mean = [0, 0]
        cov = [[1, self.rho_true], [self.rho_true, 1]]
        errors = np.random.multivariate_normal(mean, cov, self.n)
        u = errors[:, 0]
        e = errors[:, 1] * self.sigma_true

        # Selection process
        s_star = self.Z @ self.gamma_true + u
        self.selection = (s_star > 0).astype(int)

        # Outcome (latent)
        y_star = self.X @ self.beta_true + e

        # Observed outcome (only if selected)
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_two_step_estimation(self):
        """Test two-step Heckman estimation."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z, method="two_step")

        result = model.fit()

        assert result.converged
        assert result.method == "two_step"
        assert hasattr(result, "probit_params")
        assert hasattr(result, "outcome_params")
        assert hasattr(result, "sigma")
        assert hasattr(result, "rho")
        assert hasattr(result, "lambda_imr")

        # Check parameter dimensions
        assert len(result.probit_params) == self.k_selection
        assert len(result.outcome_params) == self.k_outcome

        # Check that rho is in valid range
        assert -1 <= result.rho <= 1

    def test_mle_estimation(self):
        """Test maximum likelihood estimation."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z, method="mle")

        result = model.fit()

        assert result.method == "mle"
        assert hasattr(result, "llf")
        assert result.llf is not None

        # MLE should give similar results to two-step for large samples
        two_step = model.fit(method="two_step")

        # Parameters should be reasonably close
        # (exact comparison depends on sample size and convergence)
        assert np.allclose(result.rho, two_step.rho, atol=0.2)

    def test_prediction(self):
        """Test prediction functionality."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)

        result = model.fit()

        # Unconditional prediction (latent outcome)
        pred_uncond = result.predict(type="unconditional")
        assert len(pred_uncond) == self.n

        # Conditional prediction (corrected for selection)
        pred_cond = result.predict(type="conditional")
        assert len(pred_cond) == self.n

        # Conditional should differ from unconditional when rho != 0
        if abs(result.rho) > 0.1:
            assert not np.allclose(pred_uncond, pred_cond)

    def test_selection_test(self):
        """Test for selection bias."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)

        result = model.fit()
        test = result.selection_test()

        assert "rho" in test
        assert "p_value" in test
        assert "significant" in test

        # With true rho = 0.5, should detect selection
        # (though this is stochastic)
        assert abs(test["rho"]) > 0.1

    def test_no_exclusion_restriction_warning(self):
        """Test warning when no exclusion restriction."""
        # Use same regressors for both equations
        with pytest.warns(UserWarning, match="exclusion restriction"):
            model = PanelHeckman(self.y, self.X, self.selection, self.X)  # Same as outcome equation

    def test_summary(self):
        """Test summary output."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)

        result = model.fit()
        summary = result.summary()

        assert "Panel Heckman Selection Model" in summary
        assert "Selection Equation" in summary
        assert "Outcome Equation" in summary
        assert f"Selected observations: {np.sum(self.selection)}" in summary
        assert "rho:" in summary

    def test_no_selection_case(self):
        """Test when all observations are selected."""
        # All selected
        selection_all = np.ones(self.n)
        y_all = self.X @ self.beta_true + np.random.randn(self.n)

        model = PanelHeckman(y_all, self.X, selection_all, self.Z)

        result = model.fit()

        # When all selected, should be similar to OLS
        # and rho should be near zero
        assert abs(result.rho) < 0.5

    def test_high_censoring_case(self):
        """Test with high censoring rate."""
        # Create high censoring
        selection_few = np.zeros(self.n)
        selection_few[:50] = 1  # Only 10% selected

        model = PanelHeckman(self.y, self.X, selection_few, self.Z)

        result = model.fit()

        # Should still converge even with high censoring
        assert result.converged
        assert result.n_selected == 50
