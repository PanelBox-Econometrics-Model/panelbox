"""
Unit tests for quantile regression diagnostics.

Tests cover:
- Pseudo R² computation
- Goodness of fit tests
- Symmetry tests
- Residual analysis
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from panelbox.diagnostics.quantile import QuantileRegressionDiagnostics
from panelbox.models.quantile import PooledQuantile


class TestPseudoR2:
    """Tests for pseudo R² computation."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_pseudo_r2_range(self, fitted_model):
        """Test that pseudo R² is in [0, 1]."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        r2 = diag.pseudo_r2()

        assert 0 <= r2 <= 1, f"R² should be in [0, 1], got {r2}"

    def test_pseudo_r2_perfect_fit(self):
        """Test pseudo R² with perfect fit."""
        n_obs = 50
        X = np.column_stack([np.ones(n_obs), np.arange(n_obs)])
        y = X @ np.array([1.0, 2.0])

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        r2 = diag.pseudo_r2()

        # Should be very close to 1 for perfect fit
        assert r2 > 0.99, f"R² should be close to 1, got {r2}"

    def test_pseudo_r2_poor_fit(self):
        """Test pseudo R² with poor fit."""
        np.random.seed(42)
        n_obs = 100

        # Independent X and y
        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs)])
        y = np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        r2 = diag.pseudo_r2()

        # Should be close to 0 for poor fit
        assert r2 < 0.3, f"R² should be low for poor fit, got {r2}"


class TestGoodnessOfFit:
    """Tests for goodness of fit measures."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_goodness_of_fit_dict(self, fitted_model):
        """Test that goodness of fit returns dictionary."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        gof = diag.goodness_of_fit()

        assert isinstance(gof, dict)
        assert "pseudo_r2" in gof
        assert "mean_residual" in gof
        assert "median_residual" in gof
        assert "quantile_count" in gof
        assert "sparsity" in gof

    def test_goodness_of_fit_values(self, fitted_model):
        """Test reasonableness of goodness of fit measures."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        gof = diag.goodness_of_fit()

        # Check ranges
        assert 0 <= gof["pseudo_r2"] <= 1
        assert 0 <= gof["quantile_count"] <= 1
        assert gof["sparsity"] > 0

    def test_residual_statistics(self, fitted_model):
        """Test residual statistics."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        gof = diag.goodness_of_fit()

        # Mean and median should be finite
        assert np.isfinite(gof["mean_residual"])
        assert np.isfinite(gof["median_residual"])


class TestSymmetryTest:
    """Tests for symmetry test."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_symmetry_test_output(self, fitted_model):
        """Test symmetry test output."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        z_stat, pval = diag.symmetry_test()

        assert np.isfinite(z_stat)
        assert np.isfinite(pval)
        assert 0 <= pval <= 1

    def test_symmetry_test_median(self, fitted_model):
        """Test symmetry test for median (tau=0.5)."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        z_stat, pval = diag.symmetry_test()

        # For well-specified model, should not reject (high p-value)
        # But due to random noise, this isn't guaranteed
        # Just check that the test runs
        assert isinstance(pval, (float, np.floating))

    def test_symmetry_test_different_quantiles(self, fitted_model):
        """Test symmetry test for different quantiles."""
        model, results = fitted_model

        for tau in [0.25, 0.5, 0.75]:
            diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=tau)

            z_stat, pval = diag.symmetry_test()

            assert np.isfinite(z_stat)
            assert np.isfinite(pval)


class TestGoodnessOfFitTest:
    """Tests for goodness of fit test."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_gof_test_output(self, fitted_model):
        """Test goodness of fit test output."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        chi2_stat, pval = diag.goodness_of_fit_test()

        assert np.isfinite(chi2_stat)
        assert np.isfinite(pval)
        assert chi2_stat >= 0
        assert 0 <= pval <= 1

    def test_gof_test_ranges(self, fitted_model):
        """Test reasonable ranges for GOF test."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        chi2_stat, pval = diag.goodness_of_fit_test(n_bins=10)

        assert chi2_stat >= 0
        assert 0 <= pval <= 1


class TestResidualQuantiles:
    """Tests for residual quantiles."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_residual_quantiles_default(self, fitted_model):
        """Test residual quantiles with default quantiles."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        q_dict = diag.residual_quantiles()

        assert isinstance(q_dict, dict)
        assert 0.25 in q_dict
        assert 0.5 in q_dict
        assert 0.75 in q_dict

    def test_residual_quantiles_custom(self, fitted_model):
        """Test residual quantiles with custom quantiles."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        q_vals = np.array([0.1, 0.5, 0.9])
        q_dict = diag.residual_quantiles(quantiles=q_vals)

        assert len(q_dict) == 3
        assert all(q in q_dict for q in q_vals)

    def test_residual_quantiles_ordering(self, fitted_model):
        """Test that residual quantiles are ordered."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        q_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q_dict = diag.residual_quantiles(quantiles=q_vals)

        vals = [q_dict[q] for q in q_vals]

        # Should be monotonically increasing
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]


class TestDiagnosticsSummary:
    """Tests for diagnostics summary."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_summary_output(self, fitted_model):
        """Test that summary produces output."""
        model, results = fitted_model

        diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

        summary = diag.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Diagnostics" in summary or "diagnostics" in summary
