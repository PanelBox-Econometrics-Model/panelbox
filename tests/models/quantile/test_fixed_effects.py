"""
Tests for Fixed Effects Quantile Regression using Koenker (2004) penalty method.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.models.quantile.fixed_effects import (
    FixedEffectsQuantile,
    FixedEffectsQuantilePanelResult,
    FixedEffectsQuantileResult,
)
from panelbox.utils.data import PanelData


class TestFixedEffectsQuantile:
    """Test Fixed Effects QR with penalty method."""

    @pytest.fixture
    def simple_panel_data(self):
        """Create simple panel data for testing."""
        np.random.seed(42)

        n_entities = 20
        n_time = 10
        n = n_entities * n_time

        # Generate entity and time indices
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        # Generate covariates
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)

        # Generate fixed effects
        entity_effects = np.random.randn(n_entities) * 2
        entity_effects_expanded = np.repeat(entity_effects, n_time)

        # Generate outcome with fixed effects
        # y = 1 + 2*X1 + 3*X2 + entity_effect + error
        y = 1 + 2 * X1 + 3 * X2 + entity_effects_expanded + np.random.randn(n)

        # Create DataFrame
        df = pd.DataFrame({"y": y, "X1": X1, "X2": X2})

        # Create PanelData
        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        return panel_data, entity_effects

    @pytest.fixture
    def large_panel_data(self):
        """Create larger panel data for performance testing."""
        np.random.seed(123)

        n_entities = 100
        n_time = 20
        n = n_entities * n_time

        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        X = np.random.randn(n, 5)
        entity_effects = np.random.randn(n_entities)
        entity_effects_expanded = np.repeat(entity_effects, n_time)

        y = X @ np.array([1, -0.5, 2, 0.3, -1]) + entity_effects_expanded + np.random.randn(n)

        df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(5)])
        df["y"] = y

        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        return panel_data

    def test_initialization(self, simple_panel_data):
        """Test model initialization."""
        data, _ = simple_panel_data

        # Test with auto lambda
        model = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5)
        assert model.lambda_fe == "auto"
        assert model.n_entities == 20
        assert model.tau == [0.5]

        # Test with fixed lambda
        model = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=[0.25, 0.75], lambda_fe=0.1)
        assert model.lambda_fe == 0.1
        assert model.tau == [0.25, 0.75]

    def test_check_loss_functions(self, simple_panel_data):
        """Test check loss and gradient computation."""
        data, _ = simple_panel_data
        model = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5)

        residuals = np.array([-2, -1, 0, 1, 2])
        tau = 0.5

        # Test check loss
        loss = model.check_loss(residuals, tau)
        expected = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
        assert_allclose(loss, expected)

        # Test gradient
        grad = model.check_loss_gradient(residuals, tau)
        expected_grad = np.array([-0.5, -0.5, 0.5, 0.5, 0.5])
        assert_allclose(grad, expected_grad)

    def test_lambda_selection(self, simple_panel_data):
        """Test automatic lambda selection via CV."""
        data, _ = simple_panel_data
        model = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5)

        # Test lambda max computation
        lambda_max = model._compute_lambda_max(0.5)
        assert lambda_max > 0
        assert np.isfinite(lambda_max)

        # Test CV selection (with small grid for speed)
        lambda_grid = np.array([0.001, 0.01, 0.1, 1.0])
        best_lambda = model._select_lambda_cv(
            0.5, lambda_grid=lambda_grid, cv_folds=3, verbose=False
        )
        assert best_lambda in lambda_grid
        assert hasattr(model, "cv_results_")
        assert len(model.cv_results_["cv_scores"]) == len(lambda_grid)

    def test_fit_with_fixed_lambda(self, simple_panel_data):
        """Test fitting with fixed lambda value."""
        data, true_fe = simple_panel_data
        model = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=0.1)

        result = model.fit(verbose=False)

        assert isinstance(result, FixedEffectsQuantilePanelResult)
        assert 0.5 in result.results

        # Check result properties
        res_05 = result.results[0.5]
        assert isinstance(res_05, FixedEffectsQuantileResult)
        assert res_05.lambda_fe == 0.1
        assert res_05.converged
        assert len(res_05.params) == 3  # intercept + 2 covariates
        assert len(res_05.fixed_effects) == 20  # 20 entities
        assert res_05.cov_matrix.shape == (3, 3)

        # Check that coefficients are reasonable
        assert np.abs(res_05.params[1] - 2.0) < 1.0  # X1 coef should be around 2
        assert np.abs(res_05.params[2] - 3.0) < 1.0  # X2 coef should be around 3

    def test_fit_with_auto_lambda(self, simple_panel_data):
        """Test fitting with automatic lambda selection."""
        data, _ = simple_panel_data
        model = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5)

        result = model.fit(cv_folds=3, verbose=False)

        assert isinstance(result, FixedEffectsQuantilePanelResult)
        res_05 = result.results[0.5]
        assert res_05.lambda_fe > 0
        assert res_05.converged

    def test_multiple_quantiles(self, simple_panel_data):
        """Test estimation at multiple quantiles."""
        data, _ = simple_panel_data
        tau_list = [0.1, 0.5, 0.9]
        model = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=tau_list, lambda_fe=0.1)

        result = model.fit(verbose=False)

        # Check all quantiles estimated
        for tau in tau_list:
            assert tau in result.results
            assert result.results[tau].converged

        # Check monotonicity in intercept (common pattern)
        intercepts = [result.results[tau].params[0] for tau in tau_list]
        assert intercepts[0] <= intercepts[1] <= intercepts[2]

    def test_shrinkage_effect(self, simple_panel_data):
        """Test that larger lambda leads to more shrinkage."""
        data, _ = simple_panel_data

        # Fit with small lambda
        model1 = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=0.01)
        result1 = model1.fit(verbose=False)
        fe1 = result1.results[0.5].fixed_effects

        # Fit with large lambda
        model2 = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=10.0)
        result2 = model2.fit(verbose=False)
        fe2 = result2.results[0.5].fixed_effects

        # Check shrinkage
        assert np.sum(np.abs(fe2)) < np.sum(np.abs(fe1))  # More shrinkage with larger lambda
        assert np.sum(np.abs(fe2) < 1e-6) > np.sum(np.abs(fe1) < 1e-6)  # More zeros

    def test_extreme_lambda_cases(self, simple_panel_data):
        """Test extreme values of lambda."""
        data, _ = simple_panel_data

        # Very large lambda should shrink all FE to zero (pooled QR)
        model_large = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=1000.0)
        result_large = model_large.fit(verbose=False)
        fe_large = result_large.results[0.5].fixed_effects

        assert np.max(np.abs(fe_large)) < 0.1  # All FE should be close to zero

        # Very small lambda should allow unrestricted FE
        model_small = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=0.0001)
        result_small = model_small.fit(verbose=False)
        fe_small = result_small.results[0.5].fixed_effects

        assert np.std(fe_small) > 0.5  # FE should have variation

    def test_covariance_computation(self, simple_panel_data):
        """Test covariance matrix computation."""
        data, _ = simple_panel_data
        model = FixedEffectsQuantile(data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=0.1)
        result = model.fit(verbose=False)

        res = result.results[0.5]
        cov = res.cov_matrix

        # Check properties of covariance matrix
        assert cov.shape == (3, 3)
        assert np.all(np.diag(cov) > 0)  # Positive variances
        assert np.allclose(cov, cov.T)  # Symmetric

        # Check standard errors
        se = res.bse
        assert len(se) == 3
        assert np.all(se > 0)
        assert_allclose(se, np.sqrt(np.diag(cov)))

    def test_convergence_with_difficult_data(self):
        """Test convergence with challenging data."""
        np.random.seed(999)

        # Create data with multicollinearity
        n = 200
        entity_ids = np.repeat(np.arange(20), 10)
        time_ids = np.tile(np.arange(10), 20)

        X1 = np.random.randn(n)
        X2 = X1 + np.random.randn(n) * 0.1  # Highly correlated
        y = X1 + X2 + np.random.randn(n)

        df = pd.DataFrame({"y": y, "X1": X1, "X2": X2})
        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        model = FixedEffectsQuantile(panel_data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=0.1)
        result = model.fit(verbose=False)

        # Should still converge despite multicollinearity
        assert result.results[0.5].converged

    def test_performance_metrics(self, large_panel_data):
        """Test performance with larger dataset."""
        import time

        model = FixedEffectsQuantile(large_panel_data, tau=0.5, lambda_fe=1.0)

        start = time.time()
        result = model.fit(verbose=False)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 30  # Less than 30 seconds
        assert result.results[0.5].converged

        # Check result dimensions
        assert len(result.results[0.5].params) == 6  # intercept + 5 covariates
        assert len(result.results[0.5].fixed_effects) == 100  # 100 entities


class TestFixedEffectsQuantileResult:
    """Test result class methods."""

    @pytest.fixture
    def result(self):
        """Create a result object for testing."""
        params = np.array([1.0, 2.0, 3.0])
        fixed_effects = np.random.randn(10)
        cov_matrix = np.eye(3) * 0.1
        tau = 0.5
        lambda_fe = 0.1
        converged = True

        return FixedEffectsQuantileResult(
            params=params,
            fixed_effects=fixed_effects,
            cov_matrix=cov_matrix,
            tau=tau,
            lambda_fe=lambda_fe,
            converged=converged,
            model=None,
        )

    def test_standard_errors(self, result):
        """Test standard error computation."""
        se = result.bse
        expected = np.sqrt(np.diag(result.cov_matrix))
        assert_allclose(se, expected)

    def test_summary_output(self, result, capsys):
        """Test summary printing."""
        result.summary()
        captured = capsys.readouterr()

        assert "Fixed Effects Quantile Regression" in captured.out
        assert f"τ={result.tau}" in captured.out
        assert f"λ: {result.lambda_fe}" in captured.out
        assert "Converged: True" in captured.out
        assert "Coefficients:" in captured.out
        assert "Fixed Effects Distribution:" in captured.out


class TestShrinkagePath:
    """Test shrinkage path computation and visualization."""

    def test_shrinkage_path_computation(self):
        """Test computation of shrinkage path."""
        np.random.seed(42)

        # Simple data
        n = 100
        entity_ids = np.repeat(np.arange(10), 10)
        time_ids = np.tile(np.arange(10), 10)

        X = np.random.randn(n, 2)
        y = X @ np.array([1, -1]) + np.random.randn(n)

        df = pd.DataFrame(X, columns=["X1", "X2"])
        df["y"] = y

        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        model = FixedEffectsQuantile(panel_data, tau=0.5)

        # Compute path for a small grid
        lambda_grid = np.array([0.01, 0.1, 1.0, 10.0])
        coef_paths = []
        fe_paths = []

        for lam in lambda_grid:
            result = model._fit_with_lambda(0.5, lam)
            coef_paths.append(result["beta"])
            fe_paths.append(result["alpha"])

        coef_paths = np.array(coef_paths)
        fe_paths = np.array(fe_paths)

        # Check dimensions
        assert coef_paths.shape == (4, 3)  # 4 lambdas, 3 coefficients
        assert fe_paths.shape == (4, 10)  # 4 lambdas, 10 entities

        # Check shrinkage pattern
        fe_l1_norms = np.sum(np.abs(fe_paths), axis=1)
        assert all(fe_l1_norms[i] >= fe_l1_norms[i + 1] for i in range(3))  # Decreasing L1 norm
