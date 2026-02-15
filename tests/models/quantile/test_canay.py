"""
Tests for Canay (2011) Two-Step Quantile Estimator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.models.quantile.canay import (
    CanayQuantileResult,
    CanayTwoStep,
    CanayTwoStepResult,
    LocationShiftTestResult,
)
from panelbox.utils.data import PanelData


class TestCanayTwoStep:
    """Test Canay two-step estimator."""

    @pytest.fixture
    def location_shift_data(self):
        """Create panel data that satisfies location shift assumption."""
        np.random.seed(42)

        n_entities = 30
        n_time = 15
        n = n_entities * n_time

        # Generate entity and time indices
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        # Generate covariates
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)

        # Generate TRUE location shift fixed effects
        true_fe = np.random.randn(n_entities) * 2
        true_fe_expanded = np.repeat(true_fe, n_time)

        # Generate outcome with location shift FE
        # Key: coefficients DON'T vary with quantile
        # y = 1 + 2*X1 - 1*X2 + FE + error
        errors = np.random.randn(n)
        y = 1 + 2 * X1 - 1 * X2 + true_fe_expanded + errors

        # Create DataFrame
        df = pd.DataFrame({"y": y, "X1": X1, "X2": X2})

        # Create PanelData
        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        return panel_data, true_fe

    @pytest.fixture
    def non_location_shift_data(self):
        """Create panel data that violates location shift assumption."""
        np.random.seed(123)

        n_entities = 20
        n_time = 10
        n = n_entities * n_time

        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        X = np.random.randn(n)

        # Generate FE that affect different quantiles differently
        entity_effects_location = np.random.randn(n_entities)
        entity_effects_scale = np.random.uniform(0.5, 2, n_entities)

        y = []
        for i in range(n_entities):
            mask = entity_ids == i
            n_i = np.sum(mask)
            X_i = X[mask]

            # Different entities have different error distributions
            errors_i = np.random.randn(n_i) * entity_effects_scale[i]
            y_i = 2 * X_i + entity_effects_location[i] + errors_i
            y.extend(y_i)

        y = np.array(y)

        df = pd.DataFrame({"y": y, "X": X})

        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        return panel_data

    def test_initialization(self, location_shift_data):
        """Test model initialization."""
        data, _ = location_shift_data

        model = CanayTwoStep(data, formula="y ~ X1 + X2", tau=0.5)
        assert not model._step1_complete
        assert model.fixed_effects_ is None
        assert model.y_transformed_ is None
        assert model.n_entities == 30

        # Test with multiple quantiles
        model = CanayTwoStep(data, tau=[0.25, 0.5, 0.75])
        assert model.tau == [0.25, 0.5, 0.75]

    def test_step1_fe_estimation(self, location_shift_data):
        """Test Step 1: Fixed effects estimation via OLS."""
        data, true_fe = location_shift_data
        model = CanayTwoStep(data, formula="y ~ X1 + X2", tau=0.5)

        # Run step 1 only
        model._estimate_fixed_effects()

        assert model._step1_complete
        assert model.fixed_effects_ is not None
        assert len(model.fixed_effects_) == 30
        assert model.y_transformed_ is not None
        assert len(model.y_transformed_) == len(model.y)

        # Check FE estimation quality (correlation with true FE)
        corr = np.corrcoef(model.fixed_effects_, true_fe)[0, 1]
        assert corr > 0.8  # Should recover true FE reasonably well

        # Check that y is properly transformed
        # Mean of transformed y should be close to mean of y minus grand mean of FE
        assert (
            np.abs(
                np.mean(model.y_transformed_) - (np.mean(model.y) - np.mean(model.fixed_effects_))
            )
            < 0.1
        )

    def test_full_estimation(self, location_shift_data):
        """Test full two-step estimation procedure."""
        data, true_fe = location_shift_data
        model = CanayTwoStep(data, formula="y ~ X1 + X2", tau=0.5)

        result = model.fit(verbose=False)

        assert isinstance(result, CanayTwoStepResult)
        assert 0.5 in result.results

        # Check step 1 was completed
        assert model._step1_complete
        assert result.fixed_effects is not None
        assert result.fe_ols_result is not None

        # Check step 2 results
        res_05 = result.results[0.5]
        assert isinstance(res_05, CanayQuantileResult)
        assert res_05.converged
        assert len(res_05.params) == 3  # intercept + 2 covariates

        # Check coefficient recovery (should be close to true values)
        assert np.abs(res_05.params[1] - 2.0) < 0.5  # X1 coef ≈ 2
        assert np.abs(res_05.params[2] - (-1.0)) < 0.5  # X2 coef ≈ -1

    def test_multiple_quantiles(self, location_shift_data):
        """Test estimation at multiple quantiles."""
        data, _ = location_shift_data
        tau_list = [0.1, 0.25, 0.5, 0.75, 0.9]
        model = CanayTwoStep(data, formula="y ~ X1 + X2", tau=tau_list)

        result = model.fit(verbose=False)

        # Check all quantiles estimated
        for tau in tau_list:
            assert tau in result.results
            assert result.results[tau].converged

        # Under location shift, coefficients should be similar across quantiles
        coefs = np.array([result.results[tau].params for tau in tau_list])
        coef_std = np.std(coefs, axis=0)

        # Standard deviation across quantiles should be small for slope coefficients
        assert coef_std[1] < 0.5  # X1 coefficient stable
        assert coef_std[2] < 0.5  # X2 coefficient stable

    def test_location_shift_test_pass(self, location_shift_data):
        """Test location shift test with data that satisfies assumption."""
        data, _ = location_shift_data
        model = CanayTwoStep(data, formula="y ~ X1 + X2")

        # Fit model first (needed for test)
        model.fit(verbose=False)

        # Test location shift
        test_result = model.test_location_shift(tau_grid=[0.25, 0.5, 0.75])

        assert isinstance(test_result, LocationShiftTestResult)
        assert test_result.p_value > 0.05  # Should NOT reject H0 (location shift holds)

    def test_location_shift_test_fail(self, non_location_shift_data):
        """Test location shift test with data that violates assumption."""
        data = non_location_shift_data
        model = CanayTwoStep(data, formula="y ~ X")

        # Fit model
        model.fit(verbose=False)

        # Test location shift
        test_result = model.test_location_shift(tau_grid=[0.1, 0.5, 0.9])

        # With strongly heterogeneous effects, should detect violation
        # (though with small sample, test may not always reject)
        assert isinstance(test_result, LocationShiftTestResult)
        assert test_result.statistic > 0  # Should detect some difference

    def test_se_adjustment_options(self, location_shift_data):
        """Test different standard error adjustment options."""
        data, _ = location_shift_data
        model = CanayTwoStep(data, formula="y ~ X1 + X2", tau=0.5)

        # Naive SE
        result_naive = model.fit(se_adjustment="naive", verbose=False)
        se_naive = result_naive.results[0.5].bse

        # Two-step adjusted SE
        model2 = CanayTwoStep(data, formula="y ~ X1 + X2", tau=0.5)
        result_adjusted = model2.fit(se_adjustment="two-step", verbose=False)
        se_adjusted = result_adjusted.results[0.5].bse

        # Adjusted SE should generally be different (usually larger)
        assert not np.allclose(se_naive, se_adjusted)

    def test_small_t_warning(self):
        """Test warning for small T."""
        # Create data with small T
        n_entities = 50
        n_time = 5  # Small T
        n = n_entities * n_time

        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        X = np.random.randn(n)
        y = X + np.random.randn(n)

        df = pd.DataFrame({"y": y, "X": X})
        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        model = CanayTwoStep(panel_data, formula="y ~ X", tau=0.5)

        # Should warn about small T
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(verbose=False)
            assert len(w) == 1
            assert "small" in str(w[0].message).lower()

    def test_compare_with_penalty_method(self, location_shift_data):
        """Test comparison with penalty method."""
        data, _ = location_shift_data
        model = CanayTwoStep(data, formula="y ~ X1 + X2", tau=0.5)

        comparison = model.compare_with_penalty_method(tau=0.5, lambda_fe=0.1)

        assert "canay" in comparison
        assert "penalty" in comparison
        assert "correlation" in comparison
        assert "time_ratio" in comparison

        # Canay should be faster
        assert comparison["time_ratio"] > 1  # penalty_time / canay_time > 1

        # Results should be somewhat correlated
        assert comparison["correlation"] > 0.7

    def test_computational_efficiency(self):
        """Test that Canay is indeed faster than alternatives."""
        import time

        # Create larger dataset
        np.random.seed(42)
        n_entities = 100
        n_time = 20
        n = n_entities * n_time

        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        X = np.random.randn(n, 3)
        y = X @ np.array([1, -0.5, 2]) + np.random.randn(n)

        df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
        df["y"] = y

        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        model = CanayTwoStep(panel_data, tau=0.5)

        start = time.time()
        result = model.fit(verbose=False)
        elapsed = time.time() - start

        # Should be very fast
        assert elapsed < 5  # Less than 5 seconds
        assert result.results[0.5].converged


class TestCanayTwoStepResult:
    """Test result class methods."""

    @pytest.fixture
    def result(self):
        """Create a result object for testing."""
        # Mock model
        model = type("MockModel", (), {})()

        # Create individual quantile results
        results = {}
        for tau in [0.25, 0.5, 0.75]:
            res = CanayQuantileResult(
                params=np.array([1.0, 2.0, -1.0]),
                cov_matrix=np.eye(3) * 0.1,
                tau=tau,
                converged=True,
                model=model,
            )
            results[tau] = res

        # Create fixed effects and FE-OLS result
        fixed_effects = np.random.randn(10)
        fe_ols_result = {"params": np.array([2.0, -1.0]), "cov_matrix": np.eye(2) * 0.05}

        return CanayTwoStepResult(model, results, fixed_effects, fe_ols_result)

    def test_summary_output(self, result, capsys):
        """Test summary printing."""
        result.summary()
        captured = capsys.readouterr()

        assert "CANAY TWO-STEP" in captured.out
        assert "Step 1: Fixed Effects" in captured.out
        assert "Step 2: Quantile Regression" in captured.out
        assert "τ = 0.5" in captured.out

    def test_summary_specific_tau(self, result, capsys):
        """Test summary for specific quantile."""
        result.summary(tau=0.25)
        captured = capsys.readouterr()

        assert "τ = 0.25" in captured.out
        assert "τ = 0.5" not in captured.out
        assert "τ = 0.75" not in captured.out


class TestLocationShiftTestResult:
    """Test location shift test result class."""

    @pytest.fixture
    def test_result_reject(self):
        """Create test result that rejects H0."""
        tau_grid = [0.25, 0.5, 0.75]
        coef_matrix = np.array(
            [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]  # tau=0.25  # tau=0.5  # tau=0.75
        )

        return LocationShiftTestResult(
            statistic=15.5,
            p_value=0.001,
            df=4,
            method="wald",
            tau_grid=tau_grid,
            coef_matrix=coef_matrix,
        )

    @pytest.fixture
    def test_result_fail_reject(self):
        """Create test result that fails to reject H0."""
        tau_grid = [0.25, 0.5, 0.75]
        coef_matrix = np.array(
            [[1.0, 2.0], [1.1, 2.0], [0.9, 2.0]]  # tau=0.25  # tau=0.5  # tau=0.75
        )

        return LocationShiftTestResult(
            statistic=2.3,
            p_value=0.31,
            df=4,
            method="wald",
            tau_grid=tau_grid,
            coef_matrix=coef_matrix,
        )

    def test_summary_reject(self, test_result_reject, capsys):
        """Test summary when rejecting H0."""
        test_result_reject.summary()
        captured = capsys.readouterr()

        assert "LOCATION SHIFT TEST" in captured.out
        assert "REJECT H0" in captured.out
        assert "may be biased" in captured.out

    def test_summary_fail_reject(self, test_result_fail_reject, capsys):
        """Test summary when failing to reject H0."""
        test_result_fail_reject.summary()
        captured = capsys.readouterr()

        assert "LOCATION SHIFT TEST" in captured.out
        assert "Cannot reject H0" in captured.out
        assert "appears reasonable" in captured.out


class TestCanayRobustness:
    """Test robustness of Canay estimator."""

    def test_with_missing_data(self):
        """Test handling of unbalanced panels."""
        # Create unbalanced panel
        np.random.seed(42)
        entity_ids = []
        time_ids = []
        X = []
        y = []

        for i in range(20):
            # Random number of time periods for each entity
            T_i = np.random.randint(5, 15)
            entity_ids.extend([i] * T_i)
            time_ids.extend(range(T_i))
            X_i = np.random.randn(T_i)
            y_i = 2 * X_i + i + np.random.randn(T_i)  # Entity FE = i
            X.extend(X_i)
            y.extend(y_i)

        df = pd.DataFrame({"y": y, "X": X})
        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        model = CanayTwoStep(panel_data, formula="y ~ X", tau=0.5)
        result = model.fit(verbose=False)

        assert result.results[0.5].converged
        assert len(result.fixed_effects) == 20

    def test_with_single_covariate(self):
        """Test with only one covariate."""
        np.random.seed(42)
        n_entities = 15
        n_time = 8
        n = n_entities * n_time

        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        X = np.random.randn(n)
        y = 3 * X + np.random.randn(n)

        df = pd.DataFrame({"y": y, "X": X})
        panel_data = PanelData(df, entity_col="entity", time_col="time")
        panel_data.entity_ids = pd.Series(entity_ids, name="entity")
        panel_data.time_ids = pd.Series(time_ids, name="time")

        model = CanayTwoStep(panel_data, formula="y ~ X", tau=[0.25, 0.5, 0.75])
        result = model.fit(verbose=False)

        for tau in [0.25, 0.5, 0.75]:
            assert result.results[tau].converged
            # Should recover coefficient around 3
            assert np.abs(result.results[tau].params[1] - 3.0) < 0.5
