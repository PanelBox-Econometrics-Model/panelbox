"""
Tests for Honoré Trimmed LAD Estimator.

Author: PanelBox Developers
License: MIT
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from panelbox.models.censored import HonoreTrimmedEstimator
from panelbox.models.censored.honore import ExperimentalWarning, HonoreResults


class TestHonoreTrimmedEstimator:
    """Test suite for Honoré Trimmed LAD Estimator."""

    @pytest.fixture
    def simple_panel_data(self):
        """Generate simple panel data for testing."""
        np.random.seed(42)

        # Small panel for testing
        N = 10  # Number of entities
        T = 3  # Time periods
        K = 2  # Number of covariates

        # True parameters
        beta_true = np.array([0.5, -0.3])

        # Generate data
        X = np.random.randn(N * T, K)

        # Fixed effects
        alpha_i = np.random.randn(N) * 0.5

        # Expand to panel
        alpha = np.repeat(alpha_i, T)

        # Error term
        epsilon = np.random.randn(N * T) * 0.3

        # Latent variable
        y_star = X @ beta_true + alpha + epsilon

        # Left censoring at 0
        y = np.maximum(0, y_star)

        # Panel structure
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        return {
            "y": y,
            "X": X,
            "groups": groups,
            "time": time,
            "beta_true": beta_true,
            "N": N,
            "T": T,
            "K": K,
        }

    def test_initialization_with_warning(self, simple_panel_data):
        """Test that initialization produces experimental warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model = HonoreTrimmedEstimator(
                endog=simple_panel_data["y"],
                exog=simple_panel_data["X"],
                groups=simple_panel_data["groups"],
                time=simple_panel_data["time"],
                censoring_point=0,
            )

            # Check that a warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, ExperimentalWarning)
            assert "computationally intensive" in str(w[0].message)

        assert model.n_obs == len(simple_panel_data["y"])
        assert model.n_features == simple_panel_data["K"]
        assert model.n_entities == simple_panel_data["N"]

    def test_pairwise_differences_creation(self, simple_panel_data):
        """Test creation of pairwise differences."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(
                endog=simple_panel_data["y"],
                exog=simple_panel_data["X"],
                groups=simple_panel_data["groups"],
                time=simple_panel_data["time"],
            )

            delta_y, delta_X, trim_indicator = model._create_pairwise_differences()

            # Number of pairwise differences per entity
            T = simple_panel_data["T"]
            n_pairs_per_entity = T * (T - 1) // 2
            total_pairs = simple_panel_data["N"] * n_pairs_per_entity

            assert len(delta_y) == total_pairs
            assert delta_X.shape == (total_pairs, simple_panel_data["K"])
            assert len(trim_indicator) == total_pairs

            # Check that trim indicator is binary
            assert np.all((trim_indicator == 0) | (trim_indicator == 1))

    def test_trimmed_lad_objective(self, simple_panel_data):
        """Test LAD objective function computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(
                endog=simple_panel_data["y"],
                exog=simple_panel_data["X"],
                groups=simple_panel_data["groups"],
                time=simple_panel_data["time"],
            )

            delta_y, delta_X, trim_indicator = model._create_pairwise_differences()

            # Test objective with some parameter values
            beta_test = np.array([0.3, -0.2])
            objective_val = model._trimmed_lad_objective(
                beta_test, delta_y, delta_X, trim_indicator
            )

            assert np.isfinite(objective_val)
            assert objective_val >= 0  # LAD objective is sum of absolute values

    def test_trimmed_lad_gradient(self, simple_panel_data):
        """Test gradient computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(
                endog=simple_panel_data["y"],
                exog=simple_panel_data["X"],
                groups=simple_panel_data["groups"],
                time=simple_panel_data["time"],
            )

            delta_y, delta_X, trim_indicator = model._create_pairwise_differences()

            beta_test = np.array([0.3, -0.2])
            gradient = model._trimmed_lad_gradient(beta_test, delta_y, delta_X, trim_indicator)

            assert gradient.shape == (simple_panel_data["K"],)
            assert np.all(np.isfinite(gradient))

    def test_fit_basic(self, simple_panel_data):
        """Test basic fitting functionality."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(
                endog=simple_panel_data["y"],
                exog=simple_panel_data["X"],
                groups=simple_panel_data["groups"],
                time=simple_panel_data["time"],
            )

            # Fit with very few iterations for testing
            results = model.fit(maxiter=10, verbose=False)

            assert isinstance(results, HonoreResults)
            assert results.params.shape == (simple_panel_data["K"],)
            assert results.n_obs == simple_panel_data["N"] * simple_panel_data["T"]
            assert results.n_entities == simple_panel_data["N"]
            assert results.n_trimmed >= 0

    def test_predict(self, simple_panel_data):
        """Test prediction method."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(
                endog=simple_panel_data["y"],
                exog=simple_panel_data["X"],
                groups=simple_panel_data["groups"],
                time=simple_panel_data["time"],
            )

            # Fit model
            model.fit(maxiter=10, verbose=False)

            # Test prediction
            predictions = model.predict()

            assert len(predictions) == len(simple_panel_data["y"])
            assert np.all(np.isfinite(predictions))

            # Test with new data
            X_new = np.random.randn(5, simple_panel_data["K"])
            pred_new = model.predict(exog=X_new)

            assert len(pred_new) == 5
            assert np.all(np.isfinite(pred_new))

    def test_summary_output(self, simple_panel_data):
        """Test summary output."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(
                endog=simple_panel_data["y"],
                exog=simple_panel_data["X"],
                groups=simple_panel_data["groups"],
                time=simple_panel_data["time"],
            )

            # Before fitting
            summary_before = model.summary()
            assert "not been fitted" in summary_before

            # After fitting
            model.fit(maxiter=10, verbose=False)
            summary_after = model.summary()

            assert isinstance(summary_after, str)
            assert "Honoré Trimmed LAD Estimator" in summary_after
            assert "semiparametric" in summary_after
            assert "Standard errors are not computed" in summary_after

    def test_trimming_logic(self):
        """Test the trimming logic for censored observations."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create data where some entities have all observations censored
            N, T, K = 5, 2, 1
            y = np.array(
                [
                    0,
                    0,  # Entity 0: both censored
                    0,
                    1,  # Entity 1: one censored
                    1,
                    2,  # Entity 2: none censored
                    0,
                    0,  # Entity 3: both censored
                    0.5,
                    0,  # Entity 4: one censored
                ]
            )
            X = np.random.randn(N * T, K)
            groups = np.repeat(np.arange(N), T)
            time = np.tile(np.arange(T), N)

            model = HonoreTrimmedEstimator(
                endog=y, exog=X, groups=groups, time=time, censoring_point=0
            )

            _delta_y, _delta_X, trim_indicator = model._create_pairwise_differences()

            # Entities 0 and 3 should be trimmed (both obs censored)
            # Total pairs = 5 (one per entity)
            assert len(trim_indicator) == N

            # Check specific trimming decisions
            # Entity 0 and 3 should be trimmed (indicator = 0)
            # Others should be kept (indicator = 1)
            expected_trim = np.array([0, 1, 1, 0, 1])
            assert np.array_equal(trim_indicator, expected_trim)

    def test_no_valid_differences_error(self):
        """Test error when no valid pairwise differences exist."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # All observations censored - should fail
            N, T = 3, 2
            y = np.zeros(N * T)  # All censored at 0
            X = np.random.randn(N * T, 1)
            groups = np.repeat(np.arange(N), T)
            time = np.tile(np.arange(T), N)

            model = HonoreTrimmedEstimator(
                endog=y, exog=X, groups=groups, time=time, censoring_point=0
            )

            # This should work but all observations will be trimmed
            _delta_y, _delta_X, trim_indicator = model._create_pairwise_differences()
            assert np.sum(trim_indicator) == 0  # All trimmed

    def test_single_period_entity(self):
        """Test handling of entities with single observation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Mix of entities with different numbers of observations
            y = np.array([1, 2, 3, 4, 5])  # 3 entities: 2 obs, 2 obs, 1 obs
            X = np.random.randn(5, 1)
            groups = np.array([0, 0, 1, 1, 2])
            time = np.array([0, 1, 0, 1, 0])

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)

            delta_y, _delta_X, _trim_indicator = model._create_pairwise_differences()

            # Only entities 0 and 1 can create pairwise differences
            # Entity 2 has only 1 observation, so no differences
            assert len(delta_y) == 2  # One pair from entity 0, one from entity 1


class TestHonoreResultsPredict:
    """Tests for HonoreResults.predict() method."""

    def test_predict_with_none_exog_raises(self):
        """Test that predict with None exog raises ValueError."""
        result = HonoreResults(
            params=np.array([0.5, -0.3]),
            converged=True,
            n_iter=10,
            n_obs=30,
            n_entities=10,
            n_trimmed=5,
        )

        with pytest.raises(ValueError, match="exog is required"):
            result.predict(exog=None)

    def test_predict_with_ndarray(self):
        """Test predict with numpy array input."""
        result = HonoreResults(
            params=np.array([0.5, -0.3]),
            converged=True,
            n_iter=10,
            n_obs=30,
            n_entities=10,
            n_trimmed=5,
        )

        X_new = np.array([[1.0, 2.0], [3.0, 4.0]])
        predictions = result.predict(exog=X_new)

        assert len(predictions) == 2
        np.testing.assert_allclose(predictions, X_new @ np.array([0.5, -0.3]))

    def test_predict_with_dataframe_and_exog_names(self):
        """Test predict with DataFrame and exog_names set."""
        result = HonoreResults(
            params=np.array([0.5, -0.3]),
            converged=True,
            n_iter=10,
            n_obs=30,
            n_entities=10,
            n_trimmed=5,
            exog_names=["x1", "x2"],
        )

        df = pd.DataFrame({"x1": [1.0, 3.0], "x2": [2.0, 4.0], "extra": [99.0, 100.0]})
        predictions = result.predict(exog=df)

        assert len(predictions) == 2
        expected = np.array([1.0 * 0.5 + 2.0 * (-0.3), 3.0 * 0.5 + 4.0 * (-0.3)])
        np.testing.assert_allclose(predictions, expected)

    def test_predict_with_dataframe_missing_columns(self):
        """Test predict with DataFrame missing required columns."""
        result = HonoreResults(
            params=np.array([0.5, -0.3]),
            converged=True,
            n_iter=10,
            n_obs=30,
            n_entities=10,
            n_trimmed=5,
            exog_names=["x1", "x2"],
        )

        df = pd.DataFrame({"x1": [1.0, 3.0], "other_col": [2.0, 4.0]})

        with pytest.raises(ValueError, match="Missing columns"):
            result.predict(exog=df)

    def test_predict_with_dataframe_no_exog_names(self):
        """Test predict with DataFrame when exog_names is None."""
        result = HonoreResults(
            params=np.array([0.5, -0.3]),
            converged=True,
            n_iter=10,
            n_obs=30,
            n_entities=10,
            n_trimmed=5,
            exog_names=None,
        )

        df = pd.DataFrame({"x1": [1.0, 3.0], "x2": [2.0, 4.0]})
        predictions = result.predict(exog=df)

        assert len(predictions) == 2


class TestHonoreUncoveredBranches:
    """Tests covering previously uncovered branches in honore.py."""

    def test_dimension_mismatch_endog_exog(self):
        """Test error when endog and exog have different number of observations."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with pytest.raises(ValueError, match="Dimension mismatch"):
                HonoreTrimmedEstimator(
                    endog=np.array([1.0, 2.0, 3.0]),
                    exog=np.array([[1.0], [2.0]]),  # Wrong length
                    groups=np.array([0, 0, 1]),
                    time=np.array([0, 1, 0]),
                )

    def test_groups_length_mismatch(self):
        """Test error when groups has different length from endog."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with pytest.raises(ValueError, match="groups and time must have same length"):
                HonoreTrimmedEstimator(
                    endog=np.array([1.0, 2.0, 3.0]),
                    exog=np.array([[1.0], [2.0], [3.0]]),
                    groups=np.array([0, 0]),  # Wrong length
                    time=np.array([0, 1, 0]),
                )

    def test_time_length_mismatch(self):
        """Test error when time has different length from endog."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with pytest.raises(ValueError, match="groups and time must have same length"):
                HonoreTrimmedEstimator(
                    endog=np.array([1.0, 2.0, 3.0]),
                    exog=np.array([[1.0], [2.0], [3.0]]),
                    groups=np.array([0, 0, 1]),
                    time=np.array([0, 1]),  # Wrong length
                )

    def test_no_valid_pairwise_differences_raises(self):
        """Test error when all entities have only single observation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Each entity has only 1 observation
            model = HonoreTrimmedEstimator(
                endog=np.array([1.0, 2.0, 3.0]),
                exog=np.array([[1.0], [2.0], [3.0]]),
                groups=np.array([0, 1, 2]),  # Different entities
                time=np.array([0, 0, 0]),
            )

            with pytest.raises(ValueError, match="No valid pairwise differences"):
                model._create_pairwise_differences()

    def test_fit_with_few_retained_obs_zeros_start(self):
        """Test fit starting with zeros when retained obs <= n_features."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create data where almost all observations are censored
            N, T, K = 3, 2, 3  # K=3 features, only 3 pairs possible
            y = np.array([0, 0.1, 0, 0, 0, 0.2])
            X = np.random.randn(N * T, K)
            groups = np.repeat(np.arange(N), T)
            time = np.tile(np.arange(T), N)

            model = HonoreTrimmedEstimator(
                endog=y, exog=X, groups=groups, time=time, censoring_point=0
            )

            # With many censored pairs, retained might be <= K
            results = model.fit(maxiter=5, verbose=False)
            assert isinstance(results, HonoreResults)

    def test_fit_verbose_true(self):
        """Test fit with verbose=True exercises logging branches."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        beta_true = np.array([0.5, -0.3])
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y_star = X @ beta_true + alpha + np.random.randn(N * T) * 0.3
        y = np.maximum(0, y_star)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)

            # Set logging level to capture log output
            logger = logging.getLogger("panelbox.models.censored.honore")
            logger.setLevel(logging.DEBUG)

            results = model.fit(maxiter=10, verbose=True)
            assert isinstance(results, HonoreResults)

    def test_fit_nelder_mead_method(self):
        """Test fit with non-L-BFGS-B method (Nelder-Mead)."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        beta_true = np.array([0.5, -0.3])
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y_star = X @ beta_true + alpha + np.random.randn(N * T) * 0.3
        y = np.maximum(0, y_star)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)

            results = model.fit(method="Nelder-Mead", maxiter=10, verbose=False)
            assert isinstance(results, HonoreResults)
            assert results.params.shape == (K,)

    def test_fit_powell_method(self):
        """Test fit with Powell method."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y = np.maximum(0, X @ np.array([0.5, -0.3]) + alpha + np.random.randn(N * T) * 0.3)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)
            results = model.fit(method="Powell", maxiter=10, verbose=False)
            assert isinstance(results, HonoreResults)

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            N, T = 5, 2
            y = np.random.randn(N * T)
            X = np.random.randn(N * T, 1)
            groups = np.repeat(np.arange(N), T)
            time = np.tile(np.arange(T), N)

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)

            with pytest.raises(ValueError, match="Model must be fitted"):
                model.predict()

    def test_predict_with_dataframe(self):
        """Test predict with DataFrame input on the estimator."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y = np.maximum(0, X @ np.array([0.5, -0.3]) + alpha + np.random.randn(N * T) * 0.3)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)
            model.fit(maxiter=10, verbose=False)

            # Predict with DataFrame (no exog_names set)
            df = pd.DataFrame(np.random.randn(5, K), columns=["a", "b"])
            predictions = model.predict(exog=df)
            assert len(predictions) == 5

    def test_predict_with_dataframe_and_exog_names(self):
        """Test predict with DataFrame when exog_names is set on estimator."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y = np.maximum(0, X @ np.array([0.5, -0.3]) + alpha + np.random.randn(N * T) * 0.3)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)
            model.fit(maxiter=10, verbose=False)
            model.exog_names = ["x1", "x2"]

            df = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})
            predictions = model.predict(exog=df)
            assert len(predictions) == 2

    def test_predict_with_dataframe_missing_columns(self):
        """Test predict with DataFrame missing required columns."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y = np.maximum(0, X @ np.array([0.5, -0.3]) + alpha + np.random.randn(N * T) * 0.3)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)
            model.fit(maxiter=10, verbose=False)
            model.exog_names = ["x1", "x2"]

            df = pd.DataFrame({"x1": [1.0, 2.0], "wrong_col": [3.0, 4.0]})

            with pytest.raises(ValueError, match="Missing columns"):
                model.predict(exog=df)

    def test_fit_verbose_converged_logging(self):
        """Test verbose logging when optimization converges."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y = np.maximum(0, X @ np.array([0.5, -0.3]) + alpha + np.random.randn(N * T) * 0.3)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)

            logger = logging.getLogger("panelbox.models.censored.honore")
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            logger.addHandler(handler)

            try:
                results = model.fit(maxiter=500, verbose=True, tol=1e-4)
                assert isinstance(results, HonoreResults)
            finally:
                logger.removeHandler(handler)

    def test_fit_verbose_not_converged_logging(self):
        """Test verbose logging when optimization does not converge."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y = np.maximum(0, X @ np.array([0.5, -0.3]) + alpha + np.random.randn(N * T) * 0.3)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)

            logger = logging.getLogger("panelbox.models.censored.honore")
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            logger.addHandler(handler)

            try:
                # Very few iterations to force non-convergence
                results = model.fit(maxiter=1, verbose=True, tol=1e-20)
                assert isinstance(results, HonoreResults)
            finally:
                logger.removeHandler(handler)

    def test_fit_with_custom_start_params(self):
        """Test fit with user-provided start_params."""
        np.random.seed(42)
        N, T, K = 10, 3, 2
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y = np.maximum(0, X @ np.array([0.5, -0.3]) + alpha + np.random.randn(N * T) * 0.3)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time)

            start_params = np.array([0.1, -0.1])
            results = model.fit(start_params=start_params, maxiter=10, verbose=False)
            assert isinstance(results, HonoreResults)

    def test_different_censoring_point(self):
        """Test estimator with non-zero censoring point."""
        np.random.seed(42)
        N, T, K = 10, 3, 1
        X = np.random.randn(N * T, K)
        alpha = np.repeat(np.random.randn(N) * 0.5, T)
        y_star = X @ np.array([1.0]) + alpha + np.random.randn(N * T) * 0.3
        censoring_point = -0.5
        y = np.maximum(censoring_point, y_star)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = HonoreTrimmedEstimator(
                endog=y, exog=X, groups=groups, time=time, censoring_point=censoring_point
            )

            assert model.censoring_point == censoring_point
            results = model.fit(maxiter=10, verbose=False)
            assert isinstance(results, HonoreResults)
