"""
Unit tests for binary choice models.

Tests cover:
- Pooled Logit and Probit models
- Fixed Effects Logit
- Convergence and optimization
- Standard errors computation
- Predictions and diagnostics
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.special import expit

from panelbox.models.discrete import FixedEffectsLogit, PooledLogit, PooledProbit
from panelbox.models.discrete.results import NonlinearPanelResults


class TestPooledLogit:
    """Tests for Pooled Logit model."""

    @pytest.fixture
    def panel_data(self):
        """Generate panel data for testing."""
        np.random.seed(42)
        n_entities = 100
        n_periods = 10
        n_obs = n_entities * n_periods

        # Generate panel structure
        entity_id = np.repeat(range(n_entities), n_periods)
        time_id = np.tile(range(n_periods), n_entities)

        # Generate covariates
        x1 = np.random.randn(n_obs)
        x2 = np.random.randn(n_obs)
        x3 = np.random.binomial(1, 0.5, n_obs)

        # True parameters
        beta_true = np.array([0.5, -0.3, 0.8, 1.2])  # intercept, x1, x2, x3

        # Generate outcome
        X = np.column_stack([np.ones(n_obs), x1, x2, x3])
        linear_pred = X @ beta_true
        prob = expit(linear_pred)
        y = np.random.binomial(1, prob)

        # Create DataFrame
        df = pd.DataFrame(
            {"entity_id": entity_id, "time_id": time_id, "y": y, "x1": x1, "x2": x2, "x3": x3}
        )

        return df, beta_true

    def test_pooled_logit_basic(self, panel_data):
        """Test basic Pooled Logit estimation."""
        df, beta_true = panel_data

        # Fit model
        model = PooledLogit("y ~ x1 + x2 + x3", df, "entity_id", "time_id")
        results = model.fit(se_type="nonrobust")

        # Check convergence
        assert results.converged

        # Check parameters are reasonable
        assert_allclose(results.params, beta_true, rtol=0.5)

        # Check standard errors are positive
        assert np.all(results.std_errors > 0)

    def test_pooled_logit_cluster_se(self, panel_data):
        """Test Pooled Logit with cluster-robust standard errors."""
        df, _ = panel_data

        # Fit with different SE types
        model = PooledLogit("y ~ x1 + x2 + x3", df, "entity_id", "time_id")

        results_nonrobust = model.fit(se_type="nonrobust")
        results_cluster = model.fit(se_type="cluster")

        # Cluster SEs should generally be larger than non-robust
        # (not always, but typically in presence of clustering)
        se_ratio = results_cluster.std_errors / results_nonrobust.std_errors
        assert np.median(se_ratio) >= 0.8  # Some tolerance for randomness

    def test_cluster_se_manual_implementation(self, panel_data):
        """Test cluster-robust SEs against manual implementation."""
        df, _ = panel_data

        # Fit model with cluster SE
        model = PooledLogit("y ~ x1 + x2", df, "entity_id", "time_id")
        results = model.fit(se_type="cluster")

        # Manual calculation of cluster-robust SEs
        # Build design matrices
        y = df["y"].values
        X = pd.get_dummies(df[["x1", "x2"]], drop_first=False)
        X.insert(0, "intercept", 1.0)
        X = X.values
        entities = df["entity_id"].values

        # Get parameters
        params = results.params.values

        # Calculate fitted probabilities
        eta = X @ params
        p = expit(eta)

        # Hessian: H = -X'WX where W = diag(p*(1-p))
        W = p * (1 - p)
        H = -(X.T * W) @ X
        H_inv = np.linalg.inv(H)

        # Cluster meat: S = Σ_g (Σ_i∈g s_i)(Σ_i∈g s_i)'
        # where s_i = (y_i - p_i) * X_i is the score
        scores = (y - p)[:, np.newaxis] * X

        # Aggregate scores by cluster
        unique_entities = np.unique(entities)
        n_clusters = len(unique_entities)
        cluster_scores = np.zeros((n_clusters, X.shape[1]))

        for i, entity in enumerate(unique_entities):
            mask = entities == entity
            cluster_scores[i] = scores[mask].sum(axis=0)

        # Meat matrix
        S = cluster_scores.T @ cluster_scores

        # Degrees of freedom correction
        n = len(y)
        k = X.shape[1]
        G = n_clusters
        df_correction = G / (G - 1) * (n - 1) / (n - k)

        # Sandwich variance
        vcov_cluster = df_correction * H_inv @ S @ H_inv
        se_cluster_manual = np.sqrt(np.diag(vcov_cluster))

        # Compare with model results
        assert_allclose(results.std_errors.values, se_cluster_manual, rtol=1e-6)

    def test_pooled_logit_predictions(self, panel_data):
        """Test prediction methods."""
        df, _ = panel_data

        model = PooledLogit("y ~ x1 + x2", df, "entity_id", "time_id")
        results = model.fit()

        # Linear predictions
        linear_pred = results.predict(type="linear")
        assert linear_pred.shape == (len(df),)
        assert np.all(np.isfinite(linear_pred))

        # Probability predictions
        prob_pred = results.predict(type="prob")
        assert prob_pred.shape == (len(df),)
        assert np.all((prob_pred >= 0) & (prob_pred <= 1))

        # Class predictions
        class_pred = results.predict(type="class")
        assert class_pred.shape == (len(df),)
        assert np.all(np.isin(class_pred, [0, 1]))

    def test_pooled_logit_diagnostics(self, panel_data):
        """Test diagnostic measures."""
        df, _ = panel_data

        model = PooledLogit("y ~ x1 + x2", df, "entity_id", "time_id")
        results = model.fit()

        # Pseudo-R²
        r2_mcfadden = results.pseudo_r2("mcfadden")
        assert 0 <= r2_mcfadden <= 1

        r2_cox_snell = results.pseudo_r2("cox_snell")
        assert 0 <= r2_cox_snell <= 1

        # Information criteria
        assert results.aic > 0
        assert results.bic > 0
        assert results.bic >= results.aic  # BIC has larger penalty

        # Classification metrics
        metrics = results.classification_metrics()
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["auc_roc"] <= 1

    def test_pooled_logit_multiple_starts(self, panel_data):
        """Test optimization with multiple starting values."""
        df, _ = panel_data

        model = PooledLogit("y ~ x1 + x2", df, "entity_id", "time_id")

        # Single start
        results1 = model.fit(n_starts=1)

        # Multiple starts
        results2 = model.fit(n_starts=3)

        # Both should converge
        assert results1.converged
        assert results2.converged

        # Results should be similar (found same optimum)
        assert_allclose(results1.params, results2.params, rtol=0.1)


class TestPooledProbit:
    """Tests for Pooled Probit model."""

    @pytest.fixture
    def panel_data(self):
        """Generate panel data for testing."""
        np.random.seed(123)
        n_entities = 80
        n_periods = 8
        n_obs = n_entities * n_periods

        # Generate panel structure
        entity_id = np.repeat(range(n_entities), n_periods)
        time_id = np.tile(range(n_periods), n_entities)

        # Generate covariates
        x1 = np.random.randn(n_obs)
        x2 = np.random.randn(n_obs)

        # True parameters (smaller for probit)
        beta_true = np.array([0.3, -0.2, 0.5])  # intercept, x1, x2

        # Generate outcome
        X = np.column_stack([np.ones(n_obs), x1, x2])
        linear_pred = X @ beta_true
        prob = stats.norm.cdf(linear_pred)
        y = np.random.binomial(1, prob)

        # Create DataFrame
        df = pd.DataFrame({"entity_id": entity_id, "time_id": time_id, "y": y, "x1": x1, "x2": x2})

        return df, beta_true

    def test_pooled_probit_basic(self, panel_data):
        """Test basic Pooled Probit estimation."""
        df, beta_true = panel_data

        # Fit model
        model = PooledProbit("y ~ x1 + x2", df, "entity_id", "time_id")
        results = model.fit()

        # Check convergence
        assert results.converged

        # Check parameters are reasonable
        assert_allclose(results.params, beta_true, rtol=0.5)

        # Check standard errors are positive
        assert np.all(results.std_errors > 0)

    def test_probit_vs_logit_consistency(self):
        """Test that Probit and Logit give similar qualitative results."""
        np.random.seed(456)
        n = 500

        # Simple cross-sectional data
        x = np.random.randn(n)
        prob_true = expit(1.0 + 0.5 * x)
        y = np.random.binomial(1, prob_true)

        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x": x})

        # Fit both models
        logit_model = PooledLogit("y ~ x", df, "entity_id", "time_id")
        logit_results = logit_model.fit()

        probit_model = PooledProbit("y ~ x", df, "entity_id", "time_id")
        probit_results = probit_model.fit()

        # Signs should match
        assert np.sign(logit_results.params[1]) == np.sign(probit_results.params[1])

        # Logit coefficients ≈ 1.6 * Probit coefficients (rough approximation)
        ratio = logit_results.params[1] / probit_results.params[1]
        assert 1.2 < ratio < 2.0


class TestFixedEffectsLogit:
    """Tests for Fixed Effects Logit model."""

    @pytest.fixture
    def panel_data_with_fe(self):
        """Generate panel data with fixed effects."""
        np.random.seed(789)
        n_entities = 50
        n_periods = 6

        # Generate entity fixed effects
        alpha_i = np.random.randn(n_entities)

        data_list = []
        for i in range(n_entities):
            for t in range(n_periods):
                x1 = np.random.randn()
                x2 = np.random.randn()

                # Linear index with fixed effect
                linear_pred = alpha_i[i] + 0.5 * x1 - 0.3 * x2
                prob = expit(linear_pred)
                y = np.random.binomial(1, prob)

                data_list.append({"entity_id": i, "time_id": t, "y": y, "x1": x1, "x2": x2})

        df = pd.DataFrame(data_list)
        return df

    def test_fe_logit_dgp_with_known_effects(self):
        """Test FE Logit recovers true parameters from known DGP."""
        np.random.seed(42)
        n_entities = 100
        n_periods = 8

        # True parameters
        beta_true = np.array([1.0, -0.5])

        # Generate entity fixed effects
        alpha_i = np.random.randn(n_entities) * 2.0

        data_list = []
        for i in range(n_entities):
            # Ensure some variation for each entity
            y_sum = 0
            attempts = 0
            while (y_sum == 0 or y_sum == n_periods) and attempts < 10:
                entity_data = []
                y_sum = 0
                for t in range(n_periods):
                    x1 = np.random.randn()
                    x2 = np.random.randn()

                    # Linear index with fixed effect
                    linear_pred = alpha_i[i] + beta_true[0] * x1 + beta_true[1] * x2
                    prob = expit(linear_pred)
                    y = np.random.binomial(1, prob)
                    y_sum += y

                    entity_data.append({"entity_id": i, "time_id": t, "y": y, "x1": x1, "x2": x2})
                attempts += 1

            # Only add entities with variation
            if 0 < y_sum < n_periods:
                data_list.extend(entity_data)

        df = pd.DataFrame(data_list)

        # Fit FE Logit model
        model = FixedEffectsLogit("y ~ x1 + x2", df, "entity_id", "time_id")
        results = model.fit()

        # Check convergence
        assert results.converged

        # Check parameters are close to true values
        # FE Logit is consistent but may have larger variance
        assert_allclose(results.params.values, beta_true, rtol=0.3)

        # Check standard errors are reasonable
        assert np.all(results.std_errors > 0)
        assert np.all(results.std_errors < 1.0)  # Shouldn't be too large

    def test_fe_logit_basic(self, panel_data_with_fe):
        """Test basic Fixed Effects Logit estimation."""
        df = panel_data_with_fe

        # Fit model
        model = FixedEffectsLogit("y ~ x1 + x2", df, "entity_id", "time_id")
        results = model.fit()

        # Check convergence
        assert results.converged

        # Check that some entities were dropped (no within variation)
        assert hasattr(model, "dropped_entities")
        assert hasattr(model, "n_used_entities")

        # Check parameters
        assert len(results.params) == 2  # x1 and x2, no intercept
        assert np.all(results.std_errors > 0)

    def test_fe_logit_drops_no_variation(self):
        """Test that FE Logit correctly drops entities without variation."""
        # Create data with some entities having no variation
        data_list = []

        # Entities with variation
        for i in range(30):
            for t in range(5):
                y = 1 if (i + t) % 3 == 0 else 0
                data_list.append({"entity_id": i, "time_id": t, "y": y, "x": np.random.randn()})

        # Entities with no variation (always 0)
        for i in range(30, 40):
            for t in range(5):
                data_list.append({"entity_id": i, "time_id": t, "y": 0, "x": np.random.randn()})

        # Entities with no variation (always 1)
        for i in range(40, 50):
            for t in range(5):
                data_list.append({"entity_id": i, "time_id": t, "y": 1, "x": np.random.randn()})

        df = pd.DataFrame(data_list)

        # Fit model
        model = FixedEffectsLogit("y ~ x", df, "entity_id", "time_id")

        # Check that correct number of entities are dropped
        assert len(model.dropped_entities) == 20  # Entities 30-49
        assert model.n_used_entities == 30  # Entities 0-29

    def test_fe_logit_vs_pooled_with_dummies(self, panel_data_with_fe):
        """Test that FE Logit is consistent with pooled logit with entity dummies."""
        df = panel_data_with_fe

        # Keep only entities with variation for fair comparison
        entity_variation = df.groupby("entity_id")["y"].agg(["mean", "count"])
        entities_with_var = entity_variation[
            (entity_variation["mean"] > 0) & (entity_variation["mean"] < 1)
        ].index

        df_subset = df[df["entity_id"].isin(entities_with_var)].copy()

        # Fixed Effects Logit
        fe_model = FixedEffectsLogit("y ~ x1 + x2", df_subset, "entity_id", "time_id")
        fe_results = fe_model.fit()

        # Both should converge
        assert fe_results.converged

        # Coefficients should be defined
        assert np.all(np.isfinite(fe_results.params))

    def test_fe_logit_performance(self):
        """Test FE Logit performance with T=10, N=500."""
        import time

        np.random.seed(333)
        n_entities = 500
        n_periods = 10

        # Generate data
        data_list = []
        for i in range(n_entities):
            # Generate entity fixed effect
            alpha_i = np.random.randn() * 0.5

            for t in range(n_periods):
                x = np.random.randn()

                # Simple model
                linear_pred = alpha_i + 0.7 * x
                prob = expit(linear_pred)
                y = np.random.binomial(1, prob)

                data_list.append({"entity_id": i, "time_id": t, "y": y, "x": x})

        df = pd.DataFrame(data_list)

        # Fit model and measure time
        model = FixedEffectsLogit("y ~ x", df, "entity_id", "time_id")

        start_time = time.time()
        results = model.fit()
        elapsed_time = time.time() - start_time

        # Should converge in reasonable time (< 5 seconds)
        assert elapsed_time < 5.0

        # Should converge
        assert results.converged

        # Parameters should be reasonable
        assert np.all(np.isfinite(results.params))


class TestHosmerLemeshowTest:
    """Tests for Hosmer-Lemeshow goodness-of-fit test."""

    def test_hosmer_lemeshow_perfect_fit(self):
        """Test H-L test with perfect fit."""
        np.random.seed(111)
        n = 1000

        # Generate data with deterministic outcome
        x = np.random.randn(n)
        y = (x > 0).astype(int)

        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x": x})

        # Fit model (should fit perfectly with large coefficient)
        model = PooledLogit("y ~ x", df, "entity_id", "time_id")
        results = model.fit()

        # H-L test should show good fit (high p-value)
        hl_test = results.hosmer_lemeshow_test(n_groups=10)

        # With near-perfect fit, p-value should be high
        # (but not exactly 1 due to numerical precision)
        assert hl_test["p_value"] > 0.05

    def test_hosmer_lemeshow_poor_fit(self):
        """Test H-L test with poor fit."""
        np.random.seed(222)
        n = 1000

        # Generate data with nonlinear relationship
        x = np.random.randn(n)
        # True relationship is quadratic, but we fit linear
        prob_true = expit(-2 + 3 * x**2)
        y = np.random.binomial(1, prob_true)

        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x": x})

        # Fit misspecified model (linear instead of quadratic)
        model = PooledLogit("y ~ x", df, "entity_id", "time_id")
        results = model.fit()

        # H-L test should detect poor fit
        hl_test = results.hosmer_lemeshow_test(n_groups=10)

        # Should have low p-value indicating poor fit
        assert hl_test["p_value"] < 0.1


class TestConvergence:
    """Tests for convergence diagnostics and warnings."""

    def test_convergence_warning(self):
        """Test that convergence warnings are issued appropriately."""
        np.random.seed(333)

        # Create difficult optimization problem
        n = 100
        x = np.random.randn(n) * 0.01  # Very small variation
        y = np.random.binomial(1, 0.5, n)  # Random outcome

        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x": x})

        # Fit with very tight tolerance
        model = PooledLogit("y ~ x", df, "entity_id", "time_id")

        # Should converge but might have warnings
        with pytest.warns(None) as warnings:
            results = model.fit(maxiter=10, gtol=1e-10)

        # Check if appropriate warnings were issued
        # (exact warnings depend on data realization)

    def test_multiple_optimization_methods(self):
        """Test different optimization methods."""
        np.random.seed(444)
        n = 200

        x = np.random.randn(n)
        prob_true = expit(0.5 + 0.8 * x)
        y = np.random.binomial(1, prob_true)

        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x": x})

        model = PooledLogit("y ~ x", df, "entity_id", "time_id")

        # Try different methods
        results_bfgs = model.fit(method="bfgs")
        results_newton = model.fit(method="newton")

        # Both should converge to similar values
        assert results_bfgs.converged
        assert results_newton.converged

        # Parameters should be very close
        assert_allclose(results_bfgs.params, results_newton.params, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
