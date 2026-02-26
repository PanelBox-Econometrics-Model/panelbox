"""
Test Ordered choice models against R implementations.

This module validates panelbox ordered models against R packages:
- MASS::polr for Ordered Logit/Probit
- ordinal::clm for more advanced ordered models

Requirements:
    R packages: MASS, ordinal
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.models.discrete.ordered import OrderedLogit, OrderedProbit

pytestmark = pytest.mark.r_validation


class TestVsROrdered:
    """Test ordered choice models against R implementations."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)

    def generate_ordered_data(self, n_entities=50, n_periods=10, n_categories=4, pooled=True):
        """Generate panel data for ordered choice model."""
        N = n_entities
        T = n_periods
        J = n_categories  # number of ordered categories
        K = 3  # number of covariates

        # True parameters
        beta_true = np.array([0.8, -0.5, 0.3])

        # True cutpoints (J-1 cutpoints for J categories)
        # These define thresholds for latent variable
        cutpoints_true = np.array([-1.0, 0.0, 1.5])  # for 4 categories

        # Generate covariates
        X = np.random.randn(N, T, K)

        # Generate latent variable
        if pooled:
            # Pooled model - no entity effects
            epsilon = np.random.logistic(0, 1, (N, T))  # logistic errors
            y_star = np.zeros((N, T))
            for i in range(N):
                for t in range(T):
                    y_star[i, t] = X[i, t] @ beta_true + epsilon[i, t]
        else:
            # Random effects model
            sigma_alpha = 0.5
            alpha_i = np.random.normal(0, sigma_alpha, N)
            epsilon = np.random.logistic(0, 1, (N, T))
            y_star = np.zeros((N, T))
            for i in range(N):
                for t in range(T):
                    y_star[i, t] = X[i, t] @ beta_true + alpha_i[i] + epsilon[i, t]

        # Convert latent to ordered categories
        y = np.zeros((N, T), dtype=int)
        for i in range(N):
            for t in range(T):
                if y_star[i, t] <= cutpoints_true[0]:
                    y[i, t] = 0
                elif y_star[i, t] <= cutpoints_true[1]:
                    y[i, t] = 1
                elif y_star[i, t] <= cutpoints_true[2]:
                    y[i, t] = 2
                else:
                    y[i, t] = 3

        # Flatten for panel format
        entity_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)
        y_flat = y.flatten()
        X_flat = X.reshape(-1, K)

        # Create panel DataFrame
        data = pd.DataFrame(
            {
                "entity": entity_ids,
                "time": time_ids,
                "y": y_flat,
                "X1": X_flat[:, 0],
                "X2": X_flat[:, 1],
                "X3": X_flat[:, 2],
            }
        )

        # Check category distribution
        for j in range(J):
            prop = np.mean(y_flat == j)
            print(f"Category {j}: {prop:.2%}")

        return data, beta_true, cutpoints_true

    def run_r_polr(self, data):
        """Run Ordered Logit in R using MASS::polr."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_csv(f.name, index=False)
            csv_path = f.name

        r_script = f"""
        library(MASS)

        # Read data
        data <- read.csv("{csv_path}")

        # Convert y to ordered factor
        data$y <- as.ordered(data$y)

        # Fit ordered logit model
        model <- polr(y ~ X1 + X2 + X3, data = data,
                     method = "logistic", Hess = TRUE)

        # Extract coefficients and cutpoints
        coef_est <- coef(model)
        cutpoints_est <- model$zeta

        # Get standard errors
        se_est <- sqrt(diag(vcov(model)))

        # Log-likelihood
        loglik <- as.numeric(logLik(model))

        # Create results
        results <- list(
            coefficients = as.numeric(coef_est),
            cutpoints = as.numeric(cutpoints_est),
            std_errors = as.numeric(se_est[1:length(coef_est)]),
            loglik = loglik,
            converged = model$convergence == 0
        )

        # Save results as JSON
        library(jsonlite)
        write_json(results, "{csv_path}.results.json")
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
            f.write(r_script)
            r_path = f.name

        try:
            # Run R script
            result = subprocess.run(["Rscript", r_path], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"R stderr: {result.stderr}")
                return None

            # Read results
            with open(f"{csv_path}.results.json") as f:
                results = json.load(f)

            return results

        except Exception as e:
            print(f"Error running R: {e}")
            return None

        finally:
            # Cleanup
            for path in [csv_path, r_path, f"{csv_path}.results.json"]:
                if os.path.exists(path):
                    os.remove(path)

    def run_r_clm(self, data):
        """Run Ordered Logit in R using ordinal::clm for comparison."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_csv(f.name, index=False)
            csv_path = f.name

        r_script = f"""
        library(ordinal)

        # Read data
        data <- read.csv("{csv_path}")

        # Convert y to ordered factor
        data$y <- as.ordered(data$y)

        # Fit cumulative link model (ordered logit)
        model <- clm(y ~ X1 + X2 + X3, data = data,
                    link = "logit")

        # Extract coefficients and thresholds
        coef_est <- coef(model)[4:6]  # Skip threshold parameters
        thresholds <- coef(model)[1:3]

        # Create results
        results <- list(
            coefficients = as.numeric(coef_est),
            thresholds = as.numeric(thresholds),
            loglik = as.numeric(logLik(model)),
            converged = model$convergence == 0
        )

        # Save results as JSON
        library(jsonlite)
        write_json(results, "{csv_path}.results.json")
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
            f.write(r_script)
            r_path = f.name

        try:
            # Run R script
            result = subprocess.run(["Rscript", r_path], capture_output=True, text=True)

            if result.returncode != 0:
                # Try without ordinal package
                return None

            # Read results
            with open(f"{csv_path}.results.json") as f:
                results = json.load(f)

            return results

        except Exception:
            return None

        finally:
            # Cleanup
            for path in [csv_path, r_path, f"{csv_path}.results.json"]:
                if os.path.exists(path):
                    os.remove(path)

    @pytest.mark.skipif(not os.path.exists("/usr/bin/Rscript"), reason="R not available")
    def test_ordered_logit_vs_polr(self):
        """Test Ordered Logit against MASS::polr."""
        # Generate test data (pooled for simpler comparison)
        data, beta_true, cutpoints_true = self.generate_ordered_data(
            n_entities=40, n_periods=6, n_categories=4, pooled=True
        )

        # Fit with panelbox
        y = data["y"].values
        X = data[["X1", "X2", "X3"]].values
        groups = data["entity"].values
        time = data["time"].values

        model_pb = OrderedLogit(y, X, groups, time, n_categories=4)
        result_pb = model_pb.fit()

        # Get R results from MASS::polr
        r_results = self.run_r_polr(data)

        if r_results is None:
            pytest.skip("R MASS package not available")

        # Compare coefficients
        K = X.shape[-1]
        beta_pb = result_pb.params[:K]
        beta_r = np.array(r_results["coefficients"])

        print(f"\nPanelbox coefficients: {beta_pb}")
        print(f"R polr coefficients: {beta_r}")
        print(f"True coefficients: {beta_true}")

        # Check coefficients match (within tolerance)
        assert_allclose(
            beta_pb, beta_r, rtol=1e-3, atol=1e-4, err_msg="Coefficients should match R polr"
        )

        # Compare cutpoints
        cutpoint_params = result_pb.params[K:]
        cutpoints_pb = model_pb._transform_cutpoints(cutpoint_params)
        cutpoints_r = np.array(r_results["cutpoints"])

        print(f"\nPanelbox cutpoints: {cutpoints_pb}")
        print(f"R polr cutpoints: {cutpoints_r}")
        print(f"True cutpoints: {cutpoints_true}")

        # Check cutpoints are ordered
        assert np.all(np.diff(cutpoints_pb) > 0), "Cutpoints should be ordered"
        assert np.all(np.diff(cutpoints_r) > 0), "R cutpoints should be ordered"

        # Cutpoints may differ in parameterization but relative spacing should be similar
        spacing_pb = np.diff(cutpoints_pb)
        spacing_r = np.diff(cutpoints_r)

        # Check relative spacing
        rel_spacing = spacing_pb / spacing_r
        assert np.all((rel_spacing > 0.7) & (rel_spacing < 1.3)), (
            "Relative spacing of cutpoints should be similar"
        )

        # Compare log-likelihood
        llf_pb = result_pb.llf
        llf_r = r_results["loglik"]
        # JSON may return scalar as list
        if isinstance(llf_r, list):
            llf_r = llf_r[0]

        print(f"\nPanelbox log-likelihood: {llf_pb:.4f}")
        print(f"R polr log-likelihood: {llf_r:.4f}")

        # Log-likelihoods should be very close
        assert abs(llf_pb - llf_r) < 1.0, "Log-likelihoods should be within 1.0"

    @pytest.mark.skipif(not os.path.exists("/usr/bin/Rscript"), reason="R not available")
    def test_ordered_probit_predictions(self):
        """Test ordered probit predictions."""
        # Generate simple test data
        np.random.seed(456)
        N, T = 30, 5
        K = 2
        J = 3  # 3 categories

        X = np.random.randn(N, T, K)

        # Generate ordered outcomes
        beta = np.array([0.5, -0.3])
        cutpoints = np.array([-0.5, 0.5])

        y_star = X @ beta + np.random.normal(0, 1, (N, T))
        y = np.zeros((N, T), dtype=int)
        y[y_star <= cutpoints[0]] = 0
        y[(y_star > cutpoints[0]) & (y_star <= cutpoints[1])] = 1
        y[y_star > cutpoints[1]] = 2

        # Flatten for panel format
        y_flat = y.flatten()
        X_flat = X.reshape(-1, K)
        groups = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        # Fit ordered probit
        model = OrderedProbit(y_flat, X_flat, groups, time_ids, n_categories=3)
        result = model.fit()

        assert result.converged, "Ordered probit should converge"

        # Test probability predictions
        probs = model.predict(type="prob")

        # Check probability properties
        assert probs.shape == (N * T, J), "Should have probabilities for each category"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1"
        assert np.all(probs >= 0), "Probabilities should be non-negative"
        assert np.all(probs <= 1), "Probabilities should be <= 1"

        # Test category predictions
        cats = model.predict(type="category")
        assert cats.shape == (N * T,), "Should predict one category per observation"
        assert np.all((cats >= 0) & (cats < J)), "Categories should be in valid range"

    def test_marginal_effects_ordered(self):
        """Test marginal effects for ordered models."""
        # Generate simple data
        np.random.seed(789)
        N, T = 20, 4
        K = 2
        J = 3

        X = np.random.randn(N, T, K)
        y = np.random.randint(0, J, (N, T))

        # Flatten for panel format
        y_flat = y.flatten()
        X_flat = X.reshape(-1, K)
        groups = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        # Fit model
        model = OrderedLogit(y_flat, X_flat, groups, time_ids, n_categories=J)
        model.fit()

        # Calculate marginal effects for each category
        from panelbox.marginal_effects.discrete_me import compute_ordered_ame

        ame_result = compute_ordered_ame(model)

        # Check that marginal effects sum to zero across categories
        assert ame_result.verify_sum_to_zero(tol=1e-10), (
            "Marginal effects should sum to zero across categories"
        )

        # marginal_effects is a DataFrame (categories x variables)
        ame_matrix = ame_result.marginal_effects
        me_sum = ame_matrix.sum(axis=0)

        print("\nMarginal effects by category:")
        print(ame_matrix)
        print(f"Sum across categories: {me_sum.values}")

        # Effects can be positive for some categories and negative for others
        # This is the expected behavior for ordered models
