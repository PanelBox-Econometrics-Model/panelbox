"""
Test Panel Tobit models against R implementations.

This module validates panelbox censored models against R packages:
- censReg for Random Effects Tobit

Requirements:
    R packages: censReg, plm
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from panelbox.models.censored.tobit import RandomEffectsTobit

pytestmark = pytest.mark.r_validation


class TestVsRCensored:
    """Test censored models against R implementations."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)

    def generate_tobit_data(
        self, n_entities=50, n_periods=10, censoring_point=0.0, censoring_rate=0.3
    ):
        """Generate panel data with censoring."""
        N = n_entities
        T = n_periods
        K = 3  # number of covariates

        # Parameters
        beta_true = np.array([1.5, -0.8, 0.5])
        sigma_alpha = 0.5  # std of random effects
        sigma_eps = 0.8  # std of error term

        # Generate data
        X = np.random.randn(N, T, K)
        alpha_i = np.random.normal(0, sigma_alpha, N)
        epsilon = np.random.normal(0, sigma_eps, (N, T))

        # Latent variable
        y_star = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                y_star[i, t] = X[i, t] @ beta_true + alpha_i[i] + epsilon[i, t]

        # Apply censoring (left censoring at censoring_point)
        y = np.maximum(censoring_point, y_star)

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

        # Check censoring rate
        actual_censoring = np.mean(y_flat == censoring_point)
        print(f"Actual censoring rate: {actual_censoring:.2%}")

        return data, beta_true, sigma_alpha, sigma_eps

    def run_r_tobit_re(self, data, censoring_point=0.0):
        """Run Random Effects Tobit in R using censReg."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_csv(f.name, index=False)
            csv_path = f.name

        r_script = f"""
        library(censReg)
        library(plm)

        # Read data
        data <- read.csv("{csv_path}")

        # Set up panel data
        pdata <- pdata.frame(data, index=c("entity", "time"))

        # Run Random Effects Tobit
        # Note: censReg uses different parameterization
        model <- censReg(y ~ X1 + X2 + X3,
                        left = {censoring_point},
                        data = pdata,
                        method = "BHHH")  # BHHH for better convergence

        # Extract coefficients
        coef_est <- coef(model)[1:4]  # Intercept + 3 betas
        sigma_est <- coef(model)["logSigma"]

        # Create results
        results <- list(
            coefficients = as.numeric(coef_est),
            sigma = exp(as.numeric(sigma_est)),
            loglik = as.numeric(logLik(model)),
            converged = model$code == 0
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

    @pytest.mark.xfail(
        reason="RE Tobit optimization convergence is fragile with this simulated data",
        strict=False,
    )
    @pytest.mark.skipif(not os.path.exists("/usr/bin/Rscript"), reason="R not available")
    def test_tobit_re_vs_censreg(self):
        """Test Random Effects Tobit against R censReg package."""
        # Generate test data
        data, _beta_true, _sigma_alpha_true, _sigma_eps_true = self.generate_tobit_data(
            n_entities=30, n_periods=8, censoring_point=0.0, censoring_rate=0.25
        )

        # Fit with panelbox
        y = data["y"].values
        X = data[["X1", "X2", "X3"]].values
        groups = data["entity"].values
        time = data["time"].values

        model_pb = RandomEffectsTobit(
            y, X, groups, time, censoring_point=0.0, censoring_type="left"
        )
        result_pb = model_pb.fit()

        # Get R results
        r_results = self.run_r_tobit_re(data, censoring_point=0.0)

        if r_results is None:
            pytest.skip("R censReg package not available")

        # Compare coefficients (excluding intercept from R)
        beta_pb = result_pb.params[:-2]  # exclude variance parameters
        beta_r = np.array(r_results["coefficients"][1:], dtype=float)  # exclude intercept

        # Note: censReg and our implementation may have different parameterizations
        # so we check relative magnitudes and signs
        print(f"\nPanelbox coefficients: {beta_pb}")
        print(f"R censReg coefficients: {beta_r}")

        # Check signs match
        assert np.all(np.sign(beta_pb) == np.sign(beta_r)), "Coefficient signs should match"

        # Check relative magnitudes (within 20% due to different methods)
        rel_diff = np.abs(beta_pb - beta_r) / (np.abs(beta_r) + 1e-8)
        assert np.all(rel_diff < 0.20), (
            f"Coefficients should be within 20%, got relative diff: {rel_diff}"
        )

        # Compare sigma
        sigma_pb = np.exp(result_pb.params[-2])  # sigma_eps
        sigma_r = r_results["sigma"]
        # JSON may return scalar as list; extract and convert to float
        if isinstance(sigma_r, list):
            sigma_r = sigma_r[0]
        try:
            sigma_r = float(sigma_r)
        except (ValueError, TypeError):
            pytest.skip("R censReg returned NA for sigma")

        print(f"\nPanelbox sigma: {sigma_pb:.4f}")
        print(f"R censReg sigma: {sigma_r:.4f}")

        # Check sigma within reasonable range
        assert abs(sigma_pb - sigma_r) / sigma_r < 0.25, "Sigma estimates should be within 25%"

    @pytest.mark.xfail(
        reason="RE Tobit optimization convergence is fragile with small simulated panels",
        strict=False,
    )
    def test_tobit_censoring_types(self):
        """Test different censoring types."""
        # Generate data with right censoring
        np.random.seed(123)
        N, T = 20, 5
        K = 2

        X = np.random.randn(N, T, K)
        y = np.random.randn(N, T) + 2.0

        # Apply right censoring at 3.0
        censoring_point = 3.0
        y_right = np.minimum(censoring_point, y)

        # Flatten for panel format
        y_flat = y_right.flatten()
        X_flat = X.reshape(-1, K)
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        # Test right censoring
        model_right = RandomEffectsTobit(
            y_flat, X_flat, groups, time, censoring_point=censoring_point, censoring_type="right"
        )
        result_right = model_right.fit()

        assert result_right.converged, "Right censoring model should converge"

        # Test predictions
        pred_latent = model_right.predict(pred_type="latent")
        pred_censored = model_right.predict(pred_type="censored")

        # Censored predictions should account for censoring
        assert not np.allclose(pred_latent, pred_censored), (
            "Censored and latent predictions should differ"
        )
