"""
Validation tests comparing PanelBox spatial models with R splm package.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.models.spatial import SpatialError, SpatialLag, SpatialWeights

# Path to validation data
DATA_DIR = Path(__file__).parent / "data"


def load_validation_data(test_case):
    """Load validation dataset."""
    # Load panel data
    data_file = DATA_DIR / f"sar_fe_{test_case}.csv"
    if not data_file.exists():
        pytest.skip(f"Validation data not found: {data_file}")

    data = pd.read_csv(data_file)

    # Load weight matrix
    W_file = DATA_DIR / f"W_{test_case}.csv"
    W = pd.read_csv(W_file, index_col=0).values

    return data, W


def load_r_results(model_type, test_case):
    """Load R estimation results."""
    results_file = DATA_DIR / f"{model_type}_{test_case}_results.json"
    if not results_file.exists():
        pytest.skip(f"R results not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


class TestSARValidation:
    """Validate SAR model against R splm."""

    @pytest.mark.parametrize("test_case", ["small", "medium", "large"])
    def test_sar_fe_vs_splm(self, test_case):
        """Test SAR-FE estimation against splm."""
        # Load data
        data, W_array = load_validation_data(test_case)
        W = SpatialWeights(W_array, normalized=True)

        # Load R results
        r_results = load_r_results("sar_fe", test_case)

        # Estimate with PanelBox
        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        result = model.fit(effects="fixed", method="qml", verbose=False)

        # Compare spatial parameter
        assert (
            np.abs(result.params["rho"] - r_results["rho"]) < 0.01
        ), f"Rho mismatch: {result.params['rho']:.6f} vs {r_results['rho']:.6f}"

        # Compare coefficients (allowing for some tolerance)
        for i, param in enumerate(["x1", "x2"]):
            pb_coef = result.params[param]
            r_coef = r_results["coefficients"][i]
            assert (
                np.abs(pb_coef - r_coef) < 0.05
            ), f"{param} mismatch: {pb_coef:.6f} vs {r_coef:.6f}"

        # Compare log-likelihood (with tolerance for numerical differences)
        if "logLik" in r_results:
            assert (
                np.abs(result.llf - r_results["logLik"]) < 1.0
            ), f"Log-likelihood mismatch: {result.llf:.2f} vs {r_results['logLik']:.2f}"

    def test_sar_spillover_effects(self):
        """Test spillover effects calculation."""
        # Use small dataset for quick test
        data, W_array = load_validation_data("small")
        W = SpatialWeights(W_array, normalized=True)

        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        result = model.fit(effects="fixed")

        # Check spillover effects
        assert hasattr(result, "spillover_effects")

        for var in ["x1", "x2"]:
            effects = result.spillover_effects[var]

            # With positive rho, indirect effects should be positive
            if result.params["rho"] > 0:
                assert effects["indirect"] > 0

            # Total = direct + indirect
            assert np.abs(effects["total"] - (effects["direct"] + effects["indirect"])) < 1e-10


class TestSEMValidation:
    """Validate SEM model against R splm."""

    @pytest.mark.parametrize("test_case", ["small", "medium", "large"])
    def test_sem_fe_ml_vs_splm(self, test_case):
        """Test SEM-FE ML estimation against splm."""
        # Load SEM data
        data_file = DATA_DIR / f"sem_fe_{test_case}.csv"
        if not data_file.exists():
            pytest.skip(f"SEM validation data not found")

        data = pd.read_csv(data_file)

        # Load weight matrix
        W_file = DATA_DIR / f"W_sem_{test_case}.csv"
        W_array = pd.read_csv(W_file, index_col=0).values
        W = SpatialWeights(W_array, normalized=True)

        # Load R results
        r_results = load_r_results("sem_fe", test_case)

        # Estimate with PanelBox
        model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        result = model.fit(effects="fixed", method="ml", verbose=False)

        # Compare spatial error parameter
        assert (
            np.abs(result.params["lambda"] - r_results["lambda"]) < 0.05
        ), f"Lambda mismatch: {result.params['lambda']:.6f} vs {r_results['lambda']:.6f}"

        # Compare coefficients
        for i, param in enumerate(["x1", "x2"]):
            pb_coef = result.params[param]
            r_coef = r_results["coefficients"][i]
            assert (
                np.abs(pb_coef - r_coef) < 0.05
            ), f"{param} mismatch: {pb_coef:.6f} vs {r_coef:.6f}"

    @pytest.mark.parametrize("test_case", ["small", "medium"])
    def test_sem_fe_gmm_vs_splm(self, test_case):
        """Test SEM-FE GMM estimation against splm."""
        # Load SEM data
        data_file = DATA_DIR / f"sem_fe_{test_case}.csv"
        if not data_file.exists():
            pytest.skip(f"SEM validation data not found")

        data = pd.read_csv(data_file)

        # Load weight matrix
        W_file = DATA_DIR / f"W_sem_{test_case}.csv"
        W_array = pd.read_csv(W_file, index_col=0).values
        W = SpatialWeights(W_array, normalized=True)

        # Load R GMM results
        gmm_results_file = DATA_DIR / f"sem_fe_{test_case}_results_gmm.json"
        if gmm_results_file.exists():
            with open(gmm_results_file) as f:
                r_results = json.load(f)
        else:
            pytest.skip("GMM validation results not found")

        # Estimate with PanelBox
        model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        result = model.fit(effects="fixed", method="gmm", n_lags=2, verbose=False)

        # Compare lambda (GMM might have larger tolerance)
        assert (
            np.abs(result.params["lambda"] - r_results["lambda"]) < 0.1
        ), f"Lambda mismatch: {result.params['lambda']:.6f} vs {r_results['lambda']:.6f}"


class TestModelSelection:
    """Test model selection and comparison."""

    def test_aic_bic_comparison(self):
        """Test that AIC/BIC can be used for model selection."""
        data, W_array = load_validation_data("small")
        W = SpatialWeights(W_array, normalized=True)

        # Fit SAR
        sar_model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )
        sar_result = sar_model.fit(effects="fixed")

        # Fit SEM
        sem_model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )
        sem_result = sem_model.fit(effects="fixed", method="gmm")

        # Both should have AIC/BIC
        assert hasattr(sar_result, "aic")
        assert hasattr(sar_result, "bic")
        assert hasattr(sem_result, "aic")
        assert hasattr(sem_result, "bic")

        # Can compare models
        print(f"SAR AIC: {sar_result.aic:.2f}, BIC: {sar_result.bic:.2f}")
        print(f"SEM AIC: {sem_result.aic:.2f}, BIC: {sem_result.bic:.2f}")


@pytest.mark.slow
class TestLargeScalePerformance:
    """Test performance on larger datasets."""

    def test_large_n_performance(self):
        """Test with large N (many entities)."""
        N = 200
        T = 10

        # Generate simple data
        np.random.seed(42)

        # Create banded weight matrix (sparse structure)
        W_raw = np.zeros((N, N))
        for i in range(N):
            if i > 0:
                W_raw[i, i - 1] = 1
            if i < N - 1:
                W_raw[i, i + 1] = 1

        # Row-normalize
        row_sums = W_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W_array = W_raw / row_sums

        W = SpatialWeights(W_array, normalized=True)

        # Generate panel data
        entity = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)
        y = np.random.randn(N * T)
        x1 = np.random.randn(N * T)
        x2 = np.random.randn(N * T)

        data = pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})

        # Test SAR estimation
        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        # Should complete without error
        result = model.fit(effects="fixed", rho_grid_size=10)
        assert result is not None
        assert "rho" in result.params
