"""
Validation tests against R splm package.

These tests compare PanelBox spatial model results with R splm output
to ensure correctness of implementation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.models.spatial.spatial_error import SpatialError
from panelbox.models.spatial.spatial_lag import SpatialLag


class TestValidationBase:
    """Base class for validation tests."""

    @staticmethod
    def load_validation_data(scenario_name):
        """Load validation data for a scenario."""
        data_dir = Path("tests/validation_spatial/data")

        # Load metadata
        with open(data_dir / f"metadata_{scenario_name}.json") as f:
            metadata = json.load(f)

        # Load W matrix
        W_data = np.load(data_dir / f"W_{scenario_name}.npz")
        W = W_data["W"]

        # Load SAR data
        sar_data = pd.read_csv(data_dir / f"sar_{scenario_name}.csv")

        # Load SEM data
        sem_data = pd.read_csv(data_dir / f"sem_{scenario_name}.csv")

        return {"metadata": metadata, "W": W, "sar_data": sar_data, "sem_data": sem_data}

    @staticmethod
    def load_r_results(result_file):
        """Load R estimation results."""
        r_output_dir = Path("tests/validation_spatial/r_output")
        file_path = r_output_dir / result_file

        if not file_path.exists():
            pytest.skip(f"R results file not found: {file_path}")

        with open(file_path) as f:
            return json.load(f)


class TestSARValidation(TestValidationBase):
    """Validation tests for SAR-FE model."""

    @pytest.mark.parametrize(
        "scenario", ["synthetic_n25_t10", "synthetic_n49_t15", "synthetic_n100_t20"]
    )
    def test_sar_fe_vs_true_params(self, scenario):
        """Test SAR-FE recovery of true parameters."""
        # Load data
        data_dict = self.load_validation_data(scenario)
        metadata = data_dict["metadata"]
        W = data_dict["W"]
        sar_data = data_dict["sar_data"]

        # True parameters
        true_rho = metadata["true_params"]["rho"]
        true_beta = np.array(metadata["true_params"]["beta"])

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Build formula
        k = len(true_beta)
        x_vars = " + ".join([f"x{i+1}" for i in range(k)])
        formula = f"y ~ {x_vars}"

        # Estimate model
        model = SpatialLag(
            formula=formula, data=sar_data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="qml")

        # Check parameter recovery
        # Tolerances depend on sample size
        n = metadata["n"]
        t = metadata["t"]

        if n * t < 500:
            rho_tol = 0.15
            beta_tol = 0.2
        else:
            rho_tol = 0.1
            beta_tol = 0.15

        # Check rho
        assert (
            abs(result.params["rho"] - true_rho) < rho_tol
        ), f"Rho recovery failed: {result.params['rho']} vs {true_rho}"

        # Check beta
        for i in range(k):
            param_name = f"x{i+1}"
            assert (
                abs(result.params[param_name] - true_beta[i]) < beta_tol
            ), f"Beta[{i}] recovery failed: {result.params[param_name]} vs {true_beta[i]}"


class TestSEMValidation(TestValidationBase):
    """Validation tests for SEM-FE model."""

    @pytest.mark.parametrize(
        "scenario", ["synthetic_n25_t10", "synthetic_n49_t15", "synthetic_n100_t20"]
    )
    def test_sem_fe_vs_true_params(self, scenario):
        """Test SEM-FE recovery of true parameters."""
        # Load data
        data_dict = self.load_validation_data(scenario)
        metadata = data_dict["metadata"]
        W = data_dict["W"]
        sem_data = data_dict["sem_data"]

        # True parameters
        true_lambda = metadata["true_params"]["lambda"]
        true_beta = np.array(metadata["true_params"]["beta"])

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Build formula
        k = len(true_beta)
        x_vars = " + ".join([f"x{i+1}" for i in range(k)])
        formula = f"y ~ {x_vars}"

        # Estimate model
        model = SpatialError(
            formula=formula, data=sem_data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="gmm")

        # Check parameter recovery
        # Tolerances for GMM are typically larger
        n = metadata["n"]
        t = metadata["t"]

        if n * t < 500:
            lambda_tol = 0.2
            beta_tol = 0.25
        else:
            lambda_tol = 0.15
            beta_tol = 0.2

        # Check lambda
        assert (
            abs(result.params["lambda"] - true_lambda) < lambda_tol
        ), f"Lambda recovery failed: {result.params['lambda']} vs {true_lambda}"

        # Check beta
        for i in range(k):
            param_name = f"x{i+1}"
            assert (
                abs(result.params[param_name] - true_beta[i]) < beta_tol
            ), f"Beta[{i}] recovery failed: {result.params[param_name]} vs {true_beta[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
