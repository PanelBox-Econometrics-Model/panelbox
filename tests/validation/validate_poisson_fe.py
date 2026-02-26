#!/usr/bin/env python
"""
Validate Poisson Fixed Effects against R's pglm package.
"""

import os
import subprocess

# Add panelbox to path
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from panelbox.models.count.poisson import PoissonFixedEffects


def generate_test_data():
    """Generate panel data for testing."""
    np.random.seed(42)

    # Panel structure
    n_entities = 30
    n_periods = 8
    n_obs = n_entities * n_periods

    # Generate data
    entity_id = np.repeat(np.arange(n_entities), n_periods)
    time_id = np.tile(np.arange(n_periods), n_entities)

    # Covariates (no intercept for FE)
    X = np.random.randn(n_obs, 2)

    # True parameters and fixed effects
    beta_true = np.array([0.5, -0.3])
    fe = np.random.randn(n_entities) * 0.5

    # Generate count outcomes
    eta = X @ beta_true + np.repeat(fe, n_periods)
    lambda_true = np.exp(eta)
    y = np.random.poisson(lambda_true)

    return y, X, entity_id, time_id


def validate_with_r():
    """Validate against R's pglm."""
    print("Generating test data...")
    y, X, entity_id, time_id = generate_test_data()

    # Create DataFrame for R
    df = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1], "entity": entity_id, "time": time_id})

    # Save data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        data_file = f.name

    try:
        print("\nFitting Poisson FE with R's pglm...")
        r_code = f"""
        library(pglm)
        data <- read.csv('{data_file}')

        # Convert to panel data
        data$entity <- factor(data$entity)
        data$time <- factor(data$time)

        # Fit fixed effects Poisson
        model <- pglm(y ~ x1 + x2,
                     data = data,
                     index = c("entity", "time"),
                     family = poisson,
                     model = "within")

        # Extract results
        coef <- coefficients(model)
        se <- sqrt(diag(vcov(model)))

        cat("COEF:", paste(coef, collapse=","), "\\n")
        cat("SE:", paste(se, collapse=","), "\\n")

        # Print summary
        print(summary(model))
        """

        result = subprocess.run(
            ["Rscript", "-e", r_code], capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            print(f"R Error: {result.stderr}")
            return None, None

        # Parse R output
        r_coef = None
        r_se = None

        for line in result.stdout.split("\n"):
            if line.startswith("COEF:"):
                r_coef = np.array([float(x) for x in line.split(":")[1].split(",")])
            elif line.startswith("SE:"):
                r_se = np.array([float(x) for x in line.split(":")[1].split(",")])

        print("\nR Results:")
        print(f"Coefficients: {r_coef}")
        print(f"Standard Errors: {r_se}")

        print("\nFitting Poisson FE with panelbox...")
        # Create model properly
        model = PoissonFixedEffects(y, X, entity_id, time_id)
        pb_result = model.fit()

        print("\nPanelbox Results:")
        print(f"Coefficients: {pb_result.params}")
        print(f"Standard Errors: {pb_result.se}")

        # Compare
        if r_coef is not None and pb_result.params is not None:
            print("\nComparison:")
            print(f"Coefficient difference: {np.abs(r_coef - pb_result.params)}")

            max_diff = np.max(np.abs(r_coef - pb_result.params))
            if max_diff < 1e-3:
                print(f"✓ VALIDATION PASSED: Maximum difference = {max_diff:.6f}")
                return True, max_diff
            else:
                print(f"✗ VALIDATION FAILED: Maximum difference = {max_diff:.6f}")
                return False, max_diff
        else:
            print("Could not compare results")
            return None, None

    finally:
        os.unlink(data_file)


if __name__ == "__main__":
    print("=" * 60)
    print("POISSON FIXED EFFECTS VALIDATION")
    print("=" * 60)

    success, diff = validate_with_r()

    if success is True:
        print("\n✓ Validation successful!")
        sys.exit(0)
    elif success is False:
        print("\n✗ Validation failed!")
        sys.exit(1)
    else:
        print("\n⚠ Validation could not be completed")
        sys.exit(2)
