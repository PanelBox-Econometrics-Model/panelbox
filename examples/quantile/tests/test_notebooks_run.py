"""
Tests that tutorial notebooks execute without errors.

Uses nbconvert to execute each notebook and checks for exceptions.
Skips notebooks that don't exist yet.
"""

import os
import subprocess

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

NOTEBOOKS = [
    "01_quantile_regression_fundamentals.ipynb",
    "02_multiple_quantiles_process.ipynb",
    "03_fixed_effects_canay.ipynb",
    "04_fixed_effects_penalty.ipynb",
    "05_location_scale_models.ipynb",
    "06_advanced_diagnostics.ipynb",
    "07_bootstrap_inference.ipynb",
    "08_monotonicity_non_crossing.ipynb",
    "09_quantile_treatment_effects.ipynb",
    "10_dynamic_quantile_models.ipynb",
]


@pytest.mark.parametrize("notebook", NOTEBOOKS)
@pytest.mark.slow
def test_notebook_runs(notebook):
    """Test that a notebook executes without errors."""
    path = os.path.join(NOTEBOOKS_DIR, notebook)

    if not os.path.exists(path):
        pytest.skip(f"Notebook {notebook} does not exist yet")

    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=300",
            "--output",
            "/dev/null",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Notebook {notebook} failed:\nSTDOUT: {result.stdout[:500]}\nSTDERR: {result.stderr[:500]}"
    )
