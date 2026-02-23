"""
Pytest fixtures for R validation tests.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

R_BENCHMARKS_DIR = Path(__file__).parent / "r_benchmarks"


def run_r_script(script_name: str) -> bool:
    """Run an R script and return True if successful."""
    script_path = R_BENCHMARKS_DIR / script_name
    if not script_path.exists():
        return False

    try:
        result = subprocess.run(
            ["Rscript", str(script_path)],
            cwd=str(R_BENCHMARKS_DIR),
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def load_r_results(csv_name: str) -> pd.DataFrame:
    """Load R results from CSV."""
    csv_path = R_BENCHMARKS_DIR / csv_name
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def r_available():
    """Check if R is available."""
    try:
        result = subprocess.run(["Rscript", "--version"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


@pytest.fixture(scope="module")
def cue_gmm_r_results(r_available):
    """Load CUE-GMM R benchmark results."""
    if not r_available:
        pytest.skip("R not available")

    # Run R benchmark if results don't exist
    results_path = R_BENCHMARKS_DIR / "cue_gmm_results.csv"
    if not results_path.exists():
        success = run_r_script("cue_gmm_benchmark.R")
        if not success:
            pytest.skip("Could not run R benchmark")

    results = load_r_results("cue_gmm_results.csv")
    data = load_r_results("cue_gmm_data.csv")

    if results is None or data is None:
        pytest.skip("R results not available")

    return {"results": results, "data": data}


@pytest.fixture(scope="module")
def heckman_r_results(r_available):
    """Load Heckman R benchmark results."""
    if not r_available:
        pytest.skip("R not available")

    results_path = R_BENCHMARKS_DIR / "heckman_2step_results.csv"
    if not results_path.exists():
        success = run_r_script("heckman_benchmark.R")
        if not success:
            pytest.skip("Could not run R benchmark")

    results_2step = load_r_results("heckman_2step_results.csv")
    results_mle = load_r_results("heckman_mle_results.csv")
    data = load_r_results("heckman_data.csv")

    if results_2step is None or data is None:
        pytest.skip("R results not available")

    return {
        "results_2step": results_2step,
        "results_mle": results_mle,
        "data": data
    }


@pytest.fixture(scope="module")
def multinomial_r_results(r_available):
    """Load Multinomial Logit R benchmark results."""
    if not r_available:
        pytest.skip("R not available")

    results_path = R_BENCHMARKS_DIR / "multinomial_results.csv"
    if not results_path.exists():
        success = run_r_script("multinomial_benchmark.R")
        if not success:
            pytest.skip("Could not run R benchmark")

    results = load_r_results("multinomial_results.csv")
    data = load_r_results("multinomial_data.csv")

    if results is None or data is None:
        pytest.skip("R results not available")

    return {"results": results, "data": data}


@pytest.fixture(scope="module")
def conditional_logit_r_results(r_available):
    """Load Conditional Logit R benchmark results."""
    if not r_available:
        pytest.skip("R not available")

    results_path = R_BENCHMARKS_DIR / "conditional_logit_results.csv"
    if not results_path.exists():
        success = run_r_script("conditional_logit_benchmark.R")
        if not success:
            pytest.skip("Could not run R benchmark")

    results = load_r_results("conditional_logit_results.csv")
    data = load_r_results("conditional_logit_data.csv")

    if results is None or data is None:
        pytest.skip("R results not available")

    return {"results": results, "data": data}
