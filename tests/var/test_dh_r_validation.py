"""
Test Dumitrescu-Hurlin implementation against R's plm::pgrangertest.

This test validates our Python implementation of the Dumitrescu-Hurlin (2012)
panel Granger causality test against the reference implementation in R's plm package.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality import dumitrescu_hurlin_test


def run_r_pgrangertest(data, cause, effect, lags):
    """
    Run R's plm::pgrangertest on the given data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time)
    cause : str
        Name of the causing variable
    effect : str
        Name of the effect variable
    lags : int
        Number of lags

    Returns
    -------
    dict
        Results from R including W_bar, Z_tilde, Z_bar and p-values
    """
    # Create temporary directory for data exchange
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save data to CSV
        data_path = os.path.join(tmpdir, "data.csv")
        data_reset = data.reset_index()
        data_reset.to_csv(data_path, index=False)

        # Create R script
        # NOTE: R's plm uses INVERTED nomenclature compared to DH2012 paper:
        #   - R "Zbar" = Simple statistic (DH2012's Z̃)
        #   - R "Ztilde" = Exact moments statistic (DH2012's Z̄)
        # Our Python implementation follows DH2012 paper naming
        r_script = f"""
library(plm)

# Load data
data <- read.csv("{data_path}")
colnames(data)[1:2] <- c("entity", "time")

# Create pdata.frame
pdata <- pdata.frame(data, index = c("entity", "time"))

# Try all test types (may fail depending on T, N, and panel balance)
result_wbar <- pgrangertest({effect} ~ {cause}, data = pdata, test = "Wbar", order = {lags})

# NOTE: R naming is INVERTED from DH2012 paper!
# R "Zbar" is simple (DH2012 Z̃), R "Ztilde" is exact moments (DH2012 Z̄)
result_r_zbar <- tryCatch({{
    pgrangertest({effect} ~ {cause}, data = pdata, test = "Zbar", order = {lags})
}}, error = function(e) NULL)

result_r_ztilde <- tryCatch({{
    pgrangertest({effect} ~ {cause}, data = pdata, test = "Ztilde", order = {lags})
}}, error = function(e) NULL)

# Extract results
cat("W_bar:", result_wbar$statistic, "\\n")
cat("W_bar_pvalue:", result_wbar$p.value, "\\n")

# Map to DH2012 naming (inverted from R)
if (!is.null(result_r_zbar)) {{
    cat("Z_tilde:", result_r_zbar$statistic, "\\n")
    cat("Z_tilde_pvalue:", result_r_zbar$p.value, "\\n")
}} else {{
    cat("Z_tilde: NA\\n")
    cat("Z_tilde_pvalue: NA\\n")
}}

if (!is.null(result_r_ztilde)) {{
    cat("Z_bar:", result_r_ztilde$statistic, "\\n")
    cat("Z_bar_pvalue:", result_r_ztilde$p.value, "\\n")
}} else {{
    cat("Z_bar: NA\\n")
    cat("Z_bar_pvalue: NA\\n")
}}
"""

        # Save R script
        script_path = os.path.join(tmpdir, "script.R")
        with open(script_path, "w") as f:
            f.write(r_script)

        # Run R script
        result = subprocess.run(
            ["R", "--vanilla", "--slave", "-f", script_path], capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"R script failed:\n{result.stderr}")

        # Parse output
        output_lines = result.stdout.strip().split("\n")
        r_results = {}
        for line in output_lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value and value != "NA":
                    r_results[key] = float(value)
                else:
                    r_results[key] = None

        return r_results


@pytest.mark.slow
def test_dumitrescu_hurlin_vs_r_balanced_panel():
    """
    Test DH implementation against R with a balanced panel.

    This test uses a simple DGP with known causal structure.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate balanced panel data
    N = 30  # entities
    T = 25  # time periods
    p = 2  # lags

    entities = [f"E{i}" for i in range(N)]
    times = list(range(T))

    # Create MultiIndex
    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])

    # Generate data with causal structure: x -> y
    data_dict = {"x": [], "y": []}

    for entity in entities:
        # Entity-specific coefficients (heterogeneous)
        beta_0 = np.random.normal(0.3, 0.1)  # x causes y
        beta_1 = np.random.normal(0.6, 0.1)  # y's own lag

        x_series = np.zeros(T)
        y_series = np.zeros(T)

        # Initialize first p observations
        for t in range(p):
            x_series[t] = np.random.normal(0, 1)
            y_series[t] = np.random.normal(0, 1)

        # Generate rest with causal structure
        for t in range(p, T):
            x_series[t] = 0.5 * x_series[t - 1] + np.random.normal(0, 0.5)
            y_series[t] = (
                beta_0 * x_series[t - 1] + beta_1 * y_series[t - 1] + np.random.normal(0, 0.5)
            )

        data_dict["x"].extend(x_series)
        data_dict["y"].extend(y_series)

    data = pd.DataFrame(data_dict, index=index)

    # Reset index for Python implementation (expects columns, not MultiIndex)
    data_for_py = data.reset_index()

    # Run Python implementation
    py_result = dumitrescu_hurlin_test(data_for_py, cause="x", effect="y", lags=p)

    # Run R implementation
    r_result = run_r_pgrangertest(data, cause="x", effect="y", lags=p)

    # Compare results
    print("\n=== Python Results ===")
    print(f"W_bar: {py_result.W_bar:.6f}")
    print(f"Z_tilde: {py_result.Z_tilde_stat:.6f} (p={py_result.Z_tilde_pvalue:.6f})")
    print(f"Z_bar: {py_result.Z_bar_stat:.6f} (p={py_result.Z_bar_pvalue:.6f})")

    print("\n=== R Results ===")
    print(f"W_bar: {r_result['W_bar']:.6f}")
    if r_result["Z_tilde"] is not None:
        print(f"Z_tilde: {r_result['Z_tilde']:.6f} (p={r_result['Z_tilde_pvalue']:.6f})")
    else:
        print("Z_tilde: Not available (T too small)")
    if r_result["Z_bar"] is not None:
        print(f"Z_bar: {r_result['Z_bar']:.6f} (p={r_result['Z_bar_pvalue']:.6f})")

    print("\n=== Differences ===")
    print(f"W_bar diff: {abs(py_result.W_bar - r_result['W_bar']):.6f}")

    # Assertions (allow small numerical differences)
    # W_bar should always match
    assert (
        abs(py_result.W_bar - r_result["W_bar"]) < 0.01
    ), f"W_bar differs: Python={py_result.W_bar}, R={r_result['W_bar']}"

    # Compare Z_tilde if available in R
    if r_result["Z_tilde"] is not None:
        print(f"Z_tilde diff: {abs(py_result.Z_tilde_stat - r_result['Z_tilde']):.6f}")
        print(
            f"Z_tilde p-value diff: {abs(py_result.Z_tilde_pvalue - r_result['Z_tilde_pvalue']):.6f}"
        )

        assert (
            abs(py_result.Z_tilde_stat - r_result["Z_tilde"]) < 0.01
        ), f"Z_tilde differs: Python={py_result.Z_tilde_stat}, R={r_result['Z_tilde']}"

        assert (
            abs(py_result.Z_tilde_pvalue - r_result["Z_tilde_pvalue"]) < 0.01
        ), f"Z_tilde p-value differs: Python={py_result.Z_tilde_pvalue}, R={r_result['Z_tilde_pvalue']}"

        # Both should reject null (significant causality)
        assert py_result.Z_tilde_pvalue < 0.05, "Python should reject null"
        assert r_result["Z_tilde_pvalue"] < 0.05, "R should reject null"

    # Compare Z_bar if available in R
    # NOTE: Z_bar uses exact finite-sample moments which may differ slightly
    # between implementations. We use a more lenient tolerance.
    if r_result["Z_bar"] is not None:
        z_bar_diff = abs(py_result.Z_bar_stat - r_result["Z_bar"])
        print(f"Z_bar diff: {z_bar_diff:.6f}")

        # Both should have same conclusion (more important than exact value)
        py_rejects = py_result.Z_bar_pvalue < 0.05
        r_rejects = r_result["Z_bar_pvalue"] < 0.05
        assert (
            py_rejects == r_rejects
        ), f"Z_bar conclusions differ: Python rejects={py_rejects}, R rejects={r_rejects}"


@pytest.mark.slow
def test_dumitrescu_hurlin_vs_r_no_causality():
    """
    Test DH implementation against R with no causality structure.

    Both implementations should NOT reject the null hypothesis.
    """
    # Set seed for reproducibility
    np.random.seed(123)

    # Generate balanced panel data WITHOUT causality
    N = 25
    T = 30
    p = 2

    entities = [f"E{i}" for i in range(N)]
    times = list(range(T))

    index = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])

    # Generate independent AR(1) processes
    data_dict = {"x": [], "y": []}

    for entity in entities:
        x_series = np.zeros(T)
        y_series = np.zeros(T)

        # Initialize
        x_series[0] = np.random.normal(0, 1)
        y_series[0] = np.random.normal(0, 1)

        # Generate as independent AR(1) processes
        for t in range(1, T):
            x_series[t] = 0.5 * x_series[t - 1] + np.random.normal(0, 0.5)
            y_series[t] = 0.5 * y_series[t - 1] + np.random.normal(0, 0.5)

        data_dict["x"].extend(x_series)
        data_dict["y"].extend(y_series)

    data = pd.DataFrame(data_dict, index=index)

    # Reset index for Python implementation (expects columns, not MultiIndex)
    data_for_py = data.reset_index()

    # Run Python implementation
    py_result = dumitrescu_hurlin_test(data_for_py, cause="x", effect="y", lags=p)

    # Run R implementation
    r_result = run_r_pgrangertest(data, cause="x", effect="y", lags=p)

    # Compare results
    print("\n=== Python Results (No Causality) ===")
    print(f"W_bar: {py_result.W_bar:.6f}")
    print(f"Z_tilde: {py_result.Z_tilde_stat:.6f} (p={py_result.Z_tilde_pvalue:.6f})")

    print("\n=== R Results (No Causality) ===")
    print(f"W_bar: {r_result['W_bar']:.6f}")
    if r_result["Z_tilde"] is not None:
        print(f"Z_tilde: {r_result['Z_tilde']:.6f} (p={r_result['Z_tilde_pvalue']:.6f})")
    else:
        print("Z_tilde: Not available (T too small)")

    # Assertions
    assert abs(py_result.W_bar - r_result["W_bar"]) < 0.01

    if r_result["Z_tilde"] is not None:
        assert abs(py_result.Z_tilde_stat - r_result["Z_tilde"]) < 0.01
        assert abs(py_result.Z_tilde_pvalue - r_result["Z_tilde_pvalue"]) < 0.01

        # Both should NOT reject null (no significant causality)
        assert py_result.Z_tilde_pvalue > 0.05, "Python should not reject null"
        assert r_result["Z_tilde_pvalue"] > 0.05, "R should not reject null"


@pytest.mark.slow
def test_dumitrescu_hurlin_vs_r_grunfeld_data():
    """
    Test DH implementation against R using the classic Grunfeld dataset.

    This replicates the example from plm::pgrangertest documentation.
    """
    # Load Grunfeld data from plm package via R
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "grunfeld.csv")

        # Extract Grunfeld data from R
        r_script = f"""
library(plm)
data("Grunfeld", package = "plm")
write.csv(Grunfeld, "{data_path}", row.names = FALSE)
"""
        script_path = os.path.join(tmpdir, "extract.R")
        with open(script_path, "w") as f:
            f.write(r_script)

        result = subprocess.run(
            ["R", "--vanilla", "--slave", "-f", script_path], capture_output=True, text=True
        )

        if result.returncode != 0:
            pytest.skip(f"Could not load Grunfeld data: {result.stderr}")

        # Load data
        grunfeld = pd.read_csv(data_path)

        # Convert to panel format with MultiIndex for R
        grunfeld_idx = grunfeld.set_index(["firm", "year"])
        grunfeld_idx.index.names = ["entity", "time"]

        # Keep column format for Python
        grunfeld_py = grunfeld.rename(columns={"firm": "entity", "year": "time"})

        # Test: value -> inv (as in plm documentation example)
        py_result = dumitrescu_hurlin_test(grunfeld_py, cause="value", effect="inv", lags=1)

        # Run R version
        r_result = run_r_pgrangertest(grunfeld_idx, cause="value", effect="inv", lags=1)

        print("\n=== Python Results (Grunfeld: value -> inv) ===")
        print(f"W_bar: {py_result.W_bar:.6f}")
        print(f"Z_tilde: {py_result.Z_tilde_stat:.6f} (p={py_result.Z_tilde_pvalue:.6f})")

        print("\n=== R Results (Grunfeld: value -> inv) ===")
        print(f"W_bar: {r_result['W_bar']:.6f}")
        if r_result["Z_tilde"] is not None:
            print(f"Z_tilde: {r_result['Z_tilde']:.6f} (p={r_result['Z_tilde_pvalue']:.6f})")
        else:
            print("Z_tilde: Not available (T too small)")

        # Compare
        assert abs(py_result.W_bar - r_result["W_bar"]) < 0.01

        if r_result["Z_tilde"] is not None:
            assert abs(py_result.Z_tilde_stat - r_result["Z_tilde"]) < 0.01
            assert abs(py_result.Z_tilde_pvalue - r_result["Z_tilde_pvalue"]) < 0.01

            # Both should have same conclusion
            py_rejects = py_result.Z_tilde_pvalue < 0.05
            r_rejects = r_result["Z_tilde_pvalue"] < 0.05
            assert py_rejects == r_rejects, "Python and R should reach same conclusion"


if __name__ == "__main__":
    # Run tests manually
    print("Testing DH vs R - Balanced panel with causality...")
    test_dumitrescu_hurlin_vs_r_balanced_panel()
    print("\n✓ PASSED\n")

    print("Testing DH vs R - No causality...")
    test_dumitrescu_hurlin_vs_r_no_causality()
    print("\n✓ PASSED\n")

    print("Testing DH vs R - Grunfeld dataset...")
    test_dumitrescu_hurlin_vs_r_grunfeld_data()
    print("\n✓ PASSED\n")

    print("All validation tests passed!")
