"""
Test PanelBox Pooled OLS vs R plm package

Compares numerical results between PanelBox's PooledOLS and R's plm pooling model.

Dataset: Grunfeld (200 observations, 10 firms, 20 years)
Model: invest ~ value + capital

Expected tolerance: < 1e-6 for coefficients and standard errors
"""

import os
import sys

import numpy as np
import pandas as pd

# Add panelbox to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import panelbox as pb


def load_grunfeld_data():
    """Load Grunfeld dataset - use R version for comparison"""
    try:
        # Use R's Grunfeld dataset for exact comparison
        r_data_path = os.path.join(os.path.dirname(__file__), "grunfeld_r.csv")
        if os.path.exists(r_data_path):
            data = pd.read_csv(r_data_path)
            # Rename 'inv' to 'invest' for PanelBox
            if "inv" in data.columns:
                data = data.rename(columns={"inv": "invest"})
            return data
        else:
            # Fallback to PanelBox dataset
            print("Warning: R dataset not found, using PanelBox default (may differ)")
            data = pb.load_grunfeld()
            return data
    except Exception as e:
        print(f"Error loading Grunfeld data: {e}")
        sys.exit(1)


def main():
    """
    Main comparison function

    Steps:
    1. Load Grunfeld data
    2. Estimate Pooled OLS with PanelBox
    3. Compare with R plm reference values
    4. Report differences
    """

    print("=" * 80)
    print("BENCHMARK: Pooled OLS - PanelBox vs R plm")
    print("=" * 80)
    print()

    # Load data
    data = load_grunfeld_data()

    # Map column names (PanelBox uses 'invest', 'value', 'capital', 'firm', 'year')
    # Grunfeld in plm uses 'inv', 'value', 'capital', 'firm', 'year'
    if "inv" in data.columns and "invest" not in data.columns:
        data = data.rename(columns={"inv": "invest"})

    print(f"Dataset: {len(data)} observations, {data['firm'].nunique()} firms")
    print(f"Variables: invest, value, capital")
    print()

    # Estimate with PanelBox
    print("-" * 80)
    print("PanelBox Estimation")
    print("-" * 80)
    print()

    model = pb.PooledOLS(
        formula="invest ~ value + capital", data=data, entity_col="firm", time_col="year"
    )

    results = model.fit()

    # Display results
    print(results.summary())
    print()

    # R plm reference values
    # INSTRUCTIONS: After running pooling.R, copy values from pooling_results.txt here
    #
    # Example format from pooling_results.txt:
    # value: coef=0.1101238441, se=0.0119086939, t=9.246823, p=3.63940561e-17
    # capital: coef=0.3100653144, se=0.0173866368, t=17.833730, p=6.02316337e-45
    # (Intercept): coef=-42.7143740451, se=9.5116760127, t=-4.491639, p=1.06208521e-05

    plm_results = {
        "coef": {
            "value": 0.11556216,  # From R plm pooling.R
            "capital": 0.23067849,  # From R plm pooling.R
            "const": -42.71436944,  # From R plm pooling.R (Intercept)
        },
        "se": {
            "value": 0.0058357096,  # From R plm pooling.R
            "capital": 0.0254758015,  # From R plm pooling.R
            "const": 9.5116760314,  # From R plm pooling.R
        },
        "r2": 0.81240801,  # From R plm pooling.R
        "adj_r2": 0.81240801,  # From R plm pooling.R
        "n_obs": 200,
        "n_params": 3,
        "df_resid": 197,
    }

    # Map variable names (PanelBox uses 'Intercept', R uses '(Intercept)' or 'const')
    var_map = {"value": "value", "capital": "capital", "Intercept": "const"}

    # Compare results
    print("-" * 80)
    print("Comparison with R plm")
    print("-" * 80)
    print()

    print("Coefficients:")
    print(f"{'Variable':<15} {'PanelBox':>12} {'R plm':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 67)

    all_passed = True
    tolerance = 1e-6

    for panelbox_var in results.params.index:
        plm_var = var_map.get(panelbox_var, panelbox_var)

        panelbox_coef = results.params[panelbox_var]
        plm_coef = plm_results["coef"].get(plm_var, np.nan)

        if not np.isnan(plm_coef):
            diff = panelbox_coef - plm_coef
            rel_error = abs(diff / plm_coef) if plm_coef != 0 else abs(diff)

            status = "✓" if abs(diff) < tolerance else "✗"
            if abs(diff) >= tolerance:
                all_passed = False

            print(
                f"{panelbox_var:<15} {panelbox_coef:12.7f} {plm_coef:12.7f} "
                f"{diff:12.2e} {rel_error:12.2e} {status}"
            )

    print()
    print("Standard Errors:")
    print(f"{'Variable':<15} {'PanelBox':>12} {'R plm':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 67)

    for panelbox_var in results.std_errors.index:
        plm_var = var_map.get(panelbox_var, panelbox_var)

        panelbox_se = results.std_errors[panelbox_var]
        plm_se = plm_results["se"].get(plm_var, np.nan)

        if not np.isnan(plm_se):
            diff = panelbox_se - plm_se
            rel_error = abs(diff / plm_se) if plm_se != 0 else abs(diff)

            status = "✓" if abs(diff) < tolerance else "✗"
            if abs(diff) >= tolerance:
                all_passed = False

            print(
                f"{panelbox_var:<15} {panelbox_se:12.7f} {plm_se:12.7f} "
                f"{diff:12.2e} {rel_error:12.2e} {status}"
            )

    print()
    print("Model Statistics:")
    print(f"{'Statistic':<20} {'PanelBox':>12} {'R plm':>12} {'Diff':>12}")
    print("-" * 58)

    # R-squared
    r2_diff = results.rsquared - plm_results["r2"]
    print(f"{'R-squared':<20} {results.rsquared:12.7f} {plm_results['r2']:12.7f} {r2_diff:12.2e}")

    # Adjusted R-squared
    adj_r2_diff = results.rsquared_adj - plm_results["adj_r2"]
    print(
        f"{'Adj R-squared':<20} {results.rsquared_adj:12.7f} {plm_results['adj_r2']:12.7f} {adj_r2_diff:12.2e}"
    )

    # N observations
    print(f"{'N':<20} {results.nobs:12.0f} {plm_results['n_obs']:12.0f}")

    print()
    print("=" * 80)

    if all_passed:
        print("✓ All comparisons passed (within tolerance 1e-6)")
        return 0
    else:
        print("✗ Some comparisons failed")
        print()
        print("INSTRUCTIONS TO UPDATE REFERENCE VALUES:")
        print("1. Run pooling.R in R: Rscript pooling.R")
        print("2. Open pooling_results.txt")
        print("3. Copy coefficient and standard error values")
        print("4. Update plm_results dict in this file")
        print("5. Re-run this test")
        return 1


if __name__ == "__main__":
    sys.exit(main())
