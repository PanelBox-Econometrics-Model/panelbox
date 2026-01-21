"""
Generate test data and run PanelBox validation tests.

This script creates controlled panel datasets with known properties
and runs all 7 validation tests, saving results for comparison with R.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# Add panelbox to path
import sys
sys.path.insert(0, '/home/guhaase/projetos/panelbox')

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.models.static.pooled_ols import PooledOLS


def generate_panel_data_ar1(n_entities=50, n_periods=10, rho=0.5, seed=42):
    """
    Generate panel data with AR(1) serial correlation.

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_periods : int
        Number of time periods
    rho : float
        AR(1) coefficient
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Panel data with AR(1) errors
    """
    np.random.seed(seed)

    # True parameters
    beta_0 = 10.0
    beta_1 = 2.0
    beta_2 = -1.5

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    # Regressors
    x1 = np.random.randn(n_entities * n_periods) * 5 + 50
    x2 = np.random.randn(n_entities * n_periods) * 3 + 30

    # Entity fixed effects
    entity_effects = np.repeat(np.random.randn(n_entities) * 10, n_periods)

    # Generate AR(1) errors
    errors = np.zeros(n_entities * n_periods)
    innovations = np.random.randn(n_entities * n_periods) * 5

    for i in range(n_entities):
        start_idx = i * n_periods
        for t in range(n_periods):
            idx = start_idx + t
            if t == 0:
                errors[idx] = innovations[idx]
            else:
                errors[idx] = rho * errors[idx - 1] + innovations[idx]

    # Generate y
    y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects + errors

    data = pd.DataFrame({
        'entity': entities,
        'time': times,
        'y': y,
        'x1': x1,
        'x2': x2
    })

    return data


def generate_panel_data_heteroskedastic(n_entities=50, n_periods=10, seed=43):
    """
    Generate panel data with groupwise heteroskedasticity.
    """
    np.random.seed(seed)

    beta_0 = 10.0
    beta_1 = 2.0
    beta_2 = -1.5

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    x1 = np.random.randn(n_entities * n_periods) * 5 + 50
    x2 = np.random.randn(n_entities * n_periods) * 3 + 30

    entity_effects = np.repeat(np.random.randn(n_entities) * 10, n_periods)

    # Heteroskedastic errors (variance increases with entity index)
    errors = np.zeros(n_entities * n_periods)
    for i in range(n_entities):
        sigma_i = 1.0 + 0.5 * i
        start_idx = i * n_periods
        end_idx = (i + 1) * n_periods
        errors[start_idx:end_idx] = np.random.randn(n_periods) * sigma_i

    y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects + errors

    data = pd.DataFrame({
        'entity': entities,
        'time': times,
        'y': y,
        'x1': x1,
        'x2': x2
    })

    return data


def generate_panel_data_clean(n_entities=50, n_periods=10, seed=44):
    """
    Generate clean panel data (no violations).
    """
    np.random.seed(seed)

    beta_0 = 10.0
    beta_1 = 2.0
    beta_2 = -1.5

    entities = np.repeat(range(1, n_entities + 1), n_periods)
    times = np.tile(range(1, n_periods + 1), n_entities)

    x1 = np.random.randn(n_entities * n_periods) * 5 + 50
    x2 = np.random.randn(n_entities * n_periods) * 3 + 30

    entity_effects = np.repeat(np.random.randn(n_entities) * 10, n_periods)

    # Clean i.i.d. errors
    errors = np.random.randn(n_entities * n_periods) * 5

    y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects + errors

    data = pd.DataFrame({
        'entity': entities,
        'time': times,
        'y': y,
        'x1': x1,
        'x2': x2
    })

    return data


def run_panelbox_tests(data, model_type='fe'):
    """
    Run all validation tests on panel data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    model_type : str
        'fe' for Fixed Effects, 're' for Random Effects

    Returns
    -------
    dict
        Test results
    """
    # Fit model
    if model_type == 'fe':
        model = FixedEffects("y ~ x1 + x2", data, "entity", "time")
        results = model.fit(cov_type='nonrobust')
    elif model_type == 're':
        model = RandomEffects("y ~ x1 + x2", data, "entity", "time")
        results = model.fit(cov_type='nonrobust')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run validation tests
    from panelbox.validation.serial_correlation.wooldridge_ar import WooldridgeARTest
    from panelbox.validation.serial_correlation.breusch_godfrey import BreuschGodfreyTest
    from panelbox.validation.heteroskedasticity.modified_wald import ModifiedWaldTest
    from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest
    from panelbox.validation.heteroskedasticity.white import WhiteTest
    from panelbox.validation.cross_sectional.pesaran_cd import PesaranCDTest
    from panelbox.validation.specification.mundlak import MundlakTest

    test_results = {
        'model_type': model_type,
        'coefficients': {
            'x1': float(results.params['x1']),
            'x2': float(results.params['x2'])
        },
        'tests': {}
    }

    # Wooldridge AR (FE only)
    if model_type == 'fe':
        try:
            test = WooldridgeARTest(results)
            result = test.run(alpha=0.05)
            test_results['tests']['wooldridge'] = {
                'statistic': float(result.statistic),
                'pvalue': float(result.pvalue),
                'df': result.df
            }
        except Exception as e:
            test_results['tests']['wooldridge'] = {'error': str(e)}

    # Breusch-Godfrey
    try:
        test = BreuschGodfreyTest(results)
        result = test.run(lags=1, alpha=0.05)
        test_results['tests']['breusch_godfrey'] = {
            'statistic': float(result.statistic),
            'pvalue': float(result.pvalue),
            'df': result.df
        }
    except Exception as e:
        test_results['tests']['breusch_godfrey'] = {'error': str(e)}

    # Modified Wald (FE only)
    if model_type == 'fe':
        try:
            test = ModifiedWaldTest(results)
            result = test.run(alpha=0.05)
            test_results['tests']['modified_wald'] = {
                'statistic': float(result.statistic),
                'pvalue': float(result.pvalue),
                'df': result.df
            }
        except Exception as e:
            test_results['tests']['modified_wald'] = {'error': str(e)}

    # Breusch-Pagan
    try:
        test = BreuschPaganTest(results)
        result = test.run(alpha=0.05)
        test_results['tests']['breusch_pagan'] = {
            'statistic': float(result.statistic),
            'pvalue': float(result.pvalue),
            'df': result.df if result.df is not None else None
        }
    except Exception as e:
        test_results['tests']['breusch_pagan'] = {'error': str(e)}

    # White
    try:
        test = WhiteTest(results)
        result = test.run(alpha=0.05, cross_terms=False)
        test_results['tests']['white'] = {
            'statistic': float(result.statistic),
            'pvalue': float(result.pvalue),
            'df': result.df if result.df is not None else None
        }
    except Exception as e:
        test_results['tests']['white'] = {'error': str(e)}

    # Pesaran CD
    try:
        test = PesaranCDTest(results)
        result = test.run(alpha=0.05)
        test_results['tests']['pesaran_cd'] = {
            'statistic': float(result.statistic),
            'pvalue': float(result.pvalue)
        }
    except Exception as e:
        test_results['tests']['pesaran_cd'] = {'error': str(e)}

    # Mundlak (RE only)
    if model_type == 're':
        try:
            test = MundlakTest(results)
            result = test.run(alpha=0.05)
            test_results['tests']['mundlak'] = {
                'statistic': float(result.statistic),
                'pvalue': float(result.pvalue),
                'df': result.df
            }
        except Exception as e:
            test_results['tests']['mundlak'] = {'error': str(e)}

    return test_results


def main():
    """Main validation script."""
    output_dir = Path('/home/guhaase/projetos/panelbox/scripts/validation/output')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("PANELBOX VALIDATION - GENERATING TEST DATA AND RUNNING TESTS")
    print("=" * 80)
    print()

    # Test Case 1: AR(1) Serial Correlation
    print("1. Generating data with AR(1) serial correlation (rho=0.5)...")
    data_ar1 = generate_panel_data_ar1(n_entities=50, n_periods=10, rho=0.5, seed=42)
    data_ar1.to_csv(output_dir / 'data_ar1.csv', index=False)
    print(f"   Saved to: {output_dir / 'data_ar1.csv'}")
    print(f"   Shape: {data_ar1.shape}")

    print("\n2. Running PanelBox tests on AR(1) data (Fixed Effects)...")
    results_ar1_fe = run_panelbox_tests(data_ar1, model_type='fe')
    with open(output_dir / 'panelbox_results_ar1_fe.json', 'w') as f:
        json.dump(results_ar1_fe, f, indent=2)
    print(f"   Saved to: {output_dir / 'panelbox_results_ar1_fe.json'}")

    # Test Case 2: Heteroskedasticity
    print("\n3. Generating data with groupwise heteroskedasticity...")
    data_het = generate_panel_data_heteroskedastic(n_entities=50, n_periods=10, seed=43)
    data_het.to_csv(output_dir / 'data_het.csv', index=False)
    print(f"   Saved to: {output_dir / 'data_het.csv'}")

    print("\n4. Running PanelBox tests on heteroskedastic data (Fixed Effects)...")
    results_het_fe = run_panelbox_tests(data_het, model_type='fe')
    with open(output_dir / 'panelbox_results_het_fe.json', 'w') as f:
        json.dump(results_het_fe, f, indent=2)
    print(f"   Saved to: {output_dir / 'panelbox_results_het_fe.json'}")

    # Test Case 3: Clean data
    print("\n5. Generating clean data (no violations)...")
    data_clean = generate_panel_data_clean(n_entities=50, n_periods=10, seed=44)
    data_clean.to_csv(output_dir / 'data_clean.csv', index=False)
    print(f"   Saved to: {output_dir / 'data_clean.csv'}")

    print("\n6. Running PanelBox tests on clean data (Fixed Effects)...")
    results_clean_fe = run_panelbox_tests(data_clean, model_type='fe')
    with open(output_dir / 'panelbox_results_clean_fe.json', 'w') as f:
        json.dump(results_clean_fe, f, indent=2)
    print(f"   Saved to: {output_dir / 'panelbox_results_clean_fe.json'}")

    print("\n7. Running PanelBox tests on clean data (Random Effects)...")
    results_clean_re = run_panelbox_tests(data_clean, model_type='re')
    with open(output_dir / 'panelbox_results_clean_re.json', 'w') as f:
        json.dump(results_clean_re, f, indent=2)
    print(f"   Saved to: {output_dir / 'panelbox_results_clean_re.json'}")

    print("\n" + "=" * 80)
    print("PANELBOX TESTS COMPLETED")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")

    print("\nNext: Run R validation script")
    print("  cd scripts/validation")
    print("  Rscript run_r_tests.R")


if __name__ == '__main__':
    main()
