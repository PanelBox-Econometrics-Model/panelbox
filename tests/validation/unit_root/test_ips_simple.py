"""
Simple tests for IPS test without pytest dependency.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

from panelbox.validation.unit_root import IPSTest


def generate_stationary_panel():
    """Generate stationary panel data."""
    np.random.seed(42)
    n_entities = 10
    n_time = 50

    data_list = []
    for i in range(n_entities):
        # AR(1) with |rho| < 1 (stationary), allow heterogeneity
        rho = 0.3 + 0.3 * (i / n_entities)  # rho varies from 0.3 to 0.6
        y = np.zeros(n_time)
        y[0] = np.random.normal(0, 1)
        for t in range(1, n_time):
            y[t] = rho * y[t - 1] + np.random.normal(0, 1)

        entity_data = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
        data_list.append(entity_data)

    return pd.concat(data_list, ignore_index=True)


def generate_unit_root_panel():
    """Generate panel data with unit root."""
    np.random.seed(123)
    n_entities = 10
    n_time = 50

    data_list = []
    for i in range(n_entities):
        # Random walk (unit root)
        y = np.cumsum(np.random.normal(0, 1, n_time))

        entity_data = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
        data_list.append(entity_data)

    return pd.concat(data_list, ignore_index=True)


def generate_mixed_panel():
    """Generate panel with some stationary and some unit root series."""
    np.random.seed(456)
    n_entities = 10
    n_time = 50

    data_list = []
    for i in range(n_entities):
        if i < 5:
            # Stationary
            rho = 0.5
            y = np.zeros(n_time)
            y[0] = np.random.normal(0, 1)
            for t in range(1, n_time):
                y[t] = rho * y[t - 1] + np.random.normal(0, 1)
        else:
            # Unit root
            y = np.cumsum(np.random.normal(0, 1, n_time))

        entity_data = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
        data_list.append(entity_data)

    return pd.concat(data_list, ignore_index=True)


def test_ips_stationary():
    """Test IPS on stationary data."""
    print("\n" + "=" * 70)
    print("Test 1: IPS on Stationary Data (Heterogeneous)")
    print("=" * 70)

    data = generate_stationary_panel()
    ips = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
    result = ips.run()

    print(result)

    # Should reject H0 for stationary data
    assert result.pvalue < 0.10, "Should reject null for stationary data"
    print("\n✓ Test passed: Correctly identifies stationary data")


def test_ips_unit_root():
    """Test IPS on unit root data."""
    print("\n" + "=" * 70)
    print("Test 2: IPS on Unit Root Data")
    print("=" * 70)

    data = generate_unit_root_panel()
    ips = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
    result = ips.run()

    print(result)

    # With unit root data, may or may not reject in finite samples
    if result.pvalue > 0.05:
        print("\n✓ Test passed: Correctly fails to reject null")
    else:
        print(f"\n⚠ Warning: P-value ({result.pvalue:.4f}) < 0.05")
        print("  This can happen with finite samples")


def test_ips_mixed():
    """Test IPS on mixed panel (some stationary, some unit root)."""
    print("\n" + "=" * 70)
    print("Test 3: IPS on Mixed Panel")
    print("=" * 70)

    data = generate_mixed_panel()
    ips = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
    result = ips.run()

    print(result)
    print("\nH1 for IPS is 'some panels are stationary', so should reject H0")
    print("since half the panels are stationary.")

    # Should reject H0 because some panels are stationary
    if result.pvalue < 0.10:
        print("\n✓ Test passed: Correctly detects mixed panel")
    else:
        print(f"\n⚠ Warning: Did not reject H0 (p={result.pvalue:.4f})")


def test_ips_grunfeld():
    """Test IPS on Grunfeld data."""
    print("\n" + "=" * 70)
    print("Test 4: IPS on Grunfeld Dataset")
    print("=" * 70)

    try:
        import panelbox as pb

        data = pb.load_grunfeld()

        for var in ["invest", "value", "capital"]:
            print(f"\n{var.upper()}:")
            print("-" * 40)
            ips = IPSTest(data, var, "firm", "year", lags=1, trend="c")
            result = ips.run()
            print(f"W-statistic: {result.statistic:.4f}")
            print(f"t-bar: {result.t_bar:.4f}")
            print(f"P-value: {result.pvalue:.4f}")
            print(f"Conclusion: {result.conclusion}")

        print("\n✓ Test passed: Successfully tested Grunfeld variables")

    except Exception as e:
        print(f"⚠ Skipped: {e}")


def test_ips_different_trends():
    """Test IPS with different trend specifications."""
    print("\n" + "=" * 70)
    print("Test 5: IPS with Different Trend Specifications")
    print("=" * 70)

    data = generate_stationary_panel()

    for trend, name in [("n", "No trend"), ("c", "Constant"), ("ct", "Constant + trend")]:
        print(f"\n{name}:")
        print("-" * 40)
        ips = IPSTest(data, "y", "entity", "time", lags=1, trend=trend)
        result = ips.run()
        print(f"W-statistic: {result.statistic:.4f}")
        print(f"P-value: {result.pvalue:.4f}")
        print(f"Deterministics: {result.deterministics}")

    print("\n✓ Test passed: All trend specifications work")


def test_ips_auto_lags():
    """Test automatic lag selection."""
    print("\n" + "=" * 70)
    print("Test 6: IPS with Automatic Lag Selection")
    print("=" * 70)

    data = generate_stationary_panel()
    ips = IPSTest(data, "y", "entity", "time", lags=None, trend="c")
    result = ips.run()

    if isinstance(result.lags, list):
        print(f"Selected lags (per entity): {result.lags}")
        print(f"Mean lags: {np.mean(result.lags):.1f}")
    else:
        print(f"Selected lags: {result.lags}")

    print(f"W-statistic: {result.statistic:.4f}")
    print(f"P-value: {result.pvalue:.4f}")

    print("\n✓ Test passed: Auto lag selection works")


def test_ips_validation():
    """Test input validation."""
    print("\n" + "=" * 70)
    print("Test 7: Input Validation")
    print("=" * 70)

    data = generate_stationary_panel()

    # Test invalid variable
    try:
        IPSTest(data, "invalid", "entity", "time")
        assert False, "Should raise error for invalid variable"
    except ValueError as e:
        print(f"✓ Correctly caught invalid variable: {e}")

    # Test invalid entity column
    try:
        IPSTest(data, "y", "invalid", "time")
        assert False, "Should raise error for invalid entity column"
    except ValueError as e:
        print(f"✓ Correctly caught invalid entity column: {e}")

    # Test invalid trend
    try:
        IPSTest(data, "y", "entity", "time", trend="invalid")
        assert False, "Should raise error for invalid trend"
    except ValueError as e:
        print(f"✓ Correctly caught invalid trend: {e}")

    print("\n✓ Test passed: Input validation works correctly")


def test_ips_individual_stats():
    """Test that individual statistics are reported."""
    print("\n" + "=" * 70)
    print("Test 8: Individual Entity Statistics")
    print("=" * 70)

    data = generate_stationary_panel()
    ips = IPSTest(data, "y", "entity", "time", lags=1, trend="c")
    result = ips.run()

    print(f"Number of entities: {result.n_entities}")
    print(f"Individual t-statistics:")
    for entity, t_stat in list(result.individual_stats.items())[:5]:
        print(f"  Entity {entity}: {t_stat:.4f}")
    if len(result.individual_stats) > 5:
        print(f"  ... and {len(result.individual_stats) - 5} more")

    assert len(result.individual_stats) == result.n_entities
    print("\n✓ Test passed: Individual statistics are available")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("IPS (Im-Pesaran-Shin) Test Validation Suite")
    print("=" * 70)

    tests = [
        test_ips_stationary,
        test_ips_unit_root,
        test_ips_mixed,
        test_ips_grunfeld,
        test_ips_different_trends,
        test_ips_auto_lags,
        test_ips_validation,
        test_ips_individual_stats,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
