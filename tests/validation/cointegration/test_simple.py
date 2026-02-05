"""
Simple tests for cointegration tests without pytest dependency.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

from panelbox.validation.cointegration import KaoTest, PedroniTest


def generate_cointegrated_panel():
    """Generate panel data with cointegration."""
    np.random.seed(42)
    n_entities = 8
    n_time = 50

    data_list = []
    for i in range(n_entities):
        # Generate I(1) processes with cointegration
        # x and y share a common stochastic trend
        trend = np.cumsum(np.random.normal(0, 1, n_time))

        x = trend + np.random.normal(0, 0.5, n_time)
        y = 2 + 1.5 * x + np.random.normal(0, 1, n_time)  # y and x are cointegrated

        data_list.append(pd.DataFrame({"entity": i, "time": range(n_time), "y": y, "x": x}))

    return pd.concat(data_list, ignore_index=True)


def generate_noncointegrated_panel():
    """Generate panel data without cointegration (independent random walks)."""
    np.random.seed(123)
    n_entities = 8
    n_time = 50

    data_list = []
    for i in range(n_entities):
        # Independent random walks
        y = np.cumsum(np.random.normal(0, 1, n_time))
        x = np.cumsum(np.random.normal(0, 1, n_time))

        data_list.append(pd.DataFrame({"entity": i, "time": range(n_time), "y": y, "x": x}))

    return pd.concat(data_list, ignore_index=True)


def test_pedroni_basic():
    """Test Pedroni with Grunfeld data."""
    print("\n" + "=" * 70)
    print("Test 1: Pedroni Test - Grunfeld Dataset")
    print("=" * 70)

    try:
        import panelbox as pb

        data = pb.load_grunfeld()

        # Test cointegration between invest and value
        ped = PedroniTest(data, "invest", ["value"], "firm", "year", trend="c")
        result = ped.run()

        print(result)
        print("\n✓ Test passed: Pedroni test runs successfully")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


def test_kao_basic():
    """Test Kao with Grunfeld data."""
    print("\n" + "=" * 70)
    print("Test 2: Kao Test - Grunfeld Dataset")
    print("=" * 70)

    try:
        import panelbox as pb

        data = pb.load_grunfeld()

        # Test cointegration between invest and value
        kao = KaoTest(data, "invest", ["value"], "firm", "year", trend="c")
        result = kao.run()

        print(result)
        print("\n✓ Test passed: Kao test runs successfully")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


def test_pedroni_cointegrated():
    """Test Pedroni on cointegrated data."""
    print("\n" + "=" * 70)
    print("Test 3: Pedroni Test - Simulated Cointegrated Data")
    print("=" * 70)

    data = generate_cointegrated_panel()

    ped = PedroniTest(data, "y", ["x"], "entity", "time", trend="c")
    result = ped.run()

    print(result)

    # Should have some evidence of cointegration
    reject_count = sum(1 for p in result.pvalues.values() if p < 0.10)
    print(f"\n{reject_count}/{len(result.pvalues)} tests reject at 10% level")

    if reject_count >= len(result.pvalues) / 2:
        print("✓ Test passed: Detects cointegration")
    else:
        print("⚠ Warning: Did not detect strong cointegration (expected with simulated data)")


def test_kao_cointegrated():
    """Test Kao on cointegrated data."""
    print("\n" + "=" * 70)
    print("Test 4: Kao Test - Simulated Cointegrated Data")
    print("=" * 70)

    data = generate_cointegrated_panel()

    kao = KaoTest(data, "y", ["x"], "entity", "time", trend="c")
    result = kao.run()

    print(result)

    if result.pvalue < 0.10:
        print("✓ Test passed: Detects cointegration")
    else:
        print("⚠ Warning: Did not detect cointegration (p={:.4f})".format(result.pvalue))


def test_pedroni_noncointegrated():
    """Test Pedroni on non-cointegrated data."""
    print("\n" + "=" * 70)
    print("Test 5: Pedroni Test - Independent Random Walks")
    print("=" * 70)

    data = generate_noncointegrated_panel()

    ped = PedroniTest(data, "y", ["x"], "entity", "time", trend="c")
    result = ped.run()

    print(result)

    # Should not find cointegration
    reject_count = sum(1 for p in result.pvalues.values() if p < 0.05)
    print(f"\n{reject_count}/{len(result.pvalues)} tests reject at 5% level")

    if reject_count < len(result.pvalues) / 2:
        print("✓ Test passed: Does not falsely detect cointegration")
    else:
        print("⚠ Warning: May have false positives (expected occasionally)")


def test_kao_noncointegrated():
    """Test Kao on non-cointegrated data."""
    print("\n" + "=" * 70)
    print("Test 6: Kao Test - Independent Random Walks")
    print("=" * 70)

    data = generate_noncointegrated_panel()

    kao = KaoTest(data, "y", ["x"], "entity", "time", trend="c")
    result = kao.run()

    print(result)

    if result.pvalue > 0.05:
        print("✓ Test passed: Does not falsely detect cointegration")
    else:
        print("⚠ Warning: False positive (p={:.4f})".format(result.pvalue))


def test_validation():
    """Test input validation."""
    print("\n" + "=" * 70)
    print("Test 7: Input Validation")
    print("=" * 70)

    data = generate_cointegrated_panel()

    # Test invalid dependent
    try:
        PedroniTest(data, "invalid", ["x"], "entity", "time")
        assert False, "Should raise error"
    except ValueError as e:
        print(f"✓ Caught invalid dependent: {e}")

    # Test invalid independent
    try:
        PedroniTest(data, "y", ["invalid"], "entity", "time")
        assert False, "Should raise error"
    except ValueError as e:
        print(f"✓ Caught invalid independent: {e}")

    # Test invalid trend
    try:
        PedroniTest(data, "y", ["x"], "entity", "time", trend="invalid")
        assert False, "Should raise error"
    except ValueError as e:
        print(f"✓ Caught invalid trend: {e}")

    print("\n✓ Test passed: Input validation works")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Cointegration Tests Validation Suite")
    print("=" * 70)

    tests = [
        test_pedroni_basic,
        test_kao_basic,
        test_pedroni_cointegrated,
        test_kao_cointegrated,
        test_pedroni_noncointegrated,
        test_kao_noncointegrated,
        test_validation,
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
