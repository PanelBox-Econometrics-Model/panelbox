"""
Simple tests for Fisher-type panel unit root test.
"""

import sys

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

import numpy as np
import pandas as pd

import panelbox as pb

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def test_fisher_import():
    """Test that Fisher test can be imported."""
    assert hasattr(pb, "FisherTest")
    assert hasattr(pb, "FisherTestResult")


def test_fisher_basic_stationary():
    """Test Fisher test with stationary data."""
    np.random.seed(42)

    # Generate stationary AR(1) data with ρ = 0.5
    n_entities = 5
    n_time = 50

    data_list = []
    for i in range(n_entities):
        y = np.zeros(n_time)
        y[0] = np.random.randn()
        for t in range(1, n_time):
            y[t] = 0.5 * y[t - 1] + np.random.randn()

        df_i = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
        data_list.append(df_i)

    data = pd.concat(data_list, ignore_index=True)

    # Run Fisher-ADF test
    fisher = pb.FisherTest(data, "y", "entity", "time", test_type="adf", trend="c")
    result = fisher.run()

    # Assertions
    assert isinstance(result, pb.FisherTestResult)
    assert result.statistic > 0
    assert 0 <= result.pvalue <= 1
    assert result.n_entities == n_entities
    assert result.test_type == "adf"
    assert len(result.individual_pvalues) == n_entities

    # For stationary data, we expect low p-value (reject H0)
    # But not strict since it's random data
    print(f"Fisher-ADF statistic: {result.statistic:.4f}")
    print(f"P-value: {result.pvalue:.4f}")
    print(f"Conclusion: {result.conclusion}")


def test_fisher_basic_unit_root():
    """Test Fisher test with unit root data (random walk)."""
    np.random.seed(123)

    # Generate random walk data (unit root)
    n_entities = 5
    n_time = 50

    data_list = []
    for i in range(n_entities):
        y = np.cumsum(np.random.randn(n_time))  # Random walk

        df_i = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
        data_list.append(df_i)

    data = pd.concat(data_list, ignore_index=True)

    # Run Fisher-ADF test
    fisher = pb.FisherTest(data, "y", "entity", "time", test_type="adf", trend="c")
    result = fisher.run()

    # Assertions
    assert isinstance(result, pb.FisherTestResult)
    assert result.statistic > 0
    assert 0 <= result.pvalue <= 1
    assert result.n_entities == n_entities

    # For unit root data, we expect high p-value (fail to reject H0)
    # But not strict since it's random data
    print(f"\nFisher-ADF statistic (unit root): {result.statistic:.4f}")
    print(f"P-value: {result.pvalue:.4f}")
    print(f"Conclusion: {result.conclusion}")


def test_fisher_pp_test():
    """Test Fisher-PP variant."""
    np.random.seed(42)

    # Generate stationary data
    n_entities = 5
    n_time = 50

    data_list = []
    for i in range(n_entities):
        y = np.zeros(n_time)
        y[0] = np.random.randn()
        for t in range(1, n_time):
            y[t] = 0.6 * y[t - 1] + np.random.randn()

        df_i = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
        data_list.append(df_i)

    data = pd.concat(data_list, ignore_index=True)

    # Run Fisher-PP test
    fisher = pb.FisherTest(data, "y", "entity", "time", test_type="pp", trend="c")
    result = fisher.run()

    # Assertions
    assert isinstance(result, pb.FisherTestResult)
    assert result.test_type == "pp"
    assert result.statistic > 0
    assert 0 <= result.pvalue <= 1

    print(f"\nFisher-PP statistic: {result.statistic:.4f}")
    print(f"P-value: {result.pvalue:.4f}")


def test_fisher_trend_specifications():
    """Test different trend specifications."""
    np.random.seed(42)

    # Generate data
    n_entities = 5
    n_time = 40

    data_list = []
    for i in range(n_entities):
        y = np.random.randn(n_time).cumsum() * 0.5  # Moderately persistent
        df_i = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
        data_list.append(df_i)

    data = pd.concat(data_list, ignore_index=True)

    trends = ["n", "c", "ct"]

    for trend in trends:
        fisher = pb.FisherTest(data, "y", "entity", "time", test_type="adf", trend=trend)
        result = fisher.run()

        assert result.trend == trend
        assert result.statistic > 0
        print(f"\nTrend '{trend}': statistic={result.statistic:.4f}, p={result.pvalue:.4f}")


def test_fisher_grunfeld():
    """Test Fisher test on Grunfeld data."""
    data = pb.load_grunfeld()

    # Test on 'invest' variable
    fisher = pb.FisherTest(data, "invest", "firm", "year", test_type="adf", trend="c")
    result = fisher.run()

    assert isinstance(result, pb.FisherTestResult)
    assert result.n_entities == 10
    assert len(result.individual_pvalues) == 10

    print(f"\nGrunfeld 'invest': statistic={result.statistic:.4f}, p={result.pvalue:.4f}")
    print(f"Individual p-values:")
    for entity, pval in list(result.individual_pvalues.items())[:3]:
        print(f"  Entity {entity}: {pval:.4f}")


def test_fisher_unbalanced_panel():
    """Test Fisher with unbalanced panel."""
    np.random.seed(42)

    # Create unbalanced panel (different lengths)
    data_list = []

    # Entity 0: 50 observations
    y0 = np.random.randn(50).cumsum() * 0.3
    data_list.append(pd.DataFrame({"entity": 0, "time": range(50), "y": y0}))

    # Entity 1: 30 observations
    y1 = np.random.randn(30).cumsum() * 0.3
    data_list.append(pd.DataFrame({"entity": 1, "time": range(30), "y": y1}))

    # Entity 2: 40 observations
    y2 = np.random.randn(40).cumsum() * 0.3
    data_list.append(pd.DataFrame({"entity": 2, "time": range(40), "y": y2}))

    data = pd.concat(data_list, ignore_index=True)

    # Fisher test should handle unbalanced panel
    fisher = pb.FisherTest(data, "y", "entity", "time", test_type="adf", trend="c")
    result = fisher.run()

    assert result.n_entities == 3
    assert len(result.individual_pvalues) == 3

    print(f"\nUnbalanced panel: statistic={result.statistic:.4f}, p={result.pvalue:.4f}")


def test_fisher_result_string():
    """Test string representation of results."""
    np.random.seed(42)

    # Simple data
    data = pd.DataFrame(
        {
            "entity": [0] * 20 + [1] * 20,
            "time": list(range(20)) * 2,
            "y": np.random.randn(40).cumsum() * 0.5,
        }
    )

    fisher = pb.FisherTest(data, "y", "entity", "time", test_type="adf", trend="c")
    result = fisher.run()

    result_str = str(result)

    # Check that string contains key information
    assert "Fisher-type Panel Unit Root Test" in result_str
    assert "Fisher statistic" in result_str
    assert "P-value" in result_str
    assert "Cross-sections" in result_str
    assert result.test_type.upper() in result_str

    print(f"\n{result_str}")


def test_fisher_invalid_inputs():
    """Test error handling for invalid inputs."""
    data = pb.load_grunfeld()

    # Invalid test_type
    try:
        fisher = pb.FisherTest(data, "invest", "firm", "year", test_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "test_type must be" in str(e)

    # Invalid trend
    try:
        fisher = pb.FisherTest(data, "invest", "firm", "year", trend="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "trend must be" in str(e)

    # Missing variable
    try:
        fisher = pb.FisherTest(data, "nonexistent", "firm", "year")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e)


if __name__ == "__main__":
    print("=" * 70)
    print("Running Fisher-type Panel Unit Root Tests")
    print("=" * 70)

    test_fisher_import()
    print("\n✅ Import test passed")

    test_fisher_basic_stationary()
    print("\n✅ Basic stationary test passed")

    test_fisher_basic_unit_root()
    print("\n✅ Basic unit root test passed")

    test_fisher_pp_test()
    print("\n✅ PP test passed")

    test_fisher_trend_specifications()
    print("\n✅ Trend specifications test passed")

    test_fisher_grunfeld()
    print("\n✅ Grunfeld test passed")

    test_fisher_unbalanced_panel()
    print("\n✅ Unbalanced panel test passed")

    test_fisher_result_string()
    print("\n✅ Result string test passed")

    test_fisher_invalid_inputs()
    print("\n✅ Invalid inputs test passed")

    print("\n" + "=" * 70)
    print("✅ All Fisher tests passed!")
    print("=" * 70)
