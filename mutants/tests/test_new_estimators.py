"""
Simple tests for Between and First Difference Estimators (no pytest required).
"""

import numpy as np
import pandas as pd

import panelbox as pb


def create_test_data():
    """Create simple balanced panel dataset for testing."""
    np.random.seed(42)
    n_entities = 10
    n_periods = 5

    entities = np.repeat(range(n_entities), n_periods)
    times = np.tile(range(n_periods), n_entities)

    # Create variables with clear between variation
    entity_effects = np.repeat(np.arange(n_entities) * 10, n_periods)
    x1 = entity_effects + np.random.normal(0, 1, n_entities * n_periods)
    x2 = np.random.normal(0, 1, n_entities * n_periods)
    y = 2 + 0.5 * x1 + 1.5 * x2 + entity_effects + np.random.normal(0, 1, n_entities * n_periods)

    data = pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})

    return data


def test_between_estimator():
    """Test Between Estimator."""
    print("\n" + "=" * 70)
    print("Testing Between Estimator")
    print("=" * 70)

    data = create_test_data()

    # Test 1: Basic initialization and fitting
    print("\n1. Initialization and basic fit...")
    be = pb.BetweenEstimator("y ~ x1 + x2", data, "entity", "time")
    results = be.fit(cov_type="nonrobust")
    assert len(results.params) == 3  # intercept + x1 + x2
    assert "Intercept" in results.params.index
    assert "x1" in results.params.index
    assert "x2" in results.params.index
    print("   ✓ PASSED")

    # Test 2: Entity means
    print("\n2. Entity means structure...")
    assert be.entity_means is not None
    assert len(be.entity_means) == 10  # 10 entities
    assert "entity" in be.entity_means.columns
    assert "y" in be.entity_means.columns
    print("   ✓ PASSED")

    # Test 3: Degrees of freedom
    print("\n3. Degrees of freedom...")
    assert results.nobs == 10  # 10 entities
    assert results.df_resid == 7  # 10 - 3
    print("   ✓ PASSED")

    # Test 4: R-squared
    print("\n4. R-squared measures...")
    assert results.rsquared == results.rsquared_between
    assert 0 <= results.rsquared <= 1
    assert results.rsquared_within == 0.0
    print("   ✓ PASSED")

    # Test 5: Different covariance types
    print("\n5. Different covariance types...")
    for cov_type in ["robust", "hc1", "clustered"]:
        res = be.fit(cov_type=cov_type)
        assert res.cov_type == cov_type
    print("   ✓ PASSED")

    # Test 6: Summary
    print("\n6. Summary output...")
    summary = results.summary()
    assert isinstance(summary, str)
    assert "Between Estimator" in summary
    print("   ✓ PASSED")

    # Test 7: With Grunfeld data
    print("\n7. Grunfeld dataset...")
    grunfeld = pb.load_grunfeld()
    be_grun = pb.BetweenEstimator("invest ~ value + capital", grunfeld, "firm", "year")
    results_grun = be_grun.fit(cov_type="robust")
    assert results_grun.nobs == 10  # 10 firms
    assert results_grun.rsquared_between > 0.8  # Known property
    print("   ✓ PASSED")

    print("\n" + "=" * 70)
    print("Between Estimator: ALL TESTS PASSED ✓")
    print("=" * 70)


def test_first_difference_estimator():
    """Test First Difference Estimator."""
    print("\n" + "=" * 70)
    print("Testing First Difference Estimator")
    print("=" * 70)

    data = create_test_data()

    # Test 1: Basic initialization and fitting
    print("\n1. Initialization and basic fit...")
    fd = pb.FirstDifferenceEstimator("y ~ x1 + x2", data, "entity", "time")
    results = fd.fit(cov_type="nonrobust")
    assert len(results.params) == 2  # x1 + x2 (no intercept)
    assert "Intercept" not in results.params.index
    assert "x1" in results.params.index
    assert "x2" in results.params.index
    print("   ✓ PASSED")

    # Test 2: Observations dropped
    print("\n2. Observations dropped...")
    assert fd.n_obs_original == 50  # 10 × 5
    assert fd.n_obs_differenced == 40  # 10 × 4
    assert (fd.n_obs_original - fd.n_obs_differenced) == 10
    print("   ✓ PASSED")

    # Test 3: Degrees of freedom
    print("\n3. Degrees of freedom...")
    assert results.nobs == 40
    assert results.df_model == 2
    assert results.df_resid == 38
    print("   ✓ PASSED")

    # Test 4: R-squared
    print("\n4. R-squared measures...")
    assert 0 <= results.rsquared <= 1
    assert results.rsquared == results.rsquared_within
    assert np.isnan(results.rsquared_between)
    print("   ✓ PASSED")

    # Test 5: Different covariance types (especially clustered)
    print("\n5. Different covariance types...")
    for cov_type in ["robust", "clustered", "driscoll_kraay"]:
        res = fd.fit(cov_type=cov_type)
        assert res.cov_type == cov_type
    print("   ✓ PASSED")

    # Test 6: Summary
    print("\n6. Summary output...")
    summary = results.summary()
    assert isinstance(summary, str)
    assert "First Difference" in summary
    print("   ✓ PASSED")

    # Test 7: Residuals shape
    print("\n7. Residuals shape...")
    assert len(results.resid) == 50  # Original length
    n_nan = np.sum(np.isnan(results.resid))
    assert n_nan == 10  # First period dropped for each entity
    print("   ✓ PASSED")

    # Test 8: With Grunfeld data
    print("\n8. Grunfeld dataset...")
    grunfeld = pb.load_grunfeld()
    fd_grun = pb.FirstDifferenceEstimator("invest ~ value + capital", grunfeld, "firm", "year")
    results_grun = fd_grun.fit(cov_type="clustered")
    assert results_grun.nobs == 190  # 10 × 19
    assert (fd_grun.n_obs_original - fd_grun.n_obs_differenced) == 10
    print("   ✓ PASSED")

    print("\n" + "=" * 70)
    print("First Difference Estimator: ALL TESTS PASSED ✓")
    print("=" * 70)


def test_comparison():
    """Compare all three estimators."""
    print("\n" + "=" * 70)
    print("Comparing Estimators: FE vs BE vs FD")
    print("=" * 70)

    grunfeld = pb.load_grunfeld()

    # Fixed Effects
    fe = pb.FixedEffects("invest ~ value + capital", grunfeld, "firm", "year")
    results_fe = fe.fit(cov_type="clustered")

    # Between
    be = pb.BetweenEstimator("invest ~ value + capital", grunfeld, "firm", "year")
    results_be = be.fit(cov_type="robust")

    # First Difference
    fd = pb.FirstDifferenceEstimator("invest ~ value + capital", grunfeld, "firm", "year")
    results_fd = fd.fit(cov_type="clustered")

    print("\nCoefficients:")
    print(f"{'Estimator':<20} {'value':>10} {'capital':>10}")
    print("-" * 42)
    print(
        f"{'Fixed Effects':<20} {results_fe.params['value']:>10.4f} {results_fe.params['capital']:>10.4f}"
    )
    print(
        f"{'Between':<20} {results_be.params['value']:>10.4f} {results_be.params['capital']:>10.4f}"
    )
    print(
        f"{'First Difference':<20} {results_fd.params['value']:>10.4f} {results_fd.params['capital']:>10.4f}"
    )

    print("\nR-squared:")
    print(f"Fixed Effects (within):  {results_fe.rsquared:.4f}")
    print(f"Between (between):       {results_be.rsquared:.4f}")
    print(f"First Difference (diff): {results_fd.rsquared:.4f}")

    print("\nObservations:")
    print(f"Fixed Effects:     {results_fe.nobs}")
    print(f"Between:           {results_be.nobs}")
    print(f"First Difference:  {results_fd.nobs}")

    print("\n" + "=" * 70)
    print("Comparison: COMPLETE ✓")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PANELBOX - NEW ESTIMATORS TEST SUITE")
    print("=" * 70)

    try:
        test_between_estimator()
        test_first_difference_estimator()
        test_comparison()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Between Estimator: 7 tests passed")
        print("  ✓ First Difference Estimator: 8 tests passed")
        print("  ✓ Comparison test: Complete")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
