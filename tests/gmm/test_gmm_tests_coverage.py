"""
Coverage tests for panelbox.gmm.tests module.

Targets uncovered lines: 111, 135-142, 331, 466-468, 508-550.
"""

import numpy as np
import pytest

from panelbox.gmm.tests import GMMTests


@pytest.fixture
def tester():
    """Return a GMMTests instance."""
    return GMMTests()


# ---------------------------------------------------------------------------
# Line 111: hansen_j_test with n_instruments=None, Z is not None
# When n_instruments is not supplied, the code reads Z.shape[1].
# ---------------------------------------------------------------------------


class TestHansenJFromZShape:
    """Cover line 111: n_instruments derived from Z.shape[1]."""

    def test_n_instruments_from_z_shape(self, tester):
        """n_instruments should be inferred from Z when not explicitly given."""
        np.random.seed(42)
        n, k, L = 80, 2, 5
        Z = np.random.randn(n, L)
        residuals = np.random.randn(n)
        W = np.eye(L)

        result = tester.hansen_j_test(residuals, Z, W, n_params=k)
        assert result.name == "Hansen J-test"
        assert result.df == L - k  # 5 - 2 = 3
        assert np.isfinite(result.statistic)
        assert 0 <= result.pvalue <= 1
        assert result.details["n_instruments"] == L

    def test_n_instruments_from_z_shape_2d_residuals(self, tester):
        """Also works when residuals are 2-D column vector (legacy path)."""
        np.random.seed(42)
        n, k, L = 60, 2, 4
        Z = np.random.randn(n, L)
        residuals = np.random.randn(n, 1)  # 2-D
        W = np.eye(L)

        result = tester.hansen_j_test(residuals, Z, W, n_params=k)
        assert result.df == L - k
        assert np.isfinite(result.statistic)


# ---------------------------------------------------------------------------
# Lines 135-142: Legacy observation-level formula (zs is None)
# ---------------------------------------------------------------------------


class TestHansenJLegacyFormula:
    """Cover lines 135-142: legacy Hansen J path when zs is not provided."""

    def test_legacy_formula_basic(self, tester):
        """Legacy formula computes J from observation-level Z, residuals, W."""
        np.random.seed(42)
        n, L, k = 100, 5, 2
        Z = np.random.randn(n, L)
        residuals = np.random.randn(n)
        # Use identity weight matrix
        W = np.eye(L)

        result = tester.hansen_j_test(residuals, Z, W, n_params=k, n_instruments=L)
        assert result.df == L - k
        assert np.isfinite(result.statistic)
        assert result.statistic >= 0
        assert 0 <= result.pvalue <= 1

    def test_legacy_formula_with_nan_residuals(self, tester):
        """Legacy formula cleans NaN residuals before computing J."""
        np.random.seed(42)
        n, L, k = 100, 4, 1
        Z = np.random.randn(n, L)
        residuals = np.random.randn(n)
        # Insert some NaNs
        residuals[10] = np.nan
        residuals[50] = np.nan
        W = np.eye(L)

        result = tester.hansen_j_test(residuals, Z, W, n_params=k, n_instruments=L)
        assert result.df == L - k
        assert np.isfinite(result.statistic)

    def test_legacy_formula_2d_residuals(self, tester):
        """Legacy path flattens 2-D residuals (line 135)."""
        np.random.seed(42)
        n, L, k = 80, 5, 3
        Z = np.random.randn(n, L)
        residuals = np.random.randn(n, 1)  # Column vector
        W = np.eye(L)

        result = tester.hansen_j_test(residuals, Z, W, n_params=k, n_instruments=L)
        assert result.df == L - k
        assert np.isfinite(result.statistic)

    def test_legacy_vs_manual(self, tester):
        """Verify legacy formula matches manual computation."""
        np.random.seed(42)
        n, L, k = 50, 3, 1
        Z = np.random.randn(n, L)
        residuals = np.random.randn(n)
        W = np.eye(L)

        # Manual computation
        g_n = (Z.T @ residuals) / n
        J_manual = float(n * (g_n.T @ W @ g_n))

        result = tester.hansen_j_test(residuals, Z, W, n_params=k, n_instruments=L)
        assert abs(result.statistic - J_manual) < 1e-10


# ---------------------------------------------------------------------------
# Line 331: arellano_bond_ar_test with var_product == 0
# ---------------------------------------------------------------------------


class TestArellanoBondZeroVariance:
    """Cover line 331: zero variance in AR products."""

    def test_zero_variance_constant_residuals(self, tester):
        """When all residuals are identical, products have zero variance."""
        # All residuals are the same constant => all products identical => var = 0
        n_individuals = 5
        T = 6
        ids = np.repeat(np.arange(n_individuals), T)
        # Constant residuals for each group
        residuals_diff = np.ones(n_individuals * T) * 3.0

        result = tester.arellano_bond_ar_test(residuals_diff, ids, order=1)
        assert result.conclusion == "N/A (zero variance)"
        assert np.isnan(result.statistic)
        assert np.isnan(result.pvalue)
        assert "Zero variance" in result.details["message"]

    def test_zero_variance_single_product(self, tester):
        """With exactly one product per group and one group, var is NaN/zero."""
        # Single group with exactly (order+1) observations => 1 product total
        # ddof=1 on a single observation gives NaN, but let's verify the path
        ids = np.array([0, 0])
        residuals_diff = np.array([1.0, 1.0])
        # order=1 => one product: 1.0*1.0 = 1.0, single element, var(ddof=1)=NaN
        # NaN != 0 so this might not hit line 331, but with multiple groups
        # all having identical products it will.
        # Use multiple groups with identical products instead:
        n_individuals = 3
        T = 3  # order=1 => 2 products per group, all identical
        ids = np.repeat(np.arange(n_individuals), T)
        residuals_diff = np.full(n_individuals * T, 2.0)

        result = tester.arellano_bond_ar_test(residuals_diff, ids, order=1)
        assert result.conclusion == "N/A (zero variance)"


# ---------------------------------------------------------------------------
# Lines 466-468: difference_in_hansen with df_diff <= 0
# ---------------------------------------------------------------------------


class TestDifferenceInHansenInvalidDf:
    """Cover lines 466-468: df_diff <= 0 branch."""

    def test_df_diff_zero(self, tester):
        """When subset and full have same df, df_diff=0 returns N/A."""
        np.random.seed(42)
        n, L, k = 60, 4, 2
        Z_full = np.random.randn(n, L)
        # Subset has same number of instruments as full
        Z_subset = np.random.randn(n, L)
        W_full = np.eye(L)
        W_subset = np.eye(L)
        residuals = np.random.randn(n)

        result = tester.difference_in_hansen(
            residuals, Z_full, Z_subset, W_full, W_subset, n_params=k
        )
        # df_diff = (L - k) - (L - k) = 0
        assert result.df == 0
        assert np.isnan(result.statistic)
        assert np.isnan(result.pvalue)
        assert result.conclusion == "N/A (invalid df)"

    def test_df_diff_negative(self, tester):
        """When subset has fewer instruments than full, df_diff < 0."""
        np.random.seed(42)
        n, k = 60, 2
        L_full = 5
        L_subset = 3  # fewer instruments => df_subset < df_full
        Z_full = np.random.randn(n, L_full)
        Z_subset = np.random.randn(n, L_subset)
        W_full = np.eye(L_full)
        W_subset = np.eye(L_subset)
        residuals = np.random.randn(n)

        result = tester.difference_in_hansen(
            residuals, Z_full, Z_subset, W_full, W_subset, n_params=k
        )
        # df_diff = (3 - 2) - (5 - 2) = 1 - 3 = -2
        assert result.df == -2
        assert np.isnan(result.statistic)
        assert result.conclusion == "N/A (invalid df)"

    def test_df_diff_positive_returns_result(self, tester):
        """When df_diff > 0, the normal path returns a valid result."""
        np.random.seed(42)
        n, k = 80, 2
        L_full = 3
        L_subset = 6  # more instruments => df_subset > df_full
        Z_full = np.random.randn(n, L_full)
        Z_subset = np.random.randn(n, L_subset)
        W_full = np.eye(L_full)
        W_subset = np.eye(L_subset)
        residuals = np.random.randn(n)

        result = tester.difference_in_hansen(
            residuals, Z_full, Z_subset, W_full, W_subset, n_params=k
        )
        # df_diff = (6 - 2) - (3 - 2) = 4 - 1 = 3
        assert result.df == 3
        assert np.isfinite(result.statistic)
        assert np.isfinite(result.pvalue)
        assert 0 <= result.pvalue <= 1


# ---------------------------------------------------------------------------
# Lines 508-550: weak_instruments_test (entire method)
# ---------------------------------------------------------------------------


class TestWeakInstruments:
    """Cover lines 508-550: weak_instruments_test."""

    def test_strong_instruments(self, tester):
        """Instruments that explain X well should yield F > 10 and weak=False."""
        np.random.seed(42)
        n = 200
        L = 3

        Z = np.random.randn(n, L)
        # X is a strong linear function of Z plus small noise
        beta_true = np.array([[2.0], [-1.5], [0.8]])
        X = Z @ beta_true + np.random.randn(n, 1) * 0.1

        f_stat, weak = tester.weak_instruments_test(X, Z)
        assert np.isfinite(f_stat)
        assert f_stat > 10
        assert bool(weak) is False

    def test_weak_instruments(self, tester):
        """Instruments unrelated to X should yield F < 10 and weak=True."""
        np.random.seed(42)
        n = 50
        L = 3

        Z = np.random.randn(n, L)
        # X is pure noise, uncorrelated with Z
        X = np.random.randn(n, 1) * 100

        f_stat, weak = tester.weak_instruments_test(X, Z)
        assert np.isfinite(f_stat)
        assert bool(weak) is True

    def test_multiple_endogenous_variables(self, tester):
        """With k > 1 endogenous variables, returns the minimum F-stat."""
        np.random.seed(42)
        n = 200
        L = 4

        Z = np.random.randn(n, L)
        # First endogenous variable strongly related to Z
        X1 = Z @ np.array([3.0, 1.0, -2.0, 0.5]) + np.random.randn(n) * 0.1
        # Second endogenous variable weakly related to Z
        X2 = np.random.randn(n) * 10 + Z[:, 0] * 0.01
        X = np.column_stack([X1, X2])

        f_stat, weak = tester.weak_instruments_test(X, Z)
        assert np.isfinite(f_stat)
        # The minimum should be driven by the weak second variable
        assert bool(weak) is True

    def test_nan_handling(self, tester):
        """NaN rows in X or Z are excluded before computing F-stat."""
        np.random.seed(42)
        n = 100
        L = 3

        Z = np.random.randn(n, L)
        beta_true = np.array([[2.0], [-1.0], [0.5]])
        X = Z @ beta_true + np.random.randn(n, 1) * 0.2

        # Insert NaNs
        X[5, 0] = np.nan
        Z[10, 1] = np.nan
        X[20, 0] = np.nan

        f_stat, _weak = tester.weak_instruments_test(X, Z)
        assert np.isfinite(f_stat)
        # Still strong instruments after removing 3 rows
        assert f_stat > 10

    def test_singular_ztz_uses_pinv(self, tester):
        """When Z'Z is exactly singular (zero matrix), pinv fallback is used (line 528)."""
        n = 50
        L = 3

        # Zero instrument matrix => Z'Z is zero matrix => inv raises LinAlgError
        Z = np.zeros((n, L))
        X = np.random.randn(n, 1)

        # Should not raise; should use pinv fallback
        f_stat, _weak = tester.weak_instruments_test(X, Z)
        # With zero instruments, the fit is meaningless but the code should not crash
        assert isinstance(float(f_stat), float)

    def test_r_squared_geq_one(self, tester):
        """When X is exactly Z*pi (no noise), r_squared >= 1 => f_stat = inf."""
        np.random.seed(42)
        n = 50
        L = 3

        Z = np.random.randn(n, L)
        # X is an exact linear combination of Z (perfect fit)
        pi_true = np.array([[1.0], [2.0], [3.0]])
        X = Z @ pi_true  # no noise at all

        f_stat, weak = tester.weak_instruments_test(X, Z)
        # Perfect fit => r_squared = 1.0 => f_stat = inf
        assert f_stat == np.inf
        assert bool(weak) is False

    def test_returns_tuple(self, tester):
        """Confirm the return type is (float, bool)."""
        np.random.seed(42)
        n = 60
        L = 3
        Z = np.random.randn(n, L)
        X = Z @ np.array([[1.0], [0.5], [-0.3]]) + np.random.randn(n, 1) * 0.5

        result = tester.weak_instruments_test(X, Z)
        assert isinstance(result, tuple)
        assert len(result) == 2
        f_stat, weak = result
        assert isinstance(float(f_stat), float)
        assert isinstance(weak, (bool, np.bool_))  # numpy may return np.bool_
