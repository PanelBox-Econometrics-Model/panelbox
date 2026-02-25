"""Tests for panelbox.utils.statistics module.

Covers: compute_sandwich_covariance, compute_cluster_robust_covariance,
likelihood_ratio_test, wald_test, hausman_test, compute_standard_errors,
compute_t_statistics, compute_p_values, compute_confidence_intervals,
compute_aic, compute_bic.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from panelbox.utils.statistics import (
    compute_aic,
    compute_bic,
    compute_cluster_robust_covariance,
    compute_confidence_intervals,
    compute_p_values,
    compute_sandwich_covariance,
    compute_standard_errors,
    compute_t_statistics,
    hausman_test,
    likelihood_ratio_test,
    wald_test,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regression_data(n=100, k=3, n_entities=10, seed=42):
    """Generate synthetic regression data with entity clustering."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    beta = rng.standard_normal(k)
    eps = rng.standard_normal(n)
    y = X @ beta + eps
    residuals = y - X @ beta  # == eps
    entity_id = np.repeat(np.arange(n_entities), n // n_entities)
    return X, y, beta, residuals, entity_id


# ===================================================================
# compute_sandwich_covariance
# ===================================================================


class TestComputeSandwichCovariance:
    """Tests for the sandwich (robust) covariance estimator."""

    def test_basic_sandwich_no_clustering(self):
        """Without clustering, B = G'G and vcov = H^{-1} G'G H^{-1}."""
        k = 2
        hessian = -np.eye(k) * 100  # Negative-definite Hessian
        # Gradient contributions: 5 observations, 2 params
        grads = np.array(
            [
                [1.0, 0.5],
                [-0.5, 1.0],
                [0.2, -0.3],
                [0.8, 0.1],
                [-0.1, 0.7],
            ]
        )

        vcov = compute_sandwich_covariance(hessian, grads)

        # Manual computation
        H_inv = np.linalg.inv(hessian)
        B = grads.T @ grads
        expected = H_inv @ B @ H_inv

        assert_allclose(vcov, expected, atol=1e-12)
        assert vcov.shape == (k, k)

    def test_sandwich_symmetric_positive_semidefinite(self):
        """Result should be symmetric and PSD (diagonal >= 0)."""
        X, _, _, _, _ = _make_regression_data(n=50, k=2)
        hessian = -(X.T @ X)
        grads = X * np.random.default_rng(1).standard_normal((50, 1))

        vcov = compute_sandwich_covariance(hessian, grads)

        assert_allclose(vcov, vcov.T, atol=1e-12)
        assert np.all(np.diag(vcov) >= -1e-12)

    def test_sandwich_with_entity_clustering(self):
        """With entity_id, gradients are summed within clusters."""
        k = 2
        hessian = -np.eye(k) * 50
        # 6 obs, 2 entities (3 each)
        grads = np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [-0.5, 1.0],
                [0.0, -1.0],
                [1.0, 0.5],
                [-1.0, 0.0],
            ]
        )
        entity_id = np.array([0, 0, 0, 1, 1, 1])

        vcov = compute_sandwich_covariance(hessian, grads, entity_id=entity_id)

        # Manual: clustered grads
        cg0 = grads[:3].sum(axis=0)  # [1.0, 1.5]
        cg1 = grads[3:].sum(axis=0)  # [0.0, -0.5]
        clustered = np.array([cg0, cg1])
        B = clustered.T @ clustered

        n, g = 6, 2
        correction = g / (g - 1) * n / (n - 1)
        B *= correction

        H_inv = np.linalg.inv(hessian)
        expected = H_inv @ B @ H_inv

        assert_allclose(vcov, expected, atol=1e-12)

    def test_sandwich_singular_hessian_uses_pinv(self):
        """When Hessian is singular, pseudo-inverse is used."""
        # Singular Hessian (rank 1)
        hessian = np.array([[1.0, 2.0], [2.0, 4.0]])
        grads = np.array([[1.0, 0.5], [0.5, 0.25]])

        # Should not raise
        vcov = compute_sandwich_covariance(hessian, grads)
        assert vcov.shape == (2, 2)
        assert np.all(np.isfinite(vcov))


# ===================================================================
# compute_cluster_robust_covariance
# ===================================================================


class TestComputeClusterRobustCovariance:
    """Tests for the cluster-robust covariance estimator."""

    def test_basic_cluster_robust(self):
        """Basic cluster-robust covariance with identity-like data."""
        X, _, _, residuals, entity_id = _make_regression_data(n=100, k=2, n_entities=10)

        vcov = compute_cluster_robust_covariance(residuals, X, entity_id)

        assert vcov.shape == (2, 2)
        assert_allclose(vcov, vcov.T, atol=1e-12)
        # Standard errors should be positive
        assert np.all(np.diag(vcov) > 0)

    def test_cluster_robust_with_vcov_base(self):
        """Providing vcov_base overrides (X'X)^{-1}."""
        X, _, _, residuals, entity_id = _make_regression_data(n=100, k=2, n_entities=10)
        vcov_base = np.eye(2) * 0.01

        vcov = compute_cluster_robust_covariance(residuals, X, entity_id, vcov_base=vcov_base)

        assert vcov.shape == (2, 2)
        assert np.all(np.isfinite(vcov))

    def test_cluster_robust_single_entity_per_obs(self):
        """Each observation is its own cluster (degenerate case)."""
        rng = np.random.default_rng(99)
        n, k = 20, 2
        X = rng.standard_normal((n, k))
        residuals = rng.standard_normal(n)
        entity_id = np.arange(n)  # Every obs is unique

        vcov = compute_cluster_robust_covariance(residuals, X, entity_id)

        assert vcov.shape == (k, k)
        assert np.all(np.isfinite(vcov))

    def test_cluster_robust_finite_sample_correction(self):
        """Verify the correction factor g/(g-1) * n/(n-k)."""
        n, k = 30, 2
        n_entities = 6
        rng = np.random.default_rng(7)
        X = rng.standard_normal((n, k))
        residuals = rng.standard_normal(n)
        entity_id = np.repeat(np.arange(n_entities), n // n_entities)

        vcov = compute_cluster_robust_covariance(residuals, X, entity_id)

        # Re-compute without correction
        XtX_inv = np.linalg.inv(X.T @ X)
        B = np.zeros((k, k))
        for eid in range(n_entities):
            mask = entity_id == eid
            score = X[mask].T @ residuals[mask]
            B += np.outer(score, score)

        correction = n_entities / (n_entities - 1) * n / (n - k)
        expected = correction * XtX_inv @ B @ XtX_inv
        assert_allclose(vcov, expected, atol=1e-12)


# ===================================================================
# likelihood_ratio_test
# ===================================================================


class TestLikelihoodRatioTest:
    """Tests for likelihood_ratio_test."""

    def test_basic_significant(self):
        """When unrestricted is much better, reject H0."""
        result = likelihood_ratio_test(llf_unrestricted=-50.0, llf_restricted=-60.0, df=2)

        assert result["statistic"] == pytest.approx(20.0)
        assert result["df"] == 2
        assert result["pvalue"] < 0.05
        assert result["conclusion"] == "Reject H0"
        assert result["llf_unrestricted"] == -50.0
        assert result["llf_restricted"] == -60.0

    def test_not_significant(self):
        """When models are similar, fail to reject H0."""
        result = likelihood_ratio_test(llf_unrestricted=-50.0, llf_restricted=-50.5, df=2)

        assert result["statistic"] == pytest.approx(1.0)
        assert result["pvalue"] > 0.05
        assert result["conclusion"] == "Fail to reject H0"

    def test_zero_statistic(self):
        """When both likelihoods are equal, LR = 0, p = 1."""
        result = likelihood_ratio_test(llf_unrestricted=-100.0, llf_restricted=-100.0, df=1)

        assert result["statistic"] == pytest.approx(0.0)
        assert result["pvalue"] == pytest.approx(1.0)

    def test_lr_statistic_formula(self):
        """LR = 2 * (llf_u - llf_r) matches chi-squared p-value."""
        llf_u, llf_r, df = -30.0, -35.0, 3
        result = likelihood_ratio_test(llf_u, llf_r, df)

        expected_stat = 2 * (llf_u - llf_r)
        expected_pvalue = 1 - stats.chi2.cdf(expected_stat, df)

        assert result["statistic"] == pytest.approx(expected_stat)
        assert result["pvalue"] == pytest.approx(expected_pvalue)

    def test_result_keys(self):
        """Result dict contains all expected keys."""
        result = likelihood_ratio_test(-50.0, -60.0, 2)
        expected_keys = {
            "statistic",
            "pvalue",
            "df",
            "conclusion",
            "llf_unrestricted",
            "llf_restricted",
        }
        assert set(result.keys()) == expected_keys


# ===================================================================
# wald_test
# ===================================================================


class TestWaldTest:
    """Tests for wald_test (linear restrictions)."""

    def test_single_restriction_reject(self):
        """Single restriction R*beta = 0 when beta is far from 0."""
        params = np.array([5.0, 3.0])
        vcov = np.eye(2)
        R = np.array([[1.0, 0.0]])  # Test beta_1 = 0

        result = wald_test(params, vcov, R)

        assert result["statistic"] == pytest.approx(25.0)
        assert result["df"] == 1
        assert result["pvalue"] < 0.001
        assert result["conclusion"] == "Reject H0"

    def test_single_restriction_fail_to_reject(self):
        """Single restriction with beta near 0."""
        params = np.array([0.1, 0.05])
        vcov = np.eye(2)
        R = np.array([[1.0, 0.0]])

        result = wald_test(params, vcov, R)

        assert result["statistic"] == pytest.approx(0.01)
        assert result["pvalue"] > 0.05
        assert result["conclusion"] == "Fail to reject H0"

    def test_with_nonzero_values(self):
        """Test H0: R*beta = q where q != 0."""
        params = np.array([5.0, 3.0])
        vcov = np.eye(2)
        R = np.array([[1.0, 0.0]])
        q = np.array([5.0])  # Test beta_1 = 5 (true value)

        result = wald_test(params, vcov, R, values=q)

        assert result["statistic"] == pytest.approx(0.0, abs=1e-12)
        assert result["pvalue"] == pytest.approx(1.0, abs=1e-6)

    def test_multiple_restrictions(self):
        """Joint restriction R*beta = 0 with 2 restrictions."""
        params = np.array([5.0, 3.0, 2.0])
        vcov = np.diag([1.0, 1.0, 1.0])
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        result = wald_test(params, vcov, R)

        # W = [5, 3] @ I @ [5, 3]' = 25 + 9 = 34
        assert result["statistic"] == pytest.approx(34.0)
        assert result["df"] == 2
        assert result["pvalue"] < 0.001

    def test_wald_chi2_distribution(self):
        """Verify p-value matches chi2 survival function."""
        params = np.array([2.0])
        vcov = np.array([[0.5]])
        R = np.array([[1.0]])

        result = wald_test(params, vcov, R)

        W = 2.0**2 / 0.5  # = 8.0
        expected_p = 1 - stats.chi2.cdf(W, 1)

        assert result["statistic"] == pytest.approx(W)
        assert result["pvalue"] == pytest.approx(expected_p)

    def test_1d_restriction_auto_converted(self):
        """A 1-D restriction vector is converted to 2-D via atleast_2d."""
        params = np.array([3.0, 1.0])
        vcov = np.eye(2)
        R = np.array([1.0, 0.0])  # 1-D

        result = wald_test(params, vcov, R)

        assert result["statistic"] == pytest.approx(9.0)
        assert result["df"] == 1


# ===================================================================
# hausman_test
# ===================================================================


class TestHausmanTest:
    """Tests for the Hausman FE vs RE test."""

    def test_reject_use_fe(self):
        """When FE and RE differ significantly, recommend FE."""
        params_fe = np.array([2.0, 1.0])
        params_re = np.array([1.0, 0.5])
        vcov_fe = np.eye(2) * 0.1
        vcov_re = np.eye(2) * 0.05

        result = hausman_test(params_fe, params_re, vcov_fe, vcov_re)

        assert result["df"] == 2
        assert result["pvalue"] < 0.05
        assert result["conclusion"] == "Use Fixed Effects"
        assert result["statistic"] > 0

    def test_fail_to_reject_re_consistent(self):
        """When FE and RE are similar, RE is consistent."""
        params_fe = np.array([1.01, 0.50])
        params_re = np.array([1.00, 0.49])
        vcov_fe = np.eye(2) * 0.5
        vcov_re = np.eye(2) * 0.3

        result = hausman_test(params_fe, params_re, vcov_fe, vcov_re)

        assert result["pvalue"] > 0.05
        assert result["conclusion"] == "Random Effects is consistent"

    def test_identical_params_zero_statistic(self):
        """When FE == RE, H = 0 and p = 1."""
        params = np.array([1.0, 2.0])
        vcov_fe = np.eye(2) * 0.1
        vcov_re = np.eye(2) * 0.05

        result = hausman_test(params, params, vcov_fe, vcov_re)

        assert result["statistic"] == pytest.approx(0.0, abs=1e-12)
        assert result["pvalue"] == pytest.approx(1.0)

    def test_hausman_statistic_formula(self):
        """Manual calculation: H = (b_fe - b_re)' (V_fe - V_re)^{-1} (b_fe - b_re)."""
        params_fe = np.array([3.0])
        params_re = np.array([1.0])
        vcov_fe = np.array([[2.0]])
        vcov_re = np.array([[0.5]])

        result = hausman_test(params_fe, params_re, vcov_fe, vcov_re)

        b_diff = 3.0 - 1.0  # = 2.0
        v_diff = 2.0 - 0.5  # = 1.5
        expected_H = b_diff**2 / v_diff  # = 4/1.5 = 2.6667
        expected_p = 1 - stats.chi2.cdf(expected_H, df=1)

        assert result["statistic"] == pytest.approx(expected_H)
        assert result["pvalue"] == pytest.approx(expected_p)

    def test_result_keys(self):
        """Result dict has expected keys."""
        result = hausman_test(
            np.array([1.0]), np.array([0.5]), np.array([[1.0]]), np.array([[0.5]])
        )
        assert set(result.keys()) == {"statistic", "pvalue", "df", "conclusion"}


# ===================================================================
# compute_standard_errors
# ===================================================================


class TestComputeStandardErrors:
    """Tests for compute_standard_errors."""

    def test_identity_covariance(self):
        """SE from identity matrix = [1, 1, ...]."""
        vcov = np.eye(3)
        se = compute_standard_errors(vcov)
        assert_allclose(se, np.ones(3))

    def test_diagonal_covariance(self):
        """SE = sqrt(diag(V))."""
        vcov = np.diag([4.0, 9.0, 16.0])
        se = compute_standard_errors(vcov)
        assert_allclose(se, [2.0, 3.0, 4.0])

    def test_non_diagonal_covariance(self):
        """Off-diagonal elements are ignored."""
        vcov = np.array([[4.0, 1.0], [1.0, 9.0]])
        se = compute_standard_errors(vcov)
        assert_allclose(se, [2.0, 3.0])

    def test_returns_ndarray(self):
        """Return type is ndarray."""
        se = compute_standard_errors(np.eye(2))
        assert isinstance(se, np.ndarray)


# ===================================================================
# compute_t_statistics
# ===================================================================


class TestComputeTStatistics:
    """Tests for compute_t_statistics."""

    def test_null_zero(self):
        """Default null is zero: t = params / se."""
        params = np.array([3.0, -2.0])
        se = np.array([1.0, 0.5])
        t = compute_t_statistics(params, se)
        assert_allclose(t, [3.0, -4.0])

    def test_custom_null(self):
        """With non-zero null: t = (params - null) / se."""
        params = np.array([5.0, 3.0])
        se = np.array([1.0, 1.0])
        null_values = np.array([5.0, 1.0])
        t = compute_t_statistics(params, se, null_values=null_values)
        assert_allclose(t, [0.0, 2.0])

    def test_zero_params(self):
        """When params == null (zero), t = 0."""
        params = np.zeros(3)
        se = np.ones(3)
        t = compute_t_statistics(params, se)
        assert_allclose(t, np.zeros(3))

    def test_large_t_statistics(self):
        """Very small SE leads to large t-stats."""
        params = np.array([1.0])
        se = np.array([0.001])
        t = compute_t_statistics(params, se)
        assert_allclose(t, [1000.0])


# ===================================================================
# compute_p_values
# ===================================================================


class TestComputePValues:
    """Tests for compute_p_values."""

    def test_normal_distribution_no_df(self):
        """Without df, uses standard normal for p-values."""
        t_stats = np.array([0.0, 1.96, -1.96])
        p = compute_p_values(t_stats)

        # t = 0 => p = 1.0
        assert p[0] == pytest.approx(1.0)
        # t = +/-1.96 => p ~ 0.05
        assert p[1] == pytest.approx(0.05, abs=0.001)
        assert p[2] == pytest.approx(0.05, abs=0.001)

    def test_t_distribution_with_df(self):
        """With df, uses t-distribution."""
        t_stats = np.array([2.0])

        p_normal = compute_p_values(t_stats, df=None)
        p_t_small = compute_p_values(t_stats, df=5)
        p_t_large = compute_p_values(t_stats, df=1000)

        # Small df => heavier tails => larger p-value
        assert p_t_small[0] > p_t_large[0]
        # Large df t-dist approaches normal
        assert p_t_large[0] == pytest.approx(p_normal[0], abs=0.005)

    def test_symmetric_p_values(self):
        """Positive and negative t give same two-sided p-value."""
        t_pos = np.array([3.0])
        t_neg = np.array([-3.0])

        p_pos = compute_p_values(t_pos, df=20)
        p_neg = compute_p_values(t_neg, df=20)

        assert_allclose(p_pos, p_neg)

    def test_zero_t_statistic(self):
        """t = 0 gives p = 1."""
        p = compute_p_values(np.array([0.0]))
        assert p[0] == pytest.approx(1.0)

    def test_very_large_t_gives_small_p(self):
        """Very large |t| gives p near 0."""
        p = compute_p_values(np.array([10.0]))
        assert p[0] < 1e-10

    def test_p_values_between_0_and_1(self):
        """P-values are always in [0, 1]."""
        rng = np.random.default_rng(0)
        t_stats = rng.standard_normal(100)
        p = compute_p_values(t_stats, df=30)
        assert np.all(p >= 0)
        assert np.all(p <= 1)


# ===================================================================
# compute_confidence_intervals
# ===================================================================


class TestComputeConfidenceIntervals:
    """Tests for compute_confidence_intervals."""

    def test_95ci_normal(self):
        """95% CI with normal distribution: params +/- 1.96 * SE."""
        params = np.array([1.0, 2.0])
        se = np.array([0.5, 1.0])

        lower, upper = compute_confidence_intervals(params, se, alpha=0.05)

        z = stats.norm.ppf(0.975)
        assert_allclose(lower, params - z * se, atol=1e-10)
        assert_allclose(upper, params + z * se, atol=1e-10)

    def test_95ci_t_distribution(self):
        """95% CI with t-distribution uses t critical value."""
        params = np.array([3.0])
        se = np.array([1.0])
        df = 10

        lower, upper = compute_confidence_intervals(params, se, alpha=0.05, df=df)

        t_crit = stats.t.ppf(0.975, df)
        assert_allclose(lower, [3.0 - t_crit])
        assert_allclose(upper, [3.0 + t_crit])

    def test_99ci_wider_than_95ci(self):
        """99% CI is wider than 95% CI."""
        params = np.array([1.0])
        se = np.array([1.0])

        lower95, upper95 = compute_confidence_intervals(params, se, alpha=0.05)
        lower99, upper99 = compute_confidence_intervals(params, se, alpha=0.01)

        assert lower99[0] < lower95[0]
        assert upper99[0] > upper95[0]

    def test_ci_symmetric_around_params(self):
        """CI is symmetric: params - lower == upper - params."""
        params = np.array([5.0, -3.0])
        se = np.array([2.0, 1.0])

        lower, upper = compute_confidence_intervals(params, se, alpha=0.05)

        assert_allclose(params - lower, upper - params, atol=1e-12)

    def test_zero_se_gives_point_interval(self):
        """When SE = 0, CI collapses to the point estimate."""
        params = np.array([2.0])
        se = np.array([0.0])

        lower, upper = compute_confidence_intervals(params, se)

        assert_allclose(lower, [2.0])
        assert_allclose(upper, [2.0])

    def test_ci_with_small_df(self):
        """Small df gives wider CI than large df."""
        params = np.array([1.0])
        se = np.array([1.0])

        _, upper_small = compute_confidence_intervals(params, se, df=3)
        _, upper_large = compute_confidence_intervals(params, se, df=1000)

        assert upper_small[0] > upper_large[0]

    def test_return_tuple_of_arrays(self):
        """Returns a tuple of two ndarrays."""
        lower, upper = compute_confidence_intervals(np.array([1.0]), np.array([0.5]))
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)


# ===================================================================
# compute_aic
# ===================================================================


class TestComputeAIC:
    """Tests for AIC = 2k - 2*ln(L)."""

    def test_known_values(self):
        """AIC = 2*3 - 2*(-100) = 6 + 200 = 206."""
        aic = compute_aic(llf=-100.0, n_params=3)
        assert aic == pytest.approx(206.0)

    def test_zero_log_likelihood(self):
        """AIC = 2k when log-likelihood is 0."""
        aic = compute_aic(llf=0.0, n_params=5)
        assert aic == pytest.approx(10.0)

    def test_more_params_higher_aic(self):
        """With same log-likelihood, more params gives higher AIC."""
        aic_small = compute_aic(llf=-50.0, n_params=2)
        aic_large = compute_aic(llf=-50.0, n_params=10)
        assert aic_large > aic_small

    def test_better_fit_lower_aic(self):
        """With same params, higher log-likelihood gives lower AIC."""
        aic_bad = compute_aic(llf=-200.0, n_params=3)
        aic_good = compute_aic(llf=-100.0, n_params=3)
        assert aic_good < aic_bad

    def test_positive_log_likelihood(self):
        """AIC can handle positive log-likelihood."""
        aic = compute_aic(llf=10.0, n_params=2)
        # 2*2 - 2*10 = 4 - 20 = -16
        assert aic == pytest.approx(-16.0)


# ===================================================================
# compute_bic
# ===================================================================


class TestComputeBIC:
    """Tests for BIC = k*ln(n) - 2*ln(L)."""

    def test_known_values(self):
        """BIC = 3*ln(100) - 2*(-100) = 3*4.605... + 200."""
        bic = compute_bic(llf=-100.0, n_params=3, n_obs=100)
        expected = 3 * np.log(100) + 200
        assert bic == pytest.approx(expected)

    def test_bic_penalizes_more_than_aic_for_large_n(self):
        """For n > ~8, BIC penalizes parameters more than AIC."""
        llf = -50.0
        k = 5
        n = 100

        aic = compute_aic(llf, k)
        bic = compute_bic(llf, k, n)

        # BIC penalty per param = ln(n) = ln(100) ~ 4.6 vs AIC penalty = 2
        assert bic > aic

    def test_bic_increases_with_n_obs(self):
        """More observations increase BIC penalty."""
        bic_small = compute_bic(llf=-50.0, n_params=3, n_obs=10)
        bic_large = compute_bic(llf=-50.0, n_params=3, n_obs=1000)
        assert bic_large > bic_small

    def test_bic_with_one_obs(self):
        """n_obs=1 => ln(1) = 0 => BIC = -2*llf."""
        bic = compute_bic(llf=-50.0, n_params=3, n_obs=1)
        # 3*ln(1) - 2*(-50) = 0 + 100 = 100
        assert bic == pytest.approx(100.0)

    def test_bic_formula(self):
        """Verify exact formula: k*ln(n) - 2*llf."""
        llf, k, n = -75.0, 4, 200
        bic = compute_bic(llf, k, n)
        expected = k * np.log(n) - 2 * llf
        assert bic == pytest.approx(expected)


# ===================================================================
# Integration tests
# ===================================================================


class TestStatisticsIntegration:
    """End-to-end integration tests combining multiple functions."""

    def test_params_to_inference_pipeline(self):
        """Full pipeline: params -> se -> t -> p -> CI."""
        params = np.array([2.5, -1.0, 0.3])
        vcov = np.diag([0.25, 0.16, 0.09])

        se = compute_standard_errors(vcov)
        assert_allclose(se, [0.5, 0.4, 0.3])

        t = compute_t_statistics(params, se)
        assert_allclose(t, [5.0, -2.5, 1.0])

        p = compute_p_values(t, df=50)
        # First param is highly significant, third is not
        assert p[0] < 0.001
        assert p[2] > 0.05

        lower, upper = compute_confidence_intervals(params, se, df=50)
        # Params should lie within their CI
        for i in range(3):
            assert lower[i] < params[i] < upper[i]

    def test_aic_bic_model_selection(self):
        """AIC and BIC both prefer the model with better fit."""
        # Model A: better fit, fewer params
        aic_a = compute_aic(-100.0, 3)
        bic_a = compute_bic(-100.0, 3, 200)

        # Model B: worse fit, more params
        aic_b = compute_aic(-110.0, 5)
        bic_b = compute_bic(-110.0, 5, 200)

        # Model A should be preferred (lower AIC/BIC)
        assert aic_a < aic_b
        assert bic_a < bic_b

    def test_lr_and_wald_same_direction(self):
        """LR test and Wald test should agree on direction of significance
        for a simple restriction."""
        # LR test: unrestricted is much better
        lr_result = likelihood_ratio_test(-50.0, -60.0, df=1)

        # Wald test: param far from 0
        params = np.array([4.47])  # sqrt(20) to match LR stat ~20
        vcov = np.array([[1.0]])
        R = np.array([[1.0]])
        wald_result = wald_test(params, vcov, R)

        # Both should reject H0
        assert lr_result["pvalue"] < 0.05
        assert wald_result["pvalue"] < 0.05

    def test_sandwich_produces_valid_se(self):
        """Sandwich covariance -> standard errors -> t-stats -> p-values."""
        k = 2
        hessian = -np.eye(k) * 100
        grads = np.random.default_rng(42).standard_normal((50, k))

        vcov = compute_sandwich_covariance(hessian, grads)
        se = compute_standard_errors(vcov)
        assert np.all(se > 0)

        params = np.array([0.5, -0.3])
        t = compute_t_statistics(params, se)
        p = compute_p_values(t)
        assert np.all((p >= 0) & (p <= 1))
