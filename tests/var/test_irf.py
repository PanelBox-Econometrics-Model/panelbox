"""
Tests for additional coverage of panelbox.var.irf module.

Covers:
- IRFResult.__getitem__() with valid and invalid keys (lines 132-147)
- compute_irf_cholesky() with near-singular Sigma (regularization, lines 369-378)
- compute_analytical_ci() function (lines 703-753)
- compute_cumulative_irf() (line 530-549)
"""

import warnings

import numpy as np
import pytest

from panelbox.var.irf import (
    IRFResult,
    compute_analytical_ci,
    compute_cumulative_irf,
    compute_irf_cholesky,
)


@pytest.fixture
def simple_irf_result():
    """Create a simple IRFResult for testing __getitem__."""
    np.random.seed(42)
    periods = 5
    var_names = ["y1", "y2"]

    A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

    irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods)

    return IRFResult(
        irf_matrix=irf_matrix,
        var_names=var_names,
        periods=periods,
        method="cholesky",
    )


class TestIRFResultGetItem:
    """Test IRFResult.__getitem__() with valid and invalid keys (lines 132-147)."""

    def test_getitem_valid_tuple_key(self, simple_irf_result):
        """Test accessing IRF with valid (response, impulse) tuple."""
        irf = simple_irf_result

        # Access y1's response to y2 shock
        response = irf["y1", "y2"]
        assert response.shape == (6,)  # periods+1
        assert np.allclose(response, irf.irf_matrix[:, 0, 1])

    def test_getitem_all_combinations(self, simple_irf_result):
        """Test all four variable-pair combinations."""
        irf = simple_irf_result

        for i, resp in enumerate(["y1", "y2"]):
            for j, imp in enumerate(["y1", "y2"]):
                result = irf[resp, imp]
                assert result.shape == (6,)
                assert np.allclose(result, irf.irf_matrix[:, i, j])

    def test_getitem_not_tuple_raises_keyerror(self, simple_irf_result):
        """Test that non-tuple key raises KeyError (line 132-133)."""
        with pytest.raises(KeyError, match="Key must be a tuple"):
            simple_irf_result["y1"]

    def test_getitem_wrong_length_tuple_raises_keyerror(self, simple_irf_result):
        """Test that tuple of wrong length raises KeyError."""
        with pytest.raises(KeyError, match="Key must be a tuple"):
            simple_irf_result["y1", "y2", "y3"]

    def test_getitem_invalid_response_var(self, simple_irf_result):
        """Test that invalid response variable raises KeyError (lines 139-140)."""
        with pytest.raises(KeyError, match="Response variable 'nonexistent' not found"):
            simple_irf_result["nonexistent", "y1"]

    def test_getitem_invalid_impulse_var(self, simple_irf_result):
        """Test that invalid impulse variable raises KeyError (lines 143-145)."""
        with pytest.raises(KeyError, match="Impulse variable 'nonexistent' not found"):
            simple_irf_result["y1", "nonexistent"]


class TestComputeIRFCholeskyRegularization:
    """Test compute_irf_cholesky() when Sigma is not positive definite (lines 369-378)."""

    def test_near_singular_sigma_triggers_regularization(self):
        """Test that near-singular Sigma triggers regularization warning."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]

        # Create a Sigma that is *barely* not positive definite.
        # The smallest eigenvalue will be -1e-9, so adding 1e-8 * I
        # makes it positive definite.
        # Start from PD matrix and perturb slightly
        # eigenvalues of [[1, r],[r, 1]] are 1+r and 1-r
        # For r = 1 + 1e-9: eigenvalues are 2+1e-9 and -1e-9
        r = 1.0 + 1e-9
        Sigma_bad = np.array([[1.0, r], [r, 1.0]])

        # Verify it's not PD
        eigenvalues = np.linalg.eigvalsh(Sigma_bad)
        assert np.any(eigenvalues < 0), "Sigma should not be positive definite"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Phi = compute_irf_cholesky(A_matrices, Sigma_bad, periods=5)

            # Should have issued a warning
            regularization_warnings = [x for x in w if "not positive definite" in str(x.message)]
            assert len(regularization_warnings) > 0, "Should warn about non-positive-definite Sigma"

        # Result should still be computed (regularized)
        assert Phi.shape == (6, 2, 2)
        assert np.all(np.isfinite(Phi))

    def test_positive_definite_sigma_no_warning(self):
        """Test that positive definite Sigma does not trigger warning."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Phi = compute_irf_cholesky(A_matrices, Sigma, periods=5)

            regularization_warnings = [x for x in w if "not positive definite" in str(x.message)]
            assert len(regularization_warnings) == 0

        assert Phi.shape == (6, 2, 2)


class TestComputeAnalyticalCI:
    """Test compute_analytical_ci() function (lines 703-753)."""

    def test_analytical_ci_cholesky_basic(self):
        """Test basic analytical CI computation with Cholesky method."""
        np.random.seed(42)
        K = 2
        p = 1
        periods = 5

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

        # Create a covariance matrix for parameter estimates
        n_params = K * p * K  # 2*1*2 = 4
        cov_params = np.eye(n_params) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices=A_matrices,
            Sigma=Sigma,
            cov_params=cov_params,
            periods=periods,
            method="cholesky",
            ci_level=0.95,
        )

        # Check shapes
        assert ci_lower.shape == (periods + 1, K, K)
        assert ci_upper.shape == (periods + 1, K, K)

        # Lower should be <= upper everywhere
        assert np.all(ci_lower <= ci_upper + 1e-10)

    def test_analytical_ci_generalized(self):
        """Test analytical CI with generalized method."""
        np.random.seed(42)
        K = 2
        periods = 5

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices=A_matrices,
            Sigma=Sigma,
            cov_params=cov_params,
            periods=periods,
            method="generalized",
            ci_level=0.95,
        )

        assert ci_lower.shape == (periods + 1, K, K)
        assert ci_upper.shape == (periods + 1, K, K)
        assert np.all(ci_lower <= ci_upper + 1e-10)

    def test_analytical_ci_wider_at_higher_confidence(self):
        """Test that wider confidence level produces wider intervals."""
        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.01

        ci_lower_90, ci_upper_90 = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=5,
            ci_level=0.90,
        )
        ci_lower_99, ci_upper_99 = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=5,
            ci_level=0.99,
        )

        width_90 = ci_upper_90 - ci_lower_90
        width_99 = ci_upper_99 - ci_lower_99

        # 99% intervals should be wider than 90% intervals
        assert np.all(width_99 >= width_90 - 1e-10)

    def test_analytical_ci_cumulative(self):
        """Test analytical CI with cumulative=True."""
        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=5,
            cumulative=True,
        )

        assert ci_lower.shape == (6, 2, 2)
        assert ci_upper.shape == (6, 2, 2)
        assert np.all(ci_lower <= ci_upper + 1e-10)

    def test_analytical_ci_wider_at_longer_horizons(self):
        """Test that analytical CI uncertainty grows with horizon."""
        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=20,
            ci_level=0.95,
        )

        width = ci_upper - ci_lower
        # Width at h=0 should be <= width at h=20 (uncertainty grows)
        for i in range(2):
            for j in range(2):
                assert width[0, i, j] <= width[20, i, j] + 1e-10, (
                    f"CI width should grow with horizon for [{i},{j}]"
                )


class TestCumulativeIRF:
    """Test compute_cumulative_irf() (lines 530-549)."""

    def test_cumulative_irf_shape(self):
        """Test that cumulative IRF has same shape as input."""
        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)

        irf = compute_irf_cholesky(A_matrices, Sigma, periods=10)
        cumulative = compute_cumulative_irf(irf)

        assert cumulative.shape == irf.shape

    def test_cumulative_irf_matches_cumsum(self):
        """Test that cumulative IRF matches np.cumsum."""
        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)

        irf = compute_irf_cholesky(A_matrices, Sigma, periods=10)
        cumulative = compute_cumulative_irf(irf)
        expected = np.cumsum(irf, axis=0)

        assert np.allclose(cumulative, expected)

    def test_cumulative_irf_h0_equals_regular(self):
        """Test that at h=0, cumulative equals regular IRF."""
        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)

        irf = compute_irf_cholesky(A_matrices, Sigma, periods=5)
        cumulative = compute_cumulative_irf(irf)

        assert np.allclose(cumulative[0], irf[0])

    def test_cumulative_irf_monotonic_for_positive_irf(self):
        """Test cumulative IRF is monotonically increasing when all IRF values are positive."""
        # Use identity matrices to guarantee positive IRFs on diagonal
        A_matrices = [np.array([[0.3, 0.0], [0.0, 0.3]])]
        Sigma = np.eye(2)

        irf = compute_irf_cholesky(A_matrices, Sigma, periods=10)
        cumulative = compute_cumulative_irf(irf)

        # For diagonal elements (own shock), all IRF values are positive
        # so cumulative should be monotonically increasing
        for h in range(1, 11):
            for i in range(2):
                assert cumulative[h, i, i] >= cumulative[h - 1, i, i] - 1e-10


class TestIRFCoverage:
    """
    Additional tests to improve coverage for panelbox/var/irf.py.

    Targets uncovered lines:
    - 139-145 (__getitem__ errors for invalid response/impulse vars)
    - 369-378 (Sigma not positive definite warning and regularization)
    - 703-753 (compute_analytical_ci)
    - compute_cumulative_irf
    - IRFResult.to_dataframe with impulse/response filters
    """

    # ---------------------------------------------------------------
    # IRFResult.__getitem__ with invalid keys
    # ---------------------------------------------------------------
    def test_getitem_not_tuple_raises(self):
        """Test that a non-tuple key raises KeyError (line 132-133)."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        with pytest.raises(KeyError, match="Key must be a tuple"):
            result["y1"]

    def test_getitem_wrong_length_tuple_raises(self):
        """Test that a tuple of length != 2 raises KeyError."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        with pytest.raises(KeyError, match="Key must be a tuple"):
            result[("y1",)]

        with pytest.raises(KeyError, match="Key must be a tuple"):
            result[("y1", "y2", "y3")]

    def test_getitem_invalid_response_var_raises(self):
        """Test that unknown response variable raises KeyError (line 139-140)."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        with pytest.raises(KeyError, match="Response variable 'bad' not found"):
            result["bad", "y1"]

    def test_getitem_invalid_impulse_var_raises(self):
        """Test that unknown impulse variable raises KeyError (line 143-145)."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        with pytest.raises(KeyError, match="Impulse variable 'bad' not found"):
            result["y1", "bad"]

    # ---------------------------------------------------------------
    # compute_irf_cholesky() with non-positive-definite Sigma
    # ---------------------------------------------------------------
    def test_non_pd_sigma_warns_and_regularizes(self):
        """Test that non-PD Sigma triggers a warning and still computes (lines 369-378)."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]

        # Create Sigma that is barely NOT positive definite.
        # eigenvalues of [[1, r], [r, 1]] are 1+r and 1-r.
        # For r = 1 + 1e-9: eigenvalues are 2+1e-9 and -1e-9.
        # Adding 1e-8 * I makes the smallest eigenvalue 1e-8 - 1e-9 > 0.
        r = 1.0 + 1e-9
        Sigma_bad = np.array([[1.0, r], [r, 1.0]])
        eigenvalues = np.linalg.eigvalsh(Sigma_bad)
        assert np.any(eigenvalues < 0), "Sigma should not be positive definite"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            irf = compute_irf_cholesky(A_matrices, Sigma_bad, periods=5)

            reg_warnings = [x for x in w if "not positive definite" in str(x.message)]
            assert len(reg_warnings) > 0, "Should warn about non-positive-definite Sigma"

        # Result should still be valid
        assert irf.shape == (6, 2, 2)
        assert np.all(np.isfinite(irf))

    def test_pd_sigma_no_warning(self):
        """Test that valid PD Sigma does not issue a warning."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            irf = compute_irf_cholesky(A_matrices, Sigma, periods=5)

            reg_warnings = [x for x in w if "not positive definite" in str(x.message)]
            assert len(reg_warnings) == 0

        assert irf.shape == (6, 2, 2)

    # ---------------------------------------------------------------
    # compute_analytical_ci()
    # ---------------------------------------------------------------
    def test_analytical_ci_cholesky(self):
        """Test compute_analytical_ci with Cholesky method (lines 703-753)."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.01
        periods = 5

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=periods,
            method="cholesky",
            ci_level=0.95,
        )

        assert ci_lower.shape == (periods + 1, 2, 2)
        assert ci_upper.shape == (periods + 1, 2, 2)
        assert np.all(ci_lower <= ci_upper + 1e-10)

    def test_analytical_ci_generalized(self):
        """Test compute_analytical_ci with generalized method."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=5,
            method="generalized",
            ci_level=0.95,
        )

        assert ci_lower.shape == (6, 2, 2)
        assert ci_upper.shape == (6, 2, 2)
        assert np.all(ci_lower <= ci_upper + 1e-10)

    def test_analytical_ci_cumulative(self):
        """Test compute_analytical_ci with cumulative=True."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=5,
            method="cholesky",
            ci_level=0.90,
            cumulative=True,
        )

        assert ci_lower.shape == (6, 2, 2)
        assert ci_upper.shape == (6, 2, 2)
        assert np.all(ci_lower <= ci_upper + 1e-10)

    def test_analytical_ci_width_grows_with_horizon(self):
        """Test that CI width grows with horizon."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.1

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=10,
            method="cholesky",
            ci_level=0.95,
        )

        width = ci_upper - ci_lower
        # Width at h=0 should be <= width at h=10
        for i in range(2):
            for j in range(2):
                assert width[0, i, j] <= width[10, i, j] + 1e-10

    def test_analytical_ci_wider_confidence_wider_interval(self):
        """Test that higher confidence level produces wider intervals."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.01

        ci_lower_90, ci_upper_90 = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=5,
            ci_level=0.90,
        )
        ci_lower_99, ci_upper_99 = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=5,
            ci_level=0.99,
        )

        width_90 = ci_upper_90 - ci_lower_90
        width_99 = ci_upper_99 - ci_lower_99
        assert np.all(width_99 >= width_90 - 1e-10)

    # ---------------------------------------------------------------
    # compute_cumulative_irf()
    # ---------------------------------------------------------------
    def test_cumulative_irf_basic(self):
        """Test compute_cumulative_irf returns cumsum along axis 0."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)
        irf = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        cumulative = compute_cumulative_irf(irf)
        expected = np.cumsum(irf, axis=0)

        assert np.allclose(cumulative, expected)
        assert cumulative.shape == irf.shape

    def test_cumulative_irf_h0_matches_original(self):
        """Test that cumulative IRF at h=0 equals regular IRF at h=0."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)
        irf = compute_irf_cholesky(A_matrices, Sigma, periods=5)
        cumulative = compute_cumulative_irf(irf)

        assert np.allclose(cumulative[0], irf[0])

    # ---------------------------------------------------------------
    # IRFResult.to_dataframe() with impulse/response filters
    # ---------------------------------------------------------------
    def test_to_dataframe_impulse_filter(self):
        """Test to_dataframe with impulse filter."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        df = result.to_dataframe(impulse="y1")
        assert "horizon" in df.columns
        # Should have columns for responses of y1 and y2 to y1
        assert "y1\u2190y1" in df.columns
        assert "y2\u2190y1" in df.columns
        assert df.shape[0] == 6  # periods + 1

    def test_to_dataframe_response_filter(self):
        """Test to_dataframe with response filter."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        df = result.to_dataframe(response="y2")
        assert "horizon" in df.columns
        assert "y2\u2190y1" in df.columns
        assert "y2\u2190y2" in df.columns
        assert df.shape[0] == 6

    def test_to_dataframe_impulse_and_response_filter(self):
        """Test to_dataframe with both impulse and response filters."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        df = result.to_dataframe(impulse="y1", response="y2")
        assert "horizon" in df.columns
        assert "y2\u2190y1" in df.columns
        assert df.shape[0] == 6
        # Should have exactly 2 columns: horizon and the single IRF
        assert df.shape[1] == 2

    def test_to_dataframe_no_filter(self):
        """Test to_dataframe with no filter returns all IRFs."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        df = result.to_dataframe()
        assert df.shape[0] == 6  # periods + 1
        # 1 horizon column + 4 IRF columns (2x2)
        assert df.shape[1] == 5

    # ---------------------------------------------------------------
    # IRFResult repr and summary
    # ---------------------------------------------------------------
    def test_repr(self):
        """Test IRFResult __repr__ method."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods=5)

        result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=["y1", "y2"],
            periods=5,
            method="cholesky",
        )

        repr_str = repr(result)
        assert "IRFResult" in repr_str
        assert "K=2" in repr_str
        assert "periods=5" in repr_str

    def test_irf_cholesky_with_numeric_shock_size(self):
        """Test compute_irf_cholesky with numeric shock_size."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

        irf_default = compute_irf_cholesky(A_matrices, Sigma, periods=5)
        irf_scaled = compute_irf_cholesky(
            A_matrices,
            Sigma,
            periods=5,
            shock_size=2.0,
        )

        # Scaled shock should be 2x the default at h=0
        assert np.allclose(irf_scaled[0], irf_default[0] * 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
