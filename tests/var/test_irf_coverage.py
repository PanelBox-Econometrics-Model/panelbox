"""
Additional IRF coverage tests.

Covers uncovered lines in panelbox/var/irf.py:
- IRFResult.__getitem__() error paths (lines 132-145)
- Cholesky regularization for non-PD Sigma (lines 369-378)
- compute_analytical_ci() function (lines 703-753)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from panelbox.var import PanelVAR, PanelVARData
from panelbox.var.irf import compute_analytical_ci, compute_irf_cholesky


@pytest.fixture
def simple_var_data():
    """
    Create simple VAR(1) data for testing.

    DGP:
    y1_t = 0.5 * y1_{t-1} + 0.2 * y2_{t-1} + e1_t
    y2_t = 0.1 * y1_{t-1} + 0.6 * y2_{t-1} + e2_t

    with Cov(e1, e2) = [[1.0, 0.3], [0.3, 0.8]]
    """
    np.random.seed(42)

    A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 0.8]])

    N = 30
    T = 50
    K = 2

    data_list = []

    for i in range(N):
        errors = np.random.multivariate_normal(np.zeros(K), Sigma, size=T + 20)

        y = np.zeros((T + 20, K))
        for t in range(1, T + 20):
            y[t] = A1 @ y[t - 1] + errors[t]

        y = y[20:]

        df_entity = pd.DataFrame(
            {
                "entity": i,
                "time": range(T),
                "y1": y[:, 0],
                "y2": y[:, 1],
            }
        )
        data_list.append(df_entity)

    df = pd.concat(data_list, ignore_index=True)

    return df, A1, Sigma


class TestIRFResultGetItem:
    """Tests for IRFResult.__getitem__ error paths to cover lines 132-145."""

    def test_getitem_non_tuple_key(self, simple_var_data):
        """Test that a non-tuple key raises KeyError."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")
        irf_result = result.irf(periods=10, method="cholesky")

        # Single string key (not a tuple)
        with pytest.raises(KeyError, match="Key must be a tuple"):
            irf_result["y1"]

    def test_getitem_invalid_response_var(self, simple_var_data):
        """Test that an invalid response variable raises KeyError."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")
        irf_result = result.irf(periods=10, method="cholesky")

        with pytest.raises(KeyError, match="Response variable 'nonexistent' not found"):
            irf_result["nonexistent", "y1"]

    def test_getitem_invalid_impulse_var(self, simple_var_data):
        """Test that an invalid impulse variable raises KeyError."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")
        irf_result = result.irf(periods=10, method="cholesky")

        with pytest.raises(KeyError, match="Impulse variable 'nonexistent' not found"):
            irf_result["y1", "nonexistent"]

    def test_getitem_wrong_tuple_length(self, simple_var_data):
        """Test that a tuple with wrong length raises KeyError."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")
        irf_result = result.irf(periods=10, method="cholesky")

        with pytest.raises(KeyError, match="Key must be a tuple"):
            irf_result["y1", "y2", "y3"]

    def test_getitem_valid_pair(self, simple_var_data):
        """Test that a valid variable pair returns the correct IRF slice."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")
        irf_result = result.irf(periods=10, method="cholesky")

        # Access IRF: response of y1 to shock in y2
        irf_y1_y2 = irf_result["y1", "y2"]
        assert irf_y1_y2.shape == (11,)
        # Should match irf_matrix[:, response_idx=0, impulse_idx=1]
        assert np.allclose(irf_y1_y2, irf_result.irf_matrix[:, 0, 1])

        # Access IRF: response of y2 to shock in y1
        irf_y2_y1 = irf_result["y2", "y1"]
        assert irf_y2_y1.shape == (11,)
        assert np.allclose(irf_y2_y1, irf_result.irf_matrix[:, 1, 0])


class TestIRFCholeskyRegularization:
    """Tests for regularization when Cholesky fails to cover lines 369-378."""

    def test_non_positive_definite_sigma_warns(self):
        """Test that non-PD Sigma triggers a warning and regularization."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        # Build a matrix that is barely not PD (smallest eigenvalue ~ -1e-10).
        # The regularization adds 1e-8 * I which will make it PD.
        Sigma_bad = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1, PSD but not PD
        # Subtract a tiny bit to make it non-PSD
        Sigma_bad[1, 1] -= 1e-9

        with pytest.warns(UserWarning, match="not positive definite"):
            irf = compute_irf_cholesky(A_matrices, Sigma_bad, periods=5)

        # Should still return valid IRFs after regularization
        assert irf.shape == (6, 2, 2)
        # All values should be finite
        assert np.all(np.isfinite(irf))

    def test_positive_definite_sigma_no_warning(self):
        """Test that a PD Sigma does NOT trigger a warning."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma_good = np.array([[1.0, 0.3], [0.3, 1.0]])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            irf = compute_irf_cholesky(A_matrices, Sigma_good, periods=5)

        assert irf.shape == (6, 2, 2)

    def test_nearly_singular_sigma(self):
        """Test Cholesky IRF with a nearly singular but still PD Sigma."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        # Nearly singular (det close to 0 but still PD)
        Sigma_near = np.array([[1.0, 0.999], [0.999, 1.0]])

        # This should succeed without warning (it is PD)
        irf = compute_irf_cholesky(A_matrices, Sigma_near, periods=5)
        assert irf.shape == (6, 2, 2)
        assert np.all(np.isfinite(irf))


class TestAnalyticalIRFCI:
    """Tests for compute_analytical_ci to cover lines 703-753."""

    def test_analytical_ci_basic(self):
        """Test basic analytical CI computation."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)
        # cov_params should be (K*p*K, K*p*K) = (4, 4) for K=2, p=1
        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices, Sigma, cov_params, periods=5, method="cholesky", ci_level=0.95
        )

        assert ci_lower.shape == (6, 2, 2)
        assert ci_upper.shape == (6, 2, 2)
        # Lower should be <= upper
        assert np.all(ci_lower <= ci_upper)

    def test_analytical_ci_generalized_method(self):
        """Test analytical CI with generalized method."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices, Sigma, cov_params, periods=5, method="generalized", ci_level=0.95
        )

        assert ci_lower.shape == (6, 2, 2)
        assert ci_upper.shape == (6, 2, 2)
        assert np.all(ci_lower <= ci_upper)

    def test_analytical_ci_cumulative(self):
        """Test analytical CI with cumulative IRFs."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)
        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices,
            Sigma,
            cov_params,
            periods=5,
            method="cholesky",
            ci_level=0.95,
            cumulative=True,
        )

        assert ci_lower.shape == (6, 2, 2)
        assert ci_upper.shape == (6, 2, 2)
        assert np.all(ci_lower <= ci_upper)

    def test_analytical_ci_wider_with_higher_level(self):
        """Test that wider CI level produces wider intervals."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)
        cov_params = np.eye(4) * 0.01

        ci_lower_90, ci_upper_90 = compute_analytical_ci(
            A_matrices, Sigma, cov_params, periods=5, method="cholesky", ci_level=0.90
        )

        ci_lower_99, ci_upper_99 = compute_analytical_ci(
            A_matrices, Sigma, cov_params, periods=5, method="cholesky", ci_level=0.99
        )

        # 99% CI should be wider than 90% CI
        width_90 = ci_upper_90 - ci_lower_90
        width_99 = ci_upper_99 - ci_lower_99
        assert np.all(width_99 >= width_90 - 1e-10)

    def test_analytical_ci_stderr_grows_with_horizon(self):
        """Test that CI width grows with horizon (due to sqrt(h+1) factor)."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)
        cov_params = np.eye(4) * 0.01

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices, Sigma, cov_params, periods=10, method="cholesky", ci_level=0.95
        )

        # Width at h=10 should be wider than at h=0
        width_h0 = (ci_upper[0] - ci_lower[0]).mean()
        width_h10 = (ci_upper[10] - ci_lower[10]).mean()
        assert width_h10 > width_h0

    def test_analytical_ci_finite_values(self):
        """Test that all CI values are finite."""
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[2.0, 0.5], [0.5, 3.0]])
        cov_params = np.eye(4) * 0.05

        ci_lower, ci_upper = compute_analytical_ci(
            A_matrices, Sigma, cov_params, periods=10, method="cholesky", ci_level=0.95
        )

        assert np.all(np.isfinite(ci_lower))
        assert np.all(np.isfinite(ci_upper))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
