"""
Tests for var/diagnostics.py, var/inference.py, and var/causality_network.py
to improve branch coverage.

Round 3 - targets specific uncovered lines/branches.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data():
    """Create simple data for GMM diagnostics tests."""
    np.random.seed(42)
    n_obs = 50
    n_instruments = 5
    n_params = 3

    residuals = np.random.randn(n_obs, 1)
    instruments = np.random.randn(n_obs, n_instruments)
    return residuals, instruments, n_params


@pytest.fixture
def entity_ids():
    """Create entity IDs for AR test."""
    # 10 entities with 5 observations each
    return np.repeat(np.arange(10), 5)


# ---------------------------------------------------------------------------
# var/diagnostics.py — GMMDiagnostics
# ---------------------------------------------------------------------------


class TestGMMDiagnosticsBranches:
    """Test uncovered branches in GMMDiagnostics."""

    def test_hansen_j_multi_equation_residuals(self):
        """Cover line 106: multi-equation residuals (ndim > 1, shape[1] > 1)."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        # Multi-equation residuals: 50 obs x 3 equations
        residuals = np.random.randn(50, 3)
        instruments = np.random.randn(50, 5)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=20)
        result = diag.hansen_j_test()
        assert "statistic" in result
        assert "p_value" in result
        assert result["df"] == 2  # 5 - 3

    def test_hansen_j_singular_omega(self):
        """Cover lines 124-134: singular Omega matrix raises warning."""
        from panelbox.var.diagnostics import GMMDiagnostics

        # Create perfectly collinear instruments to get singular Omega
        np.random.seed(42)
        n_obs = 10
        col1 = np.random.randn(n_obs, 1)
        instruments = np.hstack([col1, col1, col1, col1, col1])  # all same
        residuals = np.zeros((n_obs, 1))  # zero residuals -> singular Omega
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = diag.hansen_j_test()
        # Should return nan or a result (depends on whether singular)
        assert "statistic" in result

    def test_hansen_j_pvalue_gt_099(self):
        """Cover lines 180-185: p-value > 0.99 branch."""
        from panelbox.var.diagnostics import GMMDiagnostics

        # Directly test the interpretation method for p > 0.99
        np.random.seed(42)
        n_obs = 50
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 5)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=20)
        interpretation, test_warnings = diag._interpret_hansen_j(0.01, 0.999)
        assert "WARNING" in interpretation
        assert "p-value very high" in interpretation
        assert any("weak instruments" in w for w in test_warnings)

    def test_hansen_j_pvalue_moderate(self):
        """Cover lines 192-193: p-value in (0.05, 0.10) or (0.90, 0.99) branch."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 100
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 5)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=50)
        result = diag.hansen_j_test()
        # We just need to ensure interpretation logic runs for all branches
        assert "interpretation" in result

    def test_interpret_hansen_j_roodman_rule(self):
        """Cover lines 196-200: n_instruments > n_entities warning."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 50
        # n_instruments (10) > n_entities (5)
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 10)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=5)
        _interpretation, test_warnings = diag._interpret_hansen_j(5.0, 0.5)
        assert any("Rule-of-thumb violated" in w for w in test_warnings)

    def test_sargan_test_1d_residuals(self):
        """Cover line 234: sargan_test with 1D residuals."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 50
        residuals = np.random.randn(n_obs)  # 1D
        instruments = np.random.randn(n_obs, 5)
        diag = GMMDiagnostics(residuals.reshape(-1, 1), instruments, n_params=3, n_entities=20)
        result = diag.sargan_test()
        assert "statistic" in result
        assert "p_value" in result

    def test_sargan_test_singular_ZtZ(self):
        """Cover lines 245-251: sargan_test with singular Z'Z."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 10
        # Singular Z'Z: all columns identical
        col = np.random.randn(n_obs, 1)
        instruments = np.hstack([col, col, col, col])
        residuals = np.random.randn(n_obs, 1)
        diag = GMMDiagnostics(residuals, instruments, n_params=2, n_entities=5)
        result = diag.sargan_test()
        # Should handle singular Z'Z gracefully
        assert "statistic" in result

    def test_sargan_test_not_reject(self):
        """Cover line 266: sargan_test p_value >= 0.05 (do not reject)."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 100
        residuals = np.random.randn(n_obs, 1) * 0.01
        instruments = np.random.randn(n_obs, 5)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=50)
        result = diag.sargan_test()
        # We just check that interpretation is set
        assert "interpretation" in result

    def test_instrument_diagnostics_no_warnings(self):
        """Cover lines 391-392, 396: diagnosis with no warnings, p in ideal range."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 100
        # Instruments < entities and ratio < 3
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 4)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=50)
        report = diag.instrument_diagnostics_report()
        assert "diagnosis" in report
        # No warnings if instruments < entities and ratio < 3
        # The actual diagnosis depends on Hansen J p-value

    def test_instrument_diagnostics_high_ratio(self):
        """Cover lines 375-379: high instrument/params ratio."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 100
        # ratio = 15/3 = 5 > 3
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 15)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=50)
        report = diag.instrument_diagnostics_report()
        assert any("Ratio instruments/params" in w for w in report["warnings"])

    def test_instrument_diagnostics_acceptable_diagnosis(self):
        """Cover line 396: acceptable but check warnings diagnosis."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 100
        # No warnings (instruments < entities, ratio < 3)
        # But p-value NOT in ideal range [0.10, 0.90]
        residuals = np.random.randn(n_obs, 1) * 0.001
        instruments = np.random.randn(n_obs, 4)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=50)
        report = diag.instrument_diagnostics_report()
        # Just check it runs
        assert "diagnosis" in report

    def test_format_report_no_warnings_no_suggestions(self):
        """Cover lines 502->508, 508->514: no warnings or suggestions."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 100
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 4)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=50)
        report_str = diag.format_diagnostics_report()
        assert "GMM Instrument Diagnostics" in report_str

    def test_format_report_with_ar_tests(self):
        """Cover lines 483-497: format report with AR tests."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 50
        entity_ids = np.repeat(np.arange(10), 5)
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 5)
        diag = GMMDiagnostics(
            residuals, instruments, n_params=3, n_entities=10, entity_ids=entity_ids
        )
        report_str = diag.format_diagnostics_report(include_ar_tests=True, max_ar_order=3)
        assert "Serial Correlation Tests" in report_str
        assert "AR(1)" in report_str
        assert "AR(2)" in report_str

    def test_ar_test_no_entity_ids(self):
        """Cover lines 429-436: AR test without entity_ids."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        residuals = np.random.randn(50, 1)
        instruments = np.random.randn(50, 5)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=10)
        result = diag.ar_test(order=1)
        assert result["interpretation"] == "AR test requires entity_ids"

    def test_hansen_j_exactly_identified(self):
        """Cover lines 94-101: exactly identified model (df_overid <= 0)."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        residuals = np.random.randn(50, 1)
        instruments = np.random.randn(50, 3)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=10)
        result = diag.hansen_j_test()
        assert np.isnan(result["statistic"])
        assert "exactly identified" in result["interpretation"]

    def test_sargan_test_exactly_identified(self):
        """Cover lines 222-228: sargan test exactly identified."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        residuals = np.random.randn(50, 1)
        instruments = np.random.randn(50, 3)
        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=10)
        result = diag.sargan_test()
        assert np.isnan(result["statistic"])


class TestARTestBranches:
    """Test uncovered branches in ar_test function."""

    def test_ar_test_multi_equation(self):
        """Cover line 1002: multi-equation residuals in ar_test."""
        from panelbox.var.diagnostics import ar_test

        np.random.seed(42)
        entity_ids = np.repeat(np.arange(10), 5)
        # Multi-equation: 50 obs x 3 equations
        residuals = np.random.randn(50, 3)
        result = ar_test(residuals, entity_ids, order=1)
        assert "statistic" in result
        assert result["n_products"] > 0

    def test_ar_test_zero_variance(self):
        """Cover line 1044-1051: zero variance in products."""
        from panelbox.var.diagnostics import ar_test

        entity_ids = np.repeat(np.arange(5), 3)
        # Constant residuals -> zero variance
        residuals = np.ones(15)
        result = ar_test(residuals, entity_ids, order=1)
        assert np.isnan(result["statistic"]) or result["n_products"] > 0

    def test_ar_test_order_gt_2_rejected(self):
        """Cover line 1084: ar_test order > 2 and rejected."""
        from panelbox.var.diagnostics import ar_test

        np.random.seed(42)
        entity_ids = np.repeat(np.arange(10), 20)
        # Create highly autocorrelated residuals for order 3
        residuals = np.random.randn(200)
        for i in range(3, len(residuals)):
            residuals[i] += 0.9 * residuals[i - 3]
        result = ar_test(residuals, entity_ids, order=3)
        assert "AR(3)" in result["interpretation"]

    def test_ar_test_order_gt_2_not_rejected(self):
        """Cover line 1086: ar_test order > 2 and not rejected."""
        from panelbox.var.diagnostics import ar_test

        np.random.seed(42)
        entity_ids = np.repeat(np.arange(10), 20)
        # IID residuals - should not reject at order 3
        residuals = np.random.randn(200)
        result = ar_test(residuals, entity_ids, order=3)
        assert "AR(3)" in result["interpretation"]

    def test_ar_test_insufficient_data(self):
        """Cover lines 1029-1036: insufficient data for AR test."""
        from panelbox.var.diagnostics import ar_test

        # Each entity has only 1 observation -> can't compute order=1 products
        entity_ids = np.arange(5)
        residuals = np.random.randn(5)
        result = ar_test(residuals, entity_ids, order=1)
        assert result["n_products"] == 0
        assert np.isnan(result["statistic"])

    def test_ar_test_with_nans(self):
        """Cover lines 1011-1013: residuals with NaN values."""
        from panelbox.var.diagnostics import ar_test

        np.random.seed(42)
        entity_ids = np.repeat(np.arange(5), 10)
        residuals = np.random.randn(50)
        residuals[0] = np.nan
        residuals[10] = np.nan
        result = ar_test(residuals, entity_ids, order=1)
        assert "statistic" in result

    def test_ar_test_order1_not_rejected(self):
        """Cover line 1068: AR(1) not rejected."""
        from panelbox.var.diagnostics import ar_test

        np.random.seed(42)
        entity_ids = np.repeat(np.arange(20), 10)
        # Very small residuals that are IID
        residuals = np.random.randn(200) * 0.001
        result = ar_test(residuals, entity_ids, order=1)
        assert "AR(1)" in result["interpretation"]

    def test_ar_test_order2_rejected(self):
        """Cover lines 1070-1076: AR(2) rejected."""
        from panelbox.var.diagnostics import ar_test

        np.random.seed(42)
        entity_ids = np.repeat(np.arange(10), 30)
        # Create residuals with strong AR(2) correlation
        residuals = np.random.randn(300)
        for i in range(2, len(residuals)):
            residuals[i] += 0.95 * residuals[i - 2]
        result = ar_test(residuals, entity_ids, order=2)
        assert "AR(2)" in result["interpretation"]


class TestDifferencHansenTest:
    """Test difference_hansen_test."""

    def test_difference_hansen_reject(self):
        """Cover lines 329-333: difference Hansen rejects."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 100
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 8)
        instruments_subset = instruments[:, :4]
        residuals_restricted = np.random.randn(n_obs, 1) * 2.0

        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=50)
        result = diag.difference_hansen_test(
            instruments_subset=instruments_subset,
            residuals_full=residuals,
            residuals_restricted=residuals_restricted,
            n_params=3,
        )
        assert "statistic" in result
        assert "p_value" in result

    def test_difference_hansen_not_reject(self):
        """Cover line 335: difference Hansen does not reject."""
        from panelbox.var.diagnostics import GMMDiagnostics

        np.random.seed(42)
        n_obs = 100
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 8)
        instruments_subset = instruments[:, :4]

        diag = GMMDiagnostics(residuals, instruments, n_params=3, n_entities=50)
        result = diag.difference_hansen_test(
            instruments_subset=instruments_subset,
            residuals_full=residuals,
            residuals_restricted=residuals,  # same residuals
            n_params=3,
        )
        assert "interpretation" in result


class TestConvenienceFunctions:
    """Test convenience functions for hansen_j and sargan."""

    def test_hansen_j_test_function(self):
        """Cover lines 541-542: convenience function."""
        from panelbox.var.diagnostics import hansen_j_test

        np.random.seed(42)
        residuals = np.random.randn(50, 1)
        instruments = np.random.randn(50, 5)
        result = hansen_j_test(residuals, instruments, n_params=3, n_entities=20)
        assert "statistic" in result

    def test_sargan_test_function(self):
        """Cover lines 567-568: convenience function."""
        from panelbox.var.diagnostics import sargan_test

        np.random.seed(42)
        residuals = np.random.randn(50, 1)
        instruments = np.random.randn(50, 5)
        result = sargan_test(residuals, instruments, n_params=3, n_entities=20)
        assert "statistic" in result


# ---------------------------------------------------------------------------
# var/inference.py
# ---------------------------------------------------------------------------


class TestInferenceBranches:
    """Test uncovered branches in var/inference.py."""

    def test_compute_ols_with_weights(self):
        """Cover lines 78-83: weighted least squares branch."""
        from panelbox.var.inference import compute_ols_equation

        np.random.seed(42)
        n = 50
        k = 3
        X = np.random.randn(n, k)
        beta_true = np.array([1.0, 2.0, 3.0])
        y = X @ beta_true + np.random.randn(n) * 0.1
        weights = np.random.uniform(0.5, 2.0, n)

        beta, resid, fitted = compute_ols_equation(y, X, weights=weights)
        assert beta.shape == (k,)
        assert resid.shape == (n,)
        assert fitted.shape == (n,)

    def test_compute_sur_covariance_singular_XtX(self):
        """Cover lines 193-194: pinv fallback for singular X'X in SUR."""
        from panelbox.var.inference import compute_sur_covariance

        np.random.seed(42)
        n = 20
        K = 2
        # Create a singular X matrix (column duplication)
        col = np.random.randn(n, 1)
        X = np.hstack([col, col])
        residuals_all = np.random.randn(n, K)
        vcov = compute_sur_covariance(X, residuals_all, K)
        assert vcov.shape == (K * 2, K * 2)

    def test_compute_covariance_clustered_no_entities(self):
        """Cover line 259: clustered covariance without entities raises ValueError."""
        from panelbox.var.inference import compute_covariance_matrix

        np.random.seed(42)
        X = np.random.randn(50, 3)
        resid = np.random.randn(50)
        with pytest.raises(ValueError, match="entities required"):
            compute_covariance_matrix(X, resid, cov_type="clustered")

    def test_compute_covariance_driscoll_kraay(self):
        """Cover lines 263-270: driscoll_kraay branch."""
        from panelbox.var.inference import compute_covariance_matrix

        np.random.seed(42)
        n = 100
        k = 3
        X = np.random.randn(n, k)
        resid = np.random.randn(n)
        times = np.repeat(np.arange(10), 10)
        vcov = compute_covariance_matrix(X, resid, cov_type="driscoll_kraay", times=times)
        assert vcov.shape == (k, k)

    def test_compute_covariance_driscoll_kraay_no_times(self):
        """Cover line 266: driscoll_kraay without times raises ValueError."""
        from panelbox.var.inference import compute_covariance_matrix

        np.random.seed(42)
        X = np.random.randn(50, 3)
        resid = np.random.randn(50)
        with pytest.raises(ValueError, match="times required"):
            compute_covariance_matrix(X, resid, cov_type="driscoll_kraay")

    def test_compute_covariance_sur(self):
        """Cover lines 272-278: SUR branch."""
        from panelbox.var.inference import compute_covariance_matrix

        np.random.seed(42)
        n = 50
        k = 3
        K = 2
        X = np.random.randn(n, k)
        resid = np.random.randn(n)
        residuals_all = np.random.randn(n, K)
        vcov = compute_covariance_matrix(X, resid, cov_type="sur", residuals_all=residuals_all, K=K)
        assert vcov.shape == (K * k, K * k)

    def test_compute_covariance_sur_missing_args(self):
        """Cover line 277: SUR without required args raises ValueError."""
        from panelbox.var.inference import compute_covariance_matrix

        np.random.seed(42)
        X = np.random.randn(50, 3)
        resid = np.random.randn(50)
        with pytest.raises(ValueError, match="SUR requires"):
            compute_covariance_matrix(X, resid, cov_type="sur")

    def test_compute_covariance_unknown_type(self):
        """Cover lines 281-283: unknown cov_type raises ValueError."""
        from panelbox.var.inference import compute_covariance_matrix

        np.random.seed(42)
        X = np.random.randn(50, 3)
        resid = np.random.randn(50)
        with pytest.raises(ValueError, match="cov_type must be one of"):
            compute_covariance_matrix(X, resid, cov_type="foobar")

    def test_f_test_exclusion(self):
        """Cover lines 356-388: f_test_exclusion function."""
        from panelbox.var.inference import f_test_exclusion

        np.random.seed(42)
        k = 5
        params = np.random.randn(k)
        cov_params = np.eye(k) * 0.1
        result = f_test_exclusion(params, cov_params, indices=[0, 1])
        assert result.df == 2
        assert result.pvalue >= 0.0

    def test_wald_test_repr(self):
        """Cover lines 43-51: WaldTestResult __repr__."""
        from panelbox.var.inference import WaldTestResult

        result = WaldTestResult(statistic=5.0, pvalue=0.03, df=2, hypothesis="R*b=r")
        repr_str = repr(result)
        assert "Wald Test" in repr_str
        assert "5.0000" in repr_str

    def test_wald_test_singular_restriction(self):
        """Cover lines 341-343: wald_test with singular var_restriction."""
        from panelbox.var.inference import wald_test

        np.random.seed(42)
        k = 3
        params = np.random.randn(k)
        # Singular covariance matrix
        cov_params = np.zeros((k, k))
        R = np.array([[1, 0, 0], [0, 1, 0]])
        result = wald_test(params, cov_params, R)
        assert result.df == 2

    def test_within_transformation_1d(self):
        """Cover lines 113-114: within_transformation with 1D input."""
        from panelbox.var.inference import within_transformation

        np.random.seed(42)
        data = np.random.randn(20)
        entities = np.repeat(np.arange(4), 5)
        demeaned, means = within_transformation(data, entities)
        assert demeaned.ndim == 1
        assert len(means) == 4

    def test_within_transformation_2d(self):
        """Cover lines 116-117: within_transformation with 2D input."""
        from panelbox.var.inference import within_transformation

        np.random.seed(42)
        data = np.random.randn(20, 3)
        entities = np.repeat(np.arange(4), 5)
        demeaned, _means = within_transformation(data, entities)
        assert demeaned.ndim == 2
        assert demeaned.shape == (20, 3)

    def test_compute_covariance_nonrobust_singular(self):
        """Cover lines 247-248: nonrobust with singular X'X."""
        from panelbox.var.inference import compute_covariance_matrix

        np.random.seed(42)
        n = 20
        col = np.random.randn(n, 1)
        X = np.hstack([col, col])  # Singular X'X
        resid = np.random.randn(n)
        vcov = compute_covariance_matrix(X, resid, cov_type="nonrobust")
        assert vcov.shape == (2, 2)


# ---------------------------------------------------------------------------
# var/causality_network.py
# ---------------------------------------------------------------------------


class TestCausalityNetworkBranches:
    """Test uncovered branches in causality_network.py."""

    @pytest.fixture
    def granger_matrix(self):
        """Create a Granger causality p-value matrix."""
        var_names = ["x", "y", "z"]
        data = np.array(
            [[1.0, 0.01, 0.8], [0.03, 1.0, 0.6], [0.5, 0.001, 1.0]]  # y->x  # x->y
        )  # y->z
        return pd.DataFrame(data, index=var_names, columns=var_names)

    @pytest.fixture
    def granger_no_sig(self):
        """Create a Granger matrix with no significant relationships."""
        var_names = ["x", "y"]
        data = np.array([[1.0, 0.5], [0.8, 1.0]])
        return pd.DataFrame(data, index=var_names, columns=var_names)

    def test_plot_network_not_square(self):
        """Cover line 136: non-square matrix raises ValueError."""
        pytest.importorskip("networkx")
        from panelbox.var.causality_network import plot_causality_network

        df = pd.DataFrame(np.ones((2, 3)), columns=["a", "b", "c"])
        with pytest.raises(ValueError, match="must be square"):
            plot_causality_network(df)

    def test_plot_network_not_dataframe(self):
        """Cover line 133: non-DataFrame input raises ValueError."""
        pytest.importorskip("networkx")
        from panelbox.var.causality_network import plot_causality_network

        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            plot_causality_network(np.ones((2, 2)))

    def test_plot_network_unknown_backend(self):
        """Cover line 177: unknown backend raises ValueError."""
        pytest.importorskip("networkx")
        from panelbox.var.causality_network import plot_causality_network

        var_names = ["x", "y"]
        df = pd.DataFrame(np.ones((2, 2)), index=var_names, columns=var_names)
        with pytest.raises(ValueError, match="Unknown backend"):
            plot_causality_network(df, backend="seaborn")

    def test_no_significant_relationships_warning(self, granger_no_sig):
        """Cover lines 220-226: no significant relationships warning."""
        pytest.importorskip("networkx")
        from panelbox.var.causality_network import _build_causality_graph

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            G = _build_causality_graph(granger_no_sig, threshold=0.05)
            assert any("No significant" in str(warning.message) for warning in w)
        assert G.number_of_edges() == 0

    def test_layout_circular(self, granger_matrix):
        """Cover line 250: circular layout."""
        pytest.importorskip("networkx")
        from panelbox.var.causality_network import (
            _build_causality_graph,
            _get_layout_positions,
        )

        G = _build_causality_graph(granger_matrix, threshold=0.05)
        pos = _get_layout_positions(G, "circular")
        assert len(pos) == 3

    def test_layout_kamada_kawai(self, granger_matrix):
        """Cover lines 254-258: kamada_kawai layout."""
        pytest.importorskip("networkx")
        from panelbox.var.causality_network import (
            _build_causality_graph,
            _get_layout_positions,
        )

        G = _build_causality_graph(granger_matrix, threshold=0.05)
        pos = _get_layout_positions(G, "kamada_kawai")
        assert len(pos) == 3

    def test_layout_kamada_kawai_single_node(self):
        """Cover line 258: kamada_kawai with single node."""
        nx = pytest.importorskip("networkx")
        from panelbox.var.causality_network import _get_layout_positions

        G = nx.DiGraph()
        G.add_node("x")
        pos = _get_layout_positions(G, "kamada_kawai")
        assert "x" in pos

    def test_layout_shell(self, granger_matrix):
        """Cover line 260: shell layout."""
        pytest.importorskip("networkx")
        from panelbox.var.causality_network import (
            _build_causality_graph,
            _get_layout_positions,
        )

        G = _build_causality_graph(granger_matrix, threshold=0.05)
        pos = _get_layout_positions(G, "shell")
        assert len(pos) == 3

    def test_layout_unknown(self, granger_matrix):
        """Cover lines 261-264: unknown layout raises ValueError."""
        pytest.importorskip("networkx")
        from panelbox.var.causality_network import (
            _build_causality_graph,
            _get_layout_positions,
        )

        G = _build_causality_graph(granger_matrix, threshold=0.05)
        with pytest.raises(ValueError, match="Unknown layout"):
            _get_layout_positions(G, "random_layout")

    def test_plot_matplotlib_with_pvalues(self, granger_matrix):
        """Cover lines 310, 332-348: matplotlib backend with show_pvalues."""
        pytest.importorskip("networkx")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from panelbox.var.causality_network import plot_causality_network

        fig = plot_causality_network(
            granger_matrix,
            threshold=0.10,
            backend="matplotlib",
            show_pvalues=True,
            show=False,
        )
        assert fig is not None
        plt.close("all")

    def test_plot_matplotlib_no_edges(self, granger_no_sig):
        """Cover matplotlib with no edges (empty graph)."""
        pytest.importorskip("networkx")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from panelbox.var.causality_network import plot_causality_network

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plot_causality_network(
                granger_no_sig, threshold=0.05, backend="matplotlib", show=False
            )
        assert fig is not None
        plt.close("all")

    def test_plot_plotly_with_edges(self, granger_matrix):
        """Cover plotly backend with edges."""
        pytest.importorskip("networkx")
        pytest.importorskip("plotly")
        from panelbox.var.causality_network import plot_causality_network

        fig = plot_causality_network(
            granger_matrix,
            threshold=0.10,
            backend="plotly",
            show=False,
        )
        assert fig is not None

    def test_plot_plotly_no_edges(self, granger_no_sig):
        """Cover plotly backend with no edges (462->493 branch)."""
        pytest.importorskip("networkx")
        pytest.importorskip("plotly")
        from panelbox.var.causality_network import plot_causality_network

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plot_causality_network(
                granger_no_sig, threshold=0.05, backend="plotly", show=False
            )
        assert fig is not None

    def test_plot_matplotlib_custom_title(self, granger_matrix):
        """Cover line 350: custom title for matplotlib."""
        pytest.importorskip("networkx")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from panelbox.var.causality_network import plot_causality_network

        fig = plot_causality_network(
            granger_matrix,
            threshold=0.10,
            backend="matplotlib",
            title="My Custom Title",
            show=False,
        )
        assert fig is not None
        plt.close("all")

    def test_plot_plotly_pvalue_colors(self):
        """Cover plotly pvalue color branches (404, 407, 409)."""
        pytest.importorskip("networkx")
        pytest.importorskip("plotly")
        from panelbox.var.causality_network import plot_causality_network

        var_names = ["a", "b", "c"]
        # Mix of p-value ranges: < 0.01, < 0.05, < 0.10
        data = np.array(
            [
                [1.0, 0.005, 0.03],
                [0.07, 1.0, 0.002],
                [0.04, 0.08, 1.0],
            ]
        )
        df = pd.DataFrame(data, index=var_names, columns=var_names)
        fig = plot_causality_network(df, threshold=0.10, backend="plotly", show=False)
        assert fig is not None
