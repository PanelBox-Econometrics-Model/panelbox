"""
Tests for SUR (Seemingly Unrelated Regressions) estimator — Subfase 04.

Comprehensive test suite covering:
- Basic initialization (string and dict formulas)
- SUR == OLS equivalences (independent errors, identical regressors)
- Efficiency gains with correlated errors
- Sigma properties (symmetric, PSD)
- ISUR convergence
- Breusch-Pagan independence test
- Heterogeneous dict formulas
- Homogeneous SUR mode
- LR homogeneity test
- Edge cases (N=2, N>T, unbalanced, small N)
- Results (system_summary, equation_summary, McElroy R^2)
- Manual GLS comparison
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.core.results import PanelResults
from panelbox.models.static.sur import SUR, SURResults

# ---------------------------------------------------------------------------
# Helper DGP functions
# ---------------------------------------------------------------------------


def _make_panel(n_entities=5, n_periods=50, rho=0.7, seed=42):
    """Panel data with cross-entity correlated errors."""
    rng = np.random.default_rng(seed)
    sigma_true = np.eye(n_entities) * (1 - rho) + rho
    L = np.linalg.cholesky(sigma_true)

    entities, times, ys, x1s, x2s = [], [], [], [], []
    for t in range(n_periods):
        eps = L @ rng.standard_normal(n_entities)
        for i in range(n_entities):
            x1 = rng.standard_normal() + i * 0.5
            x2 = rng.standard_normal()
            y = 1.0 + 2.0 * x1 + 0.5 * x2 + eps[i]
            entities.append(i)
            times.append(t)
            x1s.append(x1)
            x2s.append(x2)
            ys.append(y)

    return pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": ys,
            "x1": x1s,
            "x2": x2s,
        }
    )


def _make_panel_diagonal(n_entities=5, n_periods=50, seed=123):
    """Panel data with independent (diagonal Sigma) errors."""
    return _make_panel(n_entities=n_entities, n_periods=n_periods, rho=0.0, seed=seed)


def _make_panel_with_extra(n_entities=4, n_periods=50, seed=42):
    """Panel data with extra columns for heterogeneous formula testing."""
    rng = np.random.default_rng(seed)
    rho = 0.5
    sigma_true = np.eye(n_entities) * (1 - rho) + rho
    L = np.linalg.cholesky(sigma_true)

    entities, times, ys, x1s, x2s, x3s = [], [], [], [], [], []
    for t in range(n_periods):
        eps = L @ rng.standard_normal(n_entities)
        for i in range(n_entities):
            x1 = rng.standard_normal() + i * 0.3
            x2 = rng.standard_normal()
            x3 = rng.standard_normal()
            y = 1.0 + 2.0 * x1 + 0.5 * x2 + 0.3 * x3 + eps[i]
            entities.append(i)
            times.append(t)
            x1s.append(x1)
            x2s.append(x2)
            x3s.append(x3)
            ys.append(y)

    return pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": ys,
            "x1": x1s,
            "x2": x2s,
            "x3": x3s,
        }
    )


def _make_identical_regressors(n_entities=3, n_periods=100, rho=0.7, seed=55):
    """
    Panel where all entities share the SAME X values (identical regressors).

    When all equations have the same X, SUR reduces to OLS.
    """
    rng = np.random.default_rng(seed)
    sigma_true = np.eye(n_entities) * (1 - rho) + rho
    L = np.linalg.cholesky(sigma_true)

    entities, times, ys, x1s, x2s = [], [], [], [], []
    for t in range(n_periods):
        eps = L @ rng.standard_normal(n_entities)
        # Same X for all entities in this period
        x1 = rng.standard_normal()
        x2 = rng.standard_normal()
        for i in range(n_entities):
            y = 1.0 + 2.0 * x1 + 0.5 * x2 + eps[i]
            entities.append(i)
            times.append(t)
            x1s.append(x1)
            x2s.append(x2)
            ys.append(y)

    return pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": ys,
            "x1": x1s,
            "x2": x2s,
        }
    )


def _make_different_x_correlated(n_entities=3, n_periods=200, rho=0.8, seed=42):
    """
    Panel with different X per entity AND correlated errors.
    This is the case where SUR is more efficient than OLS.
    """
    rng = np.random.default_rng(seed)
    sigma_true = np.eye(n_entities) * (1 - rho) + rho
    L = np.linalg.cholesky(sigma_true)

    entities, times, ys, x1s, x2s = [], [], [], [], []
    for t in range(n_periods):
        eps = L @ rng.standard_normal(n_entities)
        for i in range(n_entities):
            # Different X distributions per entity
            x1 = rng.standard_normal() * (i + 1) + i
            x2 = rng.standard_normal() * (n_entities - i)
            y = 1.0 + 2.0 * x1 + 0.5 * x2 + eps[i]
            entities.append(i)
            times.append(t)
            x1s.append(x1)
            x2s.append(x2)
            ys.append(y)

    return pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": ys,
            "x1": x1s,
            "x2": x2s,
        }
    )


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------


class TestSUR:
    """Comprehensive SUR test suite (Subfase 04)."""

    # -----------------------------------------------------------------------
    # Basicos
    # -----------------------------------------------------------------------

    def test_init_string_formula(self):
        """SUR can be initialized with a string formula."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        assert sur._formula_dict is None
        assert sur.homogeneous is False
        assert sur.iterate is False

    def test_init_dict_formula(self):
        """SUR can be initialized with a dict formula."""
        data = _make_panel_with_extra()
        formulas = {
            0: "y ~ x1 + x2",
            1: "y ~ x1",
            2: "y ~ x2 + x3",
            3: "y ~ x1 + x2 + x3",
        }
        sur = SUR(formulas, data, "entity", "time")
        assert sur._formula_dict is not None
        assert len(sur._entity_parsers) == 4

    def test_fit_returns_sur_results(self):
        """fit() returns a SURResults instance."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()
        assert isinstance(res, SURResults)
        assert isinstance(res, PanelResults)

    # -----------------------------------------------------------------------
    # Propriedades do SUR
    # -----------------------------------------------------------------------

    def test_sur_equals_ols_independent_errors(self):
        """With diagonal Sigma (independent errors), SUR ≈ OLS."""
        data = _make_panel_diagonal(n_entities=4, n_periods=200, seed=77)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        for entity in res.entity_params:
            sur_beta = res.entity_params[entity].values
            ols_beta = res.ols_params[entity].values
            np.testing.assert_allclose(
                sur_beta,
                ols_beta,
                atol=0.05,
                err_msg=f"SUR != OLS for entity {entity} with independent errors",
            )

    def test_sur_equals_ols_identical_regressors(self):
        """When X_i == X_j for all i, j, SUR reduces to OLS."""
        data = _make_identical_regressors(n_entities=3, n_periods=200, seed=55)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        for entity in res.entity_params:
            sur_beta = res.entity_params[entity].values
            ols_beta = res.ols_params[entity].values
            np.testing.assert_allclose(
                sur_beta,
                ols_beta,
                atol=0.05,
                err_msg=f"SUR != OLS for entity {entity} with identical regressors",
            )

    def test_sur_more_efficient_correlated(self):
        """With correlated errors + different X, SUR SE < OLS SE."""
        data = _make_different_x_correlated(n_entities=3, n_periods=200, rho=0.9, seed=42)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        # At least one entity should show efficiency gain (SE ratio < 1)
        any_gain = False
        for _entity, gain in res.efficiency_gain.items():
            if (gain < 1.0).any():
                any_gain = True
                break
        assert any_gain, "SUR should show efficiency gains with correlated errors + different X"

    def test_sigma_symmetric_psd(self):
        """Sigma_hat is symmetric and positive semi-definite."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        sigma = res.sigma_matrix
        # Symmetric
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)
        # PSD: all eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(sigma)
        assert np.all(eigenvalues >= -1e-10), f"Sigma not PSD, min eigenvalue = {eigenvalues.min()}"

    # -----------------------------------------------------------------------
    # ISUR
    # -----------------------------------------------------------------------

    def test_isur_convergence(self):
        """ISUR converges within max_iter."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time", iterate=True, max_iter=100)
        res = sur.fit()

        assert res.converged is True
        assert res.n_iterations < 100

    def test_isur_few_iterations(self):
        """ISUR typically converges in < 10 iterations."""
        data = _make_panel(n_entities=3, n_periods=100, rho=0.5, seed=99)
        sur = SUR("y ~ x1 + x2", data, "entity", "time", iterate=True, tol=1e-8)
        res = sur.fit()

        assert res.converged is True
        assert res.n_iterations <= 10, f"ISUR took {res.n_iterations} iterations, expected <= 10"

    # -----------------------------------------------------------------------
    # Teste BP
    # -----------------------------------------------------------------------

    def test_bp_not_reject_diagonal(self):
        """BP test does not reject with diagonal Sigma (independent errors)."""
        data = _make_panel_diagonal(n_entities=4, n_periods=100, seed=77)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        bp = res.bp_independence_test
        assert bp["pvalue"] > 0.05, (
            f"BP test should NOT reject with diagonal Sigma, p={bp['pvalue']:.4f}"
        )
        assert bp["df"] == 4 * 3 // 2  # N*(N-1)/2 = 6

    def test_bp_reject_correlated(self):
        """BP test rejects with correlated errors."""
        data = _make_panel(n_entities=4, n_periods=100, rho=0.8, seed=55)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        bp = res.bp_independence_test
        assert bp["pvalue"] < 0.05, (
            f"BP test should reject with correlated errors, p={bp['pvalue']:.4f}"
        )

    # -----------------------------------------------------------------------
    # Formulas heterogeneas
    # -----------------------------------------------------------------------

    def test_different_formulas_per_entity(self):
        """FGLS with heterogeneous K_i works correctly."""
        data = _make_panel_with_extra()
        formulas = {
            0: "y ~ x1 + x2",  # K=3
            1: "y ~ x1",  # K=2
            2: "y ~ x2 + x3",  # K=3
            3: "y ~ x1 + x2 + x3",  # K=4
        }
        sur = SUR(formulas, data, "entity", "time")
        res = sur.fit()

        assert len(res.entity_params[0]) == 3
        assert len(res.entity_params[1]) == 2
        assert len(res.entity_params[2]) == 3
        assert len(res.entity_params[3]) == 4

    def test_invalid_entity_in_formula_dict(self):
        """Error when dict contains entity not in data."""
        data = _make_panel_with_extra()
        formulas = {
            0: "y ~ x1 + x2",
            999: "y ~ x1",
        }
        with pytest.raises(ValueError, match="not found in data"):
            SUR(formulas, data, "entity", "time")

    # -----------------------------------------------------------------------
    # Homogeneo
    # -----------------------------------------------------------------------

    def test_homogeneous_single_beta(self):
        """homogeneous=True produces one unique beta vector."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time", homogeneous=True)
        res = sur.fit()

        betas = list(res.entity_params.values())
        for beta in betas[1:]:
            np.testing.assert_array_equal(beta.values, betas[0].values)

    def test_lr_test_homogeneity(self):
        """LR test works and has correct degrees of freedom."""
        rng = np.random.default_rng(77)
        n_entities = 3
        n_periods = 200
        beta_true = np.array([1.0, 2.0, 0.5])

        entities, times, ys, x1s, x2s = [], [], [], [], []
        for t in range(n_periods):
            for i in range(n_entities):
                x1 = rng.standard_normal()
                x2 = rng.standard_normal()
                y = beta_true[0] + beta_true[1] * x1 + beta_true[2] * x2 + rng.standard_normal()
                entities.append(i)
                times.append(t)
                x1s.append(x1)
                x2s.append(x2)
                ys.append(y)

        df = pd.DataFrame(
            {
                "entity": entities,
                "time": times,
                "y": ys,
                "x1": x1s,
                "x2": x2s,
            }
        )

        sur = SUR("y ~ x1 + x2", df, "entity", "time")
        lr = sur.lr_test_homogeneity()

        assert "statistic" in lr
        assert "pvalue" in lr
        assert "df" in lr
        assert lr["df"] == (n_entities - 1) * 3  # (N-1)*K
        # Homogeneous DGP: should not reject
        assert lr["pvalue"] > 0.05

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_two_equations(self):
        """SUR works with only N=2 equations."""
        data = _make_panel(n_entities=2, n_periods=50, seed=42)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        assert len(res.entity_params) == 2
        assert res.sigma_matrix.shape == (2, 2)
        assert res.bp_independence_test["df"] == 1  # 2*(2-1)/2 = 1

    def test_sigma_singular_n_gt_t(self):
        """Sigma singular when N > T — uses pinv, doesn't crash."""
        # N=10, T=5 => Sigma is 10x10 with rank <= 5
        data = _make_panel(n_entities=10, n_periods=5, rho=0.3, seed=42)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        # Should still produce results without error
        assert res.sigma_matrix.shape == (10, 10)
        for entity in res.entity_params:
            assert len(res.entity_params[entity]) == 3

    def test_unbalanced_panel(self):
        """SUR handles unbalanced panels (different T per entity)."""
        rng = np.random.default_rng(42)
        entities, times, ys, x1s, x2s = [], [], [], [], []
        # Entity 0: T=50, Entity 1: T=30, Entity 2: T=40
        t_sizes = {0: 50, 1: 30, 2: 40}
        for entity_id, T_i in t_sizes.items():
            for t in range(T_i):
                x1 = rng.standard_normal()
                x2 = rng.standard_normal()
                y = 1.0 + 2.0 * x1 + 0.5 * x2 + rng.standard_normal()
                entities.append(entity_id)
                times.append(t)
                x1s.append(x1)
                x2s.append(x2)
                ys.append(y)

        df = pd.DataFrame(
            {
                "entity": entities,
                "time": times,
                "y": ys,
                "x1": x1s,
                "x2": x2s,
            }
        )
        sur = SUR("y ~ x1 + x2", df, "entity", "time")
        res = sur.fit()

        assert len(res.entity_params) == 3
        # Coefficients should still be reasonable
        for entity in res.entity_params:
            params = res.entity_params[entity]
            assert len(params) == 3

    def test_small_n_4(self):
        """N=4, T=33 (typical user case)."""
        data = _make_panel(n_entities=4, n_periods=33, rho=0.5, seed=42)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        assert len(res.entity_params) == 4
        assert res.sigma_matrix.shape == (4, 4)
        assert 0 <= res.system_rsquared <= 1

    # -----------------------------------------------------------------------
    # Resultados
    # -----------------------------------------------------------------------

    def test_system_summary(self):
        """system_summary() returns formatted string with expected sections."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        summary = res.system_summary()
        assert isinstance(summary, str)
        assert "Sigma" in summary
        assert "Breusch-Pagan" in summary
        assert "McElroy" in summary
        assert "Correlation" in summary
        assert "Efficiency Gain" in summary

    def test_equation_summary(self):
        """equation_summary() works for each entity."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        entities = list(res.entity_params.keys())
        for entity_id in entities:
            summary = res.equation_summary(entity_id)
            assert isinstance(summary, str)
            assert str(entity_id) in summary
            assert "SUR Coef." in summary
            assert "OLS Coef." in summary

        # Invalid entity
        with pytest.raises(ValueError, match="not found"):
            res.equation_summary("nonexistent_entity")

    def test_efficiency_gain(self):
        """efficiency_gain contains pd.Series with positive values."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        for _entity, gain in res.efficiency_gain.items():
            assert isinstance(gain, pd.Series)
            assert (gain > 0).all()

    def test_system_rsquared(self):
        """McElroy system R-squared is in [0, 1]."""
        data = _make_panel(n_periods=200, seed=99)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        res = sur.fit()

        assert 0 <= res.system_rsquared <= 1
        # With well-specified model, should be decent
        assert res.system_rsquared > 0.3

    # -----------------------------------------------------------------------
    # Comparacao
    # -----------------------------------------------------------------------

    def test_empty_dict_formula_raises(self):
        """Empty dict formula raises ValueError."""
        data = _make_panel()
        with pytest.raises(ValueError, match="must not be empty"):
            SUR({}, data, "entity", "time")

    def test_isur_non_convergence(self):
        """ISUR with max_iter=1 does not converge."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time", iterate=True, max_iter=1)
        res = sur.fit()
        # With max_iter=1, only 1 iteration => not converged
        assert res.converged is False

    def test_isur_homogeneous(self):
        """Iterated homogeneous SUR converges."""
        data = _make_panel()
        sur = SUR("y ~ x1 + x2", data, "entity", "time", homogeneous=True, iterate=True)
        res = sur.fit()
        assert res.converged is True
        betas = list(res.entity_params.values())
        for beta in betas[1:]:
            np.testing.assert_array_equal(beta.values, betas[0].values)

    def test_dict_formula_efficiency_gain(self):
        """Efficiency gain works with heterogeneous formulas."""
        data = _make_panel_with_extra()
        formulas = {
            0: "y ~ x1 + x2",
            1: "y ~ x1",
            2: "y ~ x2 + x3",
            3: "y ~ x1 + x2 + x3",
        }
        sur = SUR(formulas, data, "entity", "time")
        res = sur.fit()
        for _entity, gain in res.efficiency_gain.items():
            assert isinstance(gain, pd.Series)
            assert (gain > 0).all()

    def test_estimate_coefficients(self):
        """_estimate_coefficients returns correct shape."""
        data = _make_panel(n_entities=3)
        sur = SUR("y ~ x1 + x2", data, "entity", "time")
        beta = sur._estimate_coefficients()
        assert len(beta) == 3 * 3  # 3 entities * 3 params

    def test_estimate_coefficients_homogeneous(self):
        """_estimate_coefficients returns K params when homogeneous."""
        data = _make_panel(n_entities=3)
        sur = SUR("y ~ x1 + x2", data, "entity", "time", homogeneous=True)
        beta = sur._estimate_coefficients()
        assert len(beta) == 3  # K=3 (intercept + x1 + x2)

    def test_lr_test_raises_with_dict_formula(self):
        """LR test raises ValueError with dict formulas."""
        data = _make_panel_with_extra()
        formulas = {0: "y ~ x1", 1: "y ~ x2"}
        data_sub = data[data["entity"].isin([0, 1])]
        sur = SUR(formulas, data_sub, "entity", "time")
        with pytest.raises(ValueError, match="heterogeneous"):
            sur.lr_test_homogeneity()

    def test_against_manual_gls(self):
        """Compare SUR with manual GLS computation using numpy."""
        rng = np.random.default_rng(42)
        N = 3
        T = 100

        # Generate data with known Sigma
        sigma_true = np.array(
            [
                [1.0, 0.6, 0.3],
                [0.6, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]
        )
        L = np.linalg.cholesky(sigma_true)

        entities, times, ys, x1s, x2s = [], [], [], [], []
        for t in range(T):
            eps = L @ rng.standard_normal(N)
            for i in range(N):
                x1 = rng.standard_normal()
                x2 = rng.standard_normal()
                y = 1.0 + 2.0 * x1 + 0.5 * x2 + eps[i]
                entities.append(i)
                times.append(t)
                x1s.append(x1)
                x2s.append(x2)
                ys.append(y)

        df = pd.DataFrame(
            {
                "entity": entities,
                "time": times,
                "y": ys,
                "x1": x1s,
                "x2": x2s,
            }
        )

        # SUR estimate
        sur = SUR("y ~ x1 + x2", df, "entity", "time")
        res = sur.fit()

        # Manual GLS: beta = (X' Omega^{-1} X)^{-1} X' Omega^{-1} y
        # Build stacked system
        K_full = 3  # intercept + x1 + x2
        X_list = []
        y_list = []
        for i in range(N):
            mask = df["entity"] == i
            entity_df = df[mask].sort_values("time")
            X_i = np.column_stack(
                [
                    np.ones(T),
                    entity_df["x1"].values,
                    entity_df["x2"].values,
                ]
            )
            X_list.append(X_i)
            y_list.append(entity_df["y"].values)

        # OLS residuals for Sigma estimation
        ols_resids = []
        for i in range(N):
            beta_ols = np.linalg.lstsq(X_list[i], y_list[i], rcond=None)[0]
            ols_resids.append(y_list[i] - X_list[i] @ beta_ols)

        # Estimate Sigma manually
        sigma_hat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                sigma_hat[i, j] = ols_resids[i] @ ols_resids[j] / T

        # Build block-diagonal X and stacked y
        total_k = N * K_full
        X_block = np.zeros((N * T, total_k))
        y_stack = np.zeros(N * T)
        for i in range(N):
            row_start = i * T
            col_start = i * K_full
            X_block[row_start : row_start + T, col_start : col_start + K_full] = X_list[i]
            y_stack[row_start : row_start + T] = y_list[i]

        # Omega^{-1} = Sigma^{-1} kron I_T
        sigma_inv = np.linalg.inv(sigma_hat)
        Omega_inv = np.kron(sigma_inv, np.eye(T))

        # GLS
        A = X_block.T @ Omega_inv @ X_block
        b = X_block.T @ Omega_inv @ y_stack
        beta_manual = np.linalg.solve(A, b)

        # Compare with SUR
        beta_sur = res.params.values
        np.testing.assert_allclose(
            beta_sur,
            beta_manual,
            atol=1e-8,
            err_msg="SUR params don't match manual GLS computation",
        )
