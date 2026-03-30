"""
Tests for Mean Group (MG) and Pooled Mean Group (PMG) estimators.

Covers:
- Basic functionality and interface
- Statistical properties (consistency under heterogeneity)
- Edge cases (small N, unbalanced, excluded entities)
- Results output (entity_summary, coefficient_table, summary)
- PMG convergence and long-run homogeneity
- Monte Carlo: MG consistent, FE biased under slope heterogeneity
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.core.results import PanelResults
from panelbox.models.static.mean_group import (
    MeanGroupEstimator,
    MeanGroupResults,
    PMGResults,
    PooledMeanGroupEstimator,
    hausman_mg_pmg,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_heterogeneous_panel(N=20, T=20, beta_true=None, sigma_beta=None, sigma_eps=1.0, seed=42):
    """
    DGP with slope heterogeneity.

    y_it = alpha_i + beta_i' x_it + eps_it
    beta_i ~ N(beta_true, diag(sigma_beta^2))

    Returns DataFrame and the true mean beta.
    """
    rng = np.random.default_rng(seed)

    if beta_true is None:
        beta_true = np.array([1.0, 3.0, -2.0])  # intercept, x1, x2
    if sigma_beta is None:
        sigma_beta = np.array([0.5, 1.0, 0.8])

    K = len(beta_true)
    entity_betas = rng.normal(beta_true, sigma_beta, size=(N, K))

    rows = []
    for i in range(N):
        x1 = rng.normal(0, 1, T)
        x2 = rng.normal(0, 1, T)
        alpha_i = rng.normal(0, 2)
        eps = rng.normal(0, sigma_eps, T)
        y = entity_betas[i, 0] + entity_betas[i, 1] * x1 + entity_betas[i, 2] * x2 + alpha_i + eps
        for t in range(T):
            rows.append({"entity": i, "time": t, "y": y[t], "x1": x1[t], "x2": x2[t]})

    return pd.DataFrame(rows), beta_true, entity_betas


def _make_homogeneous_panel(N=20, T=20, beta_true=None, sigma_eps=1.0, seed=42):
    """DGP with homogeneous slopes (all entities share same beta)."""
    rng = np.random.default_rng(seed)
    if beta_true is None:
        beta_true = np.array([1.0, 3.0, -2.0])

    rows = []
    for i in range(N):
        x1 = rng.normal(0, 1, T)
        x2 = rng.normal(0, 1, T)
        alpha_i = rng.normal(0, 5)
        eps = rng.normal(0, sigma_eps, T)
        y = beta_true[0] + beta_true[1] * x1 + beta_true[2] * x2 + alpha_i + eps
        for t in range(T):
            rows.append({"entity": i, "time": t, "y": y[t], "x1": x1[t], "x2": x2[t]})

    return pd.DataFrame(rows), beta_true


def _make_ecm_panel(N=15, T=30, theta_true=None, phi=-0.3, seed=42):
    """
    DGP for PMG: error-correction model.

    y_it = phi_i * (y_{i,t-1} - theta' x_{i,t-1}) + delta_i * Dx_it + mu_i + eps_it

    Long-run: y_it = theta' x_it (homogeneous)
    Short-run: entity-specific
    """
    rng = np.random.default_rng(seed)
    if theta_true is None:
        theta_true = np.array([2.0])

    len(theta_true)
    rows = []
    for i in range(N):
        phi_i = phi + rng.normal(0, 0.05)  # entity-specific EC speed
        delta_i = rng.normal(0.5, 0.3)
        mu_i = rng.normal(0, 1)

        x = np.cumsum(rng.normal(0, 1, T))  # I(1) regressor
        y = np.zeros(T)
        y[0] = theta_true[0] * x[0] + rng.normal(0, 1)

        for t in range(1, T):
            ecm = y[t - 1] - theta_true[0] * x[t - 1]
            dx = x[t] - x[t - 1]
            y[t] = y[t - 1] + phi_i * ecm + delta_i * dx + mu_i + rng.normal(0, 0.5)

        for t in range(T):
            rows.append({"entity": i, "time": t, "y": y[t], "x1": x[t]})

    return pd.DataFrame(rows), theta_true


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def het_panel():
    """Heterogeneous slope panel: N=20, T=20."""
    data, beta_true, entity_betas = _make_heterogeneous_panel()
    return data, beta_true, entity_betas


@pytest.fixture
def hom_panel():
    """Homogeneous slope panel: N=20, T=20."""
    data, beta_true = _make_homogeneous_panel()
    return data, beta_true


@pytest.fixture
def ecm_panel():
    """Error-correction model panel for PMG."""
    data, theta_true = _make_ecm_panel()
    return data, theta_true


# ===========================================================================
# TestMeanGroupEstimator
# ===========================================================================


class TestMeanGroupEstimator:
    """Tests for the Mean Group estimator."""

    # --- Basic ---

    def test_init(self, het_panel):
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        assert model.formula == "y ~ x1 + x2"
        assert model.data.entity_col == "entity"
        assert model.min_obs_per_entity == 10

    def test_fit_returns_mg_results(self, het_panel):
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()
        assert isinstance(results, MeanGroupResults)
        assert isinstance(results, PanelResults)
        assert len(results.params) == 3  # Intercept, x1, x2
        assert "Intercept" in results.params.index
        assert "x1" in results.params.index
        assert "x2" in results.params.index

    def test_mg_equals_average(self, het_panel):
        """beta_MG == mean(beta_i) exactly."""
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        # Manually compute mean of entity betas
        entity_betas = np.array(
            [results.entity_params[eid].values for eid in results.entity_params]
        )
        manual_mean = entity_betas.mean(axis=0)

        np.testing.assert_allclose(results.params.values, manual_mean, rtol=1e-12)

    def test_mg_variance_formula(self, het_panel):
        """Var == (1/N(N-1)) * sum(beta_i - beta_MG)^2."""
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        entity_betas = np.array(
            [results.entity_params[eid].values for eid in results.entity_params]
        )
        beta_mg = results.params.values
        N = len(entity_betas)

        deviations = entity_betas - beta_mg[None, :]
        manual_vcov = (deviations.T @ deviations) / (N * (N - 1))

        np.testing.assert_allclose(results.cov_params.values, manual_vcov, rtol=1e-12)

    # --- Statistical behavior ---

    def test_homogeneous_mg_close_to_fe(self, hom_panel):
        """With homogeneous slopes, MG should be close to FE."""
        from panelbox.models.static.fixed_effects import FixedEffects

        data, _beta_true = hom_panel
        mg = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        mg_res = mg.fit()

        fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
        fe_res = fe.fit()

        # Compare slopes (FE has no intercept, MG has intercept)
        for var in ["x1", "x2"]:
            assert abs(mg_res.params[var] - fe_res.params[var]) < 0.5, (
                f"MG and FE differ too much on {var}: "
                f"MG={mg_res.params[var]:.4f}, FE={fe_res.params[var]:.4f}"
            )

    def test_heterogeneous_mg_consistent(self, het_panel):
        """DGP with beta_i ~ N(beta, sigma_b): MG recovers beta."""
        data, beta_true, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        # MG should be close to true mean betas
        # Intercept, x1, x2
        for i, var in enumerate(["Intercept", "x1", "x2"]):
            assert abs(results.params[var] - beta_true[i]) < 0.5, (
                f"MG estimate for {var}: {results.params[var]:.4f}, true: {beta_true[i]:.4f}"
            )

    def test_swamy_homogeneous_not_reject(self, hom_panel):
        """Swamy test should not reject under homogeneous slopes."""
        data, _ = hom_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        sw = results.swamy_test_result
        assert "statistic" in sw
        assert "pvalue" in sw
        assert "df" in sw
        # With homogeneous slopes, p-value should be large (don't reject H0)
        # Use a lenient threshold since small samples can cause spurious rejection
        assert sw["pvalue"] > 0.01, (
            f"Swamy test rejected H0 under homogeneity: p={sw['pvalue']:.4f}"
        )

    def test_swamy_heterogeneous_reject(self, het_panel):
        """Swamy test should reject under heterogeneous slopes."""
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        sw = results.swamy_test_result
        assert sw["pvalue"] < 0.05, (
            f"Swamy test failed to reject H0 under heterogeneity: p={sw['pvalue']:.4f}"
        )
        assert sw["reject"] is True

    # --- Edge cases ---

    def test_small_n_3(self):
        """MG works with as few as N=3 entities."""
        data, _, _ = _make_heterogeneous_panel(N=3, T=15, seed=99)
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        assert results.n_entities_used == 3
        assert len(results.params) == 3
        assert np.all(np.isfinite(results.params.values))
        assert np.all(np.isfinite(results.std_errors.values))

    def test_unbalanced(self):
        """MG handles unbalanced panels."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(10):
            T_i = 15 + rng.integers(0, 10)  # varying T per entity
            x1 = rng.normal(0, 1, T_i)
            x2 = rng.normal(0, 1, T_i)
            beta_i = rng.normal([1, 3, -2], [0.5, 1, 0.8])
            y = beta_i[0] + beta_i[1] * x1 + beta_i[2] * x2 + rng.normal(0, 1, T_i)
            for t in range(T_i):
                rows.append({"entity": i, "time": t, "y": y[t], "x1": x1[t], "x2": x2[t]})

        data = pd.DataFrame(rows)
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        assert results.n_entities_used == 10
        assert np.all(np.isfinite(results.params.values))

    def test_entity_excluded_insufficient_obs(self):
        """Entities with T < min_obs_per_entity are excluded."""
        rng = np.random.default_rng(42)
        rows = []
        # 8 entities with T=20, 2 entities with T=3
        for i in range(10):
            T_i = 20 if i < 8 else 3
            x1 = rng.normal(0, 1, T_i)
            x2 = rng.normal(0, 1, T_i)
            y = 1 + 2 * x1 - x2 + rng.normal(0, 1, T_i)
            for t in range(T_i):
                rows.append({"entity": i, "time": t, "y": y[t], "x1": x1[t], "x2": x2[t]})

        data = pd.DataFrame(rows)
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time", min_obs_per_entity=10)
        with pytest.warns(UserWarning, match="excluded"):
            results = model.fit()

        assert results.n_entities_used == 8
        assert len(results.entities_excluded) == 2

    def test_entity_excluded_singular_x(self):
        """Entity with singular X matrix is excluded."""
        rng = np.random.default_rng(42)
        rows = []
        # 9 normal entities
        for i in range(9):
            T_i = 20
            x1 = rng.normal(0, 1, T_i)
            x2 = rng.normal(0, 1, T_i)
            y = 1 + 2 * x1 - x2 + rng.normal(0, 1, T_i)
            for t in range(T_i):
                rows.append({"entity": i, "time": t, "y": y[t], "x1": x1[t], "x2": x2[t]})

        # 1 entity with x2 = 2 * x1 (collinear)
        T_i = 20
        x1 = rng.normal(0, 1, T_i)
        x2 = 2 * x1  # perfectly collinear
        y = 1 + 2 * x1 - x2 + rng.normal(0, 1, T_i)
        for t in range(T_i):
            rows.append({"entity": 9, "time": t, "y": y[t], "x1": x1[t], "x2": x2[t]})

        data = pd.DataFrame(rows)
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time", min_obs_per_entity=10)
        with pytest.warns(UserWarning, match="rank-deficient"):
            results = model.fit()

        assert results.n_entities_used == 9
        assert 9 in results.entities_excluded

    def test_single_regressor(self):
        """MG works with a single regressor."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(10):
            T_i = 20
            x1 = rng.normal(0, 1, T_i)
            beta_i = rng.normal(3, 0.5)
            y = beta_i * x1 + rng.normal(0, 1, T_i)
            for t in range(T_i):
                rows.append({"entity": i, "time": t, "y": y[t], "x1": x1[t]})

        data = pd.DataFrame(rows)
        model = MeanGroupEstimator("y ~ x1", data, "entity", "time")
        results = model.fit()

        assert len(results.params) == 2  # Intercept + x1
        assert abs(results.params["x1"] - 3.0) < 1.0

    # --- Results output ---

    def test_entity_summary(self, het_panel):
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        # Pick first entity
        first_entity = next(iter(results.entity_params.keys()))
        summary_str = results.entity_summary(first_entity)

        assert isinstance(summary_str, str)
        assert str(first_entity) in summary_str
        assert "R-squared" in summary_str
        assert "x1" in summary_str

        # Invalid entity raises
        with pytest.raises(ValueError, match="not found"):
            results.entity_summary("nonexistent_entity")

    def test_coefficient_table(self, het_panel):
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        table = results.coefficient_table()
        assert isinstance(table, pd.DataFrame)
        assert table.shape[0] == results.n_entities_used
        assert "x1" in table.columns
        assert "x2" in table.columns
        assert table.index.name == "entity"

    def test_summary_output(self, het_panel):
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        summary_str = results.summary()
        assert isinstance(summary_str, str)
        assert "Swamy" in summary_str
        assert "Entities Used" in summary_str
        assert "x1" in summary_str

    def test_weighted_mg(self, het_panel):
        """Weighted MG uses entity-level weights."""
        data, _, _ = het_panel
        N = data["entity"].nunique()
        weights = np.ones(N)  # uniform weights = simple average
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time", weights=weights)
        results = model.fit()
        assert np.all(np.isfinite(results.params.values))

        # Uniform weights should give same result as unweighted
        model_uw = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results_uw = model_uw.fit()
        np.testing.assert_allclose(results.params.values, results_uw.params.values, rtol=1e-10)

    def test_weighted_mg_wrong_length(self, het_panel):
        """Wrong-length weights raise ValueError."""
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time", weights=np.ones(3))
        with pytest.raises(ValueError, match="weights must have length"):
            model.fit()

    def test_estimate_coefficients_method(self, het_panel):
        """_estimate_coefficients returns array of MG betas."""
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        betas = model._estimate_coefficients()
        assert isinstance(betas, np.ndarray)
        assert len(betas) == 3

    def test_plot_coefficient_distribution(self, het_panel):
        """plot_coefficient_distribution returns an Axes object."""
        pytest.importorskip("matplotlib")
        data, _, _ = het_panel
        model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned_ax = results.plot_coefficient_distribution("x1", ax=ax)
        assert returned_ax is ax
        plt.close(fig)

        # Also test without providing ax
        returned_ax2 = results.plot_coefficient_distribution("x1")
        assert returned_ax2 is not None
        plt.close("all")

    def test_summary_with_excluded_entities(self):
        """Summary mentions excluded entities when present."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(10):
            T_i = 20 if i < 8 else 3
            x1 = rng.normal(0, 1, T_i)
            y = 1 + 2 * x1 + rng.normal(0, 1, T_i)
            for t in range(T_i):
                rows.append({"entity": i, "time": t, "y": y[t], "x1": x1[t]})

        data = pd.DataFrame(rows)
        model = MeanGroupEstimator("y ~ x1", data, "entity", "time", min_obs_per_entity=10)
        with pytest.warns(UserWarning, match="Excluded"):
            results = model.fit()

        summary_str = results.summary()
        assert "Excluded" in summary_str


# ===========================================================================
# TestPooledMeanGroup
# ===========================================================================


class TestPooledMeanGroup:
    """Tests for the Pooled Mean Group estimator."""

    def test_convergence(self, ecm_panel):
        data, _theta_true = ecm_panel
        model = PooledMeanGroupEstimator("y ~ x1", data, "entity", "time", max_iter=200)
        results = model.fit()

        assert isinstance(results, PMGResults)
        assert isinstance(results, PanelResults)
        assert results.converged

    def test_long_run_homogeneity(self, ecm_panel):
        """PMG long-run coefficients should be close to true theta."""
        data, theta_true = ecm_panel
        model = PooledMeanGroupEstimator("y ~ x1", data, "entity", "time", max_iter=200)
        results = model.fit()

        # Long-run coefficients should be close to truth
        for i, var in enumerate(results.long_run_params.index):
            estimated = results.long_run_params[var]
            true_val = theta_true[i]
            assert abs(estimated - true_val) < 1.0, (
                f"PMG long-run {var}: {estimated:.4f}, true: {true_val:.4f}"
            )

    def test_short_run_heterogeneity(self, ecm_panel):
        """PMG entity-specific short-run coefficients should vary."""
        data, _ = ecm_panel
        model = PooledMeanGroupEstimator("y ~ x1", data, "entity", "time", max_iter=200)
        results = model.fit()

        # Error-correction speeds should be entity-specific
        phis = list(results.entity_phi.values())
        assert len(phis) == results.n_entities_used
        # There should be some variation across entities
        assert np.std(phis) > 0.001

        # Short-run coefficients should exist for each entity
        assert len(results.entity_short_run) == results.n_entities_used

    def test_pmg_summary_output(self, ecm_panel):
        """PMG summary includes long-run coefficients and convergence info."""
        data, _ = ecm_panel
        model = PooledMeanGroupEstimator("y ~ x1", data, "entity", "time", max_iter=200)
        results = model.fit()

        summary_str = results.summary()
        assert isinstance(summary_str, str)
        assert "Long-run" in summary_str
        assert "Converged" in summary_str
        assert "Error-correction" in summary_str

    def test_pmg_estimate_coefficients(self, ecm_panel):
        """_estimate_coefficients returns array."""
        data, _ = ecm_panel
        model = PooledMeanGroupEstimator("y ~ x1", data, "entity", "time", max_iter=200)
        betas = model._estimate_coefficients()
        assert isinstance(betas, np.ndarray)
        assert len(betas) > 0

    def test_pmg_entity_excluded(self):
        """PMG excludes entities with too few observations."""
        rng = np.random.default_rng(42)
        rows = []
        # 8 entities with T=30, 2 with T=3
        for i in range(10):
            T_i = 30 if i < 8 else 3
            x = np.cumsum(rng.normal(0, 1, T_i))
            y = 2.0 * x + rng.normal(0, 0.5, T_i)
            for t in range(T_i):
                rows.append({"entity": i, "time": t, "y": y[t], "x1": x[t]})

        data = pd.DataFrame(rows)
        model = PooledMeanGroupEstimator("y ~ x1", data, "entity", "time")
        with pytest.warns(UserWarning, match="excluded"):
            results = model.fit()

        assert results.n_entities_used == 8
        assert len(results.entities_excluded) == 2

    def test_hausman_mg_vs_pmg(self, ecm_panel):
        """Hausman test for MG vs PMG."""
        data, _ = ecm_panel
        mg = MeanGroupEstimator("y ~ x1", data, "entity", "time")
        mg_res = mg.fit()

        pmg = PooledMeanGroupEstimator("y ~ x1", data, "entity", "time", max_iter=200)
        pmg_res = pmg.fit()

        h_test = hausman_mg_pmg(mg_res, pmg_res)

        assert "statistic" in h_test
        assert "pvalue" in h_test
        assert "df" in h_test
        assert "reject" in h_test
        assert np.isfinite(h_test["statistic"])
        assert 0.0 <= h_test["pvalue"] <= 1.0
        assert h_test["df"] > 0


# ===========================================================================
# TestMeanGroupMonteCarlo
# ===========================================================================


class TestMeanGroupMonteCarlo:
    """Monte Carlo tests for MG consistency and FE bias."""

    @pytest.mark.slow
    def test_mg_unbiased_heterogeneous(self):
        """
        R=200 replicas, heterogeneous DGP:
        |mean(beta_MG) - beta_true| < 0.05 for slope coefficients.
        """
        R = 200
        N = 20
        T = 20
        beta_true = np.array([1.0, 3.0, -2.0])  # intercept, x1, x2
        sigma_beta = np.array([0.5, 1.0, 0.8])

        mg_estimates = []

        for r in range(R):
            data, _, _ = _make_heterogeneous_panel(
                N=N, T=T, beta_true=beta_true, sigma_beta=sigma_beta, seed=r
            )
            model = MeanGroupEstimator("y ~ x1 + x2", data, "entity", "time")
            results = model.fit()
            mg_estimates.append(results.params.values)

        mg_estimates = np.array(mg_estimates)
        mg_mean = mg_estimates.mean(axis=0)

        # MG should be unbiased: mean across replications close to truth
        # Intercept absorbs entity-level heterogeneity so allow larger tolerance
        for i, var in enumerate(["Intercept", "x1", "x2"]):
            tol = 0.10 if var == "Intercept" else 0.05
            bias = abs(mg_mean[i] - beta_true[i])
            assert bias < tol, (
                f"MG biased for {var}: mean={mg_mean[i]:.4f}, "
                f"true={beta_true[i]:.4f}, bias={bias:.4f}"
            )

    @pytest.mark.slow
    def test_fe_biased_heterogeneous(self):
        """
        Under slope heterogeneity correlated with regressor variance,
        FE is biased: |mean(beta_FE) - beta_true| > 0.10 for at least one slope.

        FE computes a variance-weighted average of entity slopes. When entities
        with higher X variance have systematically different slopes, the FE
        estimate diverges from the simple mean (which MG targets).
        """
        from panelbox.models.static.fixed_effects import FixedEffects

        R = 200
        N = 20
        T = 20
        # True mean slopes
        beta_true_slopes = np.array([3.0, -2.0])

        fe_estimates = []

        for r in range(R):
            rng = np.random.default_rng(r)
            rows = []
            for i in range(N):
                # Entities with higher index have higher X variance
                # AND systematically different slopes
                x_scale = 0.5 + 2.0 * (i / N)  # x variance increases with i
                slope_shift = 2.0 * (i / N - 0.5)  # slopes vary with i
                beta_i = beta_true_slopes + slope_shift

                alpha_i = rng.normal(0, 5)
                x1 = rng.normal(0, x_scale, T)
                x2 = rng.normal(0, x_scale, T)
                eps = rng.normal(0, 1, T)
                y = alpha_i + beta_i[0] * x1 + beta_i[1] * x2 + eps
                for t in range(T):
                    rows.append(
                        {
                            "entity": i,
                            "time": t,
                            "y": y[t],
                            "x1": x1[t],
                            "x2": x2[t],
                        }
                    )

            data = pd.DataFrame(rows)
            model = FixedEffects("y ~ x1 + x2", data, "entity", "time")
            results = model.fit()
            fe_estimates.append(results.params.values)

        fe_estimates = np.array(fe_estimates)
        fe_mean = fe_estimates.mean(axis=0)

        # FE slopes should diverge from the simple mean of entity slopes
        max_bias = max(abs(fe_mean[i] - beta_true_slopes[i]) for i in range(len(beta_true_slopes)))

        assert max_bias > 0.10, (
            f"FE not biased enough under heterogeneity: "
            f"FE_mean={fe_mean}, true_slopes={beta_true_slopes}, max_bias={max_bias:.4f}"
        )
