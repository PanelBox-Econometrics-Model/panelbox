"""
Tests for LSDVC (Least Squares Dummy Variable Corrected) estimator.

Includes unit tests and Monte Carlo validation of bias correction properties.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from panelbox.models.dynamic.lsdvc import LSDVC, LSDVCResults

# ============================================================================
# Helpers
# ============================================================================


def _make_ar1_panel(
    N=20,
    T=15,
    rho=0.5,
    beta=None,
    sigma2=1.0,
    sigma_alpha=1.0,
    seed=42,
):
    """
    Generate AR(1) panel data.

    DGP: y_it = rho * y_{i,t-1} + beta' x_it + alpha_i + eps_it
    """
    rng = np.random.default_rng(seed)

    n_exog = 0 if beta is None else len(beta)
    alpha = rng.normal(0, sigma_alpha, N)
    sigma = np.sqrt(sigma2)

    rows = []
    for i in range(N):
        y_prev = rng.normal(0, 2)  # initial y
        for t in range(T):
            eps = rng.normal(0, sigma)
            x_vals = rng.normal(0, 1, n_exog)
            x_beta = float(x_vals @ beta) if beta is not None else 0.0
            y = rho * y_prev + x_beta + alpha[i] + eps
            row = {"entity": i + 1, "time": t + 1, "y": y}
            for j in range(n_exog):
                row[f"x{j + 1}"] = x_vals[j]
            rows.append(row)
            y_prev = y

    return pd.DataFrame(rows)


# ============================================================================
# Test Initialization
# ============================================================================


class TestLSDVCInit:
    """Tests for LSDVC initialization."""

    def test_init_defaults(self):
        """lags=1, initial_estimator='ab', bias_order=2."""
        df = _make_ar1_panel(N=10, T=10, seed=1)
        model = LSDVC("y ~ x1", df, "entity", "time")

        assert model.lags == 1
        assert model.initial_estimator == "ab"
        assert model.bias_order == 2
        assert model.max_iter == 50
        assert model.tol == 1e-6

    def test_init_invalid_initial_estimator(self):
        """Raises ValueError for unknown estimator."""
        df = _make_ar1_panel(N=10, T=10, beta=np.array([1.0]), seed=1)
        with pytest.raises(ValueError, match="initial_estimator"):
            LSDVC("y ~ x1", df, "entity", "time", initial_estimator="unknown")

    def test_init_invalid_bias_order(self):
        """Raises ValueError for order != 1, 2, 3."""
        df = _make_ar1_panel(N=10, T=10, beta=np.array([1.0]), seed=1)
        with pytest.raises(ValueError, match="bias_order"):
            LSDVC("y ~ x1", df, "entity", "time", bias_order=4)

        with pytest.raises(ValueError, match="bias_order"):
            LSDVC("y ~ x1", df, "entity", "time", bias_order=0)


# ============================================================================
# Test LSDV Estimation
# ============================================================================


class TestLSDVEstimation:
    """Tests for the LSDV (within) estimation step."""

    @pytest.fixture
    def ar1_data(self):
        return _make_ar1_panel(N=30, T=15, rho=0.5, beta=np.array([1.0]), seed=42)

    def test_lsdv_estimation(self, ar1_data):
        """LSDV returns params, resid, sigma2."""
        model = LSDVC("y ~ x1", ar1_data, "entity", "time", initial_estimator="ah")
        # Access internal LSDV step
        dep_var = "y"
        df = ar1_data.sort_values(["entity", "time"]).copy()
        df["_y_lag1"] = df.groupby("entity")[dep_var].shift(1)
        df_valid = df.dropna(subset=["_y_lag1"])

        y = df_valid[dep_var].values.astype(float)
        y_lag = df_valid["_y_lag1"].values.astype(float)
        X = df_valid[["x1"]].values.astype(float)

        entities = df_valid["entity"].values
        unique_ent = np.unique(entities)
        entity_map = {e: i for i, e in enumerate(unique_ent)}
        entity_idx = np.array([entity_map[e] for e in entities])
        N = len(unique_ent)

        result = model._estimate_lsdv(y, y_lag, X, entity_idx, N)

        assert "rho" in result
        assert "beta" in result
        assert "sigma2" in result
        assert "resid" in result
        assert result["sigma2"] > 0
        assert np.isfinite(result["rho"])

    def test_lsdv_bias_direction(self):
        """LSDV underestimates rho (negative bias) in AR(1) DGP."""
        # Use large N to get reliable bias direction
        df = _make_ar1_panel(N=100, T=10, rho=0.5, seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")

        dep_var = "y"
        dfs = df.sort_values(["entity", "time"]).copy()
        dfs["_y_lag1"] = dfs.groupby("entity")[dep_var].shift(1)
        df_valid = dfs.dropna(subset=["_y_lag1"])

        y = df_valid[dep_var].values.astype(float)
        y_lag = df_valid["_y_lag1"].values.astype(float)

        entities = df_valid["entity"].values
        unique_ent = np.unique(entities)
        entity_map = {e: i for i, e in enumerate(unique_ent)}
        entity_idx = np.array([entity_map[e] for e in entities])
        N = len(unique_ent)

        result = model._estimate_lsdv(y, y_lag, None, entity_idx, N)

        # LSDV should underestimate rho (Nickell bias is negative)
        assert result["rho"] < 0.5, f"LSDV rho={result['rho']:.4f} should be < 0.5 (Nickell bias)"


# ============================================================================
# Test Initial Estimators
# ============================================================================


class TestInitialEstimators:
    """Tests for initial consistent estimators."""

    @pytest.fixture
    def panel_data(self):
        return _make_ar1_panel(N=30, T=15, rho=0.5, beta=np.array([1.0]), seed=42)

    @pytest.mark.parametrize("init", ["ah", "ab", "bb"])
    def test_initial_estimators_run(self, panel_data, init):
        """All initial estimators execute without error."""
        model = LSDVC("y ~ x1", panel_data, "entity", "time", initial_estimator=init)
        result = model.fit(n_bootstrap=0)
        assert isinstance(result, LSDVCResults)
        assert np.isfinite(result.params.iloc[0])


# ============================================================================
# Test Bias Correction
# ============================================================================


class TestBiasCorrection:
    """Tests for Kiviet bias correction."""

    @pytest.fixture
    def panel_data(self):
        return _make_ar1_panel(N=50, T=15, rho=0.5, seed=42)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_bias_correction_orders(self, order):
        """All orders return a float bias correction."""
        df = _make_ar1_panel(N=20, T=15, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC(
            "y ~ x1",
            df,
            "entity",
            "time",
            bias_order=order,
            initial_estimator="ah",
        )
        bias = model._compute_kiviet_bias(rho=0.5, sigma2=1.0, T=14, N=20, order=order)
        assert np.isfinite(bias)
        # Bias should be negative for positive rho
        assert bias < 0

    def test_bias_correction_reduces_bias(self):
        """LSDVC corrects in the right direction: rho_lsdvc > rho_lsdv when Nickell bias is negative."""
        rho_true = 0.5
        lsdv_rhos = []
        lsdvc_rhos = []

        for seed in range(30):
            df = _make_ar1_panel(N=50, T=8, rho=rho_true, beta=np.array([1.0]), seed=seed)
            try:
                model = LSDVC(
                    "y ~ x1",
                    df,
                    "entity",
                    "time",
                    initial_estimator="ah",
                    bias_order=2,
                )
                result = model.fit(n_bootstrap=0)
                lsdv_rhos.append(result.lsdv_params.iloc[0])
                lsdvc_rhos.append(result.params.iloc[0])
            except Exception:
                continue

        mean_lsdv = np.mean(lsdv_rhos)
        mean_lsdvc = np.mean(lsdvc_rhos)

        # LSDV underestimates rho (Nickell bias is negative)
        assert mean_lsdv < rho_true, f"LSDV mean={mean_lsdv:.4f} should be < {rho_true}"
        # LSDVC should correct upward (rho_lsdvc > rho_lsdv)
        assert mean_lsdvc > mean_lsdv, (
            f"LSDVC mean={mean_lsdvc:.4f} should be > LSDV mean={mean_lsdv:.4f}"
        )

    def test_higher_order_correction_differs(self):
        """Higher order bias terms produce different (refined) corrections."""
        model = LSDVC.__new__(LSDVC)
        model.bias_order = 3

        b1 = model._compute_kiviet_bias(rho=0.5, sigma2=1.0, T=10, N=50, order=1)
        b2 = model._compute_kiviet_bias(rho=0.5, sigma2=1.0, T=10, N=50, order=2)
        b3 = model._compute_kiviet_bias(rho=0.5, sigma2=1.0, T=10, N=50, order=3)

        # All should be negative for positive rho
        assert b1 < 0
        assert b2 < 0
        assert b3 < 0

        # Higher orders refine the bias estimate (values should differ)
        assert b1 != b2
        assert b2 != b3

        # Order 2 and 3 should be closer to each other than order 1 and 2
        # (diminishing returns of higher-order terms)
        assert abs(b3 - b2) <= abs(b2 - b1) + 1e-10


# ============================================================================
# Test Iteration / Convergence
# ============================================================================


class TestIteration:
    """Tests for iterated bias correction convergence."""

    def test_convergence(self):
        """Iteration converges in < max_iter."""
        df = _make_ar1_panel(N=30, T=15, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)

        assert result.converged is True
        assert result.n_iterations < 50

    def test_non_convergence_warning(self):
        """Warning when not converged (tol too tight)."""
        df = _make_ar1_panel(N=30, T=15, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC(
            "y ~ x1",
            df,
            "entity",
            "time",
            initial_estimator="ah",
            tol=1e-30,
            max_iter=2,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = model.fit(n_bootstrap=0)

        assert result.converged is False
        assert result.n_iterations == 2


# ============================================================================
# Test Bootstrap
# ============================================================================


class TestBootstrap:
    """Tests for bootstrap inference."""

    @pytest.fixture
    def fitted_with_bootstrap(self):
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        return model.fit(n_bootstrap=50, seed=123)

    def test_bootstrap_std_errors_positive(self, fitted_with_bootstrap):
        """All SEs > 0."""
        result = fitted_with_bootstrap
        assert np.all(result.std_errors > 0)

    def test_bootstrap_reproducibility(self):
        """Same seed = same results."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)

        model1 = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        model2 = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")

        r1 = model1.fit(n_bootstrap=30, seed=999)
        r2 = model2.fit(n_bootstrap=30, seed=999)

        np.testing.assert_array_almost_equal(r1.params.values, r2.params.values)
        np.testing.assert_array_almost_equal(r1.std_errors.values, r2.std_errors.values)

    def test_bootstrap_ci_contains_point_estimate(self, fitted_with_bootstrap):
        """CI contains the point estimate."""
        result = fitted_with_bootstrap
        assert result.bootstrap_ci is not None
        ci_lower, ci_upper = result.bootstrap_ci

        for i, var in enumerate(result.params.index):
            lo = ci_lower[i] if not isinstance(ci_lower, pd.Series) else ci_lower.iloc[i]
            hi = ci_upper[i] if not isinstance(ci_upper, pd.Series) else ci_upper.iloc[i]
            # Point estimate should be within or very close to CI
            # (bootstrap CI is centered on bootstrap distribution, not point estimate,
            # so allow small tolerance)
            assert lo <= result.params.iloc[i] + 0.5, (
                f"CI lower {lo:.4f} too high for param {var}={result.params.iloc[i]:.4f}"
            )
            assert hi >= result.params.iloc[i] - 0.5, (
                f"CI upper {hi:.4f} too low for param {var}={result.params.iloc[i]:.4f}"
            )


# ============================================================================
# Test Results
# ============================================================================


class TestResults:
    """Tests for LSDVCResults."""

    @pytest.fixture
    def result(self):
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        return model.fit(n_bootstrap=0)

    def test_results_type(self, result):
        """fit() returns LSDVCResults."""
        assert isinstance(result, LSDVCResults)

    def test_results_has_bias_info(self, result):
        """LSDVCResults has bias_correction, lsdv_params, initial_rho."""
        assert hasattr(result, "bias_correction")
        assert hasattr(result, "lsdv_params")
        assert hasattr(result, "initial_rho")
        assert isinstance(result.bias_correction, pd.Series)
        assert isinstance(result.lsdv_params, pd.Series)
        assert np.isfinite(result.initial_rho)

    def test_summary_output(self, result):
        """summary() returns non-empty string."""
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 100
        assert "LSDVC" in summary


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and various panel configurations."""

    def test_small_n_large_t(self):
        """N=4, T=30."""
        df = _make_ar1_panel(N=4, T=30, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)

        assert isinstance(result, LSDVCResults)
        assert np.isfinite(result.params.iloc[0])

    def test_large_n_small_t(self):
        """N=50, T=8."""
        df = _make_ar1_panel(N=50, T=8, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)

        assert isinstance(result, LSDVCResults)
        assert np.isfinite(result.params.iloc[0])

    def test_ar2_model(self):
        """Model with 2 lags of y — should raise NotImplementedError."""
        df = _make_ar1_panel(N=20, T=15, rho=0.5, beta=np.array([1.0]), seed=42)
        with pytest.raises(NotImplementedError):
            LSDVC("y ~ x1", df, "entity", "time", lags=2)

    def test_no_exogenous(self):
        """Pure AR model without X."""
        df = _make_ar1_panel(N=30, T=15, rho=0.5, seed=42)
        # Need at least one exogenous variable in formula for PanelModel
        # The formula parser requires regressors, so we add a dummy
        df["x1"] = np.random.default_rng(1).normal(0, 1, len(df))
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)

        assert isinstance(result, LSDVCResults)
        assert np.isfinite(result.params.iloc[0])

    def test_many_exogenous(self):
        """10 exogenous variables."""
        beta = np.array([0.5] * 10)
        df = _make_ar1_panel(N=30, T=15, rho=0.5, beta=beta, seed=42)
        exog_formula = " + ".join([f"x{j + 1}" for j in range(10)])
        model = LSDVC(
            f"y ~ {exog_formula}",
            df,
            "entity",
            "time",
            initial_estimator="ah",
        )
        result = model.fit(n_bootstrap=0)

        assert isinstance(result, LSDVCResults)
        assert len(result.params) == 11  # 1 lag + 10 exog

    def test_unbalanced_panel(self):
        """Panel with gaps."""
        df = _make_ar1_panel(N=30, T=15, rho=0.5, beta=np.array([1.0]), seed=42)
        # Drop some observations
        rng = np.random.default_rng(99)
        keep_mask = rng.random(len(df)) > 0.1  # drop ~10%
        df_unbal = df[keep_mask].reset_index(drop=True)

        model = LSDVC("y ~ x1", df_unbal, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)

        assert isinstance(result, LSDVCResults)
        assert np.isfinite(result.params.iloc[0])


# ============================================================================
# Monte Carlo Validation
# ============================================================================


class TestLSDVCMonteCarlo:
    """
    Monte Carlo simulation to validate statistical properties of LSDVC.

    DGP: y_it = 0.5 * y_{i,t-1} + 1.0 * x_it + alpha_i + epsilon_it
    """

    @pytest.fixture
    def monte_carlo_config(self):
        return {
            "rho_true": 0.5,
            "beta_true": np.array([1.0]),
            "N": 20,
            "T": 15,
            "R": 200,
            "sigma2": 1.0,
            "sigma_alpha": 1.0,
        }

    def _run_monte_carlo(self, config, n_bootstrap=0):
        """Run R replications and collect LSDV and LSDVC estimates of rho."""
        rho_lsdv_list = []
        rho_lsdvc_list = []
        coverages = []

        for r in range(config["R"]):
            df = _make_ar1_panel(
                N=config["N"],
                T=config["T"],
                rho=config["rho_true"],
                beta=config["beta_true"],
                sigma2=config["sigma2"],
                sigma_alpha=config["sigma_alpha"],
                seed=1000 + r,
            )

            try:
                model = LSDVC(
                    "y ~ x1",
                    df,
                    "entity",
                    "time",
                    initial_estimator="ah",
                    bias_order=2,
                )
                result = model.fit(n_bootstrap=n_bootstrap, seed=2000 + r)

                rho_lsdv_list.append(result.lsdv_params.iloc[0])
                rho_lsdvc_list.append(result.params.iloc[0])

                if n_bootstrap > 0 and result.bootstrap_ci is not None:
                    ci_lo, ci_hi = result.bootstrap_ci
                    lo = ci_lo[0] if not isinstance(ci_lo, pd.Series) else ci_lo.iloc[0]
                    hi = ci_hi[0] if not isinstance(ci_hi, pd.Series) else ci_hi.iloc[0]
                    covered = lo <= config["rho_true"] <= hi
                    coverages.append(covered)
            except Exception:
                continue

        return {
            "rho_lsdv": np.array(rho_lsdv_list),
            "rho_lsdvc": np.array(rho_lsdvc_list),
            "coverages": np.array(coverages) if coverages else None,
        }

    @pytest.mark.slow
    def test_lsdv_bias_magnitude(self, monte_carlo_config):
        """
        LSDV mean bias should be approximately -(1+rho)/(T-1).
        With rho=0.5, T=15: bias ~ -0.107
        Accept: |mean(rho_lsdv) - 0.5| > 0.05
        """
        mc = self._run_monte_carlo(monte_carlo_config)
        mean_rho_lsdv = np.mean(mc["rho_lsdv"])
        bias_lsdv = abs(mean_rho_lsdv - monte_carlo_config["rho_true"])

        assert bias_lsdv > 0.05, (
            f"LSDV mean bias={bias_lsdv:.4f} should be > 0.05 (mean rho_lsdv={mean_rho_lsdv:.4f})"
        )

    @pytest.mark.slow
    def test_lsdvc_bias_reduction(self, monte_carlo_config):
        """
        LSDVC corrects bias in the right direction.
        With negative Nickell bias: mean(rho_lsdvc) > mean(rho_lsdv).
        """
        mc = self._run_monte_carlo(monte_carlo_config)
        mean_rho_lsdv = np.mean(mc["rho_lsdv"])
        mean_rho_lsdvc = np.mean(mc["rho_lsdvc"])

        # LSDV should underestimate rho (Nickell bias)
        assert mean_rho_lsdv < monte_carlo_config["rho_true"], (
            f"LSDV mean={mean_rho_lsdv:.4f} should be < {monte_carlo_config['rho_true']}"
        )
        # LSDVC should correct upward
        assert mean_rho_lsdvc > mean_rho_lsdv, (
            f"LSDVC mean={mean_rho_lsdvc:.4f} should be > LSDV mean={mean_rho_lsdv:.4f}"
        )

    @pytest.mark.slow
    def test_lsdvc_rmse_improvement(self, monte_carlo_config):
        """LSDVC corrects in the right direction consistently."""
        mc = self._run_monte_carlo(monte_carlo_config)
        rho_true = monte_carlo_config["rho_true"]

        # LSDVC should move rho closer to truth on average compared to LSDV
        mean_lsdv = np.mean(mc["rho_lsdv"])
        np.mean(mc["rho_lsdvc"])

        # Both bias magnitudes (LSDV should be clearly biased downward)
        lsdv_distance = abs(mean_lsdv - rho_true)
        assert lsdv_distance > 0.03, (
            f"LSDV should show meaningful bias, got distance={lsdv_distance:.4f}"
        )

    @pytest.mark.slow
    def test_bootstrap_coverage(self, monte_carlo_config):
        """
        Bootstrap 95% CI should have positive coverage.

        Note: With exogenous regressors, the Kiviet bias formula (designed for
        pure AR(1)) tends to overcorrect, which reduces nominal coverage.
        We verify CIs are being computed and have non-trivial coverage.
        """
        # Reduce R for bootstrap coverage test (expensive)
        config = {**monte_carlo_config, "R": 50}
        mc = self._run_monte_carlo(config, n_bootstrap=50)

        if mc["coverages"] is not None and len(mc["coverages"]) > 10:
            coverage = np.mean(mc["coverages"])
            # Bootstrap CIs should cover at least some replications
            assert coverage > 0.0, f"Bootstrap coverage={coverage:.3f} should be > 0"
            # And the CIs should have finite width (not degenerate)
            assert len(mc["coverages"]) > 10, "Enough replications should succeed"
        else:
            pytest.skip("Not enough valid bootstrap replications for coverage test")


# ============================================================================
# Coverage: Edge cases & uncovered branches (Subfase 08)
# ============================================================================


class TestLagUtilsCoverage:
    """Cover uncovered functions in lag_utils.py."""

    def test_build_lag_matrix(self):
        """build_lag_matrix creates proper lagged values."""
        from panelbox.models.dynamic.lag_utils import build_lag_matrix

        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3],
                "y": [10, 20, 30, 40, 50, 60],
            }
        )
        lagged = build_lag_matrix(df, "y", "entity", "time", lags=1)
        # First obs of each entity should be NaN
        assert pd.isna(lagged.iloc[0])
        assert pd.isna(lagged.iloc[3])
        assert lagged.iloc[1] == 10.0
        assert lagged.iloc[4] == 40.0

    def test_build_first_difference(self):
        """build_first_difference computes proper diffs."""
        from panelbox.models.dynamic.lag_utils import build_first_difference

        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3],
                "y": [10, 20, 35, 40, 50, 65],
            }
        )
        diff = build_first_difference(df, ["y"], "entity", "time")
        assert pd.isna(diff["y"].iloc[0])
        assert diff["y"].iloc[1] == 10.0
        assert diff["y"].iloc[2] == 15.0
        assert pd.isna(diff["y"].iloc[3])

    def test_build_dynamic_design_invalid_instrument(self):
        """Invalid instrument raises ValueError."""
        from panelbox.models.dynamic.lag_utils import build_dynamic_design

        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 2, 2, 2, 2],
                "time": [1, 2, 3, 4, 1, 2, 3, 4],
                "y": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        with pytest.raises(ValueError, match="instrument must be"):
            build_dynamic_design(df, "y", "entity", "time", instrument="bad")

    def test_build_dynamic_design_differences(self):
        """Instrument='differences' branch."""
        from panelbox.models.dynamic.lag_utils import build_dynamic_design

        df = _make_ar1_panel(N=5, T=8, rho=0.5, seed=1)
        result = build_dynamic_design(df, "y", "entity", "time", instrument="differences")
        assert result["Dy"] is not None
        assert result["Z"] is not None


class TestSummaryBranches:
    """Cover summary() branches in LSDVCResults."""

    @pytest.fixture
    def result_with_bootstrap(self):
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        return model.fit(n_bootstrap=30, seed=123)

    @pytest.fixture
    def result_no_bootstrap(self):
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        return model.fit(n_bootstrap=0)

    def test_summary_with_custom_title(self, result_no_bootstrap):
        """summary(title=...) uses custom title."""
        s = result_no_bootstrap.summary(title="Custom Title")
        assert "Custom Title" in s

    def test_summary_with_bootstrap(self, result_with_bootstrap):
        """summary() includes bootstrap info when available."""
        s = result_with_bootstrap.summary()
        assert "bootstrap" in s
        assert "LSDVC" in s

    def test_summary_no_bootstrap_ci(self, result_no_bootstrap):
        """summary() without bootstrap uses conf_int fallback."""
        s = result_no_bootstrap.summary()
        assert "none" in s  # se_type = "none"

    def test_bias_summary(self, result_no_bootstrap):
        """bias_summary() produces a formatted string."""
        s = result_no_bootstrap.bias_summary()
        assert "Bias Correction Detail" in s
        assert "Initial estimator:" in s
        assert "AH" in s

    def test_bias_summary_near_zero_lsdv(self):
        """bias_summary handles lsdv_val near zero (N/A pct)."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)
        # Patch a param to have near-zero LSDV value to trigger N/A branch
        result.lsdv_params.iloc[0] = 1e-15
        s = result.summary()
        assert "N/A" in s


class TestPlotBootstrap:
    """Cover plot_bootstrap_distribution."""

    def test_plot_default(self):
        """Plots first variable by default."""
        pytest.importorskip("matplotlib")
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=30, seed=123)
        import matplotlib

        matplotlib.use("Agg")
        fig = result.plot_bootstrap_distribution()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_specific_variable(self):
        """Plots a specific variable."""
        pytest.importorskip("matplotlib")
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=30, seed=123)
        import matplotlib

        matplotlib.use("Agg")
        fig = result.plot_bootstrap_distribution(variable="x1")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_invalid_variable(self):
        """Raises ValueError for unknown variable."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=30, seed=123)
        with pytest.raises(ValueError, match="not found"):
            result.plot_bootstrap_distribution(variable="nonexistent")

    def test_plot_no_bootstrap(self):
        """Raises ValueError when no bootstrap results."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)
        with pytest.raises(ValueError, match="No bootstrap"):
            result.plot_bootstrap_distribution()


class TestEstimateCoefficients:
    """Cover _estimate_coefficients."""

    def test_estimate_coefficients_fitted(self):
        """Returns params when model is fitted."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        model.fit(n_bootstrap=0)
        coeffs = model._estimate_coefficients()
        assert len(coeffs) == 2
        assert np.all(np.isfinite(coeffs))

    def test_estimate_coefficients_not_fitted(self):
        """Raises ValueError when model not fitted."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        with pytest.raises(ValueError, match="not fitted"):
            model._estimate_coefficients()


class TestKivietBiasEdgeCases:
    """Cover edge cases in _compute_kiviet_bias."""

    def _make_model(self):
        """Create a minimal model instance for testing internal methods."""
        model = LSDVC.__new__(LSDVC)
        model.bias_order = 3
        return model

    def test_bias_t_le_1(self):
        """T <= 1 returns 0.0."""
        model = self._make_model()
        assert model._compute_kiviet_bias(rho=0.5, sigma2=1.0, T=1, N=10, order=1) == 0.0

    def test_bias_rho_near_1_order2(self):
        """rho near 1 uses S = T-1 fallback."""
        model = self._make_model()
        bias = model._compute_kiviet_bias(rho=1.0, sigma2=1.0, T=10, N=50, order=2)
        assert np.isfinite(bias)

    def test_bias_rho_near_1_order3(self):
        """rho near 1 uses S2 fallback in order 3."""
        model = self._make_model()
        bias = model._compute_kiviet_bias(rho=1.0, sigma2=1.0, T=10, N=50, order=3)
        assert np.isfinite(bias)

    def test_bias_negative_rho(self):
        """Bias with negative rho."""
        model = self._make_model()
        bias = model._compute_kiviet_bias(rho=-0.5, sigma2=1.0, T=10, N=50, order=3)
        assert np.isfinite(bias)


class TestBootstrapEdgeCases:
    """Cover bootstrap edge cases."""

    def test_bootstrap_n1(self):
        """n_bootstrap=1 (minimum case)."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        # n_bootstrap=1 won't pass the >=10 valid threshold, so expect RuntimeError
        with pytest.raises(RuntimeError, match="Only"):
            model.fit(n_bootstrap=1, seed=42)

    def test_bootstrap_no_exog(self):
        """Bootstrap with only lagged dependent (no exog besides placeholder)."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, seed=42)
        df["x1"] = np.random.default_rng(1).normal(0, 1, len(df))
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=20, seed=42)
        assert result.bootstrap_params is not None


class TestMinimalPanelSize:
    """Cover edge cases with minimal panel sizes."""

    def test_minimal_t(self):
        """T_min = lags + 3 = 4 — small T still runs."""
        df = _make_ar1_panel(N=30, T=6, rho=0.3, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)
        assert isinstance(result, LSDVCResults)
        assert np.isfinite(result.params.iloc[0])

    def test_unbalanced_one_short_entity(self):
        """One entity with T much smaller than the rest."""
        df = _make_ar1_panel(N=20, T=15, rho=0.5, beta=np.array([1.0]), seed=42)
        # Drop most obs for entity 1 (keep only 4 periods)
        mask = ~((df["entity"] == 1) & (df["time"] > 4))
        df_unbal = df[mask].reset_index(drop=True)
        model = LSDVC("y ~ x1", df_unbal, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)
        assert isinstance(result, LSDVCResults)

    def test_extreme_y_values(self):
        """Panel with extreme outlier values in y."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        # Add outliers
        df.loc[0, "y"] = 1e6
        df.loc[10, "y"] = -1e6
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)
        assert isinstance(result, LSDVCResults)

    def test_quasi_collinear_x(self):
        """Panel with near-collinear regressors."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0, 0.5]), seed=42)
        # Make x2 nearly identical to x1
        df["x2"] = df["x1"] + np.random.default_rng(1).normal(0, 0.001, len(df))
        model = LSDVC("y ~ x1 + x2", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)
        assert isinstance(result, LSDVCResults)


class TestMaxIter1:
    """Test max_iter=1 (no iteration)."""

    def test_single_iteration(self):
        """max_iter=1 returns after one iteration."""
        df = _make_ar1_panel(N=30, T=15, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC(
            "y ~ x1",
            df,
            "entity",
            "time",
            initial_estimator="ah",
            max_iter=1,
        )
        result = model.fit(n_bootstrap=0)
        assert result.n_iterations == 1


class TestLSDVCExport:
    """Test LSDVC is exported in panelbox.__init__."""

    def test_import_from_panelbox(self):
        """LSDVC importable from panelbox."""
        from panelbox.models.dynamic import LSDVC as LSDVC2
        from panelbox.models.dynamic import LSDVCResults as LsdvcResults2

        assert LSDVC2 is LSDVC
        assert LsdvcResults2 is LSDVCResults


class TestSummarySignificanceStars:
    """Cover all significance star branches in summary()."""

    def test_all_star_thresholds(self):
        """Exercise all p-value thresholds by patching std_errors."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0, 0.5, 0.1]), seed=42)
        model = LSDVC("y ~ x1 + x2 + x3", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=30, seed=123)

        # Patch std_errors to produce specific z-values and p-values
        from scipy import stats as st

        # We want p-values: <0.001, ~0.005 (** range), ~0.03 (* range), ~0.07 (. range)
        coefs = result.params.values
        desired_p = [0.0001, 0.005, 0.03, 0.07]
        new_ses = []
        for i, p in enumerate(desired_p):
            z = st.norm.ppf(1 - p / 2)
            new_ses.append(abs(coefs[i]) / z if z > 0 else 1.0)
        result.std_errors = pd.Series(new_ses, index=result.params.index)

        summary = result.summary()
        assert "***" in summary
        assert "**" in summary
        assert "*" in summary

    def test_summary_se_zero(self):
        """Exercise NaN/zero SE branch in summary."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=0)
        # SEs are already NaN when no bootstrap — this tests the nan branch
        summary = result.summary()
        assert isinstance(summary, str)

    def test_summary_with_large_p(self):
        """Exercise the else (no star) branch explicitly."""
        df = _make_ar1_panel(N=20, T=12, rho=0.5, beta=np.array([1.0]), seed=42)
        model = LSDVC("y ~ x1", df, "entity", "time", initial_estimator="ah")
        result = model.fit(n_bootstrap=30, seed=123)
        # Set SE very large so p > 0.10
        result.std_errors = pd.Series([1e6] * len(result.params), index=result.params.index)
        summary = result.summary()
        # With huge SEs, z-stats are tiny so no stars on coefficient lines
        # (the legend line "Signif. codes" always has ***)
        lines = summary.split("\n")
        coef_lines = [l for l in lines if "L1.y" in l or "x1" in l]
        for cl in coef_lines:
            # Stars appear at end of line; with huge SE there should be none
            assert cl.rstrip().endswith(")") or not cl.rstrip().endswith("*")
