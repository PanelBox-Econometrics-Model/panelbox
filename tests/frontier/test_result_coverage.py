"""Tests for panelbox/frontier/result.py coverage.

Targets uncovered methods: elasticities, efficient_scale,
compare_functional_form, PanelSFResult.summary (temporal params),
PanelSFResult.test_temporal_constancy, PanelSFResult.variance_decomposition
(TRE 3-component), PanelSFResult.plot_efficiency_evolution, and partial
paths of variance_decomposition (bootstrap), returns_to_scale_test,
compare_distributions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from unittest.mock import MagicMock, PropertyMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.data import DistributionType, FrontierType, ModelType
from panelbox.frontier.result import PanelSFResult, SFResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _make_mock_model(
    *,
    n_obs: int = 100,
    is_panel: bool = False,
    n_entities: int = 10,
    n_periods: int = 5,
    model_type: ModelType = ModelType.CROSS_SECTION,
    frontier_type: FrontierType = FrontierType.PRODUCTION,
    dist: DistributionType = DistributionType.HALF_NORMAL,
    X_df: pd.DataFrame | None = None,
    depvar: str = "ln_y",
    exog: list[str] | None = None,
    data: pd.DataFrame | None = None,
) -> MagicMock:
    """Create a mock model object with required attributes."""
    model = MagicMock()
    model.n_obs = n_obs
    model.is_panel = is_panel
    model.n_entities = n_entities
    model.n_periods = n_periods
    model.model_type = model_type
    model.frontier_type = frontier_type
    model.dist = dist
    model.depvar = depvar
    model.exog = exog or ["ln_K", "ln_L"]
    model.data = data
    model.X_df = X_df
    # y and X for residuals
    model.y = np.random.default_rng(42).normal(size=n_obs)
    model.X = np.random.default_rng(42).normal(size=(n_obs, 2))
    # Remove ols_loglik so summary skips inefficiency test by default
    del model.ols_loglik
    return model


_SENTINEL = object()


def _make_sfresult(
    *,
    param_values: np.ndarray | None = None,
    param_names: list[str] | None = None,
    hessian: np.ndarray | None | object = _SENTINEL,
    model: MagicMock | None = None,
    loglik: float = -150.0,
    converged: bool = True,
) -> SFResult:
    """Factory to create SFResult with controlled parameters."""
    if param_names is None:
        param_names = ["const", "ln_K", "ln_L", "sigma_v", "sigma_u"]
    if param_values is None:
        param_values = np.array([1.0, 0.5, 0.4, 0.3, 0.2])
    if model is None:
        model = _make_mock_model()
    if hessian is _SENTINEL:
        n = len(param_values)
        hessian = -np.eye(n) * 10.0  # Negative definite => cov = I/10
    return SFResult(
        params=param_values,
        param_names=param_names,
        hessian=hessian,
        converged=converged,
        model=model,
        loglik=loglik,
    )


def _make_panel_sfresult(
    *,
    param_values: np.ndarray | None = None,
    param_names: list[str] | None = None,
    hessian: np.ndarray | None | object = _SENTINEL,
    model: MagicMock | None = None,
    loglik: float = -150.0,
    converged: bool = True,
    panel_type: str = "pitt_lee",
    temporal_params: dict | None = None,
) -> PanelSFResult:
    """Factory to create PanelSFResult with controlled parameters."""
    if param_names is None:
        param_names = ["const", "ln_K", "ln_L", "sigma_v", "sigma_u"]
    if param_values is None:
        param_values = np.array([1.0, 0.5, 0.4, 0.3, 0.2])
    if model is None:
        model = _make_mock_model(is_panel=True, model_type=ModelType.PITT_LEE)
    if hessian is _SENTINEL:
        n = len(param_values)
        hessian = -np.eye(n) * 10.0
    return PanelSFResult(
        params=param_values,
        param_names=param_names,
        hessian=hessian,
        converged=converged,
        model=model,
        loglik=loglik,
        panel_type=panel_type,
        temporal_params=temporal_params,
    )


# ===========================================================================
# Etapa 2: SFResult.elasticities
# ===========================================================================


class TestElasticities:
    """Tests for SFResult.elasticities."""

    def test_elasticities_cobb_douglas(self):
        """Test elasticities for Cobb-Douglas specification (constant)."""
        result = _make_sfresult(
            param_values=np.array([1.0, 0.6, 0.3, 0.3, 0.2]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
        )
        elas = result.elasticities(input_vars=["ln_K", "ln_L"])
        assert isinstance(elas, pd.Series)
        assert len(elas) == 2
        assert np.isclose(elas["ln_K"], 0.6)
        assert np.isclose(elas["ln_L"], 0.3)

    def test_elasticities_cobb_douglas_auto_select(self):
        """Test elasticities auto-selects input vars for Cobb-Douglas."""
        result = _make_sfresult(
            param_values=np.array([1.0, 0.6, 0.3, 0.3, 0.2]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
        )
        # input_vars=None => auto-selects non-sigma, non-ln_, non-const
        elas = result.elasticities()
        assert isinstance(elas, pd.Series)
        # Should pick ln_K and ln_L (they don't have "sigma", "ln_" prefix in
        # the exclude filter -- wait, the filter excludes "ln_" prefixed names.
        # Let's check the code: it excludes "ln_" in name.lower().
        # Actually "ln_K".lower() = "ln_k", and "ln_" IS in "ln_k", so
        # these would be excluded. Use non-ln_ names.
        result2 = _make_sfresult(
            param_values=np.array([1.0, 0.6, 0.3, 0.3, 0.2]),
            param_names=["const", "labor", "capital", "sigma_v", "sigma_u"],
        )
        elas2 = result2.elasticities()
        assert isinstance(elas2, pd.Series)
        assert "labor" in elas2.index
        assert "capital" in elas2.index

    def test_elasticities_translog(self):
        """Test elasticities for Translog specification (varying by obs)."""
        rng = np.random.default_rng(42)
        n_obs = 50
        X_df = pd.DataFrame({"ln_K": rng.normal(5, 1, n_obs), "ln_L": rng.normal(4, 1, n_obs)})
        model = _make_mock_model(n_obs=n_obs, X_df=X_df)

        # Translog params: ln_K, ln_L, ln_K_sq, ln_L_sq, ln_K_ln_L
        param_names = [
            "const",
            "ln_K",
            "ln_L",
            "ln_K_sq",
            "ln_L_sq",
            "ln_K_ln_L",
            "sigma_v",
            "sigma_u",
        ]
        param_values = np.array([1.0, 0.6, 0.3, 0.05, 0.04, 0.02, 0.3, 0.2])
        hessian = -np.eye(len(param_values)) * 10.0

        result = SFResult(
            params=param_values,
            param_names=param_names,
            hessian=hessian,
            converged=True,
            model=model,
            loglik=-100.0,
        )

        elas = result.elasticities(translog=True, translog_vars=["ln_K", "ln_L"])
        assert isinstance(elas, pd.DataFrame)
        assert elas.shape == (n_obs, 2)
        assert "ε_K" in elas.columns
        assert "ε_L" in elas.columns

    def test_elasticities_translog_missing_vars_raises(self):
        """Test translog elasticities raises when translog_vars is None."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="must specify translog_vars"):
            result.elasticities(translog=True, translog_vars=None)

    def test_elasticities_translog_missing_data_vars_raises(self):
        """Test translog elasticities raises when data vars are missing."""
        X_df = pd.DataFrame({"other_col": [1, 2, 3]})
        model = _make_mock_model(n_obs=3, X_df=X_df)
        result = _make_sfresult(model=model)
        with pytest.raises(ValueError, match="Variables not found in data"):
            result.elasticities(translog=True, translog_vars=["ln_K", "ln_L"])


# ===========================================================================
# Etapa 3: SFResult.efficient_scale
# ===========================================================================


class TestEfficientScale:
    """Tests for SFResult.efficient_scale."""

    def test_efficient_scale_basic(self):
        """Test efficient_scale finds input levels where RTS ~ 1."""
        rng = np.random.default_rng(42)
        n_obs = 50
        X_df = pd.DataFrame({"ln_K": rng.normal(5, 1, n_obs), "ln_L": rng.normal(4, 1, n_obs)})
        model = _make_mock_model(n_obs=n_obs, X_df=X_df)

        # Translog coefficients designed so RTS=1 is achievable
        param_names = [
            "const",
            "ln_K",
            "ln_L",
            "ln_K_sq",
            "ln_L_sq",
            "ln_K_ln_L",
            "sigma_v",
            "sigma_u",
        ]
        param_values = np.array([1.0, 0.6, 0.3, -0.05, -0.04, 0.02, 0.3, 0.2])
        hessian = -np.eye(len(param_values)) * 10.0

        result = SFResult(
            params=param_values,
            param_names=param_names,
            hessian=hessian,
            converged=True,
            model=model,
            loglik=-100.0,
        )

        eff_scale = result.efficient_scale(translog_vars=["ln_K", "ln_L"])
        assert "efficient_scale" in eff_scale
        assert "rts_at_efficient" in eff_scale
        assert "elasticities" in eff_scale
        assert "converged" in eff_scale
        assert "objective_value" in eff_scale
        # RTS at efficient scale should be close to 1
        assert abs(eff_scale["rts_at_efficient"] - 1.0) < 0.1

    def test_efficient_scale_with_initial_scale(self):
        """Test efficient_scale with explicit initial_scale."""
        rng = np.random.default_rng(42)
        n_obs = 50
        X_df = pd.DataFrame({"ln_K": rng.normal(5, 1, n_obs), "ln_L": rng.normal(4, 1, n_obs)})
        model = _make_mock_model(n_obs=n_obs, X_df=X_df)

        param_names = [
            "const",
            "ln_K",
            "ln_L",
            "ln_K_sq",
            "ln_L_sq",
            "ln_K_ln_L",
            "sigma_v",
            "sigma_u",
        ]
        param_values = np.array([1.0, 0.6, 0.3, -0.05, -0.04, 0.02, 0.3, 0.2])
        hessian = -np.eye(len(param_values)) * 10.0

        result = SFResult(
            params=param_values,
            param_names=param_names,
            hessian=hessian,
            converged=True,
            model=model,
            loglik=-100.0,
        )

        eff_scale = result.efficient_scale(
            translog_vars=["ln_K", "ln_L"],
            initial_scale=np.array([5.0, 4.0]),
        )
        assert "efficient_scale" in eff_scale

    def test_efficient_scale_raises_without_translog_vars(self):
        """Test efficient_scale raises without translog_vars."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="Must specify translog_vars"):
            result.efficient_scale(translog_vars=None)


# ===========================================================================
# Etapa 4: SFResult.compare_functional_form
# ===========================================================================


class TestCompareFunctionalForm:
    """Tests for SFResult.compare_functional_form."""

    def test_compare_functional_form_translog_preferred(self):
        """Test LR test where Translog is preferred (p < 0.05)."""
        cd_result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 0.3, 0.2]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
            loglik=-200.0,
        )
        # Translog has more params and higher loglik
        tl_result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 0.05, 0.04, 0.02, 0.3, 0.2]),
            param_names=[
                "const",
                "ln_K",
                "ln_L",
                "ln_K_sq",
                "ln_L_sq",
                "ln_K_ln_L",
                "sigma_v",
                "sigma_u",
            ],
            loglik=-180.0,
        )
        comparison = cd_result.compare_functional_form(tl_result)
        assert comparison["conclusion"] == "translog"
        assert comparison["df"] == 3
        assert comparison["lr_statistic"] > 0
        assert comparison["pvalue"] < 0.05
        assert comparison["aic_cd"] is not None
        assert comparison["aic_tl"] is not None

    def test_compare_functional_form_cd_preferred(self):
        """Test when Cobb-Douglas is preferred (p > 0.05)."""
        cd_result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 0.3, 0.2]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
            loglik=-200.0,
        )
        # Translog has only marginally higher loglik => not significant
        tl_result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 0.001, 0.001, 0.001, 0.3, 0.2]),
            param_names=[
                "const",
                "ln_K",
                "ln_L",
                "ln_K_sq",
                "ln_L_sq",
                "ln_K_ln_L",
                "sigma_v",
                "sigma_u",
            ],
            loglik=-199.5,
        )
        comparison = cd_result.compare_functional_form(tl_result)
        assert comparison["conclusion"] == "cobb_douglas"
        assert comparison["pvalue"] > 0.05
        assert "Cobb-Douglas" in comparison["interpretation"]

    def test_compare_functional_form_invalid_df(self):
        """Test raises when Translog has fewer params than CD."""
        cd_result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 0.05, 0.04, 0.02, 0.3, 0.2]),
            param_names=[
                "const",
                "ln_K",
                "ln_L",
                "ln_K_sq",
                "ln_L_sq",
                "ln_K_ln_L",
                "sigma_v",
                "sigma_u",
            ],
            loglik=-200.0,
        )
        tl_result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 0.3, 0.2]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
            loglik=-180.0,
        )
        with pytest.raises(ValueError, match="more parameters"):
            cd_result.compare_functional_form(tl_result)


# ===========================================================================
# Etapa 5: SFResult.variance_decomposition bootstrap and fallback paths
# ===========================================================================


class TestVarianceDecompositionPaths:
    """Tests for variance_decomposition partial paths."""

    def test_variance_decomposition_bootstrap_with_cov(self):
        """Test bootstrap method runs with covariance matrix."""
        result = _make_sfresult()
        decomp = result.variance_decomposition(method="bootstrap", ci_level=0.95)
        assert "gamma" in decomp
        assert "gamma_ci" in decomp
        assert decomp["method"] == "bootstrap"

    def test_variance_decomposition_no_cov(self):
        """Test variance_decomposition when cov is None (fallback path)."""
        result = _make_sfresult(hessian=None)
        assert result.cov is None
        decomp = result.variance_decomposition(method="delta")
        assert decomp["gamma_ci"] == (np.nan, np.nan)
        assert decomp["lambda_ci"] == (np.nan, np.nan)

    def test_variance_decomposition_invalid_method(self):
        """Test variance_decomposition with unknown method raises."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="Unknown method"):
            result.variance_decomposition(method="jackknife")

    def test_variance_decomposition_bootstrap_no_cov(self):
        """Test bootstrap fallback when cov is None."""
        result = _make_sfresult(hessian=None)
        decomp = result.variance_decomposition(method="bootstrap")
        assert "gamma" in decomp
        # Bootstrap without cov just repeats point estimates
        assert decomp["method"] == "bootstrap"

    def test_variance_decomposition_interpretation_low_gamma(self):
        """Test interpretation when gamma < 0.1 (OLS adequate)."""
        # sigma_v >> sigma_u => gamma ~ 0
        result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 1.0, 0.01]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
        )
        decomp = result.variance_decomposition()
        assert "OLS" in decomp["interpretation"]

    def test_variance_decomposition_interpretation_high_gamma(self):
        """Test interpretation when gamma > 0.9 (near deterministic)."""
        # sigma_u >> sigma_v => gamma ~ 1
        result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 0.01, 1.0]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
        )
        decomp = result.variance_decomposition()
        assert "deterministic" in decomp["interpretation"]


# ===========================================================================
# Etapa 6: SFResult.returns_to_scale_test partial paths
# ===========================================================================


class TestReturnsToScale:
    """Tests for SFResult.returns_to_scale_test partial paths."""

    def test_returns_to_scale_no_cov(self):
        """Test returns_to_scale_test when covariance is unavailable."""
        result = _make_sfresult(hessian=None)
        rts = result.returns_to_scale_test(input_vars=["ln_K", "ln_L"])
        assert rts["conclusion"] == "unknown"
        assert np.isnan(rts["rts_se"])
        assert np.isnan(rts["pvalue"])

    def test_returns_to_scale_conclusion_irs(self):
        """Test RTS conclusion for increasing returns (RTS > 1)."""
        # Large coefficients => RTS > 1, tight cov for significance
        n = 5
        hessian = -np.eye(n) * 1000.0  # Tight cov => small SE
        result = _make_sfresult(
            param_values=np.array([1.0, 0.7, 0.6, 0.3, 0.2]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
            hessian=hessian,
        )
        rts = result.returns_to_scale_test(input_vars=["ln_K", "ln_L"])
        assert rts["rts"] == pytest.approx(1.3)
        assert rts["conclusion"] == "IRS"

    def test_returns_to_scale_conclusion_drs(self):
        """Test RTS conclusion for decreasing returns (RTS < 1)."""
        # Small coefficients => RTS < 1, tight cov for significance
        n = 5
        hessian = -np.eye(n) * 1000.0
        result = _make_sfresult(
            param_values=np.array([1.0, 0.2, 0.1, 0.3, 0.2]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
            hessian=hessian,
        )
        rts = result.returns_to_scale_test(input_vars=["ln_K", "ln_L"])
        assert rts["rts"] == pytest.approx(0.3)
        assert rts["conclusion"] == "DRS"

    def test_returns_to_scale_conclusion_crs(self):
        """Test RTS conclusion for constant returns (RTS ~ 1)."""
        # Coefficients sum to 1, large SE => cannot reject CRS
        # Need large cov so SE is big enough for p > 0.05
        n = 5
        hessian = -np.eye(n) * 0.5  # Large cov = inv(0.5*I) = 2*I
        result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.5, 0.3, 0.2]),
            param_names=["const", "ln_K", "ln_L", "sigma_v", "sigma_u"],
            hessian=hessian,
        )
        rts = result.returns_to_scale_test(input_vars=["ln_K", "ln_L"])
        assert rts["rts"] == pytest.approx(1.0)
        assert rts["conclusion"] == "CRS"

    def test_returns_to_scale_missing_vars_raises(self):
        """Test returns_to_scale_test raises for missing variable names."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="Variables not found"):
            result.returns_to_scale_test(input_vars=["nonexistent_var"])

    def test_returns_to_scale_auto_select_vars(self):
        """Test auto-selection of input vars (excludes sigma, ln_, const)."""
        result = _make_sfresult(
            param_values=np.array([1.0, 0.5, 0.4, 0.3, 0.2]),
            param_names=["const", "labor", "capital", "sigma_v", "sigma_u"],
        )
        rts = result.returns_to_scale_test()
        assert set(rts["input_vars"]) == {"labor", "capital"}


# ===========================================================================
# Etapa 7: SFResult.compare_distributions auto-estimation
# ===========================================================================


class TestCompareDistributions:
    """Tests for SFResult.compare_distributions."""

    def test_compare_distributions_with_pre_estimated(self):
        """Test compare_distributions with pre-estimated results."""
        result1 = _make_sfresult(loglik=-200.0)
        result2 = _make_sfresult(
            loglik=-190.0,
            model=_make_mock_model(dist=DistributionType.EXPONENTIAL),
        )
        # Mock mean_efficiency to avoid calling actual efficiency estimation
        with patch.object(SFResult, "mean_efficiency", new_callable=PropertyMock, return_value=0.8):
            df = result1.compare_distributions(other_results=[result2])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Distribution" in df.columns
        assert "Best AIC" in df.columns

    def test_compare_distributions_auto_estimation(self):
        """Test compare_distributions with distributions param (auto-estimation)."""
        result = _make_sfresult()

        # Mock StochasticFrontier and DistributionType to avoid actual fitting
        mock_new_result = MagicMock(spec=SFResult)
        mock_new_result.model = _make_mock_model(dist=DistributionType.EXPONENTIAL)
        mock_new_result.loglik = -190.0
        mock_new_result.aic = 390.0
        mock_new_result.bic = 400.0
        mock_new_result.sigma_v = 0.3
        mock_new_result.sigma_u = 0.2
        mock_new_result.lambda_param = 0.67
        mock_new_result.mean_efficiency = 0.85
        mock_new_result.converged = True

        mock_model_cls = MagicMock()
        mock_model_cls.return_value.fit.return_value = mock_new_result

        with (
            patch("panelbox.frontier.model.StochasticFrontier", mock_model_cls),
            patch.object(SFResult, "mean_efficiency", new_callable=PropertyMock, return_value=0.8),
        ):
            df = result.compare_distributions(distributions=["half_normal", "exponential"])
        assert isinstance(df, pd.DataFrame)
        # Only 1 extra (exponential), since half_normal is current dist
        assert len(df) == 2

    def test_compare_distributions_auto_estimation_failure(self):
        """Test compare_distributions handles failed estimation gracefully."""
        result = _make_sfresult()

        mock_model_cls = MagicMock()
        mock_model_cls.return_value.fit.side_effect = RuntimeError("Failed")

        with (
            patch("panelbox.frontier.model.StochasticFrontier", mock_model_cls),
            patch.object(SFResult, "mean_efficiency", new_callable=PropertyMock, return_value=0.8),
        ):
            df = result.compare_distributions(distributions=["exponential"])
        assert isinstance(df, pd.DataFrame)
        # Only the original model since exponential failed
        assert len(df) == 1

    def test_compare_distributions_no_args(self):
        """Test compare_distributions with no other_results or distributions."""
        result = _make_sfresult()
        with patch.object(SFResult, "mean_efficiency", new_callable=PropertyMock, return_value=0.8):
            df = result.compare_distributions()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Only current model


# ===========================================================================
# Etapa 8: PanelSFResult.summary with temporal_params
# ===========================================================================


class TestPanelSummaryTemporalParams:
    """Tests for PanelSFResult.summary with temporal_params."""

    def test_panel_summary_with_eta_positive(self):
        """Test PanelSFResult.summary includes eta > 0 (efficiency improves)."""
        model = _make_mock_model(
            is_panel=True,
            model_type=ModelType.BATTESE_COELLI_92,
        )
        result = _make_panel_sfresult(
            model=model,
            panel_type="bc92",
            temporal_params={"eta": 0.05},
        )
        summary = result.summary()
        assert "Temporal Parameters:" in summary
        assert "η (decay parameter)" in summary
        assert "Efficiency improves over time" in summary

    def test_panel_summary_with_eta_negative(self):
        """Test PanelSFResult.summary includes eta < 0 (efficiency worsens)."""
        model = _make_mock_model(
            is_panel=True,
            model_type=ModelType.BATTESE_COELLI_92,
        )
        result = _make_panel_sfresult(
            model=model,
            panel_type="bc92",
            temporal_params={"eta": -0.03},
        )
        summary = result.summary()
        assert "Efficiency worsens over time" in summary

    def test_panel_summary_with_eta_zero(self):
        """Test PanelSFResult.summary includes eta = 0 (constant)."""
        model = _make_mock_model(
            is_panel=True,
            model_type=ModelType.BATTESE_COELLI_92,
        )
        result = _make_panel_sfresult(
            model=model,
            panel_type="bc92",
            temporal_params={"eta": 0.0},
        )
        summary = result.summary()
        assert "Efficiency constant over time" in summary

    def test_panel_summary_with_kumbhakar_params(self):
        """Test PanelSFResult.summary includes b and c Kumbhakar params."""
        model = _make_mock_model(
            is_panel=True,
            model_type=ModelType.KUMBHAKAR_1990,
        )
        result = _make_panel_sfresult(
            model=model,
            panel_type="kumbhakar",
            temporal_params={"b": 0.1, "c": -0.02},
        )
        summary = result.summary()
        assert "b (linear term)" in summary
        assert "c (quadratic term)" in summary
        assert "Kumbhakar" in summary

    def test_panel_summary_with_delta_t(self):
        """Test PanelSFResult.summary includes delta_t Lee-Schmidt params."""
        model = _make_mock_model(
            is_panel=True,
            model_type=ModelType.LEE_SCHMIDT_1993,
        )
        result = _make_panel_sfresult(
            model=model,
            panel_type="lee_schmidt",
            temporal_params={"delta_t": [1.0, 0.9, 0.85, 0.8, 1.0]},
        )
        summary = result.summary()
        assert "δ_t (time loadings)" in summary
        assert "5 time periods" in summary

    def test_panel_summary_no_temporal_params(self):
        """Test PanelSFResult.summary without temporal params (Pitt-Lee)."""
        result = _make_panel_sfresult(panel_type="pitt_lee", temporal_params=None)
        summary = result.summary()
        assert "PITT_LEE" in summary
        assert "Temporal Parameters:" not in summary


# ===========================================================================
# Etapa 9: PanelSFResult.test_temporal_constancy
# ===========================================================================


class TestTemporalConstancy:
    """Tests for PanelSFResult.test_temporal_constancy."""

    def test_temporal_constancy_time_varying(self):
        """Test for BC92 model (has_time_varying=True)."""
        model = _make_mock_model(is_panel=True, model_type=ModelType.BATTESE_COELLI_92)
        result = _make_panel_sfresult(
            model=model,
            panel_type="bc92",
            temporal_params={"eta": 0.05},
        )
        assert result.has_time_varying is True
        tc = result.test_temporal_constancy()
        # Returns placeholder - check structure
        assert "test_statistic" in tc
        assert "p_value" in tc
        assert "df" in tc
        assert "conclusion" in tc

    def test_temporal_constancy_not_time_varying(self):
        """Test for Pitt-Lee model (has_time_varying=False)."""
        result = _make_panel_sfresult(panel_type="pitt_lee")
        assert result.has_time_varying is False
        tc = result.test_temporal_constancy()
        assert tc["df"] == 0
        assert "does not allow time variation" in tc["conclusion"]


# ===========================================================================
# Etapa 10: PanelSFResult.variance_decomposition TRE
# ===========================================================================


class TestVarianceDecompositionTRE:
    """Tests for PanelSFResult.variance_decomposition TRE (3-component)."""

    def _make_tre_result(self, *, cov_available: bool = True):
        """Create a TRE PanelSFResult with sigma_w."""
        param_names = ["const", "ln_K", "ln_L", "sigma_v", "sigma_u", "sigma_w"]
        param_values = np.array([1.0, 0.5, 0.4, 0.3, 0.2, 0.15])
        n = len(param_values)
        hessian = -np.eye(n) * 10.0 if cov_available else None
        model = _make_mock_model(is_panel=True, model_type=ModelType.TRUE_RANDOM_EFFECTS)
        return _make_panel_sfresult(
            param_values=param_values,
            param_names=param_names,
            hessian=hessian,
            model=model,
            panel_type="tre",
        )

    def test_variance_decomposition_tre_three_component(self):
        """Test 3-component decomposition for True RE model."""
        result = self._make_tre_result()
        decomp = result.variance_decomposition()
        assert decomp["is_three_component"] is True
        assert "gamma_v" in decomp
        assert "gamma_u" in decomp
        assert "gamma_w" in decomp
        # Gammas should sum to 1
        total = decomp["gamma_v"] + decomp["gamma_u"] + decomp["gamma_w"]
        assert total == pytest.approx(1.0)

    def test_variance_decomposition_tre_delta_method_ci(self):
        """Test delta method confidence intervals for gamma_v, gamma_u, gamma_w."""
        result = self._make_tre_result(cov_available=True)
        decomp = result.variance_decomposition(method="delta")
        # CIs should be tuples of length 2
        assert len(decomp["gamma_ci_v"]) == 2
        assert len(decomp["gamma_ci_u"]) == 2
        assert len(decomp["gamma_ci_w"]) == 2
        assert len(decomp["lambda_ci"]) == 2
        # CIs should not be nan when cov is available
        assert not np.isnan(decomp["gamma_ci_v"][0])
        assert not np.isnan(decomp["gamma_ci_u"][0])
        assert not np.isnan(decomp["gamma_ci_w"][0])

    def test_variance_decomposition_tre_no_cov(self):
        """Test TRE decomposition when covariance is None."""
        result = self._make_tre_result(cov_available=False)
        decomp = result.variance_decomposition(method="delta")
        assert decomp["is_three_component"] is True
        # CIs should be NaN
        assert np.isnan(decomp["gamma_ci_v"][0])
        assert np.isnan(decomp["gamma_ci_u"][0])
        assert np.isnan(decomp["gamma_ci_w"][0])

    def test_variance_decomposition_tre_interpretation_low_inefficiency(self):
        """Test interpretation when gamma_u < 0.1."""
        param_names = ["const", "ln_K", "ln_L", "sigma_v", "sigma_u", "sigma_w"]
        # sigma_u very small relative to sigma_v and sigma_w
        param_values = np.array([1.0, 0.5, 0.4, 0.5, 0.01, 0.5])
        hessian = -np.eye(len(param_values)) * 10.0
        model = _make_mock_model(is_panel=True, model_type=ModelType.TRUE_RANDOM_EFFECTS)
        result = _make_panel_sfresult(
            param_values=param_values,
            param_names=param_names,
            hessian=hessian,
            model=model,
            panel_type="tre",
        )
        decomp = result.variance_decomposition()
        assert "SFA" in decomp["interpretation"]

    def test_variance_decomposition_tre_interpretation_high_heterogeneity(self):
        """Test interpretation when gamma_w > 0.5."""
        param_names = ["const", "ln_K", "ln_L", "sigma_v", "sigma_u", "sigma_w"]
        # sigma_w dominant
        param_values = np.array([1.0, 0.5, 0.4, 0.1, 0.1, 1.0])
        hessian = -np.eye(len(param_values)) * 10.0
        model = _make_mock_model(is_panel=True, model_type=ModelType.TRUE_RANDOM_EFFECTS)
        result = _make_panel_sfresult(
            param_values=param_values,
            param_names=param_names,
            hessian=hessian,
            model=model,
            panel_type="tre",
        )
        decomp = result.variance_decomposition()
        assert "Heterogeneity" in decomp["interpretation"]

    def test_variance_decomposition_tre_bootstrap_raises(self):
        """Test TRE bootstrap method raises NotImplementedError."""
        result = self._make_tre_result()
        with pytest.raises(NotImplementedError, match="Bootstrap"):
            result.variance_decomposition(method="bootstrap")

    def test_variance_decomposition_tre_fallback_no_sigma_w(self):
        """Test TRE falls back to base class when no sigma_w param."""
        # Panel result without sigma_w => base class decomposition
        result = _make_panel_sfresult(panel_type="tre")
        decomp = result.variance_decomposition()
        # Should NOT have three-component flag
        assert decomp.get("is_three_component") is not True


# ===========================================================================
# Etapa 11: PanelSFResult.plot_efficiency_evolution
# ===========================================================================


class TestPlotEfficiencyEvolution:
    """Tests for PanelSFResult.plot_efficiency_evolution."""

    @pytest.fixture
    def panel_result_with_efficiency(self):
        """Create a PanelSFResult with mocked efficiency data."""
        model = _make_mock_model(
            is_panel=True,
            model_type=ModelType.BATTESE_COELLI_92,
            n_entities=5,
            n_periods=4,
            n_obs=20,
        )
        result = _make_panel_sfresult(
            model=model,
            panel_type="bc92",
            temporal_params={"eta": 0.05},
        )
        return result

    def _mock_efficiency_df(self):
        """Create mock efficiency DataFrame with entity/time structure."""
        rng = np.random.default_rng(42)
        entities = [f"firm_{i}" for i in range(5)]
        times = [2000, 2001, 2002, 2003]
        rows = []
        for e in entities:
            for t in times:
                rows.append(
                    {
                        "entity": e,
                        "time": t,
                        "efficiency": rng.uniform(0.6, 0.95),
                        "ci_lower": 0.5,
                        "ci_upper": 0.98,
                    }
                )
        return pd.DataFrame(rows)

    def test_plot_efficiency_evolution_timeseries(self, panel_result_with_efficiency):
        """Test plot_efficiency_evolution with kind='timeseries'."""
        eff_df = self._mock_efficiency_df()
        mock_fig = plt.figure()
        with (
            patch.object(panel_result_with_efficiency, "efficiency", return_value=eff_df),
            patch(
                "panelbox.frontier.visualization.evolution_plots.plot_efficiency_timeseries",
                return_value=mock_fig,
            ),
        ):
            fig = panel_result_with_efficiency.plot_efficiency_evolution(
                kind="timeseries", backend="matplotlib"
            )
            assert fig is not None

    def test_plot_efficiency_evolution_spaghetti(self, panel_result_with_efficiency):
        """Test plot_efficiency_evolution with kind='spaghetti'."""
        eff_df = self._mock_efficiency_df()
        mock_fig = plt.figure()
        with (
            patch.object(panel_result_with_efficiency, "efficiency", return_value=eff_df),
            patch(
                "panelbox.frontier.visualization.evolution_plots.plot_efficiency_spaghetti",
                return_value=mock_fig,
            ),
        ):
            fig = panel_result_with_efficiency.plot_efficiency_evolution(
                kind="spaghetti", backend="matplotlib"
            )
            assert fig is not None

    def test_plot_efficiency_evolution_heatmap(self, panel_result_with_efficiency):
        """Test plot_efficiency_evolution with kind='heatmap'."""
        eff_df = self._mock_efficiency_df()
        mock_fig = plt.figure()
        with (
            patch.object(panel_result_with_efficiency, "efficiency", return_value=eff_df),
            patch(
                "panelbox.frontier.visualization.evolution_plots.plot_efficiency_heatmap",
                return_value=mock_fig,
            ),
        ):
            fig = panel_result_with_efficiency.plot_efficiency_evolution(
                kind="heatmap", backend="matplotlib"
            )
            assert fig is not None

    def test_plot_efficiency_evolution_fanchart(self, panel_result_with_efficiency):
        """Test plot_efficiency_evolution with kind='fanchart'."""
        eff_df = self._mock_efficiency_df()
        mock_fig = plt.figure()
        with (
            patch.object(panel_result_with_efficiency, "efficiency", return_value=eff_df),
            patch(
                "panelbox.frontier.visualization.evolution_plots.plot_efficiency_fanchart",
                return_value=mock_fig,
            ),
        ):
            fig = panel_result_with_efficiency.plot_efficiency_evolution(
                kind="fanchart", backend="matplotlib"
            )
            assert fig is not None

    def test_plot_efficiency_evolution_invalid_kind(self, panel_result_with_efficiency):
        """Test plot_efficiency_evolution raises for invalid kind."""
        eff_df = self._mock_efficiency_df()
        with (
            patch.object(panel_result_with_efficiency, "efficiency", return_value=eff_df),
            pytest.raises(ValueError, match="Unknown kind"),
        ):
            panel_result_with_efficiency.plot_efficiency_evolution(kind="scatter")


# ===========================================================================
# Etapa 12: plot_frontier and plot_efficiency error paths
# ===========================================================================


class TestPlotErrorPaths:
    """Tests for plot_frontier and plot_efficiency error handling."""

    def test_plot_frontier_invalid_kind(self):
        """Test plot_frontier raises ValueError for unknown kind."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="Unknown kind"):
            result.plot_frontier(kind="waterfall")

    def test_plot_frontier_2d_missing_input_var(self):
        """Test plot_frontier 2D raises when input_var is None."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="must provide 'input_var'"):
            result.plot_frontier(kind="2d", input_var=None)

    def test_plot_frontier_3d_missing_input_vars(self):
        """Test plot_frontier 3D raises when input_vars is None."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="must provide 'input_vars'"):
            result.plot_frontier(kind="3d", input_vars=None)

    def test_plot_frontier_3d_wrong_number_of_vars(self):
        """Test plot_frontier 3D raises when input_vars has wrong length."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="must provide 'input_vars' with 2"):
            result.plot_frontier(kind="3d", input_vars=["a"])

    def test_plot_frontier_contour_missing_vars(self):
        """Test plot_frontier contour raises when input_vars is None."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="must provide 'input_vars'"):
            result.plot_frontier(kind="contour", input_vars=None)

    def test_plot_frontier_partial_missing_var(self):
        """Test plot_frontier partial raises when input_var is None."""
        result = _make_sfresult()
        with pytest.raises(ValueError, match="must provide 'input_var'"):
            result.plot_frontier(kind="partial", input_var=None)

    def test_plot_efficiency_invalid_kind(self):
        """Test plot_efficiency raises ValueError for unknown kind."""
        result = _make_sfresult()
        mock_eff_df = pd.DataFrame({"efficiency": [0.8, 0.9]})
        with (
            patch.object(result, "efficiency", return_value=mock_eff_df),
            pytest.raises(ValueError, match="Unknown kind"),
        ):
            result.plot_efficiency(kind="scatter")

    def test_plot_efficiency_boxplot_missing_group_var(self):
        """Test plot_efficiency boxplot raises without group_var."""
        result = _make_sfresult()
        mock_eff_df = pd.DataFrame({"efficiency": [0.8, 0.9]})
        with (
            patch.object(result, "efficiency", return_value=mock_eff_df),
            pytest.raises(ValueError, match="must provide 'group_var'"),
        ):
            result.plot_efficiency(kind="boxplot")


# ===========================================================================
# Additional coverage: SFResult.__init__ edge cases
# ===========================================================================


class TestSFResultInitEdgeCases:
    """Tests for SFResult initialization edge cases."""

    def test_init_no_loglik(self):
        """Test SFResult with loglik=None (non-MLE model)."""
        model = _make_mock_model()
        result = SFResult(
            params=np.array([1.0, 0.5, 0.3, 0.2]),
            param_names=["const", "ln_K", "sigma_v", "sigma_u"],
            hessian=-np.eye(4) * 10.0,
            converged=True,
            model=model,
            loglik=None,
        )
        assert result.aic is None
        assert result.bic is None

    def test_init_singular_hessian(self):
        """Test SFResult when Hessian is singular (LinAlgError)."""
        model = _make_mock_model()
        # Singular matrix
        hessian = np.zeros((4, 4))
        result = SFResult(
            params=np.array([1.0, 0.5, 0.3, 0.2]),
            param_names=["const", "ln_K", "sigma_v", "sigma_u"],
            hessian=hessian,
            converged=True,
            model=model,
            loglik=-100.0,
        )
        assert result.cov is None
        assert all(np.isnan(result.se))

    def test_init_no_variance_components(self):
        """Test SFResult when params don't contain sigma_v or sigma_u."""
        model = _make_mock_model()
        result = SFResult(
            params=np.array([1.0, 0.5, 0.3]),
            param_names=["const", "labor", "capital"],
            hessian=-np.eye(3) * 10.0,
            converged=True,
            model=model,
            loglik=-100.0,
        )
        assert np.isnan(result.sigma_v)
        assert np.isnan(result.sigma_u)
        assert np.isnan(result.gamma)

    def test_vcov_property(self):
        """Test vcov property returns cov."""
        result = _make_sfresult()
        assert result.vcov is result.cov

    def test_repr(self):
        """Test __repr__ for SFResult."""
        result = _make_sfresult()
        r = repr(result)
        assert "SFResult" in r
        assert "half_normal" in r

    def test_panel_repr(self):
        """Test __repr__ for PanelSFResult."""
        result = _make_panel_sfresult()
        r = repr(result)
        assert "PanelSFResult" in r
        assert "pitt_lee" in r

    def test_hessian_smaller_than_params(self):
        """Test SFResult when Hessian is smaller than param vector."""
        model = _make_mock_model()
        # 5 params but 4x4 hessian (e.g., Lee-Schmidt normalized param)
        params = np.array([1.0, 0.5, 0.3, 0.2, 1.0])
        param_names = ["const", "ln_K", "sigma_v", "sigma_u", "delta_t5"]
        hessian = -np.eye(4) * 10.0
        result = SFResult(
            params=params,
            param_names=param_names,
            hessian=hessian,
            converged=True,
            model=model,
            loglik=-100.0,
        )
        # cov should be expanded with NaN
        assert result.cov.shape == (5, 5)
        assert np.isnan(result.cov[4, 4])

    def test_delta_method_ln_sigma_params(self):
        """Test delta method is applied for ln_sigma parameters."""
        model = _make_mock_model()
        params = np.array([1.0, 0.5, np.log(0.3), np.log(0.2)])
        param_names = ["const", "ln_K", "ln_sigma_v_sq", "ln_sigma_u_sq"]
        hessian = -np.eye(4) * 10.0
        result = SFResult(
            params=params,
            param_names=param_names,
            hessian=hessian,
            converged=True,
            model=model,
            loglik=-100.0,
        )
        # SE for ln_sigma params should be adjusted by exp(param) * raw_se
        assert not np.isnan(result.se["ln_sigma_v_sq"])

    def test_gamma_distribution_params(self):
        """Test extraction of gamma distribution parameters."""
        model = _make_mock_model(dist=DistributionType.GAMMA)
        params = np.array([1.0, 0.5, 0.3, 0.2, 2.0, 3.0])
        param_names = [
            "const",
            "ln_K",
            "sigma_v",
            "sigma_u",
            "gamma_P",
            "gamma_theta",
        ]
        hessian = -np.eye(6) * 10.0
        result = SFResult(
            params=params,
            param_names=param_names,
            hessian=hessian,
            converged=True,
            model=model,
            loglik=-100.0,
        )
        assert result.gamma_P == 2.0
        assert result.gamma_theta == 3.0

    def test_summary_with_gamma_params(self):
        """Test summary includes gamma distribution parameters."""
        model = _make_mock_model(dist=DistributionType.GAMMA)
        params = np.array([1.0, 0.5, 0.3, 0.2, 2.0, 3.0])
        param_names = [
            "const",
            "ln_K",
            "sigma_v",
            "sigma_u",
            "gamma_P",
            "gamma_theta",
        ]
        hessian = -np.eye(6) * 10.0
        result = SFResult(
            params=params,
            param_names=param_names,
            hessian=hessian,
            converged=True,
            model=model,
            loglik=-100.0,
        )
        summary = result.summary(include_diagnostics=False)
        assert "Gamma Distribution Parameters" in summary
        assert "P (shape)" in summary

    def test_summary_with_panel_info(self):
        """Test summary includes panel-specific info for panel models."""
        model = _make_mock_model(is_panel=True, n_entities=10, n_periods=5)
        result = _make_sfresult(model=model)
        summary = result.summary(include_diagnostics=False)
        assert "No. Entities" in summary
        assert "No. Time Periods" in summary
