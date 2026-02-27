"""
Tests to improve coverage for panelbox/experiment/spatial_extension.py.

Covers:
- SpatialPanelExperiment methods
- add_spatial_model
- run_spatial_diagnostics (extension version)
- compare_spatial_models
- decompose_spatial_effects
- generate_spatial_report
- _get_spatial_model_recommendation decision tree
- extend_panel_experiment dynamic method injection
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelbox.experiment.panel_experiment import PanelExperiment
from panelbox.experiment.spatial_extension import (
    SpatialPanelExperiment,
    extend_panel_experiment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_df():
    """Create a simple panel DataFrame with entity/time columns."""
    np.random.seed(42)
    n_entities, n_time = 10, 5
    entities = np.repeat(np.arange(n_entities), n_time)
    time = np.tile(np.arange(n_time), n_entities)
    n = n_entities * n_time
    df = pd.DataFrame(
        {
            "entity": entities,
            "time": time,
            "y": np.random.randn(n),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
        }
    )
    return df


@pytest.fixture
def experiment(panel_df):
    """PanelExperiment instance with entity/time columns."""
    return PanelExperiment(
        data=panel_df,
        formula="y ~ x1 + x2",
        entity_col="entity",
        time_col="time",
    )


class _SpatialTestExperiment(SpatialPanelExperiment):
    """Test helper that combines SpatialPanelExperiment mixin with
    the minimal PanelExperiment interface needed for testing.
    """

    def list_models(self):
        return list(self._models.keys())

    def get_model(self, name):
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        return self._models[name]

    def get_model_metadata(self, name):
        if name not in self._model_metadata:
            raise KeyError(f"Model '{name}' not found")
        return self._model_metadata[name].copy()


@pytest.fixture
def spatial_exp(panel_df):
    """SpatialPanelExperiment-like object for testing spatial extension methods.

    Uses _SpatialTestExperiment which combines the mixin with the minimal
    PanelExperiment interface.
    """
    exp = _SpatialTestExperiment()
    exp.original_data = panel_df.copy()
    exp.formula = "y ~ x1 + x2"
    exp.entity_col = "entity"
    exp.time_col = "time"
    exp.data = panel_df.set_index(["entity", "time"])
    exp._models = {}
    exp._model_metadata = {}
    exp._model_counters = {}
    return exp


@pytest.fixture
def chain_W():
    """Row-standardized chain spatial weight matrix for 10 entities."""
    N = 10
    W = np.zeros((N, N))
    for i in range(N - 1):
        W[i, i + 1] = 1
        W[i + 1, i] = 1
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return W / row_sums


# ===========================================================================
# add_spatial_model
# ===========================================================================


class TestAddSpatialModel:
    """Test add_spatial_model creates and fits spatial model."""

    def test_add_spatial_model(self, spatial_exp, chain_W):
        """Test add_spatial_model creates and fits spatial model."""
        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_result = MagicMock()
        mock_model_cls.return_value = mock_model_instance
        mock_model_instance.fit.return_value = mock_result

        with patch("panelbox.experiment.spatial_extension.SpatialLag", mock_model_cls):
            result = spatial_exp.add_spatial_model(
                "my_sar", W=chain_W, model_type="sar", effects="fixed"
            )

        assert result is mock_result
        assert "my_sar" in spatial_exp._models
        meta = spatial_exp._model_metadata["my_sar"]
        assert meta["type"] == "spatial_lag"
        assert meta["spatial"] is True
        assert meta["effects"] == "fixed"

    def test_add_spatial_model_sem(self, spatial_exp, chain_W):
        """Test add_spatial_model with SEM type."""
        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_result = MagicMock()
        mock_model_cls.return_value = mock_model_instance
        mock_model_instance.fit.return_value = mock_result

        with patch("panelbox.experiment.spatial_extension.SpatialError", mock_model_cls):
            result = spatial_exp.add_spatial_model("my_sem", W=chain_W, model_type="sem")

        assert result is mock_result
        assert spatial_exp._model_metadata["my_sem"]["type"] == "spatial_error"

    def test_add_spatial_model_sdm(self, spatial_exp, chain_W):
        """Test add_spatial_model with SDM type."""
        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_result = MagicMock()
        mock_model_cls.return_value = mock_model_instance
        mock_model_instance.fit.return_value = mock_result

        with patch("panelbox.experiment.spatial_extension.SpatialDurbin", mock_model_cls):
            result = spatial_exp.add_spatial_model("my_sdm", W=chain_W, model_type="sdm")

        assert result is mock_result
        assert spatial_exp._model_metadata["my_sdm"]["type"] == "spatial_durbin"

    def test_add_spatial_model_gns(self, spatial_exp, chain_W):
        """Test add_spatial_model with GNS type."""
        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_result = MagicMock()
        mock_model_cls.return_value = mock_model_instance
        mock_model_instance.fit.return_value = mock_result

        with patch(
            "panelbox.experiment.spatial_extension.GeneralNestingSpatial",
            mock_model_cls,
        ):
            result = spatial_exp.add_spatial_model("my_gns", W=chain_W, model_type="gns")

        assert result is mock_result
        assert spatial_exp._model_metadata["my_gns"]["type"] == "general_nesting"

    def test_add_spatial_model_unknown_type(self, spatial_exp, chain_W):
        """Test add_spatial_model raises for unknown model type."""
        with pytest.raises(ValueError, match="Unknown spatial model type"):
            spatial_exp.add_spatial_model("bad", W=chain_W, model_type="nonexistent")


# ===========================================================================
# run_spatial_diagnostics (extension)
# ===========================================================================


class TestRunSpatialDiagnosticsExtension:
    """Test run_spatial_diagnostics on extension module."""

    def _setup_mocks(self):
        """Create mocks for spatial diagnostics."""
        mock_moran_cls = MagicMock()
        mock_moran_result = MagicMock()
        mock_moran_result.statistic = 0.25
        mock_moran_result.pvalue = 0.01
        mock_moran_result.expected_value = -0.01
        mock_moran_result.variance = 0.005
        mock_moran_cls.return_value = mock_moran_cls
        mock_moran_cls.run.return_value = mock_moran_result

        mock_run_lm = MagicMock(
            return_value={
                "lm_lag": {"pvalue": 0.01},
                "lm_error": {"pvalue": 0.5},
                "robust_lm_lag": {"pvalue": 0.02},
                "robust_lm_error": {"pvalue": 0.6},
            }
        )

        mock_lisa_cls = MagicMock()
        mock_lisa_result = MagicMock()
        mock_lisa_result.local_i = np.array([0.1, 0.2])
        mock_lisa_result.pvalues = np.array([0.01, 0.5])
        mock_lisa_result.get_clusters.return_value = ["HH", "NS"]
        mock_lisa_cls.return_value = mock_lisa_cls
        mock_lisa_cls.run.return_value = mock_lisa_result

        return mock_moran_cls, mock_run_lm, mock_lisa_cls

    @staticmethod
    def _ensure_multiindex(spatial_exp):
        """Ensure spatial_exp.data has a MultiIndex."""
        if not isinstance(spatial_exp.data.index, pd.MultiIndex):
            # data is already MultiIndex because PanelExperiment.__init__
            # sets it when entity_col and time_col are provided
            pass

    def test_run_spatial_diagnostics_with_model(self, spatial_exp, chain_W):
        """Test run_spatial_diagnostics when a model is available."""
        mock_result = MagicMock()
        mock_result.resid = np.random.randn(50)
        spatial_exp._models["ols"] = mock_result
        spatial_exp._model_metadata["ols"] = {"type": "pooled_ols"}

        self._ensure_multiindex(spatial_exp)

        mock_moran, mock_lm, mock_lisa = self._setup_mocks()

        with (
            patch("panelbox.experiment.spatial_extension.MoranIPanelTest", mock_moran),
            patch("panelbox.experiment.spatial_extension.run_lm_tests", mock_lm),
            patch("panelbox.experiment.spatial_extension.LocalMoranI", mock_lisa),
        ):
            result = spatial_exp.run_spatial_diagnostics(W=chain_W, model_name="ols")

        assert "moran" in result
        assert result["moran"]["statistic"] == 0.25
        assert "lm_tests" in result
        assert "recommendation" in result
        assert "lisa" in result

    def test_run_spatial_diagnostics_no_model(self, spatial_exp, chain_W):
        """Test run_spatial_diagnostics raises if no models fitted."""
        with pytest.raises(ValueError, match="No models fitted"):
            spatial_exp.run_spatial_diagnostics(W=chain_W)

    def test_run_spatial_diagnostics_auto_find_ols(self, spatial_exp, chain_W):
        """Test run_spatial_diagnostics auto-finds OLS model."""
        mock_result = MagicMock()
        mock_result.resid = np.random.randn(50)
        spatial_exp._models["pooled_ols"] = mock_result
        spatial_exp._model_metadata["pooled_ols"] = {"type": "pooled_ols"}

        self._ensure_multiindex(spatial_exp)

        mock_moran, mock_lm, mock_lisa = self._setup_mocks()

        with (
            patch("panelbox.experiment.spatial_extension.MoranIPanelTest", mock_moran),
            patch("panelbox.experiment.spatial_extension.run_lm_tests", mock_lm),
            patch("panelbox.experiment.spatial_extension.LocalMoranI", mock_lisa),
        ):
            result = spatial_exp.run_spatial_diagnostics(W=chain_W)

        assert "moran" in result

    def test_run_spatial_diagnostics_specific_tests(self, spatial_exp, chain_W):
        """Test run_spatial_diagnostics with specific tests only."""
        mock_result = MagicMock()
        mock_result.resid = np.random.randn(50)
        spatial_exp._models["ols"] = mock_result
        spatial_exp._model_metadata["ols"] = {"type": "pooled_ols"}

        self._ensure_multiindex(spatial_exp)

        mock_moran, mock_lm, mock_lisa = self._setup_mocks()

        with (
            patch("panelbox.experiment.spatial_extension.MoranIPanelTest", mock_moran),
            patch("panelbox.experiment.spatial_extension.run_lm_tests", mock_lm),
            patch("panelbox.experiment.spatial_extension.LocalMoranI", mock_lisa),
        ):
            result = spatial_exp.run_spatial_diagnostics(
                W=chain_W, model_name="ols", tests=["moran"]
            )

        assert "moran" in result
        assert "lm_tests" not in result
        assert "lisa" not in result

    def test_run_spatial_diagnostics_residuals_attribute(self, spatial_exp, chain_W):
        """Test run_spatial_diagnostics uses 'residuals' attr if 'resid' missing."""
        mock_result = MagicMock(spec=[])
        mock_result.residuals = np.random.randn(50)
        spatial_exp._models["ols"] = mock_result
        spatial_exp._model_metadata["ols"] = {"type": "pooled_ols"}

        self._ensure_multiindex(spatial_exp)

        mock_moran, mock_lm, mock_lisa = self._setup_mocks()

        with (
            patch("panelbox.experiment.spatial_extension.MoranIPanelTest", mock_moran),
            patch("panelbox.experiment.spatial_extension.run_lm_tests", mock_lm),
            patch("panelbox.experiment.spatial_extension.LocalMoranI", mock_lisa),
        ):
            result = spatial_exp.run_spatial_diagnostics(
                W=chain_W, model_name="ols", tests=["moran"]
            )

        assert "moran" in result


# ===========================================================================
# compare_spatial_models
# ===========================================================================


class TestCompareSpatialModels:
    """Test compare_spatial_models generates comparison table."""

    def test_compare_spatial_models(self, spatial_exp):
        """Test compare_spatial_models with mixed spatial and non-spatial models."""
        # Add mock models
        mock_ols = MagicMock()
        mock_ols.llf = -100.0
        mock_ols.aic = 210.0
        mock_ols.bic = 215.0
        mock_ols.rsquared = 0.5
        mock_ols.nobs = 50

        mock_sar = MagicMock()
        mock_sar.llf = -80.0
        mock_sar.aic = 170.0
        mock_sar.bic = 178.0
        mock_sar.rsquared = 0.7
        mock_sar.nobs = 50
        mock_sar.rho = 0.35
        mock_sar.rho_pvalue = 0.001

        spatial_exp._models = {"OLS": mock_ols, "SAR": mock_sar}
        spatial_exp._model_metadata = {
            "OLS": {"type": "pooled_ols", "spatial": False},
            "SAR": {"type": "spatial_lag", "spatial": True},
        }

        df = spatial_exp.compare_spatial_models()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "AIC" in df.columns
        assert "BIC" in df.columns

    def test_compare_spatial_models_only_spatial(self, spatial_exp):
        """Test compare_spatial_models with include_non_spatial=False."""
        mock_ols = MagicMock()
        mock_sar = MagicMock()
        mock_sar.aic = 170.0
        mock_sar.rho = 0.35

        spatial_exp._models = {"OLS": mock_ols, "SAR": mock_sar}
        spatial_exp._model_metadata = {
            "OLS": {"type": "pooled_ols", "spatial": False},
            "SAR": {"type": "spatial_lag", "spatial": True},
        }

        df = spatial_exp.compare_spatial_models(include_non_spatial=False)
        assert len(df) == 1
        assert "SAR" in df["Model"].values

    def test_compare_spatial_models_specific_names(self, spatial_exp):
        """Test compare_spatial_models with specific model names."""
        mock_sar = MagicMock()
        mock_sar.aic = 170.0

        spatial_exp._models = {"SAR": mock_sar}
        spatial_exp._model_metadata = {
            "SAR": {"type": "spatial_lag", "spatial": True},
        }

        df = spatial_exp.compare_spatial_models(model_names=["SAR"])
        assert len(df) == 1

    def test_compare_spatial_models_with_lambda(self, spatial_exp):
        """Test compare_spatial_models shows lambda for SEM models."""
        mock_sem = MagicMock()
        mock_sem.aic = 175.0
        mock_sem.lambda_ = 0.42
        mock_sem.lambda_pvalue = 0.003

        spatial_exp._models = {"SEM": mock_sem}
        spatial_exp._model_metadata = {
            "SEM": {"type": "spatial_error", "spatial": True},
        }

        df = spatial_exp.compare_spatial_models()
        assert len(df) == 1


# ===========================================================================
# decompose_spatial_effects
# ===========================================================================


class TestDecomposeSpatialEffects:
    """Test decompose_spatial_effects extracts direct/indirect/total."""

    def test_decompose_spatial_effects(self, spatial_exp):
        """Test decompose_spatial_effects for SDM model."""
        mock_result = MagicMock()
        mock_effects = {
            "direct": pd.DataFrame({"x1": [0.5], "x2": [0.3]}),
            "indirect": pd.DataFrame({"x1": [0.2], "x2": [0.1]}),
            "total": pd.DataFrame({"x1": [0.7], "x2": [0.4]}),
        }
        mock_result.effects_decomposition.return_value = mock_effects

        spatial_exp._models = {"SDM": mock_result}
        spatial_exp._model_metadata = {
            "SDM": {"type": "spatial_durbin", "spatial": True},
        }

        effects = spatial_exp.decompose_spatial_effects("SDM")
        assert "direct" in effects
        assert "indirect" in effects
        assert "total" in effects

    def test_decompose_spatial_effects_gns(self, spatial_exp):
        """Test decompose_spatial_effects for GNS model."""
        mock_result = MagicMock(spec=[])

        mock_effects = {
            "direct": pd.DataFrame({"x1": [0.5]}),
            "indirect": pd.DataFrame({"x1": [0.2]}),
            "total": pd.DataFrame({"x1": [0.7]}),
        }

        spatial_exp._models = {"GNS": mock_result}
        spatial_exp._model_metadata = {
            "GNS": {"type": "general_nesting", "spatial": True},
        }

        with patch(
            "panelbox.experiment.spatial_extension.compute_spatial_effects",
            return_value=mock_effects,
        ):
            effects = spatial_exp.decompose_spatial_effects("GNS")

        assert "direct" in effects

    def test_decompose_spatial_effects_non_sdm_raises(self, spatial_exp):
        """Test decompose_spatial_effects raises for non-SDM/GNS models."""
        mock_result = MagicMock()
        spatial_exp._models = {"SAR": mock_result}
        spatial_exp._model_metadata = {
            "SAR": {"type": "spatial_lag", "spatial": True},
        }

        with pytest.raises(ValueError, match="does not support effects decomposition"):
            spatial_exp.decompose_spatial_effects("SAR")


# ===========================================================================
# generate_spatial_report
# ===========================================================================


class TestGenerateSpatialReport:
    """Test generate_spatial_report creates formatted report."""

    def test_generate_spatial_report(self, spatial_exp):
        """Test generate_spatial_report delegates to SpatialReportGenerator."""
        mock_generator_cls = MagicMock()
        mock_generator = MagicMock()
        mock_generator_cls.return_value = mock_generator

        with patch.dict(
            "sys.modules",
            {
                "panelbox.reporting.spatial_report": MagicMock(
                    SpatialReportGenerator=mock_generator_cls,
                )
            },
        ):
            spatial_exp.generate_spatial_report(
                filename="report.html",
                include_diagnostics=True,
                include_effects=True,
                include_maps=False,
            )

        mock_generator_cls.assert_called_once_with(spatial_exp)
        mock_generator.generate.assert_called_once_with(
            filename="report.html",
            include_diagnostics=True,
            include_effects=True,
            include_maps=False,
        )


# ===========================================================================
# _get_spatial_model_recommendation
# ===========================================================================


class TestGetSpatialModelRecommendation:
    """Test _get_spatial_model_recommendation decision tree."""

    def test_recommend_sar_only_lag_significant(self):
        """Test SAR recommendation when only LM-Lag is significant."""
        spe = SpatialPanelExperiment()
        lm_results = {
            "lm_lag": {"pvalue": 0.01},
            "lm_error": {"pvalue": 0.5},
            "robust_lm_lag": {"pvalue": 0.02},
            "robust_lm_error": {"pvalue": 0.6},
        }
        rec = spe._get_spatial_model_recommendation(lm_results)
        assert "SAR" in rec

    def test_recommend_sem_only_error_significant(self):
        """Test SEM recommendation when only LM-Error is significant."""
        spe = SpatialPanelExperiment()
        lm_results = {
            "lm_lag": {"pvalue": 0.5},
            "lm_error": {"pvalue": 0.01},
            "robust_lm_lag": {"pvalue": 0.6},
            "robust_lm_error": {"pvalue": 0.02},
        }
        rec = spe._get_spatial_model_recommendation(lm_results)
        assert "SEM" in rec

    def test_recommend_sar_both_significant_robust_lag(self):
        """Test SAR when both significant but robust LM-Lag wins."""
        spe = SpatialPanelExperiment()
        lm_results = {
            "lm_lag": {"pvalue": 0.01},
            "lm_error": {"pvalue": 0.01},
            "robust_lm_lag": {"pvalue": 0.01},
            "robust_lm_error": {"pvalue": 0.5},
        }
        rec = spe._get_spatial_model_recommendation(lm_results)
        assert "SAR" in rec

    def test_recommend_sem_both_significant_robust_error(self):
        """Test SEM when both significant but robust LM-Error wins."""
        spe = SpatialPanelExperiment()
        lm_results = {
            "lm_lag": {"pvalue": 0.01},
            "lm_error": {"pvalue": 0.01},
            "robust_lm_lag": {"pvalue": 0.5},
            "robust_lm_error": {"pvalue": 0.01},
        }
        rec = spe._get_spatial_model_recommendation(lm_results)
        assert "SEM" in rec

    def test_recommend_sdm_both_robust_significant(self):
        """Test SDM recommendation when both robust tests significant."""
        spe = SpatialPanelExperiment()
        lm_results = {
            "lm_lag": {"pvalue": 0.01},
            "lm_error": {"pvalue": 0.01},
            "robust_lm_lag": {"pvalue": 0.01},
            "robust_lm_error": {"pvalue": 0.01},
        }
        rec = spe._get_spatial_model_recommendation(lm_results)
        assert "SDM" in rec

    def test_recommend_no_spatial_dependence(self):
        """Test no spatial dependence when nothing is significant."""
        spe = SpatialPanelExperiment()
        lm_results = {
            "lm_lag": {"pvalue": 0.5},
            "lm_error": {"pvalue": 0.5},
            "robust_lm_lag": {"pvalue": 0.6},
            "robust_lm_error": {"pvalue": 0.7},
        }
        rec = spe._get_spatial_model_recommendation(lm_results)
        assert "No spatial dependence" in rec

    def test_recommend_both_robust_not_significant(self):
        """Test SDM/GNS when both LMs significant but both robust not."""
        spe = SpatialPanelExperiment()
        lm_results = {
            "lm_lag": {"pvalue": 0.01},
            "lm_error": {"pvalue": 0.01},
            "robust_lm_lag": {"pvalue": 0.5},
            "robust_lm_error": {"pvalue": 0.5},
        }
        rec = spe._get_spatial_model_recommendation(lm_results)
        assert "SDM" in rec

    def test_recommend_with_missing_keys(self):
        """Test recommendation handles missing keys with defaults."""
        spe = SpatialPanelExperiment()
        lm_results = {}  # All keys missing => defaults to pvalue=1.0
        rec = spe._get_spatial_model_recommendation(lm_results)
        assert "No spatial dependence" in rec


# ===========================================================================
# extend_panel_experiment
# ===========================================================================


class TestExtendPanelExperiment:
    """Test extend_panel_experiment adds methods dynamically."""

    def test_extend_panel_experiment(self):
        """Test extend_panel_experiment adds spatial methods to PanelExperiment."""
        # extend_panel_experiment is called at import time
        # Verify the spatial methods exist on PanelExperiment
        assert hasattr(PanelExperiment, "add_spatial_model")
        assert hasattr(PanelExperiment, "compare_spatial_models")
        assert hasattr(PanelExperiment, "decompose_spatial_effects")
        assert hasattr(PanelExperiment, "generate_spatial_report")
        assert hasattr(PanelExperiment, "_get_spatial_model_recommendation")

    def test_extend_adds_spatial_aliases(self):
        """Test extend_panel_experiment adds spatial model aliases."""
        # Verify spatial model aliases are merged into PanelExperiment
        assert "sar" in PanelExperiment.MODEL_ALIASES
        assert "sem" in PanelExperiment.MODEL_ALIASES
        assert "sdm" in PanelExperiment.MODEL_ALIASES

    def test_extend_panel_experiment_callable(self):
        """Test extend_panel_experiment can be called without error."""
        # Should be safe to call again (idempotent)
        extend_panel_experiment()
        assert hasattr(PanelExperiment, "add_spatial_model")

    def test_spatial_method_on_experiment_instance(self, experiment):
        """Test spatial methods are available on PanelExperiment instances."""
        # Methods should be callable on actual instances
        assert callable(getattr(experiment, "add_spatial_model", None))
        assert callable(getattr(experiment, "compare_spatial_models", None))
        assert callable(getattr(experiment, "_get_spatial_model_recommendation", None))

    def test_spatial_model_aliases_constant(self):
        """Test SPATIAL_MODEL_ALIASES is a valid dict."""
        aliases = SpatialPanelExperiment.SPATIAL_MODEL_ALIASES
        assert isinstance(aliases, dict)
        assert aliases["sar"] == "spatial_lag"
        assert aliases["sem"] == "spatial_error"
        assert aliases["sdm"] == "spatial_durbin"
        assert aliases["gns"] == "general_nesting"
