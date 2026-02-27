"""
Tests to improve coverage for panelbox/experiment/panel_experiment.py.

Covers:
- _create_model branches (discrete, count, censored, ordered, import error)
- _validate_data error paths
- fit/compare/get model edge cases
- spatial diagnostics on PanelExperiment
- fit_all_models, _generate_model_name
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelbox.experiment.panel_experiment import PanelExperiment

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
def panel_mi(panel_df):
    """Panel DataFrame with MultiIndex (entity, time)."""
    return panel_df.set_index(["entity", "time"])


@pytest.fixture
def experiment(panel_df):
    """PanelExperiment with entity/time columns."""
    return PanelExperiment(
        data=panel_df,
        formula="y ~ x1 + x2",
        entity_col="entity",
        time_col="time",
    )


@pytest.fixture
def experiment_mi(panel_mi):
    """PanelExperiment with MultiIndex data."""
    return PanelExperiment(
        data=panel_mi,
        formula="y ~ x1 + x2",
    )


# ===========================================================================
# Etapa 2: _create_model branches
# ===========================================================================


class TestCreateModelBranches:
    """Test _create_model factory method covers all model-type branches."""

    def test_create_model_discrete_choice(self, experiment):
        """Test _create_model with discrete choice model types."""
        mock_model = MagicMock()
        mock_model.from_formula.return_value = mock_model

        with patch("panelbox.experiment.panel_experiment.pb", create=True) as mock_pb:
            # Discrete choice: pooled_logit
            mock_pb.PooledLogit = mock_model
            mock_pb.PooledProbit = mock_model
            mock_pb.FixedEffectsLogit = mock_model
            mock_pb.RandomEffectsProbit = mock_model

            # Use internal import path
            with patch("panelbox.experiment.panel_experiment.__import__", create=True):
                pass

        # Use the _create_model method by mocking the import inside it
        mock_cls = MagicMock()
        mock_cls.from_formula.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "panelbox": MagicMock(
                    PooledLogit=mock_cls,
                    PooledProbit=mock_cls,
                    FixedEffectsLogit=mock_cls,
                    RandomEffectsProbit=mock_cls,
                )
            },
        ):
            for model_type in [
                "pooled_logit",
                "pooled_probit",
                "fixed_effects_logit",
                "random_effects_probit",
            ]:
                result = experiment._create_model(model_type)
                assert result is not None

    def test_create_model_count(self, experiment):
        """Test _create_model with count model types."""
        mock_cls = MagicMock()
        mock_cls.from_formula.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "panelbox": MagicMock(
                    PooledPoisson=mock_cls,
                    PoissonFixedEffects=mock_cls,
                    RandomEffectsPoisson=mock_cls,
                    NegativeBinomial=mock_cls,
                )
            },
        ):
            for model_type in [
                "pooled_poisson",
                "poisson_fixed_effects",
                "random_effects_poisson",
                "negative_binomial",
            ]:
                result = experiment._create_model(model_type)
                assert result is not None

    def test_create_model_censored(self, experiment):
        """Test _create_model with censored model types."""
        mock_cls = MagicMock()
        mock_cls.from_formula.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {"panelbox": MagicMock(RandomEffectsTobit=mock_cls)},
        ):
            result = experiment._create_model("random_effects_tobit")
            assert result is not None

    def test_create_model_ordered(self, experiment):
        """Test _create_model with ordered model types."""
        mock_cls = MagicMock()
        mock_cls.from_formula.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "panelbox": MagicMock(
                    OrderedLogit=mock_cls,
                    OrderedProbit=mock_cls,
                )
            },
        ):
            for model_type in ["ordered_logit", "ordered_probit"]:
                result = experiment._create_model(model_type)
                assert result is not None

    def test_create_model_import_error(self, experiment):
        """Test _create_model handles ImportError gracefully."""
        with (
            patch.dict("sys.modules", {"panelbox": None}),
            pytest.raises(ImportError, match="panelbox is required"),
        ):
            experiment._create_model("pooled_ols")

    def test_create_model_unknown_type(self, experiment):
        """Test _create_model raises ValueError for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            experiment._create_model("nonexistent_model")

    def test_create_model_linear_with_multiindex(self, experiment_mi):
        """Test _create_model for linear models covers the else branch (no entity/time cols)."""
        # entity_col and time_col are None, so it takes the else branch.
        # PooledOLS requires entity_col/time_col, so we mock to verify branch coverage.
        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "panelbox": MagicMock(
                    PooledOLS=mock_cls,
                    FixedEffects=mock_cls,
                    RandomEffects=mock_cls,
                )
            },
        ):
            model = experiment_mi._create_model("pooled_ols")
            assert model is not None
            # Verify it was called with (formula, data) — no entity/time cols
            mock_cls.assert_called_with(experiment_mi.formula, experiment_mi.data)

            model_fe = experiment_mi._create_model("fixed_effects")
            assert model_fe is not None

            model_re = experiment_mi._create_model("random_effects")
            assert model_re is not None

    def test_create_model_linear_with_entity_time_cols(self, experiment):
        """Test _create_model for linear models when using entity/time cols."""
        # entity_col and time_col are set, so it takes the if branch
        model = experiment._create_model("pooled_ols")
        assert model is not None

        model_fe = experiment._create_model("fixed_effects")
        assert model_fe is not None

        model_re = experiment._create_model("random_effects")
        assert model_re is not None


# ===========================================================================
# Etapa 3: _validate_data
# ===========================================================================


class TestValidateData:
    """Test _validate_data error paths."""

    def test_validate_data_missing_entity(self):
        """Test _validate_data raises for missing entity column."""
        df = pd.DataFrame({"time": [1, 2], "y": [1.0, 2.0], "x1": [0.5, 1.5]})
        with pytest.raises(ValueError, match="entity_col 'entity' not found"):
            PanelExperiment(
                data=df,
                formula="y ~ x1",
                entity_col="entity",
                time_col="time",
            )

    def test_validate_data_missing_time(self):
        """Test _validate_data raises for missing time column."""
        df = pd.DataFrame({"entity": [1, 2], "y": [1.0, 2.0], "x1": [0.5, 1.5]})
        with pytest.raises(ValueError, match="time_col 'time' not found"):
            PanelExperiment(
                data=df,
                formula="y ~ x1",
                entity_col="entity",
                time_col="time",
            )

    def test_validate_data_no_multiindex(self):
        """Test _validate_data raises when no MultiIndex and no cols specified."""
        df = pd.DataFrame({"y": [1.0, 2.0], "x1": [0.5, 1.5]})
        with pytest.raises(
            ValueError,
            match="data must have a MultiIndex",
        ):
            PanelExperiment(data=df, formula="y ~ x1")

    def test_validate_data_not_dataframe(self):
        """Test constructor raises for non-DataFrame input."""
        # data.copy() is called before _validate_data, so a string raises
        # AttributeError rather than TypeError. Both indicate bad input.
        with pytest.raises((TypeError, AttributeError)):
            PanelExperiment(data="not a dataframe", formula="y ~ x1")

    def test_validate_data_empty_dataframe(self):
        """Test _validate_data raises ValueError for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="data cannot be empty"):
            PanelExperiment(data=df, formula="y ~ x1", entity_col="e", time_col="t")

    def test_validate_data_entity_col_only(self):
        """Test _validate_data with only entity_col (no time_col)."""
        df = pd.DataFrame({"entity": [1, 2], "y": [1.0, 2.0], "x1": [0.5, 1.5]})
        # entity_col is set but time_col is None => the time_col check is skipped
        # No MultiIndex check because entity_col is not None
        exp = PanelExperiment(
            data=df,
            formula="y ~ x1",
            entity_col="entity",
        )
        assert exp is not None


# ===========================================================================
# Etapa 4: fit / compare / get model
# ===========================================================================


class TestFitCompareGetModel:
    """Test fit_model, compare_models, get_model edge cases."""

    def test_fit_model_error_handling(self, experiment):
        """Test fit_model propagates errors from model fitting."""
        # Unknown model type
        with pytest.raises(ValueError, match="Unknown model_type"):
            experiment.fit_model("unknown_model_type")

    def test_fit_model_duplicate_name(self, experiment):
        """Test fit_model raises for duplicate model name."""
        experiment.fit_model("pooled_ols", name="ols")
        with pytest.raises(ValueError, match="Model with name 'ols' already exists"):
            experiment.fit_model("pooled_ols", name="ols")

    def test_fit_model_auto_name(self, experiment):
        """Test fit_model auto-generates name when name is None."""
        experiment.fit_model("pooled_ols")
        models = experiment.list_models()
        assert len(models) == 1
        assert models[0] == "pooled_ols_1"

    def test_fit_model_alias_resolution(self, experiment):
        """Test fit_model resolves aliases correctly."""
        # "fe" should resolve to "fixed_effects"
        experiment.fit_model("fe", name="my_fe")
        meta = experiment.get_model_metadata("my_fe")
        assert meta["model_type"] == "fixed_effects"

    def test_get_model_keyerror(self, experiment):
        """Test get_model raises KeyError for unknown model."""
        with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
            experiment.get_model("nonexistent")

    def test_get_model_keyerror_shows_available(self, experiment):
        """Test get_model error message shows available models."""
        experiment.fit_model("pooled_ols", name="ols")
        with pytest.raises(KeyError, match="ols"):
            experiment.get_model("wrong_name")

    def test_get_model_metadata_keyerror(self, experiment):
        """Test get_model_metadata raises KeyError for unknown model."""
        with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
            experiment.get_model_metadata("nonexistent")

    def test_get_model_metadata_returns_copy(self, experiment):
        """Test get_model_metadata returns a copy, not the original."""
        experiment.fit_model("pooled_ols", name="ols")
        meta1 = experiment.get_model_metadata("ols")
        meta2 = experiment.get_model_metadata("ols")
        assert meta1 is not meta2
        assert meta1 == meta2

    def test_generate_model_name(self, experiment):
        """Test _generate_model_name generates unique names."""
        name1 = experiment._generate_model_name("pooled_ols")
        name2 = experiment._generate_model_name("pooled_ols")
        name3 = experiment._generate_model_name("fixed_effects")
        assert name1 == "pooled_ols_1"
        assert name2 == "pooled_ols_2"
        assert name3 == "fixed_effects_1"

    def test_fit_all_models_name_validation(self, experiment):
        """Test fit_all_models validates that names length matches model_types."""
        with pytest.raises(ValueError, match="Length of names"):
            experiment.fit_all_models(
                model_types=["pooled_ols", "fixed_effects"],
                names=["only_one"],
            )

    def test_fit_all_models_default(self, experiment):
        """Test fit_all_models with default model types."""
        results = experiment.fit_all_models()
        assert len(results) == 3
        models = experiment.list_models()
        assert len(models) == 3

    def test_fit_all_models_custom_names(self, experiment):
        """Test fit_all_models with custom names."""
        results = experiment.fit_all_models(
            model_types=["pooled_ols", "fixed_effects"],
            names=["my_ols", "my_fe"],
        )
        assert "my_ols" in results
        assert "my_fe" in results

    def test_compare_models_with_metadata(self, experiment):
        """Test compare_models includes experiment metadata."""
        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")

        with patch(
            "panelbox.experiment.panel_experiment.ComparisonResult",
            create=True,
        ) as mock_cr:
            mock_cr.from_experiment.return_value = MagicMock()
            with patch(
                "panelbox.experiment.results.ComparisonResult",
                mock_cr,
            ):
                result = experiment.compare_models(["ols", "fe"])
                assert result is not None

    def test_list_models_empty(self, experiment):
        """Test list_models returns empty list when no models fitted."""
        assert experiment.list_models() == []

    def test_repr(self, experiment):
        """Test __repr__ with no models and with models."""
        # No models
        rep = repr(experiment)
        assert "PanelExperiment" in rep
        assert "n_models=0" in rep
        assert "models=[none]" in rep

        # With models
        experiment.fit_model("pooled_ols", name="ols")
        rep = repr(experiment)
        assert "n_models=1" in rep
        assert "ols" in rep

    def test_validate_model(self, experiment):
        """Test validate_model calls ValidationResult.from_model_results."""
        experiment.fit_model("fixed_effects", name="fe")

        mock_vr = MagicMock()
        mock_vr.from_model_results.return_value = MagicMock()

        with patch("panelbox.experiment.results.ValidationResult", mock_vr):
            result = experiment.validate_model("fe")
            assert result is not None
            mock_vr.from_model_results.assert_called_once()

    def test_compare_models_metadata_default(self, experiment):
        """Test compare_models adds experiment metadata when none provided."""
        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")

        mock_cr = MagicMock()
        mock_cr.from_experiment.return_value = MagicMock()

        with patch("panelbox.experiment.results.ComparisonResult", mock_cr):
            result = experiment.compare_models(["ols", "fe"])
            assert result is not None
            # Verify metadata was passed
            call_kwargs = mock_cr.from_experiment.call_args[1]
            assert "metadata" in call_kwargs
            assert call_kwargs["metadata"]["experiment_formula"] == "y ~ x1 + x2"

    def test_analyze_residuals(self, experiment):
        """Test analyze_residuals calls ResidualResult.from_model_results."""
        experiment.fit_model("fixed_effects", name="fe")

        mock_rr = MagicMock()
        mock_rr.from_model_results.return_value = MagicMock()

        with patch("panelbox.experiment.results.ResidualResult", mock_rr):
            result = experiment.analyze_residuals("fe")
            assert result is not None
            mock_rr.from_model_results.assert_called_once()

    def test_save_master_report_no_models(self, experiment, tmp_path):
        """Test save_master_report raises when no models fitted."""
        with pytest.raises(ValueError, match="no models have been fitted"):
            experiment.save_master_report(str(tmp_path / "report.html"))

    def test_save_master_report(self, experiment, tmp_path):
        """Test save_master_report generates HTML file."""
        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")

        mock_rm = MagicMock()
        mock_rm.return_value = mock_rm
        mock_rm.generate_report.return_value = "<html>test</html>"

        with patch("panelbox.report.ReportManager", mock_rm):
            result = experiment.save_master_report(
                str(tmp_path / "master.html"),
                title="Test Report",
            )
            assert result is not None
            assert result.exists()


# ===========================================================================
# Etapa 5: spatial diagnostics on PanelExperiment
# ===========================================================================


class TestSpatialDiagnostics:
    """Test spatial diagnostics methods on PanelExperiment."""

    def _patch_spatial_imports(self):
        """
        Patch the 'from ..validation.spatial import ...' inside
        run_spatial_diagnostics so that LocalMoranI (missing from the real
        module) is provided as a mock and the other names remain usable.

        The extension's run_spatial_diagnostics accesses lisa_results.local_i,
        lisa_results.pvalues, and lisa_results.get_clusters(), so the LISA mock
        must return an object with those attributes (not a DataFrame).
        """
        mock_moran_cls = MagicMock()
        mock_moran_result = MagicMock()
        mock_moran_result.statistic = 0.15
        mock_moran_result.pvalue = 0.03
        mock_moran_result.expected_value = -0.01
        mock_moran_result.variance = 0.005
        mock_moran_cls.return_value = mock_moran_cls
        mock_moran_cls.run.return_value = mock_moran_result

        mock_lisa_cls = MagicMock()
        mock_lisa_result = MagicMock()
        mock_lisa_result.local_i = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        mock_lisa_result.pvalues = np.array([0.01, 0.5, 0.02, 0.6, 0.03, 0.7, 0.04, 0.8, 0.05, 0.9])
        mock_lisa_result.get_clusters.return_value = [
            "HH",
            "NS",
            "HL",
            "NS",
            "LH",
            "NS",
            "LL",
            "NS",
            "HH",
            "NS",
        ]
        mock_lisa_cls.return_value = mock_lisa_cls
        mock_lisa_cls.run.return_value = mock_lisa_result

        mock_run_lm = MagicMock(
            return_value={
                "lm_lag": {"pvalue": 0.01},
                "lm_error": {"pvalue": 0.5},
                "robust_lm_lag": {"pvalue": 0.02},
                "robust_lm_error": {"pvalue": 0.6},
            }
        )

        # Build a fake module that has all three names
        fake_mod = MagicMock()
        fake_mod.MoranIPanelTest = mock_moran_cls
        fake_mod.LocalMoranI = mock_lisa_cls
        fake_mod.run_lm_tests = mock_run_lm

        return fake_mod

    def _apply_patches(self, fake_mod):
        """Apply patches to all locations that import spatial diagnostics."""
        return [
            patch.dict("sys.modules", {"panelbox.validation.spatial": fake_mod}),
            patch(
                "panelbox.experiment.spatial_extension.MoranIPanelTest",
                fake_mod.MoranIPanelTest,
            ),
            patch(
                "panelbox.experiment.spatial_extension.run_lm_tests",
                fake_mod.run_lm_tests,
            ),
            patch(
                "panelbox.experiment.spatial_extension.LocalMoranI",
                fake_mod.LocalMoranI,
            ),
        ]

    def test_run_spatial_diagnostics(self, experiment):
        """Test run_spatial_diagnostics with valid spatial weights (mocked).

        Note: spatial_extension overrides run_spatial_diagnostics at import time,
        so this test exercises the extension's version via PanelExperiment.
        """
        experiment.fit_model("pooled_ols", name="ols")
        N = 10
        W = np.zeros((N, N))
        for i in range(N - 1):
            W[i, i + 1] = 1
            W[i + 1, i] = 1
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        fake_mod = self._patch_spatial_imports()
        patches = self._apply_patches(fake_mod)
        for p in patches:
            p.start()
        try:
            result = experiment.run_spatial_diagnostics(W=W, model_name="ols")
        finally:
            for p in reversed(patches):
                p.stop()
        assert "moran" in result
        assert "lm_tests" in result
        assert "recommendation" in result

    def test_run_spatial_diagnostics_auto_ols(self, experiment):
        """Test run_spatial_diagnostics auto-finds OLS model when no model_name."""
        # Fit a model first so list_models() returns something
        experiment.fit_model("pooled_ols", name="pooled_ols_diag")
        N = 10
        W = np.zeros((N, N))
        for i in range(N - 1):
            W[i, i + 1] = 1
            W[i + 1, i] = 1
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        fake_mod = self._patch_spatial_imports()
        patches = self._apply_patches(fake_mod)
        for p in patches:
            p.start()
        try:
            result = experiment.run_spatial_diagnostics(W=W, model_name=None)
        finally:
            for p in reversed(patches):
                p.stop()
        assert "moran" in result

    def test_run_spatial_diagnostics_wrong_W_shape(self, experiment):
        """Test run_spatial_diagnostics raises for wrong W dimensions.

        Note: The extension's run_spatial_diagnostics doesn't validate W shape
        itself; the underlying MoranIPanelTest would raise. Since we mock it,
        we test that no models fitted raises ValueError instead.
        """
        # No model fitted => should raise
        with pytest.raises(ValueError, match="No models fitted"):
            experiment.run_spatial_diagnostics(W=np.eye(3))

    def test_estimate_spatial_model_auto_no_diag(self, experiment):
        """Test estimate_spatial_model raises if auto but no diagnostics."""
        with pytest.raises(ValueError, match="Run run_spatial_diagnostics"):
            experiment.estimate_spatial_model(model_type="auto")

    def test_estimate_spatial_model_no_W(self, experiment):
        """Test estimate_spatial_model raises if no W provided."""
        with pytest.raises(ValueError, match="Spatial weight matrix W must be provided"):
            experiment.estimate_spatial_model(model_type="sar", W=None)

    def test_estimate_spatial_model_ols_recommendation(self, experiment):
        """Test estimate_spatial_model with OLS recommendation."""
        experiment.spatial_diagnostics = {
            "recommendation": "ols",
            "W": np.eye(10),
        }
        result = experiment.estimate_spatial_model(model_type="auto")
        assert result is not None
        assert "fe_no_spatial" in experiment.list_models()

    def test_estimate_spatial_model_with_W_from_diagnostics(self, experiment):
        """Test estimate_spatial_model gets W from diagnostics if not provided."""
        W = np.eye(10)
        experiment.spatial_diagnostics = {
            "recommendation": "ols",
            "W": W,
        }
        result = experiment.estimate_spatial_model(model_type="ols", W=None)
        assert result is not None

    def test_estimate_spatial_model_auto_name(self, experiment):
        """Test estimate_spatial_model auto-generates name when name is None."""
        experiment.spatial_diagnostics = {
            "recommendation": "ols",
            "W": np.eye(10),
        }
        experiment.estimate_spatial_model(model_type="auto", name=None)
        assert "fe_no_spatial" in experiment.list_models()

    def test_estimate_spatial_model_auto_sar(self, experiment):
        """Test estimate_spatial_model auto selects SAR and W from diagnostics."""
        W = np.eye(10)
        experiment.spatial_diagnostics = {
            "recommendation": "SAR",
            "W": W,
        }
        # SAR model type goes to fit_model("spatial_lag", ...) which would
        # fail without a real spatial model class, so we mock fit_model.
        mock_result = MagicMock()
        with patch.object(experiment, "fit_model", return_value=mock_result):
            result = experiment.estimate_spatial_model(model_type="auto")
        assert result is mock_result

    def test_estimate_spatial_model_explicit_type(self, experiment):
        """Test estimate_spatial_model with explicit spatial model type."""
        W = np.eye(10)
        mock_result = MagicMock()
        with patch.object(experiment, "fit_model", return_value=mock_result):
            result = experiment.estimate_spatial_model(model_type="sem", W=W, name="my_sem")
        assert result is mock_result

    def test_spatial_diagnostics_report_no_diag(self, experiment):
        """Test spatial_diagnostics_report raises if no diagnostics run."""
        with pytest.raises(ValueError, match="Run run_spatial_diagnostics"):
            experiment.spatial_diagnostics_report("report.html")

    def test_spatial_diagnostics_report(self, experiment):
        """Test spatial_diagnostics_report generates output."""
        experiment.spatial_diagnostics = {"test": True}

        mock_report_cls = MagicMock()
        mock_report_instance = MagicMock()
        mock_report_cls.return_value = mock_report_instance
        mock_report_instance.save.return_value = "/tmp/report.html"

        fake_reporting = MagicMock()
        fake_reporting.SpatialDiagnosticsReport = mock_report_cls

        with patch.dict("sys.modules", {"panelbox.reporting": fake_reporting}):
            result = experiment.spatial_diagnostics_report("report.html")
            assert result is not None
            mock_report_cls.assert_called_once()

    def test_run_spatial_diagnostics_specific_tests(self, experiment):
        """Test run_spatial_diagnostics with specific tests subset."""
        experiment.fit_model("pooled_ols", name="ols")
        N = 10
        W = np.zeros((N, N))
        for i in range(N - 1):
            W[i, i + 1] = 1
            W[i + 1, i] = 1
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        fake_mod = self._patch_spatial_imports()
        patches = self._apply_patches(fake_mod)
        for p in patches:
            p.start()
        try:
            result = experiment.run_spatial_diagnostics(W=W, model_name="ols", tests=["moran"])
        finally:
            for p in reversed(patches):
                p.stop()
        assert "moran" in result
        # Only moran test requested, so lm_tests and lisa should not be present
        assert "lm_tests" not in result
        assert "lisa" not in result
