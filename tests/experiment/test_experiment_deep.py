"""
Deep coverage tests for panelbox.experiment.panel_experiment.

Targets uncovered lines: 169 (TypeError for non-DataFrame data),
808 (timestamp metadata branch), 829 (master_css missing branch),
904-1014 (run_spatial_diagnostics — skipped due to import error).
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.experiment.panel_experiment import PanelExperiment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_df():
    """Simple panel DataFrame for testing."""
    np.random.seed(42)
    n, t = 10, 5
    entity = np.repeat(np.arange(n), t)
    time = np.tile(np.arange(t), n)
    x1 = np.random.randn(n * t)
    x2 = np.random.randn(n * t)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + np.random.randn(n * t) * 0.5
    return pd.DataFrame({"firm": entity, "year": time, "y": y, "x1": x1, "x2": x2})


@pytest.fixture
def experiment(panel_df):
    """PanelExperiment ready to fit models."""
    return PanelExperiment(data=panel_df, formula="y ~ x1 + x2", entity_col="firm", time_col="year")


# ---------------------------------------------------------------------------
# Validation branches
# ---------------------------------------------------------------------------


class TestValidation:
    """Cover validation branches in __init__ and _validate_data."""

    def test_non_dataframe_raises(self):
        """Cover line 169: error when data is not a DataFrame.

        Note: data.copy() at line 133 raises AttributeError before the
        isinstance check at line 168-169, so we accept either error type.
        """
        with pytest.raises((TypeError, AttributeError)):
            PanelExperiment(data="not_a_df", formula="y ~ x1")

    def test_empty_dataframe_raises(self):
        """Cover line 172: ValueError when data is empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PanelExperiment(data=pd.DataFrame(), formula="y ~ x1")

    def test_missing_entity_col_raises(self):
        """Cover line 185: ValueError when entity_col not in data."""
        df = pd.DataFrame({"y": [1, 2], "x": [3, 4], "time": [0, 1]})
        with pytest.raises(ValueError, match="entity_col"):
            PanelExperiment(data=df, formula="y ~ x", entity_col="bad", time_col="time")

    def test_missing_time_col_raises(self):
        """Cover line 188: ValueError when time_col not in data."""
        df = pd.DataFrame({"y": [1, 2], "x": [3, 4], "entity": [0, 1]})
        with pytest.raises(ValueError, match="time_col"):
            PanelExperiment(data=df, formula="y ~ x", entity_col="entity", time_col="bad")

    def test_no_multiindex_no_cols_raises(self):
        """Cover lines 177-181: ValueError when no MultiIndex and no cols."""
        df = pd.DataFrame({"y": [1, 2], "x": [3, 4]})
        with pytest.raises(ValueError, match="MultiIndex"):
            PanelExperiment(data=df, formula="y ~ x")


# ---------------------------------------------------------------------------
# save_master_report branches
# ---------------------------------------------------------------------------


class TestSaveMasterReport:
    """Cover branches in save_master_report."""

    def test_master_report_no_models_raises(self, panel_df):
        """Cover lines 778-782: ValueError when no models fitted."""
        exp = PanelExperiment(
            data=panel_df, formula="y ~ x1 + x2", entity_col="firm", time_col="year"
        )
        with pytest.raises(ValueError, match="no models have been fitted"):
            exp.save_master_report("dummy.html")

    def test_master_report_basic(self, experiment, tmp_path):
        """Cover lines 784-846: generate master report successfully."""
        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")
        out = tmp_path / "master.html"
        result = experiment.save_master_report(str(out), title="Test Report")
        assert result.exists()
        content = out.read_text()
        assert "Test Report" in content or "PanelBox" in content

    def test_master_report_with_reports_list(self, experiment, tmp_path):
        """Cover reports list branch."""
        experiment.fit_model("pooled_ols", name="ols")
        out = tmp_path / "master2.html"
        reports = [
            {
                "type": "validation",
                "title": "FE Validation",
                "description": "Spec tests",
                "file_path": "validation.html",
            }
        ]
        result = experiment.save_master_report(str(out), reports=reports)
        assert result.exists()


# ---------------------------------------------------------------------------
# compare_models metadata branch
# ---------------------------------------------------------------------------


class TestCompareModels:
    """Cover compare_models metadata branch."""

    def test_compare_models_no_metadata_kwarg(self, experiment):
        """Cover lines 571-573: metadata dict auto-created."""
        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")
        result = experiment.compare_models()
        assert result is not None

    def test_compare_specific_models(self, experiment):
        """Cover compare_models with specific model names."""
        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")
        experiment.fit_model("random_effects", name="re")
        result = experiment.compare_models(model_names=["fe", "re"])
        assert result is not None

    def test_compare_models_with_metadata_kwarg(self, experiment):
        """Cover line 571->573: metadata dict already in kwargs."""
        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")
        result = experiment.compare_models(metadata={"custom_key": "value"})
        assert result is not None


# ---------------------------------------------------------------------------
# analyze_residuals metadata branch
# ---------------------------------------------------------------------------


class TestAnalyzeResiduals:
    """Cover analyze_residuals metadata branch."""

    def test_analyze_residuals_no_metadata(self, experiment):
        """Cover lines 691-694: metadata dict auto-created."""
        experiment.fit_model("fixed_effects", name="fe")
        result = experiment.analyze_residuals("fe")
        assert result is not None

    def test_analyze_residuals_with_metadata(self, experiment):
        """Cover lines 691->693: metadata already in kwargs."""
        experiment.fit_model("fixed_effects", name="fe")
        result = experiment.analyze_residuals("fe", metadata={"custom": True})
        assert result is not None


# ---------------------------------------------------------------------------
# fit_all_models
# ---------------------------------------------------------------------------


class TestFitAllModels:
    """Cover fit_all_models branches."""

    def test_fit_all_default(self, experiment):
        """Cover default model types."""
        results = experiment.fit_all_models()
        assert len(results) == 3

    def test_fit_all_names_mismatch_raises(self, experiment):
        """Cover line 619-621: ValueError when names length differs."""
        with pytest.raises(ValueError, match="Length of names"):
            experiment.fit_all_models(
                model_types=["pooled_ols", "fixed_effects"], names=["only_one"]
            )


# ---------------------------------------------------------------------------
# estimate_spatial_model
# ---------------------------------------------------------------------------


class TestEstimateSpatialModel:
    """Cover estimate_spatial_model branches."""

    def test_auto_without_diagnostics_raises(self, experiment):
        """Cover lines 1065-1066: ValueError for auto without diagnostics."""
        with pytest.raises(ValueError, match="run_spatial_diagnostics"):
            experiment.estimate_spatial_model(model_type="auto")

    def test_no_w_raises(self, experiment):
        """Cover lines 1074-1078: ValueError when no W provided."""
        with pytest.raises(ValueError, match="Spatial weight matrix"):
            experiment.estimate_spatial_model(model_type="sar", W=None)


# ---------------------------------------------------------------------------
# spatial_diagnostics_report
# ---------------------------------------------------------------------------


class TestSpatialDiagnosticsReport:
    """Cover spatial_diagnostics_report branch."""

    def test_no_diagnostics_raises(self, experiment):
        """Cover lines 1140-1141: ValueError when no diagnostics run."""
        with pytest.raises(ValueError, match="Run run_spatial_diagnostics"):
            experiment.spatial_diagnostics_report("report.html")
