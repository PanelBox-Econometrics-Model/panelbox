"""
Integration test for complete Sprint 8 workflow.

This test validates the complete end-to-end workflow including:
- Fitting multiple models
- Running validation tests
- Comparing models
- Analyzing residuals
- Generating master report
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.experiment import PanelExperiment


@pytest.fixture
def panel_data():
    """Create panel data for testing."""
    np.random.seed(42)
    n_entities, n_time = 30, 10

    entities = np.repeat(range(n_entities), n_time)
    time = np.tile(range(n_time), n_entities)

    df = pd.DataFrame(
        {
            "entity": entities,
            "time": time,
            "y": np.random.randn(n_entities * n_time) + entities[:, np.newaxis].ravel() * 0.5,
            "x1": np.random.randn(n_entities * n_time),
            "x2": np.random.randn(n_entities * n_time),
        }
    )

    return df


class TestFullWorkflowSprint8:
    """Test complete end-to-end workflow from Sprint 8."""

    def test_complete_workflow(self, panel_data, tmp_path):
        """Test complete workflow from data to reports."""

        # Step 1: Create experiment
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )

        assert experiment is not None
        assert len(experiment.list_models()) == 0

        # Step 2: Fit multiple models
        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")
        experiment.fit_model("random_effects", name="re")

        assert len(experiment.list_models()) == 3
        assert "ols" in experiment.list_models()
        assert "fe" in experiment.list_models()
        assert "re" in experiment.list_models()

        # Step 3: Run validation on one model
        validation = experiment.validate_model("fe")

        assert validation is not None
        assert hasattr(validation, "save_html")
        assert hasattr(validation, "save_json")

        # Save validation report
        val_path = tmp_path / "validation.html"
        validation.save_html(str(val_path), test_type="validation")

        assert val_path.exists()
        assert val_path.stat().st_size > 0

        # Step 4: Compare models
        comparison = experiment.compare_models(["ols", "fe", "re"])

        assert comparison is not None
        assert len(comparison.model_names) == 3
        assert hasattr(comparison, "save_html")
        assert hasattr(comparison, "save_json")

        # Save comparison report
        comp_path = tmp_path / "comparison.html"
        comparison.save_html(str(comp_path), test_type="comparison")

        assert comp_path.exists()
        assert comp_path.stat().st_size > 0

        # Step 5: Analyze residuals
        residuals = experiment.analyze_residuals("fe")

        assert residuals is not None
        assert hasattr(residuals, "save_html")
        assert hasattr(residuals, "save_json")

        # Save residual report
        res_path = tmp_path / "residuals.html"
        residuals.save_html(str(res_path), test_type="residuals")

        assert res_path.exists()
        assert res_path.stat().st_size > 0

        # Step 6: Generate master report
        master_path = tmp_path / "master.html"
        experiment.save_master_report(
            str(master_path),
            reports=[
                {
                    "type": "validation",
                    "title": "Fixed Effects Validation",
                    "description": "Specification tests for FE model",
                    "file_path": "validation.html",
                },
                {
                    "type": "comparison",
                    "title": "Model Comparison",
                    "description": "Comparison of OLS, FE, and RE models",
                    "file_path": "comparison.html",
                },
                {
                    "type": "residuals",
                    "title": "Residual Diagnostics",
                    "description": "Residual analysis for FE model",
                    "file_path": "residuals.html",
                },
            ],
        )

        assert master_path.exists()
        assert master_path.stat().st_size > 0

        # Step 7: Verify all files exist
        assert val_path.exists()
        assert comp_path.exists()
        assert res_path.exists()
        assert master_path.exists()

        # Step 8: Verify HTML content
        val_html = val_path.read_text()
        assert "validation" in val_html.lower()

        comp_html = comp_path.read_text()
        assert "comparison" in comp_html.lower()

        res_html = res_path.read_text()
        assert "residual" in res_html.lower()

        master_html = master_path.read_text()
        assert "master" in master_html.lower() or "experiment" in master_html.lower()
        assert "ols" in master_html.lower()
        assert "fe" in master_html.lower()

    def test_master_report_without_subreports(self, panel_data, tmp_path):
        """Test generating master report without sub-reports."""

        # Create experiment and fit models
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )

        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")

        # Generate master report without reports list
        master_path = tmp_path / "master_simple.html"
        experiment.save_master_report(str(master_path))

        assert master_path.exists()
        assert master_path.stat().st_size > 0

        # Verify content
        master_html = master_path.read_text()
        assert "ols" in master_html.lower()
        assert "fe" in master_html.lower()

    def test_master_report_with_no_models_raises_error(self, panel_data, tmp_path):
        """Test that master report raises error if no models fitted."""

        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )

        master_path = tmp_path / "master_error.html"

        with pytest.raises(ValueError, match="no models have been fitted"):
            experiment.save_master_report(str(master_path))

    def test_json_exports(self, panel_data, tmp_path):
        """Test JSON exports for all result types."""

        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )

        experiment.fit_model("pooled_ols", name="ols")
        experiment.fit_model("fixed_effects", name="fe")

        # Validation JSON
        validation = experiment.validate_model("fe")
        val_json = tmp_path / "validation.json"
        validation.save_json(str(val_json))

        assert val_json.exists()

        # Comparison JSON
        comparison = experiment.compare_models(["ols", "fe"])
        comp_json = tmp_path / "comparison.json"
        comparison.save_json(str(comp_json))

        assert comp_json.exists()

        # Residuals JSON
        residuals = experiment.analyze_residuals("fe")
        res_json = tmp_path / "residuals.json"
        residuals.save_json(str(res_json))

        assert res_json.exists()
