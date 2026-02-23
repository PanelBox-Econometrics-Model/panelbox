"""
Tests for ValidationTest runner.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.experiment import PanelExperiment
from panelbox.experiment.results import ValidationResult
from panelbox.experiment.tests import ValidationTest


@pytest.fixture
def panel_data():
    """Create panel data for testing."""
    np.random.seed(42)
    n_entities, n_time = 20, 8

    entities = np.repeat(range(n_entities), n_time)
    time = np.tile(range(n_time), n_entities)

    df = pd.DataFrame(
        {
            "entity": entities,
            "time": time,
            "y": np.random.randn(n_entities * n_time),
            "x1": np.random.randn(n_entities * n_time),
            "x2": np.random.randn(n_entities * n_time),
        }
    )

    return df


class TestValidationTest:
    """Test ValidationTest runner."""

    def test_init(self):
        """Test initialization."""
        runner = ValidationTest()
        assert runner is not None
        assert hasattr(runner, "CONFIGS")

    def test_configs_available(self):
        """Test that all configs are defined."""
        runner = ValidationTest()
        assert "quick" in runner.CONFIGS
        assert "basic" in runner.CONFIGS
        assert "full" in runner.CONFIGS

    def test_run_with_invalid_config(self, panel_data):
        """Test running with invalid config raises ValueError."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        results = experiment.fit_model("pooled_ols", name="ols")

        runner = ValidationTest()

        with pytest.raises(ValueError, match="config must be one of"):
            runner.run(results, config="invalid")

    def test_run_basic_config(self, panel_data):
        """Test running with basic config."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        results = experiment.fit_model("pooled_ols", name="ols")

        runner = ValidationTest()
        validation = runner.run(results, config="basic")

        assert validation is not None
        assert isinstance(validation, ValidationResult)

    def test_run_quick_config(self, panel_data):
        """Test running with quick config."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        results = experiment.fit_model("pooled_ols", name="ols")

        runner = ValidationTest()
        validation = runner.run(results, config="quick")

        assert validation is not None
        assert isinstance(validation, ValidationResult)

    def test_run_full_config(self, panel_data):
        """Test running with full config."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        results = experiment.fit_model("pooled_ols", name="ols")

        runner = ValidationTest()
        validation = runner.run(results, config="full")

        assert validation is not None
        assert isinstance(validation, ValidationResult)

    def test_extract_model_info(self, panel_data):
        """Test extracting model information."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        results = experiment.fit_model("pooled_ols", name="ols")

        runner = ValidationTest()
        model_info = runner._extract_model_info(results)

        assert "model_type" in model_info
        assert "n_obs" in model_info
        assert "n_params" in model_info

    def test_extract_warnings(self, panel_data):
        """Test extracting warnings."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        results = experiment.fit_model("pooled_ols", name="ols")

        runner = ValidationTest()
        warnings = runner._extract_warnings(results)

        assert isinstance(warnings, list)

    def test_repr(self):
        """Test string representation."""
        runner = ValidationTest()
        repr_str = repr(runner)

        assert "ValidationTest" in repr_str
        assert "configs" in repr_str
