"""
Tests for ComparisonTest runner.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.experiment import PanelExperiment
from panelbox.experiment.results import ComparisonResult
from panelbox.experiment.tests import ComparisonTest


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


class TestComparisonTest:
    """Test ComparisonTest runner."""

    def test_init(self):
        """Test initialization."""
        runner = ComparisonTest()
        assert runner is not None

    def test_run_with_invalid_input(self):
        """Test running with invalid input raises TypeError."""
        runner = ComparisonTest()

        with pytest.raises(TypeError, match="models must be a dictionary"):
            runner.run("not a dict")

    def test_run_with_one_model(self, panel_data):
        """Test running with only one model raises ValueError."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        ols_res = experiment.fit_model("pooled_ols", name="ols")

        models = {"ols": ols_res}

        runner = ComparisonTest()

        with pytest.raises(ValueError, match="at least 2 models"):
            runner.run(models)

    def test_run_with_two_models(self, panel_data):
        """Test running with two models."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        ols_res = experiment.fit_model("pooled_ols", name="ols")
        fe_res = experiment.fit_model("fixed_effects", name="fe")

        models = {"ols": ols_res, "fe": fe_res}

        runner = ComparisonTest()
        comparison = runner.run(models)

        assert comparison is not None
        assert isinstance(comparison, ComparisonResult)
        assert len(comparison.models) == 2

    def test_run_with_three_models(self, panel_data):
        """Test running with three models."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        ols_res = experiment.fit_model("pooled_ols", name="ols")
        fe_res = experiment.fit_model("fixed_effects", name="fe")
        re_res = experiment.fit_model("random_effects", name="re")

        models = {"ols": ols_res, "fe": fe_res, "re": re_res}

        runner = ComparisonTest()
        comparison = runner.run(models)

        assert comparison is not None
        assert isinstance(comparison, ComparisonResult)
        assert len(comparison.models) == 3

    def test_extract_metrics(self, panel_data):
        """Test extracting metrics."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        ols_res = experiment.fit_model("pooled_ols", name="ols")
        fe_res = experiment.fit_model("fixed_effects", name="fe")

        models = {"ols": ols_res, "fe": fe_res}

        runner = ComparisonTest()
        metrics = runner._extract_metrics(models)

        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) == 2  # Two models
        assert "rsquared" in metrics.columns
        assert "aic" in metrics.columns
        assert "bic" in metrics.columns

    def test_extract_coefficients(self, panel_data):
        """Test extracting coefficients."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        ols_res = experiment.fit_model("pooled_ols", name="ols")
        fe_res = experiment.fit_model("fixed_effects", name="fe")

        models = {"ols": ols_res, "fe": fe_res}

        runner = ComparisonTest()
        coefficients = runner._extract_coefficients(models)

        assert isinstance(coefficients, pd.DataFrame)
        assert len(coefficients.columns) == 2  # Two models
        assert "ols" in coefficients.columns
        assert "fe" in coefficients.columns

    def test_run_without_coefficients(self, panel_data):
        """Test running without extracting coefficients."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        ols_res = experiment.fit_model("pooled_ols", name="ols")
        fe_res = experiment.fit_model("fixed_effects", name="fe")

        models = {"ols": ols_res, "fe": fe_res}

        runner = ComparisonTest()
        comparison = runner.run(models, include_coefficients=False)

        # ComparisonResult always has models
        assert comparison is not None
        assert comparison.models is not None

    def test_run_without_statistics(self, panel_data):
        """Test running without extracting statistics."""
        experiment = PanelExperiment(
            data=panel_data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
        )
        ols_res = experiment.fit_model("pooled_ols", name="ols")
        fe_res = experiment.fit_model("fixed_effects", name="fe")

        models = {"ols": ols_res, "fe": fe_res}

        runner = ComparisonTest()
        comparison = runner.run(models, include_statistics=False)

        # ComparisonResult always has comparison_metrics
        assert comparison is not None
        assert comparison.comparison_metrics is not None

    def test_repr(self):
        """Test string representation."""
        runner = ComparisonTest()
        repr_str = repr(runner)

        assert "ComparisonTest" in repr_str
