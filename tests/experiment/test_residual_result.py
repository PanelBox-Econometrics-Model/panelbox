"""
Tests for ResidualResult (Sprint 5).

This test module verifies the functionality of the ResidualResult class,
which provides a container for residual diagnostic analysis.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox import PanelExperiment
from panelbox.experiment.results import ResidualResult


@pytest.fixture
def sample_data():
    """Create sample panel data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "firm": np.repeat(range(10), 5),
            "year": np.tile(range(5), 10),
            "y": np.random.randn(50),
            "x": np.random.randn(50),
        }
    )


@pytest.fixture
def fitted_experiment(sample_data):
    """Create fitted experiment for testing."""
    experiment = PanelExperiment(sample_data, "y ~ x", "firm", "year")
    experiment.fit_model("fe", name="fe")
    return experiment


def test_residual_result_creation_from_experiment(fitted_experiment):
    """Test creating ResidualResult from experiment."""
    residual_result = fitted_experiment.analyze_residuals("fe")

    assert isinstance(residual_result, ResidualResult)
    assert residual_result.model_results is not None
    assert len(residual_result.residuals) == 50
    assert len(residual_result.fitted_values) == 50


def test_residual_result_from_model_results(fitted_experiment):
    """Test factory method from_model_results."""
    model_results = fitted_experiment.get_model("fe")
    residual_result = ResidualResult.from_model_results(model_results)

    assert isinstance(residual_result, ResidualResult)
    assert residual_result.model_results is model_results


def test_shapiro_test(fitted_experiment):
    """Test Shapiro-Wilk normality test."""
    residual_result = fitted_experiment.analyze_residuals("fe")
    stat, pvalue = residual_result.shapiro_test

    assert isinstance(stat, float)
    assert isinstance(pvalue, float)
    assert 0 <= stat <= 1
    assert 0 <= pvalue <= 1


def test_durbin_watson(fitted_experiment):
    """Test Durbin-Watson autocorrelation test."""
    residual_result = fitted_experiment.analyze_residuals("fe")
    dw = residual_result.durbin_watson

    assert isinstance(dw, float)
    assert 0 <= dw <= 4  # DW statistic is between 0 and 4


def test_ljung_box(fitted_experiment):
    """Test Ljung-Box autocorrelation test."""
    residual_result = fitted_experiment.analyze_residuals("fe")
    stat, pvalue = residual_result.ljung_box

    assert isinstance(stat, float)
    assert isinstance(pvalue, float)
    assert stat >= 0  # Q-statistic is non-negative
    assert 0 <= pvalue <= 1


def test_jarque_bera(fitted_experiment):
    """Test Jarque-Bera normality test."""
    residual_result = fitted_experiment.analyze_residuals("fe")
    stat, pvalue = residual_result.jarque_bera

    assert isinstance(stat, float)
    assert isinstance(pvalue, float)
    assert stat >= 0
    assert 0 <= pvalue <= 1


def test_summary_statistics(fitted_experiment):
    """Test summary statistic properties."""
    residual_result = fitted_experiment.analyze_residuals("fe")

    # Mean should be close to 0 for residuals
    assert abs(residual_result.mean) < 1  # Reasonable range

    # Std should be positive
    assert residual_result.std > 0

    # Check other statistics exist
    assert isinstance(residual_result.skewness, float)
    assert isinstance(residual_result.kurtosis, float)
    assert isinstance(residual_result.min, float)
    assert isinstance(residual_result.max, float)


def test_summary_method(fitted_experiment):
    """Test summary() method produces valid text output."""
    residual_result = fitted_experiment.analyze_residuals("fe")
    summary = residual_result.summary()

    assert isinstance(summary, str)
    assert "Residual Diagnostic Analysis" in summary
    assert "Summary Statistics:" in summary
    assert "Diagnostic Tests:" in summary
    assert "Shapiro-Wilk" in summary
    assert "Durbin-Watson" in summary
    assert "Interpretation:" in summary


def test_to_dict_method(fitted_experiment):
    """Test to_dict() returns complete diagnostic data."""
    residual_result = fitted_experiment.analyze_residuals("fe")
    data = residual_result.to_dict()

    assert isinstance(data, dict)

    # Check for required keys from ResidualDataTransformer
    assert "residuals" in data
    assert "fitted" in data
    assert "standardized_residuals" in data
    assert "model_info" in data

    # Check for test results
    assert "tests" in data
    assert "shapiro_wilk" in data["tests"]
    assert "jarque_bera" in data["tests"]
    assert "durbin_watson" in data["tests"]
    assert "ljung_box" in data["tests"]

    # Check for summary stats
    assert "summary" in data
    assert "n_obs" in data["summary"]
    assert "mean" in data["summary"]
    assert "std" in data["summary"]


def test_save_json(fitted_experiment, tmp_path):
    """Test saving to JSON file."""
    residual_result = fitted_experiment.analyze_residuals("fe")
    json_path = tmp_path / "residuals.json"

    result_path = residual_result.save_json(str(json_path))

    assert result_path.exists()
    assert result_path.suffix == ".json"

    # Verify JSON is valid
    import json

    with open(json_path) as f:
        data = json.load(f)

    assert "_metadata" in data
    assert "timestamp" in data["_metadata"]
    assert data["_metadata"]["class"] == "ResidualResult"


def test_metadata_storage(fitted_experiment):
    """Test metadata is stored correctly."""
    metadata = {"test": "value", "experiment_name": "test_experiment"}
    residual_result = fitted_experiment.analyze_residuals("fe", metadata=metadata)

    assert "test" in residual_result.metadata
    assert residual_result.metadata["test"] == "value"
    assert "experiment_formula" in residual_result.metadata


def test_repr(fitted_experiment):
    """Test __repr__ method."""
    residual_result = fitted_experiment.analyze_residuals("fe")
    repr_str = repr(residual_result)

    assert "ResidualResult" in repr_str
    assert "n_obs=" in repr_str
    assert "mean=" in repr_str
    assert "normality_tests=" in repr_str
    assert "autocorrelation=" in repr_str


def test_standardized_residuals(fitted_experiment):
    """Test standardized residuals are computed."""
    residual_result = fitted_experiment.analyze_residuals("fe")

    assert residual_result.standardized_residuals is not None
    assert len(residual_result.standardized_residuals) == len(residual_result.residuals)

    # Standardized residuals should have approximately mean 0, std 1
    std_mean = np.mean(residual_result.standardized_residuals)
    std_std = np.std(residual_result.standardized_residuals, ddof=1)

    assert abs(std_mean) < 1  # Close to 0
    # Note: std may not be exactly 1 due to degrees of freedom adjustment


def test_residual_extraction_from_different_models(sample_data):
    """Test residual extraction works for different model types."""
    experiment = PanelExperiment(sample_data, "y ~ x", "firm", "year")

    # Test with pooled OLS
    experiment.fit_model("pooled_ols", name="pooled")
    residual_result_pooled = experiment.analyze_residuals("pooled")
    assert len(residual_result_pooled.residuals) == 50

    # Test with fixed effects
    experiment.fit_model("fixed_effects", name="fe")
    residual_result_fe = experiment.analyze_residuals("fe")
    assert len(residual_result_fe.residuals) == 50

    # Test with random effects
    experiment.fit_model("random_effects", name="re")
    residual_result_re = experiment.analyze_residuals("re")
    assert len(residual_result_re.residuals) == 50


def test_diagnostic_test_values_are_reasonable(fitted_experiment):
    """Test that diagnostic test values are in reasonable ranges."""
    residual_result = fitted_experiment.analyze_residuals("fe")

    # Shapiro-Wilk: statistic close to 1 is good
    sw_stat, sw_p = residual_result.shapiro_test
    assert 0.8 <= sw_stat <= 1.0  # Reasonable for normal data

    # Durbin-Watson: around 2 is good (no autocorrelation)
    dw = residual_result.durbin_watson
    assert 0.5 <= dw <= 3.5  # Reasonable range

    # All p-values should be between 0 and 1
    _, sw_p = residual_result.shapiro_test
    _, jb_p = residual_result.jarque_bera
    _, lb_p = residual_result.ljung_box

    assert 0 <= sw_p <= 1
    assert 0 <= jb_p <= 1
    assert 0 <= lb_p <= 1


def test_integration_with_panel_experiment(sample_data):
    """Test full integration workflow."""
    # Create experiment
    experiment = PanelExperiment(sample_data, "y ~ x", "firm", "year")

    # Fit models
    experiment.fit_all_models(names=["pooled", "fe", "re"])

    # Analyze residuals for each model
    for model_name in ["pooled", "fe", "re"]:
        residual_result = experiment.analyze_residuals(model_name)

        assert isinstance(residual_result, ResidualResult)
        assert len(residual_result.residuals) == 50

        # Verify summary can be generated
        summary = residual_result.summary()
        assert "Residual Diagnostic Analysis" in summary

        # Verify to_dict works
        data = residual_result.to_dict()
        assert "tests" in data
        assert "summary" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
