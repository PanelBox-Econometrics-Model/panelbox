"""
Tests for experiment/panel_experiment.py, experiment/results/residual_result.py,
experiment/tests/validation_test.py, and experiment/tests/comparison_test.py
to improve branch coverage.

Round 3 - targets specific uncovered lines/branches.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_data():
    """Create simple panel data for experiment tests."""
    np.random.seed(42)
    n_firms = 5
    n_years = 10
    n = n_firms * n_years

    data = pd.DataFrame(
        {
            "firm": np.repeat(np.arange(1, n_firms + 1), n_years),
            "year": np.tile(np.arange(2000, 2000 + n_years), n_firms),
            "invest": np.random.randn(n) * 10 + 50,
            "value": np.random.randn(n) * 100 + 500,
            "capital": np.random.randn(n) * 20 + 100,
        }
    )
    return data


@pytest.fixture
def experiment(panel_data):
    """Create a PanelExperiment."""
    from panelbox.experiment.panel_experiment import PanelExperiment

    return PanelExperiment(
        data=panel_data,
        formula="invest ~ value + capital",
        entity_col="firm",
        time_col="year",
    )


@pytest.fixture
def fitted_experiment(experiment):
    """Create experiment with fitted models."""
    experiment.fit_model("pooled_ols", name="ols")
    experiment.fit_model("fixed_effects", name="fe")
    return experiment


@pytest.fixture
def model_results(fitted_experiment):
    """Get model results from experiment."""
    return fitted_experiment.get_model("fe")


# ---------------------------------------------------------------------------
# experiment/panel_experiment.py
# ---------------------------------------------------------------------------


class TestPanelExperimentBranches:
    """Test uncovered branches in PanelExperiment."""

    def test_validate_data_empty_dataframe(self):
        """Cover line 169/171-172: empty DataFrame raises ValueError."""
        from panelbox.experiment.panel_experiment import PanelExperiment

        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            PanelExperiment(
                data=empty_df,
                formula="y ~ x",
                entity_col="firm",
                time_col="year",
            )

    def test_validate_data_not_dataframe(self):
        """Cover line 168-169: non-DataFrame raises TypeError.
        Note: data.copy() runs before _validate_data, so we need
        an object that has .copy() but is not a DataFrame."""
        from panelbox.experiment.panel_experiment import PanelExperiment

        class FakeData:
            def copy(self):
                return self

            @property
            def empty(self):
                return False

        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            PanelExperiment(
                data=FakeData(),
                formula="y ~ x",
                entity_col="firm",
                time_col="year",
            )

    def test_validate_data_missing_entity_col(self, panel_data):
        """Cover line 184-185: missing entity column."""
        from panelbox.experiment.panel_experiment import PanelExperiment

        with pytest.raises(ValueError, match=r"entity_col .* not found"):
            PanelExperiment(
                data=panel_data,
                formula="invest ~ value",
                entity_col="nonexistent",
                time_col="year",
            )

    def test_validate_data_missing_time_col(self, panel_data):
        """Cover line 187-188: missing time column."""
        from panelbox.experiment.panel_experiment import PanelExperiment

        with pytest.raises(ValueError, match=r"time_col .* not found"):
            PanelExperiment(
                data=panel_data,
                formula="invest ~ value",
                entity_col="firm",
                time_col="nonexistent",
            )

    def test_validate_data_multiindex_required(self, panel_data):
        """Cover lines 175-181: MultiIndex required when no entity/time cols."""
        from panelbox.experiment.panel_experiment import PanelExperiment

        with pytest.raises(ValueError, match="MultiIndex"):
            PanelExperiment(
                data=panel_data,
                formula="invest ~ value",
                # No entity_col or time_col -> requires MultiIndex
            )

    def test_init_with_multiindex(self, panel_data):
        """Cover lines 143-145: init with MultiIndex data."""
        from panelbox.experiment.panel_experiment import PanelExperiment

        mi_data = panel_data.set_index(["firm", "year"])
        exp = PanelExperiment(data=mi_data, formula="invest ~ value + capital")
        assert exp.entity_col is None
        assert exp.time_col is None

    def test_fit_model_unknown_type(self, experiment):
        """Cover lines 271-274: unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            experiment.fit_model("unknown_model_type")

    def test_fit_model_duplicate_name(self, experiment):
        """Cover line 282: duplicate model name raises ValueError."""
        experiment.fit_model("pooled_ols", name="ols")
        with pytest.raises(ValueError, match="already exists"):
            experiment.fit_model("pooled_ols", name="ols")

    def test_generate_model_name(self, experiment):
        """Cover lines 387-396: auto-generate model name."""
        name = experiment._generate_model_name("pooled_ols")
        assert name == "pooled_ols_1"
        name2 = experiment._generate_model_name("pooled_ols")
        assert name2 == "pooled_ols_2"

    def test_get_model_not_found(self, experiment):
        """Cover lines 342-346: get non-existent model raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            experiment.get_model("nonexistent")

    def test_get_model_metadata_not_found(self, experiment):
        """Cover line 370: get metadata for non-existent model."""
        with pytest.raises(KeyError, match="not found"):
            experiment.get_model_metadata("nonexistent")

    def test_list_models(self, fitted_experiment):
        """Test listing models."""
        models = fitted_experiment.list_models()
        assert "ols" in models
        assert "fe" in models

    def test_fit_all_models_default(self, experiment):
        """Cover lines 613-634: fit_all_models default types."""
        results = experiment.fit_all_models()
        assert len(results) == 3

    def test_fit_all_models_with_names(self, experiment):
        """Cover lines 618-628: fit_all_models with custom names."""
        results = experiment.fit_all_models(
            model_types=["pooled_ols", "fixed_effects"],
            names=["my_ols", "my_fe"],
        )
        assert "my_ols" in results
        assert "my_fe" in results

    def test_fit_all_models_names_mismatch(self, experiment):
        """Cover lines 618-622: names length mismatch."""
        with pytest.raises(ValueError, match="Length of names"):
            experiment.fit_all_models(
                model_types=["pooled_ols", "fixed_effects"],
                names=["only_one"],
            )

    def test_repr(self, fitted_experiment):
        """Cover lines 1154-1166: __repr__."""
        repr_str = repr(fitted_experiment)
        assert "PanelExperiment" in repr_str
        assert "n_models=2" in repr_str

    def test_repr_no_models(self, experiment):
        """Cover line 1157: __repr__ with no models."""
        repr_str = repr(experiment)
        assert "none" in repr_str

    def test_save_master_report_no_models(self, experiment, tmp_path):
        """Cover lines 778-782: save_master_report with no models."""
        with pytest.raises(ValueError, match="no models have been fitted"):
            experiment.save_master_report(str(tmp_path / "master.html"))

    def test_save_master_report_with_timestamp(self, fitted_experiment, tmp_path):
        """Cover line 808: model metadata with 'timestamp' key."""
        from datetime import datetime

        # Add timestamp to metadata
        fitted_experiment._model_metadata["ols"]["timestamp"] = datetime.now()

        output = fitted_experiment.save_master_report(str(tmp_path / "master.html"))
        assert output.exists()

    def test_compare_models(self, fitted_experiment):
        """Cover lines 536-576: compare_models method."""
        comp = fitted_experiment.compare_models()
        assert comp is not None

    def test_compare_models_specific(self, fitted_experiment):
        """Cover compare_models with specific model names."""
        comp = fitted_experiment.compare_models(model_names=["ols", "fe"])
        assert comp is not None

    def test_validate_model(self, fitted_experiment):
        """Cover lines 491-534: validate_model method."""
        val = fitted_experiment.validate_model("fe")
        assert val is not None

    def test_analyze_residuals(self, fitted_experiment):
        """Cover lines 636-697: analyze_residuals method."""
        resid_result = fitted_experiment.analyze_residuals("fe")
        assert resid_result is not None
        assert hasattr(resid_result, "residuals")


# ---------------------------------------------------------------------------
# experiment/results/residual_result.py
# ---------------------------------------------------------------------------


class TestResidualResultBranches:
    """Test uncovered branches in ResidualResult."""

    def test_standardized_residuals_provided(self, model_results):
        """Cover line 117: standardized_residuals provided externally."""
        from panelbox.experiment.results import ResidualResult

        std_resid = np.random.randn(len(model_results.resid))
        rr = ResidualResult(
            model_results=model_results,
            standardized_residuals=std_resid,
        )
        np.testing.assert_array_equal(rr.standardized_residuals, std_resid)

    def test_extract_residuals_attr_name(self):
        """Cover lines 125-128: fallback to 'residuals' attribute."""
        from panelbox.experiment.results import ResidualResult

        mock_model = MagicMock()
        del mock_model.resid  # Remove resid attribute
        mock_model.residuals = np.random.randn(50)
        mock_model.fittedvalues = np.random.randn(50)
        del mock_model.scale
        del mock_model.resid_std_err
        rr = ResidualResult(model_results=mock_model)
        assert len(rr.residuals) == 50

    def test_extract_residuals_raises(self):
        """Cover line 128: no residuals attribute raises ValueError."""
        from panelbox.experiment.results import ResidualResult

        mock_model = MagicMock(spec=[])
        with pytest.raises(ValueError, match="Could not extract residuals"):
            ResidualResult(model_results=mock_model)

    def test_extract_fitted_attr_name(self):
        """Cover lines 134-135: fallback to 'fitted_values' attribute."""
        from panelbox.experiment.results import ResidualResult

        mock_model = MagicMock()
        mock_model.resid = np.random.randn(50)
        del mock_model.fittedvalues
        mock_model.fitted_values = np.random.randn(50)
        del mock_model.scale
        del mock_model.resid_std_err
        rr = ResidualResult(model_results=mock_model)
        assert len(rr.fitted_values) == 50

    def test_extract_fitted_predict(self):
        """Cover lines 136-137: fallback to predict() method."""
        from panelbox.experiment.results import ResidualResult

        mock_model = MagicMock()
        mock_model.resid = np.random.randn(50)
        del mock_model.fittedvalues
        del mock_model.fitted_values
        mock_model.predict.return_value = np.random.randn(50)
        del mock_model.scale
        del mock_model.resid_std_err
        rr = ResidualResult(model_results=mock_model)
        assert len(rr.fitted_values) == 50

    def test_extract_fitted_raises(self):
        """Cover lines 138-139: no fitted values raises ValueError."""
        from panelbox.experiment.results import ResidualResult

        mock_model = MagicMock(spec=["resid"])
        mock_model.resid = np.random.randn(50)
        with pytest.raises(ValueError, match="Could not extract fitted"):
            ResidualResult(model_results=mock_model)

    def test_compute_standardized_residuals_resid_std_err(self):
        """Cover line 147: scale from resid_std_err attribute."""
        from panelbox.experiment.results import ResidualResult

        mock_model = MagicMock()
        mock_model.resid = np.random.randn(50)
        mock_model.fittedvalues = np.random.randn(50)
        del mock_model.scale
        mock_model.resid_std_err = 2.0
        rr = ResidualResult(model_results=mock_model)
        expected = np.asarray(mock_model.resid) / 2.0
        np.testing.assert_allclose(rr.standardized_residuals, expected)

    def test_interpret_dw_positive_autocorr(self, model_results):
        """Cover line 390: DW < 1.5 -> positive autocorrelation."""
        from panelbox.experiment.results import ResidualResult

        rr = ResidualResult(model_results=model_results)
        assert rr._interpret_dw(0.5) == "Positive autocorrelation"

    def test_interpret_dw_negative_autocorr(self, model_results):
        """Cover line 392: DW > 2.5 -> negative autocorrelation."""
        from panelbox.experiment.results import ResidualResult

        rr = ResidualResult(model_results=model_results)
        assert rr._interpret_dw(3.5) == "Negative autocorrelation"

    def test_interpret_dw_no_autocorr(self, model_results):
        """Cover line 394: DW in [1.5, 2.5] -> no autocorrelation."""
        from panelbox.experiment.results import ResidualResult

        rr = ResidualResult(model_results=model_results)
        assert rr._interpret_dw(2.0) == "No significant autocorrelation"

    def test_summary_not_normal(self):
        """Cover lines 475, 480, 485: summary with non-normal, autocorrelated residuals."""
        from panelbox.experiment.results import ResidualResult

        np.random.seed(42)
        mock_model = MagicMock()
        # Create non-normal residuals (heavy-tailed)
        resids = np.concatenate([np.random.randn(48), [100.0, -100.0]])
        mock_model.resid = resids
        mock_model.fittedvalues = np.random.randn(50)
        del mock_model.scale
        del mock_model.resid_std_err
        rr = ResidualResult(model_results=mock_model)
        summary_text = rr.summary()
        assert "Residual Diagnostic Analysis" in summary_text
        # Should have at least one pass/fail interpretation
        assert "distributed" in summary_text

    def test_summary_autocorrelation_present(self):
        """Cover line 480: summary with autocorrelation present (DW far from 2)."""
        from panelbox.experiment.results import ResidualResult

        np.random.seed(42)
        mock_model = MagicMock()
        # Create strongly autocorrelated residuals (AR(1) with rho=0.99)
        n = 100
        resids = np.zeros(n)
        resids[0] = np.random.randn()
        for i in range(1, n):
            resids[i] = 0.99 * resids[i - 1] + np.random.randn() * 0.01
        mock_model.resid = resids
        mock_model.fittedvalues = np.random.randn(n)
        del mock_model.scale
        del mock_model.resid_std_err
        rr = ResidualResult(model_results=mock_model)
        summary_text = rr.summary()
        assert "Autocorrelation may be present" in summary_text

    def test_summary_all_pass(self, model_results):
        """Cover lines 472-483: summary with all tests passing."""
        from panelbox.experiment.results import ResidualResult

        rr = ResidualResult(model_results=model_results)
        summary_text = rr.summary()
        assert "Residual Diagnostic Analysis" in summary_text

    def test_repr(self, model_results):
        """Cover lines 530-541: __repr__."""
        from panelbox.experiment.results import ResidualResult

        rr = ResidualResult(model_results=model_results)
        repr_str = repr(rr)
        assert "ResidualResult" in repr_str

    def test_pass_mark(self, model_results):
        """Cover line 491: _pass_mark function."""
        from panelbox.experiment.results import ResidualResult

        rr = ResidualResult(model_results=model_results)
        assert "PASS" in rr._pass_mark(0.1)
        assert "FAIL" in rr._pass_mark(0.01)

    def test_from_model_results_factory(self, model_results):
        """Cover lines 495-528: from_model_results factory method."""
        from panelbox.experiment.results import ResidualResult

        rr = ResidualResult.from_model_results(model_results, metadata={"model_type": "fe"})
        assert rr is not None
        assert rr.metadata["model_type"] == "fe"


# ---------------------------------------------------------------------------
# experiment/tests/validation_test.py
# ---------------------------------------------------------------------------


class TestValidationTestBranches:
    """Test uncovered branches in ValidationTest."""

    def test_run_invalid_config(self):
        """Cover line 100-101: invalid config raises ValueError."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        with pytest.raises(ValueError, match="config must be one of"):
            runner.run(MagicMock(), config="invalid")

    def test_run_with_explicit_tests(self, model_results):
        """Cover lines 104->108: explicit tests list (not None)."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        result = runner.run(model_results, tests=["heteroskedasticity", "normality"])
        assert result is not None

    def test_run_no_validate_method(self):
        """Cover lines 118-124: model without validate() raises NotImplementedError."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock(spec=[])  # No validate method
        with pytest.raises(NotImplementedError, match="requires panelbox models"):
            runner.run(mock_results)

    def test_extract_model_info_basic(self, model_results):
        """Cover lines 140-173: _extract_model_info with various attributes."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        info = runner._extract_model_info(model_results)
        assert "model_type" in info
        assert "n_obs" in info
        assert "rsquared" in info

    def test_extract_model_info_with_f_stat_float(self):
        """Cover line 171: f_statistic as float."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock()
        mock_results.nobs = 100
        mock_results.params = pd.Series([1.0, 2.0])
        mock_results.rsquared = 0.8
        mock_results.rsquared_adj = 0.78
        mock_results.f_statistic = 25.5  # float
        del mock_results.aic
        del mock_results.bic
        del mock_results.loglik
        info = runner._extract_model_info(mock_results)
        assert info["f_statistic"] == 25.5

    def test_extract_model_info_with_f_stat_dict(self):
        """Cover line 167: f_statistic as dict."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock()
        mock_results.nobs = 100
        mock_results.params = pd.Series([1.0, 2.0])
        mock_results.rsquared = 0.8
        mock_results.rsquared_adj = 0.78
        mock_results.f_statistic = {"stat": 25.5, "pvalue": 0.001}
        del mock_results.aic
        del mock_results.bic
        del mock_results.loglik
        info = runner._extract_model_info(mock_results)
        assert info["f_statistic"] == 25.5

    def test_extract_model_info_with_f_stat_object(self):
        """Cover line 169: f_statistic as object with .stat."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock()
        mock_results.nobs = 100
        mock_results.params = pd.Series([1.0, 2.0])
        mock_results.rsquared = 0.8
        mock_results.rsquared_adj = 0.78
        f_obj = MagicMock()
        f_obj.stat = 25.5
        mock_results.f_statistic = f_obj
        del mock_results.aic
        del mock_results.bic
        del mock_results.loglik
        info = runner._extract_model_info(mock_results)
        assert info["f_statistic"] == 25.5

    def test_extract_model_info_no_rsquared_adj(self):
        """Cover line 152-153: rsquared_adj fallback to rsquared."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock()
        mock_results.nobs = 100
        mock_results.params = pd.Series([1.0, 2.0])
        mock_results.rsquared = 0.7
        del mock_results.rsquared_adj
        mock_results.f_statistic = None
        del mock_results.aic
        del mock_results.bic
        del mock_results.loglik
        info = runner._extract_model_info(mock_results)
        assert info["rsquared_adj"] == 0.7

    def test_extract_model_info_with_aic_bic_loglik(self):
        """Cover lines 155-162: aic, bic, loglik attributes."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock()
        mock_results.nobs = 100
        mock_results.params = pd.Series([1.0, 2.0])
        mock_results.rsquared = 0.8
        mock_results.rsquared_adj = 0.78
        mock_results.aic = 150.0
        mock_results.bic = 160.0
        mock_results.loglik = -70.0
        mock_results.f_statistic = None
        info = runner._extract_model_info(mock_results)
        assert info["aic"] == 150.0
        assert info["bic"] == 160.0
        assert info["log_likelihood"] == -70.0

    def test_extract_warnings_low_rsquared(self):
        """Cover lines 192-193: low R-squared warning."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock()
        mock_results.rsquared = 0.1
        del mock_results.condition_number
        mock_results.nobs = 100
        warnings = runner._extract_warnings(mock_results)
        assert any("Low R" in w for w in warnings)

    def test_extract_warnings_high_condition_number(self):
        """Cover lines 196-197: high condition number warning."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock()
        mock_results.rsquared = 0.8
        mock_results.condition_number = 50
        mock_results.nobs = 100
        warnings = runner._extract_warnings(mock_results)
        assert any("condition number" in w for w in warnings)

    def test_extract_warnings_small_sample(self):
        """Cover lines 200-201: small sample warning."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        mock_results = MagicMock()
        mock_results.rsquared = 0.8
        del mock_results.condition_number
        mock_results.nobs = 15
        warnings = runner._extract_warnings(mock_results)
        assert any("Small sample" in w for w in warnings)

    def test_repr(self):
        """Cover line 207: __repr__."""
        from panelbox.experiment.tests.validation_test import ValidationTest

        runner = ValidationTest()
        repr_str = repr(runner)
        assert "ValidationTest" in repr_str


# ---------------------------------------------------------------------------
# experiment/tests/comparison_test.py
# ---------------------------------------------------------------------------


class TestComparisonTestBranches:
    """Test uncovered branches in ComparisonTest."""

    def test_run_not_dict(self):
        """Cover lines 97-98: non-dict input raises TypeError."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        with pytest.raises(TypeError, match="must be a dictionary"):
            runner.run([])

    def test_run_too_few_models(self):
        """Cover lines 100-103: fewer than 2 models raises ValueError."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        with pytest.raises(ValueError, match="at least 2 models"):
            runner.run({"only_one": MagicMock()})

    def test_run_comparison(self, fitted_experiment):
        """Cover lines 105-117: run comparison with 2 models."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        models = {
            "ols": fitted_experiment.get_model("ols"),
            "fe": fitted_experiment.get_model("fe"),
        }
        result = runner.run(models)
        assert result is not None

    def test_extract_metrics_no_rsquared(self):
        """Cover line 142: model without rsquared."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        mock1 = MagicMock(spec=[])
        mock2 = MagicMock(spec=[])
        metrics = runner._extract_metrics({"m1": mock1, "m2": mock2})
        assert metrics.loc["m1", "rsquared"] is None or pd.isna(metrics.loc["m1", "rsquared"])

    def test_extract_metrics_rsquared_adj_fallback(self):
        """Cover lines 147-148: rsquared_adj fallback to rsquared."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        mock1 = MagicMock()
        mock1.rsquared = 0.7
        del mock1.rsquared_adj
        del mock1.aic
        del mock1.bic
        del mock1.loglik
        del mock1.f_statistic
        del mock1.nobs
        del mock1.params
        mock2 = MagicMock()
        mock2.rsquared = 0.8
        del mock2.rsquared_adj
        del mock2.aic
        del mock2.bic
        del mock2.loglik
        del mock2.f_statistic
        del mock2.nobs
        del mock2.params
        metrics = runner._extract_metrics({"m1": mock1, "m2": mock2})
        assert metrics.loc["m1", "rsquared_adj"] == 0.7

    def test_extract_metrics_no_rsquared_no_adj(self):
        """Cover lines 149-150: no rsquared or rsquared_adj."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        mock1 = MagicMock(spec=["nobs", "params"])
        mock1.nobs = 50
        mock1.params = pd.Series([1.0])
        mock2 = MagicMock(spec=["nobs", "params"])
        mock2.nobs = 50
        mock2.params = pd.Series([1.0])
        metrics = runner._extract_metrics({"m1": mock1, "m2": mock2})
        # None gets converted to NaN in DataFrame
        assert metrics.loc["m1", "rsquared_adj"] is None or pd.isna(
            metrics.loc["m1", "rsquared_adj"]
        )

    def test_extract_metrics_with_aic_bic(self):
        """Cover lines 153-168: aic, bic, loglik attributes."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        mock1 = MagicMock()
        mock1.rsquared = 0.7
        mock1.rsquared_adj = 0.68
        mock1.aic = 100
        mock1.bic = 110
        mock1.loglik = -45
        mock1.nobs = 50
        mock1.params = pd.Series([1.0])
        # f_statistic with .stat attribute
        f_obj = MagicMock()
        f_obj.stat = 10.0
        mock1.f_statistic = f_obj
        mock2 = MagicMock()
        mock2.rsquared = 0.8
        mock2.rsquared_adj = 0.78
        mock2.aic = 95
        mock2.bic = 105
        mock2.loglik = -42
        mock2.nobs = 50
        mock2.params = pd.Series([1.0, 2.0])
        mock2.f_statistic = 12.0  # float
        metrics = runner._extract_metrics({"m1": mock1, "m2": mock2})
        assert metrics.loc["m1", "aic"] == 100
        assert metrics.loc["m2", "f_stat"] == 12.0

    def test_extract_metrics_no_f_stat(self):
        """Cover line 177: no f_statistic attribute."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        mock1 = MagicMock()
        mock1.rsquared = 0.7
        mock1.rsquared_adj = 0.68
        del mock1.aic
        del mock1.bic
        del mock1.loglik
        del mock1.f_statistic
        mock1.nobs = 50
        mock1.params = pd.Series([1.0])
        mock2 = MagicMock()
        mock2.rsquared = 0.8
        mock2.rsquared_adj = 0.78
        del mock2.aic
        del mock2.bic
        del mock2.loglik
        del mock2.f_statistic
        mock2.nobs = 50
        mock2.params = pd.Series([1.0])
        metrics = runner._extract_metrics({"m1": mock1, "m2": mock2})
        assert metrics.loc["m1", "f_stat"] is None or pd.isna(metrics.loc["m1", "f_stat"])

    def test_extract_coefficients_no_params(self):
        """Cover line 218: model without params attribute."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        mock1 = MagicMock()
        mock1.params = pd.Series({"x": 1.0, "y": 2.0})
        mock2 = MagicMock(spec=[])  # No params
        coef_df = runner._extract_coefficients({"m1": mock1, "m2": mock2})
        assert "m1" in coef_df.columns
        assert "m2" in coef_df.columns

    def test_repr(self):
        """Cover line 227: __repr__."""
        from panelbox.experiment.tests.comparison_test import ComparisonTest

        runner = ComparisonTest()
        assert "ComparisonTest" in repr(runner)
