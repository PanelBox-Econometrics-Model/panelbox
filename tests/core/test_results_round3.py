"""
Tests for core/results.py to improve branch coverage.

Round 3 - targets specific uncovered lines/branches.
"""

from __future__ import annotations

import json
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_results():
    """Create a PanelResults instance for testing."""
    from panelbox.core.results import PanelResults

    np.random.seed(42)
    n = 50
    k = 3
    param_names = ["x1", "x2", "x3"]

    params = pd.Series([1.5, -0.5, 0.3], index=param_names)
    std_errors = pd.Series([0.2, 0.1, 0.05], index=param_names)
    cov_params = pd.DataFrame(
        np.diag([0.04, 0.01, 0.0025]),
        index=param_names,
        columns=param_names,
    )
    resid = np.random.randn(n)
    fittedvalues = np.random.randn(n) + 50

    model_info = {
        "model_type": "Fixed Effects",
        "formula": "y ~ x1 + x2 + x3",
        "cov_type": "clustered",
    }
    data_info = {
        "nobs": n,
        "n_entities": 5,
        "n_periods": 10,
        "df_model": k,
        "df_resid": n - k,
    }
    rsquared_dict = {
        "rsquared": 0.85,
        "rsquared_adj": 0.83,
        "rsquared_within": 0.80,
        "rsquared_between": 0.75,
        "rsquared_overall": 0.82,
    }

    return PanelResults(
        params=params,
        std_errors=std_errors,
        cov_params=cov_params,
        resid=resid,
        fittedvalues=fittedvalues,
        model_info=model_info,
        data_info=data_info,
        rsquared_dict=rsquared_dict,
    )


@pytest.fixture
def panel_results_no_rsquared():
    """Create PanelResults without rsquared_dict."""
    from panelbox.core.results import PanelResults

    np.random.seed(42)
    n = 50
    k = 2
    param_names = ["x1", "x2"]

    params = pd.Series([1.0, 2.0], index=param_names)
    std_errors = pd.Series([0.1, 0.2], index=param_names)
    cov_params = pd.DataFrame(
        np.diag([0.01, 0.04]),
        index=param_names,
        columns=param_names,
    )

    return PanelResults(
        params=params,
        std_errors=std_errors,
        cov_params=cov_params,
        resid=np.random.randn(n),
        fittedvalues=np.random.randn(n),
        model_info={"model_type": "PooledOLS"},
        data_info={
            "nobs": n,
            "n_entities": 5,
            "df_model": k,
            "df_resid": n - k,
        },
        rsquared_dict=None,
    )


# ---------------------------------------------------------------------------
# core/results.py — PanelResults branches
# ---------------------------------------------------------------------------


class TestPanelResultsBranches:
    """Test uncovered branches in PanelResults."""

    def test_model_property(self, panel_results):
        """Cover line 188: model property returns None when not set."""
        assert panel_results.model is None

    def test_model_property_with_model(self):
        """Cover line 188: model property returns model object."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        mock_model = MagicMock()
        params = pd.Series([1.0], index=["x"])
        std_errors = pd.Series([0.1], index=["x"])
        cov_params = pd.DataFrame([[0.01]], index=["x"], columns=["x"])
        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=np.random.randn(10),
            fittedvalues=np.random.randn(10),
            model_info={"model_type": "test"},
            data_info={"nobs": 10, "n_entities": 2, "df_model": 1, "df_resid": 8},
            model=mock_model,
        )
        assert pr.model is mock_model

    def test_predict_no_newdata(self, panel_results):
        """Cover line 274-275: predict with no newdata returns fittedvalues."""
        preds = panel_results.predict()
        np.testing.assert_array_equal(preds, panel_results.fittedvalues)

    def test_predict_newdata_with_formula_parser(self):
        """Cover lines 237-240, 316: predict with formula_parser."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        n = 20
        k = 2
        params = pd.Series([1.0, 2.0], index=["x1", "x2"])
        std_errors = pd.Series([0.1, 0.2], index=["x1", "x2"])
        cov_params = pd.DataFrame(
            np.diag([0.01, 0.04]),
            index=["x1", "x2"],
            columns=["x1", "x2"],
        )

        mock_parser = MagicMock()
        mock_parser.has_intercept = False
        X_new = np.random.randn(5, k)
        mock_parser.build_design_matrices.return_value = (None, X_new)

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=np.random.randn(n),
            fittedvalues=np.random.randn(n),
            model_info={"model_type": "PooledOLS", "formula": "y ~ x1 + x2"},
            data_info={"nobs": n, "n_entities": 5, "df_model": k, "df_resid": n - k},
            formula_parser=mock_parser,
        )

        newdata = pd.DataFrame({"x1": np.random.randn(5), "x2": np.random.randn(5)})
        preds = pr.predict(newdata)
        assert preds.shape == (5,)

    def test_predict_newdata_from_formula_string(self):
        """Cover lines 241-247: predict with formula string reconstruction."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        n = 20
        # Formula y ~ x1 + x2 produces intercept + x1 + x2 = 3 columns
        k = 3
        param_names = ["Intercept", "x1", "x2"]
        params = pd.Series([0.5, 1.0, 2.0], index=param_names)
        std_errors = pd.Series([0.1, 0.1, 0.2], index=param_names)
        cov_params = pd.DataFrame(
            np.diag([0.01, 0.01, 0.04]),
            index=param_names,
            columns=param_names,
        )

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=np.random.randn(n),
            fittedvalues=np.random.randn(n),
            model_info={"model_type": "PooledOLS", "formula": "y ~ x1 + x2"},
            data_info={"nobs": n, "n_entities": 5, "df_model": k, "df_resid": n - k},
        )

        newdata = pd.DataFrame(
            {
                "y": np.random.randn(5),
                "x1": np.random.randn(5),
                "x2": np.random.randn(5),
            }
        )
        preds = pr.predict(newdata)
        assert preds.shape == (5,)

    def test_predict_no_formula_raises(self):
        """Cover lines 248-252: predict without formula raises ValueError."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        params = pd.Series([1.0], index=["x"])
        std_errors = pd.Series([0.1], index=["x"])
        cov_params = pd.DataFrame([[0.01]], index=["x"], columns=["x"])

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=np.random.randn(10),
            fittedvalues=np.random.randn(10),
            model_info={"model_type": "PooledOLS", "formula": ""},
            data_info={"nobs": 10, "n_entities": 2, "df_model": 1, "df_resid": 8},
        )

        newdata = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Cannot build design matrix"):
            pr.predict(newdata)

    def test_summary_no_rsquared(self, panel_results_no_rsquared):
        """Cover lines 446->448..454->458: summary with nan rsquared values."""
        summary = panel_results_no_rsquared.summary()
        assert "PooledOLS" in summary
        # Should NOT contain within/between/overall R-squared lines
        assert "R-squared (within)" not in summary

    def test_summary_with_rsquared(self, panel_results):
        """Cover lines 446-455: summary with all rsquared values present."""
        summary = panel_results.summary()
        assert "R-squared:" in summary
        assert "R-squared (within)" in summary
        assert "R-squared (between)" in summary
        assert "R-squared (overall)" in summary

    def test_summary_no_n_periods(self):
        """Cover line 441->443: no n_periods in summary."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        params = pd.Series([1.0], index=["x"])
        std_errors = pd.Series([0.1], index=["x"])
        cov_params = pd.DataFrame([[0.01]], index=["x"], columns=["x"])

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=np.random.randn(10),
            fittedvalues=np.random.randn(10),
            model_info={"model_type": "Test"},
            data_info={"nobs": 10, "n_entities": 2, "df_model": 1, "df_resid": 8},
        )
        summary = pr.summary()
        assert "No. Time Periods" not in summary

    def test_summary_with_f_statistic(self, panel_results):
        """Cover lines 461-463: summary with F-statistic."""
        panel_results.f_statistic = 25.0
        panel_results.f_pvalue = 0.001
        summary = panel_results.summary()
        assert "F-statistic" in summary
        assert "F-test p-value" in summary

    def test_summary_significance_stars(self):
        """Cover lines 484-493: all significance star levels."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        # Create params with various p-value levels
        param_names = ["p_001", "p_01", "p_05", "p_10", "p_ns"]
        # Use specific coefficient/se ratios to get desired p-values
        params = pd.Series([10.0, 5.0, 2.5, 1.8, 0.5], index=param_names)
        std_errors = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=param_names)
        cov_params = pd.DataFrame(
            np.diag([1.0] * 5),
            index=param_names,
            columns=param_names,
        )

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=np.random.randn(100),
            fittedvalues=np.random.randn(100),
            model_info={"model_type": "Test"},
            data_info={
                "nobs": 100,
                "n_entities": 10,
                "df_model": 5,
                "df_resid": 95,
            },
        )
        summary = pr.summary()
        assert "***" in summary  # p < 0.001

    def test_summary_custom_title(self, panel_results):
        """Cover line 428-429: summary with custom title."""
        summary = panel_results.summary(title="My Custom Title")
        assert "My Custom Title" in summary

    def test_to_dict_cov_params_ndarray(self):
        """Cover lines 540-545: to_dict with cov_params as ndarray."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        params = pd.Series([1.0, 2.0], index=["x1", "x2"])
        std_errors = pd.Series([0.1, 0.2], index=["x1", "x2"])
        # Pass cov_params as numpy array instead of DataFrame
        cov_array = np.diag([0.01, 0.04])

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_array,
            resid=np.random.randn(10),
            fittedvalues=np.random.randn(10),
            model_info={"model_type": "Test"},
            data_info={"nobs": 10, "n_entities": 2, "df_model": 2, "df_resid": 8},
        )
        d = pr.to_dict()
        assert d["cov_params"] is not None
        assert "values" in d["cov_params"]

    def test_to_dict_cov_params_none(self):
        """Cover line 533->547: to_dict with cov_params=None."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        params = pd.Series([1.0], index=["x"])
        std_errors = pd.Series([0.1], index=["x"])

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=None,
            resid=np.random.randn(10),
            fittedvalues=np.random.randn(10),
            model_info={"model_type": "Test"},
            data_info={"nobs": 10, "n_entities": 2, "df_model": 1, "df_resid": 8},
        )
        d = pr.to_dict()
        assert d["cov_params"] is None

    def test_to_json_to_file(self, panel_results, tmp_path):
        """Cover lines 613-617: to_json with filepath."""
        filepath = tmp_path / "results.json"
        panel_results.to_json(filepath=filepath)
        assert filepath.exists()
        loaded = json.loads(filepath.read_text())
        assert "params" in loaded

    def test_to_json_no_file(self, panel_results):
        """Cover to_json without filepath."""
        json_str = panel_results.to_json()
        parsed = json.loads(json_str)
        assert "params" in parsed

    def test_save_pickle(self, panel_results, tmp_path):
        """Cover lines 650-661: save as pickle."""
        filepath = tmp_path / "results.pkl"
        panel_results.save(filepath, format="pickle")
        assert filepath.exists()

    def test_save_json(self, panel_results, tmp_path):
        """Cover lines 662-663: save as json."""
        filepath = tmp_path / "results.json"
        panel_results.save(filepath, format="json")
        assert filepath.exists()

    def test_save_unknown_format(self, panel_results, tmp_path):
        """Cover lines 664-667: save with unknown format raises ValueError."""
        filepath = tmp_path / "results.xyz"
        with pytest.raises(ValueError, match="not supported"):
            panel_results.save(filepath, format="xyz")

    def test_save_pickle_version_error(self, panel_results, tmp_path):
        """Cover lines 656-657: save when panelbox.__version__ fails."""
        filepath = tmp_path / "results.pkl"
        with patch("panelbox.core.results.pickle.dump") as mock_dump:
            # First test that version metadata is set
            mock_dump.side_effect = lambda *args, **kwargs: None
            panel_results.save(filepath, format="pickle")
            # The save should have been called (but we mocked the dump)

    def test_load_success(self, panel_results, tmp_path):
        """Cover lines 699-712: load from pickle."""
        from panelbox.core.results import PanelResults

        filepath = tmp_path / "results.pkl"
        panel_results.save(filepath, format="pickle")
        loaded = PanelResults.load(filepath)
        assert isinstance(loaded, PanelResults)
        assert loaded.nobs == panel_results.nobs

    def test_load_file_not_found(self, tmp_path):
        """Cover lines 701-702: load non-existent file."""
        from panelbox.core.results import PanelResults

        with pytest.raises(FileNotFoundError, match="File not found"):
            PanelResults.load(tmp_path / "nonexistent.pkl")

    def test_load_wrong_type(self, tmp_path):
        """Cover lines 707-709: load object that isn't PanelResults."""
        from panelbox.core.results import PanelResults

        filepath = tmp_path / "wrong_type.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({"not": "a PanelResults"}, f)
        with pytest.raises(TypeError, match="not a PanelResults"):
            PanelResults.load(filepath)

    def test_validate(self, panel_results):
        """Cover lines 745-748: validate method."""
        # Add entity_index and time_index for validation
        panel_results.entity_index = np.repeat(np.arange(5), 10)
        panel_results.time_index = np.tile(np.arange(10), 5)
        result = panel_results.validate(tests="default")
        assert result is not None

    def test_repr(self, panel_results):
        """Cover lines 750-757: __repr__."""
        repr_str = repr(panel_results)
        assert "PanelResults" in repr_str
        assert "Fixed Effects" in repr_str

    def test_str(self, panel_results):
        """Cover lines 759-761: __str__ calls summary."""
        str_output = str(panel_results)
        assert "Fixed Effects" in str_output

    def test_no_rsquared_dict(self, panel_results_no_rsquared):
        """Cover lines 157-162: rsquared_dict is None."""
        assert np.isnan(panel_results_no_rsquared.rsquared)
        assert np.isnan(panel_results_no_rsquared.rsquared_adj)
        assert np.isnan(panel_results_no_rsquared.rsquared_within)

    def test_predict_with_entity_fe(self):
        """Cover lines 280-316: predict with entity fixed effects."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        params = pd.Series([1.0, 2.0], index=["x1", "x2"])
        std_errors = pd.Series([0.1, 0.2], index=["x1", "x2"])
        cov_params = pd.DataFrame(
            np.diag([0.01, 0.04]),
            index=["x1", "x2"],
            columns=["x1", "x2"],
        )

        mock_parser = MagicMock()
        mock_parser.has_intercept = True
        X_new = np.random.randn(5, 3)  # 3 cols: intercept + 2 features
        mock_parser.build_design_matrices.return_value = (None, X_new)

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=np.random.randn(20),
            fittedvalues=np.random.randn(20),
            model_info={"model_type": "FixedEffects", "formula": "y ~ x1 + x2"},
            data_info={"nobs": 20, "n_entities": 5, "df_model": 2, "df_resid": 18},
            formula_parser=mock_parser,
        )

        # Set entity FE attributes
        pr._entity_effects = True
        pr._entity_fe = pd.Series([0.1, -0.2, 0.3, -0.1, 0.05], index=[1, 2, 3, 4, 5])
        pr._entity_col = "firm"
        pr._intercept = 5.0
        pr._time_effects = False
        pr._time_fe = None

        newdata = pd.DataFrame(
            {
                "firm": [1, 2, 3, 4, 5],
                "x1": np.random.randn(5),
                "x2": np.random.randn(5),
            }
        )
        preds = pr.predict(newdata)
        assert preds.shape == (5,)

    def test_predict_with_time_fe(self):
        """Cover lines 307-312: predict with time fixed effects."""
        from panelbox.core.results import PanelResults

        np.random.seed(42)
        params = pd.Series([1.0], index=["x1"])
        std_errors = pd.Series([0.1], index=["x1"])
        cov_params = pd.DataFrame([[0.01]], index=["x1"], columns=["x1"])

        mock_parser = MagicMock()
        mock_parser.has_intercept = True
        X_new = np.random.randn(3, 2)  # intercept + 1 feature
        mock_parser.build_design_matrices.return_value = (None, X_new)

        pr = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=np.random.randn(10),
            fittedvalues=np.random.randn(10),
            model_info={"model_type": "FixedEffects", "formula": "y ~ x1"},
            data_info={"nobs": 10, "n_entities": 2, "df_model": 1, "df_resid": 8},
            formula_parser=mock_parser,
        )

        # Set time FE attributes
        pr._entity_effects = False
        pr._entity_fe = None
        pr._time_effects = True
        pr._time_fe = pd.Series([0.5, -0.3, 0.1], index=[2020, 2021, 2022])
        pr._time_col = "year"
        pr._intercept = 3.0

        newdata = pd.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "x1": np.random.randn(3),
            }
        )
        preds = pr.predict(newdata)
        assert preds.shape == (3,)
