"""Tests for AutoExperiment HTML report generation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from panelbox.autoexperiment.report import AutoExperimentReportGenerator
from panelbox.autoexperiment.results import AutoExperimentResults

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_mock_model():
    """Create a mock PanelResults object."""
    model = MagicMock()
    model.params = pd.Series({"x1": 0.5, "x2": -0.3, "const": 1.2})
    model.std_errors = pd.Series({"x1": 0.1, "x2": 0.15, "const": 0.4})
    model.tstats = pd.Series({"x1": 5.0, "x2": -2.0, "const": 3.0})
    model.pvalues = pd.Series({"x1": 0.0001, "x2": 0.045, "const": 0.003})
    return model


def _make_ranking(n=5):
    """Create a ranking DataFrame."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "model_type": ["fe", "re", "pooled_ols", "fd", "fe"][i % 5],
                "formula": f"y ~ x1 + x{i + 2}",
                "score": 0.1 * (i + 1),
                "bic": 1000 + i * 50,
                "aic": 980 + i * 50,
                "rsq": 0.8 - i * 0.05,
                "rsq_adj": 0.78 - i * 0.05,
                "classification": ["VALID", "VALID", "WARNING", "INVALID", "VALID"][i % 5],
                "n_vars": 2 + i,
                "cov_type": "robust",
                "tests_passed": "3/4",
                "signs_ok": "2/2",
            }
        )
    return pd.DataFrame(rows)


def _make_results(datamining=False, empty_ranking=False, no_model=False):
    """Build an AutoExperimentResults for testing."""
    model = None if no_model else _make_mock_model()
    ranking = pd.DataFrame() if empty_ranking else _make_ranking()

    all_results = {}
    if model is not None:
        all_results["fe_y~x1+x2"] = {
            "results": model,
            "model_type": "fe",
            "formula": "y ~ x1 + x2",
            "cov_type": "robust",
            "metrics": {
                "rsq": 0.80,
                "rsq_adj": 0.78,
                "bic": 1000.0,
                "aic": 980.0,
                "rmse": 0.25,
                "nobs": 500,
            },
            "classification": "VALID",
            "failed_tests": [],
        }

    test_results = {
        "fe_y~x1+x2": {
            "classification": "VALID",
            "passed": ["Pesaran CD", "RESET", "Modified Wald"],
            "failed": ["Wooldridge"],
        },
        "re_y~x1+x3": {
            "classification": "WARNING",
            "passed": ["Breusch-Pagan"],
            "failed": ["Mundlak", "RESET"],
        },
    }

    variable_selection = {
        "fe": {
            "selected_vars": ["x1", "x2"],
            "rejected_vars": {"x3": "did not improve criterion"},
            "bic_path": [1200.0, 1050.0, 1000.0],
            "n_steps": 2,
        },
        "re": {
            "selected_vars": ["x1", "x3"],
            "rejected_vars": {"x2": "sign constraint violated"},
            "bic_path": [1300.0, 1100.0, 1080.0],
            "n_steps": 2,
        },
    }

    transformations_used = {
        "L1_spread": ("spread", "lag1"),
        "D_selic": ("selic", "diff"),
        "log_income": ("income", "log"),
    }

    return AutoExperimentResults(
        best_model=model,
        best_formula="y ~ x1 + x2",
        best_estimator="fe",
        best_cov_type="robust",
        ranking=ranking,
        all_results=all_results,
        transformations_used=transformations_used,
        variable_selection=variable_selection,
        test_results=test_results,
        n_combinations_tested=150 if datamining else 20,
        datamining_warning=datamining,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestAutoExperimentReportGenerator:
    """Tests for AutoExperimentReportGenerator."""

    def test_generate_returns_html_string(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")

    def test_html_contains_report_title(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "AutoExperiment Report" in html

    def test_ranking_table_present(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "ranking-table" in html
        assert "Fixed Effects" in html or "fe" in html

    def test_classification_colors(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "ae-badge-valid" in html
        assert "ae-badge-warning" in html

    def test_coefficients_table_present(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "coef-table" in html
        assert "x1" in html
        assert "0.5" in html or "0.5000" in html

    def test_test_matrix_present(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "ae-test-matrix" in html or "ae-cell-pass" in html

    def test_bic_chart_rendered(self):
        pytest.importorskip("matplotlib")
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "data:image/png;base64," in html

    def test_datamining_warning_shown(self):
        results = _make_results(datamining=True)
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "Data Mining Warning" in html
        assert "150" in html

    def test_datamining_warning_absent(self):
        results = _make_results(datamining=False)
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        # The CSS class may exist in the stylesheet, but the warning text should not appear
        assert "Data Mining Warning" not in html

    def test_transformations_listed(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "L1_spread" in html
        assert "lag1" in html

    def test_self_contained_no_external_deps(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        # Should not reference external stylesheets or scripts (except Plotly CDN which is excluded)
        assert '<link rel="stylesheet" href="http' not in html

    def test_save_to_file(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.html")
            html = gen.generate(filepath=path)
            assert Path(path).exists()
            content = Path(path).read_text(encoding="utf-8")
            assert content == html

    def test_empty_ranking(self):
        results = _make_results(empty_ranking=True)
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_no_best_model(self):
        results = _make_results(no_model=True)
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert isinstance(html, str)

    def test_sortable_table_js(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "sortTable" in html

    def test_inherits_base_v2(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        # base_v2.html includes the report-container class
        assert "report-container" in html

    def test_works_in_browser(self):
        """Verify the HTML is valid enough to render — check structure."""
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html or "<head " in html
        assert "<body>" in html
        assert "</body>" in html

    def test_variable_selection_details(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "did not improve criterion" in html
        assert "sign constraint violated" in html

    def test_se_justification(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "robust" in html.lower()
        assert "ae-se-justification" in html

    def test_transparency_section(self):
        results = _make_results()
        gen = AutoExperimentReportGenerator(results)
        html = gen.generate()
        assert "Transparency" in html
        assert "Disclaimer" in html
