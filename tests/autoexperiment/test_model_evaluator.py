"""Tests for ModelEvaluator and SpecificationFilter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from panelbox.autoexperiment.model_evaluator import ModelEvaluator
from panelbox.autoexperiment.specification_filter import SpecificationFilter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grunfeld():
    from panelbox import load_grunfeld

    return load_grunfeld()


def _make_test_result(reject_null: bool, pvalue: float = 0.01):
    return SimpleNamespace(reject_null=reject_null, pvalue=pvalue)


def _make_validation(
    reset_rejects: bool = False,
    mundlak_rejects: bool = False,
    cd_rejects: bool = False,
    het_rejects: bool = False,
    serial_rejects: bool = False,
):
    """Create a minimal ValidationReport-like object."""
    spec = {"RESET": _make_test_result(reset_rejects)}
    if mundlak_rejects is not None:
        spec["Mundlak"] = _make_test_result(mundlak_rejects)

    return SimpleNamespace(
        specification_tests=spec,
        serial_tests={"Wooldridge": _make_test_result(serial_rejects)},
        het_tests={
            "Modified Wald": _make_test_result(het_rejects),
            "Breusch-Pagan": _make_test_result(False),
        },
        cd_tests={"Pesaran CD": _make_test_result(cd_rejects)},
    )


# ===================================================================
# ModelEvaluator Tests
# ===================================================================


class TestModelEvaluator:
    def test_evaluate_returns_all_fields(self, grunfeld):
        """evaluate() must return dict with all expected keys."""
        evaluator = ModelEvaluator(grunfeld, "firm", "year")
        result = evaluator.evaluate(
            formula="invest ~ value + capital",
            model_type="fe",
        )

        expected_keys = {
            "results",
            "model_type",
            "formula",
            "cov_type",
            "metrics",
            "validation",
            "classification",
            "failed_tests",
        }
        assert set(result.keys()) == expected_keys
        assert result["model_type"] == "fe"
        assert result["formula"] == "invest ~ value + capital"
        assert isinstance(result["metrics"], dict)
        assert "bic" in result["metrics"]
        assert "aic" in result["metrics"]
        assert "rsq" in result["metrics"]
        assert "rmse" in result["metrics"]
        assert "nobs" in result["metrics"]
        assert result["classification"] in ("VALID", "WARNING", "INVALID")

    def test_evaluate_pooled(self, grunfeld):
        evaluator = ModelEvaluator(grunfeld, "firm", "year")
        result = evaluator.evaluate("invest ~ value + capital", "pooled_ols")
        assert result["model_type"] == "pooled_ols"
        assert result["results"] is not None

    def test_evaluate_re(self, grunfeld):
        evaluator = ModelEvaluator(grunfeld, "firm", "year")
        result = evaluator.evaluate("invest ~ value + capital", "re")
        assert result["model_type"] == "re"

    def test_evaluate_fd(self, grunfeld):
        evaluator = ModelEvaluator(grunfeld, "firm", "year")
        result = evaluator.evaluate("invest ~ value + capital", "fd")
        assert result["model_type"] == "fd"

    def test_auto_cov_type_driscoll_kraay(self):
        """Pesaran CD rejecting -> driscoll_kraay."""
        report = _make_validation(cd_rejects=True)
        cov = ModelEvaluator._auto_select_cov_type(report)
        assert cov == "driscoll_kraay"

    def test_auto_cov_type_robust(self):
        """Heteroskedasticity rejecting -> robust."""
        report = _make_validation(het_rejects=True)
        cov = ModelEvaluator._auto_select_cov_type(report)
        assert cov == "robust"

    def test_auto_cov_type_clustered(self):
        """Serial correlation rejecting -> clustered."""
        report = _make_validation(serial_rejects=True)
        cov = ModelEvaluator._auto_select_cov_type(report)
        assert cov == "clustered"

    def test_auto_cov_type_nonrobust(self):
        """Nothing rejecting -> nonrobust."""
        report = _make_validation()
        cov = ModelEvaluator._auto_select_cov_type(report)
        assert cov == "nonrobust"

    def test_unknown_model_type_raises(self, grunfeld):
        evaluator = ModelEvaluator(grunfeld, "firm", "year")
        with pytest.raises(ValueError, match="Unknown model_type"):
            evaluator.evaluate("invest ~ value", "xyz_model")


# ===================================================================
# SpecificationFilter Tests
# ===================================================================


class TestSpecificationFilter:
    def test_classification_valid(self):
        """All tests pass -> VALID."""
        sf = SpecificationFilter()
        report = _make_validation()
        result = sf.classify("fe", report)
        assert result["classification"] == "VALID"
        assert len(result["failed_tests"]) == 0

    def test_classification_invalid_reset(self):
        """RESET failing (critical test) -> INVALID."""
        sf = SpecificationFilter()
        report = _make_validation(reset_rejects=True)
        result = sf.classify("fe", report)
        assert result["classification"] == "INVALID"
        assert "RESET" in result["critical_failures"]

    def test_classification_invalid_mundlak(self):
        """Mundlak failing (critical test) -> INVALID for RE."""
        sf = SpecificationFilter()
        report = _make_validation(mundlak_rejects=True)
        result = sf.classify("re", report)
        assert result["classification"] == "INVALID"
        assert "Mundlak" in result["critical_failures"]

    def test_classification_warning(self):
        """Non-critical test failing -> WARNING."""
        sf = SpecificationFilter()
        # Pesaran CD is not critical for FE, but is a default test
        report = _make_validation(cd_rejects=True)
        result = sf.classify("fe", report)
        assert result["classification"] == "WARNING"
        assert "Pesaran CD" in result["failed_tests"]
        assert len(result["critical_failures"]) == 0

    def test_hausman_decision(self, grunfeld):
        """Hausman test returns 'fe' or 're'."""
        sf = SpecificationFilter()
        decision = sf.hausman_decision(
            data=grunfeld,
            formula="invest ~ value + capital",
            entity_col="firm",
            time_col="year",
        )
        assert decision in ("fe", "re")

    def test_default_tests_per_model(self):
        sf = SpecificationFilter()
        assert "RESET" in sf.DEFAULT_TESTS["pooled_ols"]
        assert "Pesaran CD" in sf.DEFAULT_TESTS["fe"]
        assert "Mundlak" in sf.DEFAULT_TESTS["re"]

    def test_custom_required_tests(self):
        """Override required tests."""
        sf = SpecificationFilter(required_tests=["RESET"])
        report = _make_validation(cd_rejects=True)
        # Only RESET is checked, which passes
        result = sf.classify("fe", report)
        assert result["classification"] == "VALID"

    def test_auto_cov_type_recommendation(self):
        sf = SpecificationFilter()
        report = _make_validation(cd_rejects=True)
        result = sf.classify("fe", report)
        assert result["auto_cov_type"] == "driscoll_kraay"
