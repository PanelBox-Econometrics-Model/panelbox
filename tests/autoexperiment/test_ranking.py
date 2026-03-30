"""Tests for ModelRanker and AutoCovTypeSelector."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from panelbox.autoexperiment.ranking import AutoCovTypeSelector, ModelRanker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_result(reject_null: bool, pvalue: float = 0.01):
    """Create a minimal ValidationTestResult-like object."""
    return SimpleNamespace(reject_null=reject_null, pvalue=pvalue)


def _make_validation(
    cd_rejects: bool = False,
    serial_rejects: bool = False,
    het_rejects: bool = False,
):
    """Create a minimal ValidationReport-like object."""
    cd_tests = {"Pesaran CD": _make_test_result(cd_rejects)}
    serial_tests = {
        "Wooldridge": _make_test_result(serial_rejects),
        "Breusch-Godfrey": _make_test_result(False),
    }
    het_tests = {
        "Modified Wald": _make_test_result(het_rejects),
        "Breusch-Pagan": _make_test_result(False),
    }
    return SimpleNamespace(
        specification_tests={"RESET": _make_test_result(False)},
        serial_tests=serial_tests,
        het_tests=het_tests,
        cd_tests=cd_tests,
    )


def _make_evaluations():
    """Build a list of model evaluation dicts for ranking tests."""
    return [
        {
            "model_type": "fe",
            "formula": "y ~ x1 + x2",
            "metrics": {"bic": 100.0, "aic": 90.0, "rsq_adj": 0.85},
            "classification": "VALID",
            "cov_type": "robust",
            "failed_tests": [],
            "validation": _make_validation(),
            "sign_violations": 0,
            "sign_total": 2,
        },
        {
            "model_type": "re",
            "formula": "y ~ x1 + x2 + x3",
            "metrics": {"bic": 110.0, "aic": 95.0, "rsq_adj": 0.82},
            "classification": "WARNING",
            "cov_type": "clustered",
            "failed_tests": ["Mundlak"],
            "validation": _make_validation(),
            "sign_violations": 1,
            "sign_total": 2,
        },
        {
            "model_type": "pooled_ols",
            "formula": "y ~ x1",
            "metrics": {"bic": 150.0, "aic": 140.0, "rsq_adj": 0.60},
            "classification": "INVALID",
            "cov_type": "nonrobust",
            "failed_tests": ["RESET", "Breusch-Pagan"],
            "validation": _make_validation(),
            "sign_violations": 0,
            "sign_total": 0,
        },
    ]


# ===================================================================
# ModelRanker Tests
# ===================================================================


class TestModelRanker:
    def test_composite_score_calculated(self):
        ranker = ModelRanker()
        evals = _make_evaluations()
        result = ranker.rank(evals)

        assert "score" in result.columns
        assert len(result) == 3
        assert result["score"].notna().all()

    def test_normalization_zero_one(self):
        ranker = ModelRanker()
        values = np.array([100.0, 200.0, 300.0])
        normalised = ranker._normalize_criterion(values, prefer_lower=True)

        assert normalised.min() >= 0.0
        assert normalised.max() <= 1.0
        # Lowest BIC (100) should have highest score (1.0)
        assert normalised[0] == pytest.approx(1.0)
        assert normalised[2] == pytest.approx(0.0)

    def test_normalization_identical_values(self):
        ranker = ModelRanker()
        values = np.array([50.0, 50.0, 50.0])
        normalised = ranker._normalize_criterion(values)

        # All identical → all get 1.0
        np.testing.assert_array_equal(normalised, [1.0, 1.0, 1.0])

    def test_normalization_with_inf(self):
        ranker = ModelRanker()
        values = np.array([100.0, np.inf, 200.0])
        normalised = ranker._normalize_criterion(values, prefer_lower=True)

        # Inf gets 0 score
        assert normalised[1] == 0.0
        assert normalised[0] == pytest.approx(1.0)

    def test_normalization_all_inf(self):
        ranker = ModelRanker()
        values = np.array([np.inf, np.inf])
        normalised = ranker._normalize_criterion(values)
        np.testing.assert_array_equal(normalised, [0.0, 0.0])

    def test_invalid_models_last(self):
        ranker = ModelRanker()
        evals = _make_evaluations()
        result = ranker.rank(evals)

        # INVALID must be last
        assert result.iloc[-1]["classification"] == "INVALID"
        # VALID should be first
        assert result.iloc[0]["classification"] == "VALID"

    def test_custom_weights(self):
        custom = {"criterion": 1.0, "tests": 0.0, "signs": 0.0, "parsimony": 0.0}
        ranker = ModelRanker(weights=custom)
        evals = _make_evaluations()
        result = ranker.rank(evals)

        assert len(result) == 3
        assert "score" in result.columns

    def test_empty_evaluations(self):
        ranker = ModelRanker()
        result = ranker.rank([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_rank_column_sequential(self):
        ranker = ModelRanker()
        evals = _make_evaluations()
        result = ranker.rank(evals)

        assert list(result["rank"]) == [1, 2, 3]

    def test_output_columns(self):
        ranker = ModelRanker()
        evals = _make_evaluations()
        result = ranker.rank(evals)

        expected_cols = {
            "rank",
            "model_type",
            "formula",
            "score",
            "bic",
            "aic",
            "rsq_adj",
            "classification",
            "n_vars",
            "cov_type",
            "tests_passed",
            "signs_ok",
        }
        assert set(result.columns) == expected_cols

    def test_signs_ok_na_when_no_constraints(self):
        ranker = ModelRanker()
        evals = [
            {
                "model_type": "fe",
                "formula": "y ~ x1",
                "metrics": {"bic": 100.0, "aic": 90.0, "rsq_adj": 0.8},
                "classification": "VALID",
                "cov_type": "robust",
                "failed_tests": [],
                "validation": _make_validation(),
                "sign_violations": 0,
                "sign_total": 0,
            }
        ]
        result = ranker.rank(evals)
        assert result.iloc[0]["signs_ok"] == "n/a"

    def test_score_between_0_and_1(self):
        """All composite scores should be in [0, 1]."""
        ranker = ModelRanker()
        evals = _make_evaluations()
        result = ranker.rank(evals)

        for score in result["score"]:
            assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"


# ===================================================================
# AutoCovTypeSelector Tests
# ===================================================================


class TestAutoCovTypeSelector:
    def test_driscoll_kraay_when_pesaran_rejects(self):
        selector = AutoCovTypeSelector()
        report = _make_validation(cd_rejects=True)
        assert selector.select(report) == "driscoll_kraay"

    def test_newey_west_when_serial_rejects(self):
        selector = AutoCovTypeSelector()
        report = _make_validation(serial_rejects=True)
        assert selector.select(report) == "newey_west"

    def test_robust_when_het_rejects(self):
        selector = AutoCovTypeSelector()
        report = _make_validation(het_rejects=True)
        assert selector.select(report) == "robust"

    def test_clustered_when_nothing_rejects_small_n(self):
        selector = AutoCovTypeSelector()
        report = _make_validation()
        assert selector.select(report, n_entities=10) == "clustered"

    def test_nonrobust_when_nothing_rejects_large_n(self):
        selector = AutoCovTypeSelector()
        report = _make_validation()
        assert selector.select(report, n_entities=50) == "nonrobust"

    def test_priority_cd_over_het(self):
        """CD takes priority even when het also rejects."""
        selector = AutoCovTypeSelector()
        report = _make_validation(cd_rejects=True, het_rejects=True)
        assert selector.select(report) == "driscoll_kraay"

    def test_priority_serial_over_het(self):
        selector = AutoCovTypeSelector()
        report = _make_validation(serial_rejects=True, het_rejects=True)
        assert selector.select(report) == "newey_west"

    def test_explain_pesaran_cd(self):
        selector = AutoCovTypeSelector()
        report = _make_validation(cd_rejects=True)
        explanation = selector.explain(report)
        assert "Pesaran CD" in explanation
        assert "Driscoll-Kraay" in explanation

    def test_explain_het(self):
        selector = AutoCovTypeSelector()
        report = _make_validation(het_rejects=True)
        explanation = selector.explain(report)
        assert "Modified Wald" in explanation
        assert "robust" in explanation.lower()

    def test_explain_nothing_rejects_large_n(self):
        selector = AutoCovTypeSelector()
        report = _make_validation()
        explanation = selector.explain(report, n_entities=50)
        assert "nonrobust" in explanation.lower() or "conventional" in explanation.lower()

    def test_explain_nothing_rejects_small_n(self):
        selector = AutoCovTypeSelector()
        report = _make_validation()
        explanation = selector.explain(report, n_entities=10)
        assert "clustered" in explanation.lower()

    def test_explain_serial_correlation(self):
        selector = AutoCovTypeSelector()
        report = _make_validation(serial_rejects=True)
        explanation = selector.explain(report)
        assert "Wooldridge" in explanation
        assert "Newey-West" in explanation
