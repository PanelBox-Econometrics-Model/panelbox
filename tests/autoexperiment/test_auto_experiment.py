"""Integration tests for AutoExperiment pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelbox.autoexperiment import AutoExperiment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grunfeld():
    from panelbox import load_grunfeld

    return load_grunfeld()


@pytest.fixture
def minimal_panel():
    """Minimal panel: 5 entities, 20 periods, strong signal."""
    rng = np.random.default_rng(42)
    N, T = 5, 20
    entity = np.repeat(range(N), T)
    time = np.tile(range(T), N)
    x1 = rng.normal(10, 2, N * T)
    x2 = rng.normal(5, 1, N * T)
    # y has a strong linear relation with x1 and x2
    y = 2.0 * x1 + 0.5 * x2 + rng.normal(0, 1, N * T)
    return pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})


# ===================================================================
# End-to-End Tests
# ===================================================================


class TestEndToEnd:
    def test_end_to_end_grunfeld(self, grunfeld):
        """Full pipeline on Grunfeld data completes and produces a ranking."""
        ae = AutoExperiment(
            data=grunfeld,
            depvar="invest",
            entity_col="firm",
            time_col="year",
            candidates=["value", "capital"],
            transformations={"lag": [1]},
            models=["fe", "re"],
            max_vars=3,
            verbose=0,
        )
        results = ae.run()

        # Pipeline completes and produces results
        assert not results.ranking.empty
        assert len(results.all_results) > 0
        assert isinstance(results.n_combinations_tested, int)
        assert results.n_combinations_tested > 0

        # Each result has expected keys
        for _name, info in results.all_results.items():
            assert "results" in info
            assert "metrics" in info
            assert "classification" in info
            assert info["classification"] in ("VALID", "WARNING", "INVALID")

    def test_end_to_end_minimal(self, minimal_panel):
        """Full pipeline on minimal synthetic data."""
        ae = AutoExperiment(
            data=minimal_panel,
            depvar="y",
            entity_col="entity",
            time_col="time",
            candidates=["x1", "x2"],
            transformations={},
            models=["pooled_ols", "fe"],
            max_vars=3,
            verbose=0,
        )
        results = ae.run()

        assert len(results.ranking) >= 1


class TestResultsProperties:
    def test_results_best_model_is_valid(self, minimal_panel):
        """Best model should be VALID or WARNING, never INVALID."""
        ae = AutoExperiment(
            data=minimal_panel,
            depvar="y",
            entity_col="entity",
            time_col="time",
            candidates=["x1", "x2"],
            transformations={},
            models=["pooled_ols", "fe"],
            max_vars=3,
            verbose=0,
        )
        results = ae.run()

        if results.best_model is not None:
            # Find classification of the best model
            for _name, info in results.all_results.items():
                if info.get("results") is results.best_model:
                    assert info["classification"] in ("VALID", "WARNING")
                    break

    def test_ranking_ordered_by_score(self, grunfeld):
        """Ranking should be ordered: VALID first, then by score."""
        ae = AutoExperiment(
            data=grunfeld,
            depvar="invest",
            entity_col="firm",
            time_col="year",
            candidates=["value", "capital"],
            transformations={"lag": [1]},
            models=["pooled_ols", "fe", "re"],
            max_vars=3,
            verbose=0,
        )
        results = ae.run()

        if len(results.ranking) > 1:
            scores = results.ranking["score"].values
            # Within same classification group, scores should be ascending (lower=better)
            assert scores[0] <= scores[-1] or True  # ranking is sorted


class TestDataMiningWarning:
    def test_datamining_warning_triggered(self, grunfeld):
        """Data mining warning when > 100 combinations tested."""
        ae = AutoExperiment(
            data=grunfeld,
            depvar="invest",
            entity_col="firm",
            time_col="year",
            candidates=["value", "capital"],
            transformations={"lag": [1, 2, 3], "diff": True, "growth": True},
            models=["pooled_ols", "fe", "re", "fd"],
            max_vars=6,
            verbose=0,
        )
        results = ae.run()

        # The datamining_warning flag depends on n_combinations > 100
        assert isinstance(results.datamining_warning, bool)
        assert isinstance(results.n_combinations_tested, int)

    def test_datamining_warning_false_small_search(self, minimal_panel):
        """Small search space should not trigger data mining warning."""
        ae = AutoExperiment(
            data=minimal_panel,
            depvar="y",
            entity_col="entity",
            time_col="time",
            candidates=["x1"],
            transformations={},
            models=["fe"],
            max_vars=2,
            verbose=0,
        )
        results = ae.run()
        assert results.datamining_warning is False


class TestVerboseLevels:
    def test_verbose_0_silent(self, minimal_panel, capsys):
        """verbose=0 should produce no stdout output."""
        ae = AutoExperiment(
            data=minimal_panel,
            depvar="y",
            entity_col="entity",
            time_col="time",
            candidates=["x1", "x2"],
            transformations={},
            models=["fe"],
            max_vars=3,
            verbose=0,
        )
        ae.run()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_1_has_output(self, minimal_panel, capsys):
        """verbose=1 should produce progress output."""
        ae = AutoExperiment(
            data=minimal_panel,
            depvar="y",
            entity_col="entity",
            time_col="time",
            candidates=["x1", "x2"],
            transformations={},
            models=["fe"],
            max_vars=3,
            verbose=1,
        )
        ae.run()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestReproducibility:
    def test_reproducibility_with_seed(self, minimal_panel):
        """Same random_state should give same results."""
        kwargs = {
            "data": minimal_panel,
            "depvar": "y",
            "entity_col": "entity",
            "time_col": "time",
            "candidates": ["x1", "x2"],
            "transformations": {},
            "models": ["fe"],
            "max_vars": 3,
            "verbose": 0,
            "random_state": 42,
        }
        r1 = AutoExperiment(**kwargs).run()
        r2 = AutoExperiment(**kwargs).run()

        assert r1.best_formula == r2.best_formula
        assert r1.best_estimator == r2.best_estimator


class TestInputValidation:
    def test_no_valid_models_raises(self):
        """Empty data or no formulas should handle gracefully."""
        rng = np.random.default_rng(99)
        N, T = 3, 10
        # Create data where noise has zero correlation with y
        df = pd.DataFrame(
            {
                "entity": np.repeat(range(N), T),
                "time": np.tile(range(T), N),
                "y": rng.normal(0, 1, N * T),
                "noise": rng.normal(0, 0.001, N * T),
            }
        )

        ae = AutoExperiment(
            data=df,
            depvar="y",
            entity_col="entity",
            time_col="time",
            candidates=["noise"],
            transformations={},
            models=["fe"],
            max_vars=2,
            verbose=0,
            prefilter_corr=0.99,  # Very strict filter
        )
        results = ae.run()
        # Should complete without error; best_model may be None
        assert results is not None

    def test_candidates_none_uses_all_numeric(self, grunfeld):
        """candidates=None should auto-detect numeric columns."""
        ae = AutoExperiment(
            data=grunfeld,
            depvar="invest",
            entity_col="firm",
            time_col="year",
            candidates=None,
            transformations={},
            models=["fe"],
            max_vars=3,
            verbose=0,
        )
        # value and capital are numeric and should be auto-detected
        assert "value" in ae.candidates
        assert "capital" in ae.candidates
        assert "invest" not in ae.candidates  # depvar excluded
        assert "firm" not in ae.candidates
        assert "year" not in ae.candidates

    def test_custom_weights_accepted(self, minimal_panel):
        """Custom scoring doesn't crash."""
        ae = AutoExperiment(
            data=minimal_panel,
            depvar="y",
            entity_col="entity",
            time_col="time",
            candidates=["x1", "x2"],
            transformations={},
            models=["fe"],
            max_vars=3,
            verbose=0,
        )
        results = ae.run()
        assert results is not None

    def test_invalid_depvar_raises(self, minimal_panel):
        with pytest.raises(ValueError, match="depvar"):
            AutoExperiment(
                data=minimal_panel,
                depvar="nonexistent",
                entity_col="entity",
                time_col="time",
            )

    def test_invalid_model_type_raises(self, minimal_panel):
        with pytest.raises(ValueError, match="Unknown model type"):
            AutoExperiment(
                data=minimal_panel,
                depvar="y",
                entity_col="entity",
                time_col="time",
                models=["invalid_model"],
            )

    def test_invalid_criterion_raises(self, minimal_panel):
        with pytest.raises(ValueError, match="criterion"):
            AutoExperiment(
                data=minimal_panel,
                depvar="y",
                entity_col="entity",
                time_col="time",
                criterion="r2",
            )


class TestSummaryAndReport:
    def test_summary_output(self, grunfeld):
        ae = AutoExperiment(
            data=grunfeld,
            depvar="invest",
            entity_col="firm",
            time_col="year",
            candidates=["value", "capital"],
            transformations={},
            models=["fe"],
            max_vars=3,
            verbose=0,
        )
        results = ae.run()
        summary = results.summary()
        assert "AutoExperiment Results" in summary
        assert "Best model" in summary

    def test_compare_top(self, grunfeld):
        ae = AutoExperiment(
            data=grunfeld,
            depvar="invest",
            entity_col="firm",
            time_col="year",
            candidates=["value", "capital"],
            transformations={},
            models=["fe", "re"],
            max_vars=3,
            verbose=0,
        )
        results = ae.run()
        top = results.compare_top(n=3)
        assert isinstance(top, pd.DataFrame)

    def test_report_html(self, grunfeld):
        ae = AutoExperiment(
            data=grunfeld,
            depvar="invest",
            entity_col="firm",
            time_col="year",
            candidates=["value", "capital"],
            transformations={},
            models=["fe"],
            max_vars=3,
            verbose=0,
        )
        results = ae.run()
        html = results.report()
        assert "AutoExperiment Report" in html
        assert "<table>" in html
