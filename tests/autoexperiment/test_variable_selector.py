"""Tests for VariableSelector: forward stepwise selection with BIC/AIC."""

from __future__ import annotations

import numpy as np
import pytest

from panelbox.autoexperiment.variable_selector import (
    VariableSelector,
    _get_source_variable,
    _get_transform_type,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grunfeld():
    from panelbox import load_grunfeld

    return load_grunfeld()


@pytest.fixture
def panel_with_transforms(grunfeld):
    """Grunfeld data enriched with some transformed columns."""
    from panelbox.autoexperiment import VariableTransformer

    vt = VariableTransformer(
        data=grunfeld,
        entity_col="firm",
        time_col="year",
        transformations={"lag": [1], "diff": True, "log": True},
    )
    return vt.transform(["value", "capital"])


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_get_source_variable_lag(self):
        assert _get_source_variable("L1_selic") == "selic"
        assert _get_source_variable("L2_spread") == "spread"

    def test_get_source_variable_diff(self):
        assert _get_source_variable("D_ibc_br") == "ibc_br"

    def test_get_source_variable_log(self):
        assert _get_source_variable("log_value") == "value"

    def test_get_source_variable_growth(self):
        assert _get_source_variable("growth_x") == "x"

    def test_get_source_variable_sq(self):
        assert _get_source_variable("sq_capital") == "capital"

    def test_get_source_variable_acum(self):
        assert _get_source_variable("acum3_gdp") == "gdp"
        assert _get_source_variable("acum12_gdp") == "gdp"

    def test_get_source_variable_original(self):
        assert _get_source_variable("value") is None
        assert _get_source_variable("capital") is None

    def test_get_transform_type(self):
        assert _get_transform_type("L1_x") == "lag"
        assert _get_transform_type("D_x") == "diff"
        assert _get_transform_type("log_x") == "log"
        assert _get_transform_type("growth_x") == "growth"
        assert _get_transform_type("sq_x") == "sq"
        assert _get_transform_type("acum6_x") == "acum"
        assert _get_transform_type("x") is None


# ---------------------------------------------------------------------------
# Prefilter tests
# ---------------------------------------------------------------------------


class TestPrefilter:
    def test_prefilter_keeps_correlated_vars(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
            prefilter_corr=0.05,
        )
        surviving = selector.prefilter(grunfeld)
        assert "value" in surviving
        assert "capital" in surviving

    def test_prefilter_eliminates_low_corr(self, grunfeld):
        # Add a random noise column with (almost) no correlation
        rng = np.random.default_rng(42)
        grunfeld = grunfeld.copy()
        grunfeld["noise"] = rng.normal(size=len(grunfeld))

        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital", "noise"],
            entity_col="firm",
            time_col="year",
            prefilter_corr=0.15,
        )
        surviving = selector.prefilter(grunfeld)
        assert "value" in surviving
        assert "capital" in surviving
        # noise should be eliminated (very low correlation)
        assert "noise" not in surviving

    def test_prefilter_missing_column(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "nonexistent"],
            entity_col="firm",
            time_col="year",
        )
        surviving = selector.prefilter(grunfeld)
        assert "nonexistent" not in surviving


# ---------------------------------------------------------------------------
# Forward stepwise tests
# ---------------------------------------------------------------------------


class TestForwardStepwise:
    def test_selects_variables_fe(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
            criterion="bic",
            max_vars=5,
        )
        result = selector.forward_stepwise(grunfeld, model_class="fe")

        assert len(result["selected_vars"]) > 0
        assert all(v in ["value", "capital"] for v in result["selected_vars"])
        assert result["n_steps"] == len(result["selected_vars"])

    def test_selects_variables_pooled(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
            criterion="bic",
            max_vars=5,
        )
        result = selector.forward_stepwise(grunfeld, model_class="pooled")

        assert len(result["selected_vars"]) > 0
        assert result["n_steps"] == len(result["selected_vars"])

    def test_bic_path_monotonically_decreasing(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
            criterion="bic",
        )
        result = selector.forward_stepwise(grunfeld, model_class="fe")

        path = result["bic_path"]
        # After the baseline, each step should improve (decrease) BIC
        for i in range(1, len(path)):
            assert path[i] <= path[i - 1], (
                f"BIC increased at step {i}: {path[i - 1]:.4f} -> {path[i]:.4f}"
            )

    def test_max_vars_respected(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
            max_vars=1,
        )
        result = selector.forward_stepwise(grunfeld, model_class="fe")
        assert len(result["selected_vars"]) <= 1

    def test_rejected_vars_documented(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
            max_vars=1,
        )
        result = selector.forward_stepwise(grunfeld, model_class="fe")

        # With max_vars=1, at least one candidate should be rejected
        assert len(result["rejected_vars"]) > 0
        # Each rejection should have a reason string
        for _var, reason in result["rejected_vars"].items():
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_aic_criterion(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
            criterion="aic",
        )
        result = selector.forward_stepwise(grunfeld, model_class="fe")
        assert len(result["selected_vars"]) > 0

    def test_with_transformed_vars(self, panel_with_transforms):
        data = panel_with_transforms
        candidates = [c for c in data.columns if c not in ["firm", "year", "invest"]]

        selector = VariableSelector(
            depvar="invest",
            candidates=candidates,
            entity_col="firm",
            time_col="year",
            criterion="bic",
            max_vars=4,
        )
        result = selector.forward_stepwise(data, model_class="fe")
        assert len(result["selected_vars"]) > 0


# ---------------------------------------------------------------------------
# Sign constraint tests
# ---------------------------------------------------------------------------


class TestSignConstraints:
    def test_sign_constraint_positive_satisfied(self):
        selector = VariableSelector(
            depvar="y",
            candidates=["x"],
            entity_col="e",
            time_col="t",
            sign_constraints={"x": "+"},
        )
        assert selector._check_sign_constraint("x", 0.5, {"x": "+"}) is True

    def test_sign_constraint_positive_violated(self):
        selector = VariableSelector(
            depvar="y",
            candidates=["x"],
            entity_col="e",
            time_col="t",
            sign_constraints={"x": "+"},
        )
        assert selector._check_sign_constraint("x", -0.3, {"x": "+"}) is False

    def test_sign_constraint_negative_satisfied(self):
        selector = VariableSelector(
            depvar="y",
            candidates=["x"],
            entity_col="e",
            time_col="t",
        )
        assert selector._check_sign_constraint("x", -2.0, {"x": "-"}) is True

    def test_sign_constraint_negative_violated(self):
        selector = VariableSelector(
            depvar="y",
            candidates=["x"],
            entity_col="e",
            time_col="t",
        )
        assert selector._check_sign_constraint("x", 1.0, {"x": "-"}) is False

    def test_sign_constraint_inherited_from_source(self):
        """L1_x should inherit constraint from x."""
        selector = VariableSelector(
            depvar="y",
            candidates=["L1_x"],
            entity_col="e",
            time_col="t",
            sign_constraints={"x": "+"},
        )
        # L1_x with positive coef should pass (inherits '+' from x)
        assert selector._check_sign_constraint("L1_x", 0.5, {"x": "+"}) is True
        # L1_x with negative coef should fail
        assert selector._check_sign_constraint("L1_x", -0.5, {"x": "+"}) is False

    def test_sign_constraint_diff_not_enforced(self):
        """D_x (diff) can invert sign, so constraint is not enforced."""
        selector = VariableSelector(
            depvar="y",
            candidates=["D_x"],
            entity_col="e",
            time_col="t",
            sign_constraints={"x": "+"},
        )
        # Should pass even with negative coef
        assert selector._check_sign_constraint("D_x", -0.5, {"x": "+"}) is True

    def test_sign_constraint_growth_not_enforced(self):
        """growth_x can invert sign, so constraint is not enforced."""
        selector = VariableSelector(
            depvar="y",
            candidates=["growth_x"],
            entity_col="e",
            time_col="t",
            sign_constraints={"x": "+"},
        )
        assert selector._check_sign_constraint("growth_x", -0.5, {"x": "+"}) is True

    def test_sign_constraint_no_constraint(self):
        """Variables without constraints always pass."""
        selector = VariableSelector(
            depvar="y",
            candidates=["x"],
            entity_col="e",
            time_col="t",
        )
        assert selector._check_sign_constraint("x", -99.0, {}) is True

    def test_sign_constraint_rejects_in_stepwise(self, grunfeld):
        """Forward stepwise should reject a variable that violates its sign constraint."""
        # In Grunfeld, both value and capital have positive coefficients.
        # Setting constraint to '-' should cause rejection.
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
            sign_constraints={"value": "-", "capital": "-"},
        )
        result = selector.forward_stepwise(grunfeld, model_class="fe")

        # Both should be rejected for sign violation
        assert len(result["selected_vars"]) == 0
        assert len(result["rejected_vars"]) > 0
        for _var, reason in result["rejected_vars"].items():
            assert "sign constraint" in reason or "not selected" in reason


# ---------------------------------------------------------------------------
# Backward elimination tests
# ---------------------------------------------------------------------------


class TestBackwardElimination:
    def test_backward_basic(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=["value", "capital"],
            entity_col="firm",
            time_col="year",
        )
        result = selector.backward_elimination(
            grunfeld, model_class="fe", selected_vars=["value", "capital"]
        )
        assert "selected_vars" in result
        assert "removed_vars" in result
        assert "bic_path" in result
        # Both value and capital are highly significant in Grunfeld — none removed
        assert len(result["selected_vars"]) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_criterion(self):
        with pytest.raises(ValueError, match="criterion"):
            VariableSelector(
                depvar="y",
                candidates=["x"],
                entity_col="e",
                time_col="t",
                criterion="r2",
            )

    def test_empty_candidates(self, grunfeld):
        selector = VariableSelector(
            depvar="invest",
            candidates=[],
            entity_col="firm",
            time_col="year",
        )
        result = selector.forward_stepwise(grunfeld, model_class="fe")
        assert result["selected_vars"] == []
        assert result["n_steps"] == 0

    def test_no_candidates_after_filter(self, grunfeld):
        """When prefilter_corr is extremely high, all candidates are eliminated."""
        rng = np.random.default_rng(42)
        grunfeld = grunfeld.copy()
        grunfeld["noise"] = rng.normal(size=len(grunfeld))

        selector = VariableSelector(
            depvar="invest",
            candidates=["noise"],
            entity_col="firm",
            time_col="year",
            prefilter_corr=0.99,  # Very high threshold
        )
        surviving = selector.prefilter(grunfeld)
        assert surviving == []

        # Forward stepwise with no surviving candidates
        result = selector.forward_stepwise(grunfeld, model_class="fe")
        assert result["selected_vars"] == []
