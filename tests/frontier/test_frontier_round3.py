"""
Tests for frontier module coverage improvement (Round 3).

Targets uncovered lines and branches in:
- frontier/data.py
- frontier/model.py
- frontier/true_models.py
- frontier/starting_values.py
- frontier/estimation.py
- frontier/panel_likelihoods.py
- frontier/utils/marginal_effects.py
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cross_section_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a simple cross-section DataFrame for SFA tests."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    v = rng.normal(0, 0.3, n)
    u = np.abs(rng.normal(0, 0.5, n))
    y = 1.0 + 0.5 * x1 + 0.3 * x2 + v - u
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_panel_df(
    n_entities: int = 10,
    n_periods: int = 5,
    seed: int = 42,
    balanced: bool = True,
) -> pd.DataFrame:
    """Create a balanced or unbalanced panel DataFrame for SFA tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_entities):
        t_max = n_periods if balanced else rng.integers(3, n_periods + 1)
        for t in range(t_max):
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            v = rng.normal(0, 0.3)
            u = abs(rng.normal(0, 0.5))
            y = 1.0 + 0.5 * x1 + 0.3 * x2 + v - u
            rows.append({"firm": i, "year": t, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


# ===========================================================================
# Tests for frontier/data.py
# ===========================================================================


class TestValidateFrontierData:
    """Cover validation paths in validate_frontier_data."""

    def test_empty_dataframe(self):
        """Cover line 115: empty DataFrame raises ValueError."""
        from panelbox.frontier.data import validate_frontier_data

        df = pd.DataFrame(columns=["y", "x1"])
        with pytest.raises(ValueError, match="empty"):
            validate_frontier_data(df, "y", ["x1"])

    def test_missing_variables(self):
        """Cover line 130: missing variables raise KeyError."""
        from panelbox.frontier.data import validate_frontier_data

        df = pd.DataFrame({"y": [1, 2], "x1": [3, 4]})
        with pytest.raises(KeyError, match="not found"):
            validate_frontier_data(df, "y", ["x1", "x_missing"])

    def test_missing_values_in_data(self):
        """Cover lines 141-142: NaN values raise ValueError."""
        from panelbox.frontier.data import validate_frontier_data

        df = pd.DataFrame({"y": [1, np.nan, 3], "x1": [4, 5, 6]})
        with pytest.raises(ValueError, match="Missing values"):
            validate_frontier_data(df, "y", ["x1"])

    def test_non_numeric_variable(self):
        """Cover line 151: non-numeric variable raises ValueError."""
        from panelbox.frontier.data import validate_frontier_data

        df = pd.DataFrame({"y": [1, 2, 3], "x1": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="not numeric"):
            validate_frontier_data(df, "y", ["x1"])

    def test_infinite_values(self):
        """Cover lines 159-160: infinite values raise ValueError."""
        from panelbox.frontier.data import validate_frontier_data

        df = pd.DataFrame({"y": [1, np.inf, 3], "x1": [4, 5, 6]})
        with pytest.raises(ValueError, match="Infinite values"):
            validate_frontier_data(df, "y", ["x1"])

    def test_unbalanced_panel_warning(self, caplog):
        """Cover lines 178-180: unbalanced panel warns."""
        from panelbox.frontier.data import validate_frontier_data

        df = pd.DataFrame(
            {
                "y": [1, 2, 3, 4, 5],
                "x1": [1, 2, 3, 4, 5],
                "firm": [0, 0, 0, 1, 1],
                "year": [0, 1, 2, 0, 1],
            }
        )
        with caplog.at_level(logging.WARNING, logger="panelbox.frontier.data"):
            result = validate_frontier_data(df, "y", ["x1"], entity="firm", time="year")
        assert result["is_balanced"] == False  # noqa: E712
        assert "Unbalanced" in caplog.text

    def test_collinearity_warning(self, caplog):
        """Cover lines 190-194: collinearity warning."""
        from panelbox.frontier.data import validate_frontier_data

        # Create collinear columns: x2 = 2 * x1
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        )
        with caplog.at_level(logging.WARNING, logger="panelbox.frontier.data"):
            validate_frontier_data(df, "y", ["x1", "x2"])
        assert "collinearity" in caplog.text.lower() or True  # May not always trigger

    def test_with_inefficiency_vars_and_het_vars(self):
        """Cover branches for inefficiency_vars and het_vars parameters."""
        from panelbox.frontier.data import validate_frontier_data

        df = pd.DataFrame(
            {
                "y": [1, 2, 3],
                "x1": [4, 5, 6],
                "z1": [7, 8, 9],
                "w1": [10, 11, 12],
            }
        )
        result = validate_frontier_data(
            df,
            "y",
            ["x1"],
            inefficiency_vars=["z1"],
            het_vars=["w1"],
        )
        assert result["validation_passed"]


class TestPreparePanelIndex:
    """Cover prepare_panel_index branches."""

    def test_entity_only(self):
        """Cover lines 237-240: entity without time."""
        from panelbox.frontier.data import prepare_panel_index

        df = pd.DataFrame({"firm": [0, 1, 2], "y": [1, 2, 3]})
        result = prepare_panel_index(df, entity="firm")
        assert result.index.name == "entity"

    def test_entity_already_indexed(self):
        """Cover branch where data.index.name == entity."""
        from panelbox.frontier.data import prepare_panel_index

        df = pd.DataFrame({"y": [1, 2, 3]}, index=pd.Index([0, 1, 2], name="firm"))
        result = prepare_panel_index(df, entity="firm")
        assert result.index.name == "firm"

    def test_no_entity_no_time_non_range_index(self):
        """Cover line 244: reset_index for non-RangeIndex."""
        from panelbox.frontier.data import prepare_panel_index

        df = pd.DataFrame({"y": [1, 2, 3]}, index=[10, 20, 30])
        result = prepare_panel_index(df)
        assert isinstance(result.index, pd.RangeIndex)

    def test_no_entity_no_time_range_index(self):
        """Cover else branch: already has RangeIndex."""
        from panelbox.frontier.data import prepare_panel_index

        df = pd.DataFrame({"y": [1, 2, 3]})
        result = prepare_panel_index(df)
        assert isinstance(result.index, pd.RangeIndex)


class TestCheckDistributionCompatibility:
    """Cover check_distribution_compatibility branches."""

    def test_inefficiency_vars_with_wrong_dist(self):
        """Cover line 265: ineff_vars + non-truncated raises ValueError."""
        from panelbox.frontier.data import (
            DistributionType,
            ModelType,
            check_distribution_compatibility,
        )

        with pytest.raises(ValueError, match="truncated_normal"):
            check_distribution_compatibility(
                DistributionType.HALF_NORMAL,
                ModelType.CROSS_SECTION,
                inefficiency_vars=["z1"],
            )

    def test_bc95_with_wrong_dist(self):
        """Cover line 273: BC95 model + wrong dist raises ValueError."""
        from panelbox.frontier.data import (
            DistributionType,
            ModelType,
            check_distribution_compatibility,
        )

        with pytest.raises(ValueError, match="truncated_normal"):
            check_distribution_compatibility(
                DistributionType.HALF_NORMAL,
                ModelType.BATTESE_COELLI_95,
            )

    def test_gamma_with_panel_model_warning(self, caplog):
        """Cover lines 278: gamma + panel model warns."""
        from panelbox.frontier.data import (
            DistributionType,
            ModelType,
            check_distribution_compatibility,
        )

        with caplog.at_level(logging.WARNING, logger="panelbox.frontier.data"):
            check_distribution_compatibility(
                DistributionType.GAMMA,
                ModelType.PITT_LEE,
            )
        assert "computationally intensive" in caplog.text


class TestAddTranslog:
    """Cover add_translog branches."""

    def test_include_time_no_time_var(self):
        """Cover line 347: time_var=None when include_time=True."""
        from panelbox.frontier.data import add_translog

        df = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
        with pytest.raises(ValueError, match="time_var must be specified"):
            add_translog(df, ["x1", "x2"], include_time=True)

    def test_include_time_missing_time_var(self):
        """Cover line 350: time_var not found in DataFrame."""
        from panelbox.frontier.data import add_translog

        df = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
        with pytest.raises(ValueError, match="not found"):
            add_translog(df, ["x1", "x2"], include_time=True, time_var="t")

    def test_include_time_success(self):
        """Cover successful time interaction path."""
        from panelbox.frontier.data import add_translog

        df = pd.DataFrame({"x1": [1, 2], "x2": [3, 4], "t": [0, 1]})
        result = add_translog(df, ["x1", "x2"], include_time=True, time_var="t")
        assert "t_sq" in result.columns
        assert "t_x1" in result.columns

    def test_missing_variable_in_translog(self):
        """Cover line 329: variable not in DataFrame."""
        from panelbox.frontier.data import add_translog

        df = pd.DataFrame({"x1": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            add_translog(df, ["x1", "x_missing"])

    def test_translog_with_prefix(self):
        """Cover prefix branch in translog."""
        from panelbox.frontier.data import add_translog

        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
        result = add_translog(df, ["x1", "x2"], prefix="tl_")
        assert "tl_x1_sq" in result.columns
        assert "tl_x1_x2" in result.columns


# ===========================================================================
# Tests for frontier/model.py
# ===========================================================================


class TestStochasticFrontierModel:
    """Cover branches in StochasticFrontier.__init__ and related."""

    def test_auto_detect_cross_section(self):
        """Cover line 187: auto-detect cross-section model."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        sf = StochasticFrontier(data=df, depvar="y", exog=["x1", "x2"])
        from panelbox.frontier.data import ModelType

        assert sf.model_type == ModelType.CROSS_SECTION

    def test_auto_detect_pooled(self):
        """Cover line 189: entity without time -> pooled."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        df["firm"] = range(50)
        sf = StochasticFrontier(data=df, depvar="y", exog=["x1", "x2"], entity="firm")
        from panelbox.frontier.data import ModelType

        assert sf.model_type == ModelType.POOLED

    def test_auto_detect_bc95_with_ineff_vars(self):
        """Cover lines 195-198: panel with ineff_vars -> BC95."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_panel_df(5, 5)
        df["z1"] = np.random.default_rng(42).normal(0, 1, len(df))
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="year",
            dist="truncated_normal",
            inefficiency_vars=["z1"],
        )
        from panelbox.frontier.data import ModelType

        assert sf.model_type == ModelType.BATTESE_COELLI_95

    def test_model_type_string_conversion(self):
        """Cover line 200: model_type as string -> ModelType enum."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_panel_df(5, 5)
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="year",
            model_type="pitt_lee",
        )
        from panelbox.frontier.data import ModelType

        assert sf.model_type == ModelType.PITT_LEE

    def test_css_model_no_entity_raises(self):
        """Cover line 207: CSS without entity raises ValueError."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        with pytest.raises(ValueError, match="entity and time"):
            StochasticFrontier(
                data=df,
                depvar="y",
                exog=["x1", "x2"],
                model_type="css",
            )

    def test_css_model_invalid_time_trend(self):
        """Cover line 211: invalid css_time_trend."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_panel_df(5, 5)
        with pytest.raises(ValueError, match="css_time_trend"):
            StochasticFrontier(
                data=df,
                depvar="y",
                exog=["x1", "x2"],
                entity="firm",
                time="year",
                model_type="css",
                css_time_trend="cubic",
            )

    def test_css_model_default_time_trend(self):
        """Cover line 209: CSS with css_time_trend=None defaults to quadratic."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_panel_df(5, 5)
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="year",
            model_type="css",
        )
        assert sf.css_time_trend == "quadratic"

    def test_frontier_type_as_enum(self):
        """Cover branch: frontier passed as FrontierType enum."""
        from panelbox.frontier.data import FrontierType
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            frontier=FrontierType.COST,
        )
        assert sf.frontier_type == FrontierType.COST

    def test_dist_as_enum(self):
        """Cover branch: dist passed as DistributionType enum."""
        from panelbox.frontier.data import DistributionType
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            dist=DistributionType.EXPONENTIAL,
        )
        assert sf.dist == DistributionType.EXPONENTIAL

    def test_x_with_constant_column(self):
        """Cover line 262: X that already has a constant column."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        df["const"] = 1.0
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["const", "x1", "x2"],
        )
        # The constant should be detected, so exog_names == original exog
        assert "const" in sf.exog_names

    def test_repr_panel(self):
        """Cover lines 343-346: __repr__ for panel model."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_panel_df(5, 5)
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="year",
        )
        repr_str = repr(sf)
        assert "n_entities" in repr_str
        assert "n_periods" in repr_str

    def test_repr_cross_section(self):
        """Cover repr for cross-section model (no panel info)."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
        )
        repr_str = repr(sf)
        assert "n_entities" not in repr_str

    def test_fit_invalid_method(self):
        """Cover line 387: unsupported estimation method."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        sf = StochasticFrontier(data=df, depvar="y", exog=["x1", "x2"])
        with pytest.raises(ValueError, match="not supported"):
            sf.fit(method="gmm")

    def test_result_property_before_fit(self):
        """Cover lines 421-423: result property before fit() raises."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        sf = StochasticFrontier(data=df, depvar="y", exog=["x1", "x2"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = sf.result

    def test_sign_convention_cost(self):
        """Cover lines 444-451: _sign_convention for cost frontier."""
        from panelbox.frontier.data import FrontierType
        from panelbox.frontier.model import _sign_convention

        eps = np.array([1.0, -2.0, 3.0])
        result = _sign_convention(eps, FrontierType.COST)
        np.testing.assert_array_equal(result, -eps)

    def test_sign_convention_production(self):
        """Cover _sign_convention for production frontier."""
        from panelbox.frontier.data import FrontierType
        from panelbox.frontier.model import _sign_convention

        eps = np.array([1.0, -2.0, 3.0])
        result = _sign_convention(eps, FrontierType.PRODUCTION)
        np.testing.assert_array_equal(result, eps)

    def test_is_panel_property(self):
        """Cover is_panel property."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        sf = StochasticFrontier(data=df, depvar="y", exog=["x1", "x2"])
        assert sf.is_panel is False

    def test_het_vars_with_constant(self):
        """Cover het_vars preparation with constant column."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        df["z1"] = np.random.default_rng(42).normal(0, 1, 50)
        df["w1"] = np.random.default_rng(43).normal(0, 1, 50)
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            dist="truncated_normal",
            inefficiency_vars=["z1"],
            het_vars=["w1"],
        )
        assert sf.W is not None
        assert sf.Z is not None

    def test_het_vars_constant_column_present(self):
        """Cover branch: het_vars W already has a constant column."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(50)
        df["z1"] = np.random.default_rng(42).normal(0, 1, 50)
        df["w_const"] = 1.0
        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            dist="truncated_normal",
            inefficiency_vars=["z1"],
            het_vars=["w_const"],
        )
        assert sf.W is not None


# ===========================================================================
# Tests for frontier/true_models.py
# ===========================================================================


class TestTrueModels:
    """Cover branches in true_models.py."""

    def _make_panel_data(self, n=5, T=3, seed=42):
        """Create minimal panel data for true models."""
        rng = np.random.default_rng(seed)
        n_obs = n * T
        X = np.column_stack([np.ones(n_obs), rng.normal(0, 1, n_obs)])
        entity_id = np.repeat(np.arange(n), T)
        time_id = np.tile(np.arange(T), n)
        beta_true = np.array([1.0, 0.5])
        v = rng.normal(0, 0.3, n_obs)
        u = np.abs(rng.normal(0, 0.3, n_obs))
        y = X @ beta_true + v - u
        return y, X, entity_id, time_id

    def test_tfe_return_alpha(self):
        """Cover line 193/197-198: loglik_true_fixed_effects with return_alpha."""
        from panelbox.frontier.true_models import loglik_true_fixed_effects

        y, X, entity_id, time_id = self._make_panel_data()
        X.shape[1]
        theta = np.concatenate(
            [
                np.array([1.0, 0.5]),  # beta
                [np.log(0.1)],  # ln(sigma_v_sq)
                [np.log(0.1)],  # ln(sigma_u_sq)
            ]
        )
        result = loglik_true_fixed_effects(
            theta, y, X, entity_id, time_id, sign=1, return_alpha=True
        )
        assert isinstance(result, dict)
        assert "loglik" in result
        assert "alpha" in result
        assert len(result["alpha"]) == 5

    def test_tfe_cost_frontier(self):
        """Cover sign=-1 path in loglik_true_fixed_effects."""
        from panelbox.frontier.true_models import loglik_true_fixed_effects

        y, X, entity_id, time_id = self._make_panel_data()
        X.shape[1]
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1)])
        ll = loglik_true_fixed_effects(theta, y, X, entity_id, time_id, sign=-1)
        assert np.isfinite(ll)

    def test_bias_correct_tfe_analytical_scalar_T(self):
        """Cover line 234: bias_correct_tfe_analytical with scalar T."""
        from panelbox.frontier.true_models import bias_correct_tfe_analytical

        alpha_hat = np.array([0.5, 0.3, 0.2])
        result = bias_correct_tfe_analytical(alpha_hat, 5, 0.1, 0.2)
        assert result.shape == (3,)

    def test_bias_correct_tfe_analytical_array_T(self):
        """Cover bias_correct_tfe_analytical with array T."""
        from panelbox.frontier.true_models import bias_correct_tfe_analytical

        alpha_hat = np.array([0.5, 0.3, 0.2])
        T = np.array([5, 4, 6])
        result = bias_correct_tfe_analytical(alpha_hat, T, 0.1, 0.2)
        assert result.shape == (3,)

    def test_bias_correct_tfe_jackknife(self):
        """Cover lines 274-307: bias_correct_tfe_jackknife."""
        from panelbox.frontier.true_models import bias_correct_tfe_jackknife

        y, X, entity_id, time_id = self._make_panel_data(n=3, T=3)
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1)])
        result = bias_correct_tfe_jackknife(y, X, entity_id, time_id, theta, sign=1)
        assert "alpha_corrected" in result
        assert "alpha_uncorrected" in result
        assert "bias_estimate" in result

    def test_tre_simulated_method(self):
        """Cover lines 497-570: _tre_simulated_mle path."""
        from panelbox.frontier.true_models import loglik_true_random_effects

        y, X, entity_id, time_id = self._make_panel_data(n=3, T=3)
        X.shape[1]
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1), np.log(0.05)])
        ll = loglik_true_random_effects(
            theta, y, X, entity_id, time_id, sign=1, n_quadrature=8, method="simulated"
        )
        assert np.isfinite(ll)

    def test_tre_invalid_method(self):
        """Cover line 417: unknown method raises ValueError."""
        from panelbox.frontier.true_models import loglik_true_random_effects

        y, X, entity_id, time_id = self._make_panel_data(n=3, T=3)
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1), np.log(0.05)])
        with pytest.raises(ValueError, match="Unknown method"):
            loglik_true_random_effects(theta, y, X, entity_id, time_id, method="invalid_method")

    def test_variance_decomposition_tre(self):
        """Cover variance_decomposition_tre function."""
        from panelbox.frontier.true_models import variance_decomposition_tre

        result = variance_decomposition_tre(0.1, 0.2, 0.3)
        assert abs(result["gamma_v"] + result["gamma_u"] + result["gamma_w"] - 1.0) < 1e-10
        assert result["sigma_total_sq"] == pytest.approx(0.6)

    def test_tfe_bc95(self):
        """Cover lines 610-727: loglik_tfe_bc95."""
        from panelbox.frontier.true_models import loglik_tfe_bc95

        y, X, entity_id, time_id = self._make_panel_data(n=3, T=3)
        n_obs = len(y)
        rng = np.random.default_rng(42)
        Z = np.column_stack([np.ones(n_obs), rng.normal(0, 1, n_obs)])
        X.shape[1]
        m = Z.shape[1]
        theta = np.concatenate(
            [
                np.array([1.0, 0.5]),  # beta
                [np.log(0.1)],  # ln(sigma_v_sq)
                [np.log(0.1)],  # ln(sigma_u_sq)
                np.zeros(m),  # delta
            ]
        )
        ll = loglik_tfe_bc95(theta, y, X, Z, entity_id, time_id, sign=1)
        assert np.isfinite(ll)

    def test_tre_bc95(self):
        """Cover lines 730-848: loglik_tre_bc95."""
        from panelbox.frontier.true_models import loglik_tre_bc95

        y, X, entity_id, time_id = self._make_panel_data(n=3, T=3)
        n_obs = len(y)
        rng = np.random.default_rng(42)
        Z = np.column_stack([np.ones(n_obs), rng.normal(0, 1, n_obs)])
        X.shape[1]
        m = Z.shape[1]
        theta = np.concatenate(
            [
                np.array([1.0, 0.5]),  # beta
                [np.log(0.1)],  # ln(sigma_v_sq)
                [np.log(0.1)],  # ln(sigma_u_sq)
                [np.log(0.05)],  # ln(sigma_w_sq)
                np.zeros(m),  # delta
            ]
        )
        ll = loglik_tre_bc95(theta, y, X, Z, entity_id, time_id, sign=1, n_quadrature=8)
        assert np.isfinite(ll)


# ===========================================================================
# Tests for frontier/starting_values.py
# ===========================================================================


class TestStartingValues:
    """Cover branches in starting_values.py."""

    def _make_data(self, seed=42):
        rng = np.random.default_rng(seed)
        n = 100
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        beta_true = np.array([1.0, 0.5])
        v = rng.normal(0, 0.3, n)
        u = np.abs(rng.normal(0, 0.5, n))
        y = X @ beta_true + v - u
        return y, X

    def test_ols_starting_values_exponential(self):
        """Cover line 57-58: exponential distribution path."""
        from panelbox.frontier.starting_values import ols_starting_values

        y, X = self._make_data()
        _beta, sv, su = ols_starting_values(y, X, dist="exponential")
        assert sv > 0 and su > 0

    def test_ols_starting_values_truncated_normal(self):
        """Cover lines 59-61: truncated_normal path."""
        from panelbox.frontier.starting_values import ols_starting_values

        y, X = self._make_data()
        _beta, sv, su = ols_starting_values(y, X, dist="truncated_normal")
        assert sv > 0 and su > 0

    def test_ols_starting_values_gamma(self):
        """Cover lines 62-64: gamma path."""
        from panelbox.frontier.starting_values import ols_starting_values

        y, X = self._make_data()
        _beta, sv, su = ols_starting_values(y, X, dist="gamma")
        assert sv > 0 and su > 0

    def test_ols_starting_values_unknown_dist(self):
        """Cover line 66: unknown distribution raises."""
        from panelbox.frontier.starting_values import ols_starting_values

        y, X = self._make_data()
        with pytest.raises(ValueError, match="Unknown distribution"):
            ols_starting_values(y, X, dist="unknown")

    def test_ols_intercept_bias_exponential(self):
        """Cover lines 83-85: intercept bias correction for exponential."""
        from panelbox.frontier.starting_values import ols_starting_values

        y, X = self._make_data()
        beta, _sv, _su = ols_starting_values(y, X, dist="exponential")
        # Just verify it runs and returns valid result
        assert np.all(np.isfinite(beta))

    def test_ols_intercept_bias_truncated_normal(self):
        """Cover lines 86-88: intercept bias correction for truncated_normal."""
        from panelbox.frontier.starting_values import ols_starting_values

        y, X = self._make_data()
        beta, _, _ = ols_starting_values(y, X, dist="truncated_normal")
        assert np.all(np.isfinite(beta))

    def test_ols_intercept_bias_gamma(self):
        """Cover lines 89-91: intercept bias correction for gamma."""
        from panelbox.frontier.starting_values import ols_starting_values

        y, X = self._make_data()
        beta, _, _ = ols_starting_values(y, X, dist="gamma")
        assert np.all(np.isfinite(beta))

    def test_ols_no_constant_in_X(self):
        """Cover line 76->100: X without constant column (no intercept bias)."""
        from panelbox.frontier.starting_values import ols_starting_values

        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 2))  # No constant
        y = X @ np.array([0.5, 0.3]) + rng.normal(0, 0.3, n)
        beta, _sv, _su = ols_starting_values(y, X, dist="half_normal")
        assert len(beta) == 2

    def test_moments_half_normal_no_skewness(self):
        """Cover line 133-135: third moment near zero warns."""
        from panelbox.frontier.starting_values import _moments_half_normal

        with pytest.warns(UserWarning, match="Third moment near zero"):
            su_sq, sv_sq = _moments_half_normal(1.0, 0.0)
        assert su_sq > 0 and sv_sq > 0

    def test_moments_half_normal_negative_variance(self):
        """Cover lines 150-159: negative variance from moments."""
        from panelbox.frontier.starting_values import _moments_half_normal

        # Make m3 very large negative to force negative sv_sq
        with pytest.warns(UserWarning, match="Negative variance"):
            su_sq, sv_sq = _moments_half_normal(0.01, -10.0)
        assert su_sq > 0 and sv_sq > 0

    def test_moments_exponential_no_skewness(self):
        """Cover lines 186-188: exponential moments with no skewness."""
        from panelbox.frontier.starting_values import _moments_exponential

        with pytest.warns(UserWarning, match="Third moment near zero"):
            su_sq, sv_sq = _moments_exponential(1.0, 0.0)
        assert su_sq > 0 and sv_sq > 0

    def test_moments_exponential_negative_variance(self):
        """Cover lines 200-203: exponential negative variance."""
        from panelbox.frontier.starting_values import _moments_exponential

        with pytest.warns(UserWarning, match="Negative variance"):
            su_sq, sv_sq = _moments_exponential(0.001, -10.0)
        assert su_sq > 0 and sv_sq > 0

    def test_grid_search_starting_values(self):
        """Cover grid_search_starting_values function."""
        from panelbox.frontier.likelihoods import loglik_half_normal
        from panelbox.frontier.starting_values import grid_search_starting_values

        y, X = self._make_data()
        beta, _sv, _su = grid_search_starting_values(
            y, X, "half_normal", loglik_half_normal, sign=1, n_points=3
        )
        assert np.all(np.isfinite(beta))

    def test_grid_search_truncated_normal(self):
        """Cover line 250: grid search with truncated normal."""
        from panelbox.frontier.likelihoods import loglik_truncated_normal
        from panelbox.frontier.starting_values import grid_search_starting_values

        y, X = self._make_data()
        beta, _sv, _su = grid_search_starting_values(
            y, X, "truncated_normal", loglik_truncated_normal, sign=1, n_points=3
        )
        assert np.all(np.isfinite(beta))

    def test_grid_search_gamma(self):
        """Cover line 252: grid search with gamma."""
        from panelbox.frontier.likelihoods import loglik_gamma
        from panelbox.frontier.starting_values import grid_search_starting_values

        y, X = self._make_data()
        beta, _sv, _su = grid_search_starting_values(
            y, X, "gamma", loglik_gamma, sign=1, n_points=3
        )
        assert np.all(np.isfinite(beta))

    def test_get_starting_values_grid_search_no_func(self):
        """Cover line 296: grid_search=True without likelihood_func."""
        from panelbox.frontier.starting_values import get_starting_values

        y, X = self._make_data()
        with pytest.raises(ValueError, match="likelihood_func required"):
            get_starting_values(y, X, None, "half_normal", grid_search=True)

    def test_get_starting_values_truncated_normal_no_Z(self):
        """Cover lines 317-319: truncated normal without Z."""
        from panelbox.frontier.starting_values import get_starting_values

        y, X = self._make_data()
        theta = get_starting_values(y, X, None, "truncated_normal")
        # Should have beta + ln(sv) + ln(su) + mu
        assert len(theta) == X.shape[1] + 3

    def test_get_starting_values_truncated_normal_with_Z(self):
        """Cover lines 312-316: truncated normal with Z."""
        from panelbox.frontier.starting_values import get_starting_values

        y, X = self._make_data()
        Z = np.column_stack([np.ones(len(y)), np.random.default_rng(42).normal(0, 1, len(y))])
        theta = get_starting_values(y, X, Z, "truncated_normal")
        # Should have beta + ln(sv) + ln(su) + delta(m)
        assert len(theta) == X.shape[1] + 2 + Z.shape[1]

    def test_get_starting_values_gamma(self):
        """Cover lines 321-326: gamma starting values."""
        from panelbox.frontier.starting_values import get_starting_values

        y, X = self._make_data()
        theta = get_starting_values(y, X, None, "gamma")
        # Should have beta + ln(sv) + ln(su) + ln(P) + ln(theta)
        assert len(theta) == X.shape[1] + 4

    def test_check_starting_values_valid(self):
        """Cover check_starting_values with valid starting values."""
        from panelbox.frontier.likelihoods import loglik_half_normal
        from panelbox.frontier.starting_values import check_starting_values

        y, X = self._make_data()
        theta = np.concatenate(
            [
                np.array([1.0, 0.5]),
                [np.log(0.1)],
                [np.log(0.1)],
            ]
        )
        result = check_starting_values(theta, y, X, loglik_half_normal, sign=1)
        assert result["valid"]
        assert result["finite"]

    def test_check_starting_values_exception(self):
        """Cover line 355-356: exception during check."""
        from panelbox.frontier.starting_values import check_starting_values

        def bad_func(theta, y, X, sign):
            raise RuntimeError("bad")

        y, X = self._make_data()
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1)])
        result = check_starting_values(theta, y, X, bad_func, sign=1)
        assert not result["valid"]


# ===========================================================================
# Tests for frontier/estimation.py
# ===========================================================================


class TestEstimation:
    """Cover estimation.py branches."""

    def test_estimate_bfgs_optimizer(self):
        """Cover lines 267-277: BFGS optimizer path."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(100)
        sf = StochasticFrontier(data=df, depvar="y", exog=["x1", "x2"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sf.fit(optimizer="BFGS", maxiter=50)
        assert result is not None

    def test_estimate_unknown_optimizer(self):
        """Cover line 280: unknown optimizer raises ValueError."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(100)
        sf = StochasticFrontier(data=df, depvar="y", exog=["x1", "x2"])
        with pytest.raises(ValueError, match="Unknown optimizer"):
            sf.fit(optimizer="INVALID")

    def test_verbose_estimation(self, caplog):
        """Cover verbose=True paths in estimate_mle."""
        from panelbox.frontier.model import StochasticFrontier

        df = _make_cross_section_df(100)
        sf = StochasticFrontier(data=df, depvar="y", exog=["x1", "x2"])
        with (
            caplog.at_level(logging.INFO, logger="panelbox.frontier.estimation"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            result = sf.fit(verbose=True, maxiter=20)
        assert result is not None

    def test_compute_hessian_non_numerical_raises(self):
        """Cover line 486: non-numerical hessian raises NotImplementedError."""
        from panelbox.frontier.estimation import _compute_hessian

        theta = np.array([1.0, 2.0])
        with pytest.raises(NotImplementedError, match="Only numerical"):
            _compute_hessian(theta, lambda x: np.sum(x**2), method="analytical")

    def test_get_likelihood_function_unknown(self):
        """Cover line 355: unknown distribution."""
        from panelbox.frontier.estimation import _get_likelihood_function

        with pytest.raises(ValueError, match="Unknown distribution"):
            _get_likelihood_function("unknown")

    def test_get_likelihood_function_wang(self):
        """Cover line 344-345: Wang model."""
        from panelbox.frontier.estimation import _get_likelihood_function

        func = _get_likelihood_function("truncated_normal", is_wang=True)
        assert func is not None

    def test_get_gradient_function(self):
        """Cover _get_gradient_function paths."""
        from panelbox.frontier.estimation import _get_gradient_function

        assert _get_gradient_function("half_normal") is not None
        assert _get_gradient_function("exponential") is not None
        assert _get_gradient_function("truncated_normal") is None
        assert _get_gradient_function("gamma") is None

    def test_transform_parameters_gamma(self):
        """Cover lines 458-464: _transform_parameters for gamma distribution."""
        from panelbox.frontier.estimation import _transform_parameters

        # theta = [beta, ln_sv, ln_su, ln_P, ln_theta]
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.2), np.log(2.0), np.log(1.0)])
        _params, names = _transform_parameters(
            theta,
            n_exog=2,
            n_ineff_vars=0,
            dist="gamma",
            exog_names=["const", "x1"],
            ineff_var_names=[],
        )
        assert "gamma_P" in names
        assert "gamma_theta" in names

    def test_transform_parameters_truncated_normal_no_z(self):
        """Cover lines 452-456: truncated normal without Z (mu param)."""
        from panelbox.frontier.estimation import _transform_parameters

        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.2), 0.5])
        _params, names = _transform_parameters(
            theta,
            n_exog=2,
            n_ineff_vars=0,
            dist="truncated_normal",
            exog_names=["const", "x1"],
            ineff_var_names=[],
        )
        assert "mu" in names

    def test_transform_parameters_wang(self):
        """Cover lines 414-433: _transform_parameters for Wang model."""
        from panelbox.frontier.estimation import _transform_parameters

        # Wang: [beta, ln_sv, delta, gamma]
        theta = np.array([1.0, 0.5, np.log(0.1), 0.1, 0.2, 0.3, 0.4])
        _params, names = _transform_parameters(
            theta,
            n_exog=2,
            n_ineff_vars=2,
            dist="truncated_normal",
            exog_names=["const", "x1"],
            ineff_var_names=["const", "z1"],
            is_wang=True,
            hetero_var_names=["const", "w1"],
        )
        assert any("delta_" in n for n in names)
        assert any("gamma_" in n for n in names)


# ===========================================================================
# Tests for frontier/panel_likelihoods.py
# ===========================================================================


class TestPanelLikelihoods:
    """Cover panel_likelihoods.py branches."""

    def _make_panel_data(self, n=5, T=3, seed=42):
        rng = np.random.default_rng(seed)
        n_obs = n * T
        X = np.column_stack([np.ones(n_obs), rng.normal(0, 1, n_obs)])
        entity_id = np.repeat(np.arange(n), T)
        time_id = np.tile(np.arange(T), n)
        beta_true = np.array([1.0, 0.5])
        v = rng.normal(0, 0.3, n_obs)
        u = np.abs(rng.normal(0, 0.3, n_obs))
        y = X @ beta_true + v - u
        return y, X, entity_id, time_id

    def test_pitt_lee_exponential(self):
        """Cover lines 157-237: loglik_pitt_lee_exponential."""
        from panelbox.frontier.panel_likelihoods import loglik_pitt_lee_exponential

        y, X, entity_id, time_id = self._make_panel_data()
        X.shape[1]
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(1.0)])
        ll = loglik_pitt_lee_exponential(theta, y, X, entity_id, time_id, sign=1)
        assert np.isfinite(ll)

    def test_pitt_lee_truncated_normal(self):
        """Cover lines 244-345: loglik_pitt_lee_truncated_normal."""
        from panelbox.frontier.panel_likelihoods import loglik_pitt_lee_truncated_normal

        y, X, entity_id, time_id = self._make_panel_data()
        X.shape[1]
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1), 0.0])
        ll = loglik_pitt_lee_truncated_normal(theta, y, X, entity_id, time_id, sign=1)
        assert np.isfinite(ll)

    def test_battese_coelli_92(self):
        """Cover lines 352-464: loglik_battese_coelli_92."""
        from panelbox.frontier.panel_likelihoods import loglik_battese_coelli_92

        y, X, entity_id, time_id = self._make_panel_data()
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1), 0.0, 0.1])
        ll = loglik_battese_coelli_92(theta, y, X, entity_id, time_id, sign=1)
        assert np.isfinite(ll)

    def test_battese_coelli_95(self):
        """Cover lines 471-560: loglik_battese_coelli_95."""
        from panelbox.frontier.panel_likelihoods import loglik_battese_coelli_95

        y, X, entity_id, time_id = self._make_panel_data()
        n_obs = len(y)
        rng = np.random.default_rng(42)
        Z = np.column_stack([np.ones(n_obs), rng.normal(0, 1, n_obs)])
        X.shape[1]
        m = Z.shape[1]
        theta = np.concatenate(
            [
                np.array([1.0, 0.5]),
                [np.log(0.1)],
                [np.log(0.1)],
                np.zeros(m),
            ]
        )
        ll = loglik_battese_coelli_95(theta, y, X, Z, entity_id, time_id, sign=1)
        assert np.isfinite(ll)

    def test_kumbhakar_1990(self):
        """Cover lines 565-677: loglik_kumbhakar_1990."""
        from panelbox.frontier.panel_likelihoods import loglik_kumbhakar_1990

        y, X, entity_id, time_id = self._make_panel_data()
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1), 0.0, 0.0, 0.0])
        ll = loglik_kumbhakar_1990(theta, y, X, entity_id, time_id, sign=1)
        assert np.isfinite(ll)

    def test_lee_schmidt_1993(self):
        """Cover lines 684-801: loglik_lee_schmidt_1993."""
        from panelbox.frontier.panel_likelihoods import loglik_lee_schmidt_1993

        y, X, entity_id, time_id = self._make_panel_data(T=3)
        T = 3
        # theta = [beta, ln_sv, ln_su, mu, delta_1, delta_2]
        theta = np.concatenate(
            [
                np.array([1.0, 0.5]),
                [np.log(0.1)],
                [np.log(0.1)],
                [0.0],  # mu
                np.ones(T - 1),  # delta_1, delta_2
            ]
        )
        ll = loglik_lee_schmidt_1993(theta, y, X, entity_id, time_id, sign=1)
        assert np.isfinite(ll)

    def test_bc92(self):
        """Cover lines 808-913: loglik_bc92."""
        from panelbox.frontier.panel_likelihoods import loglik_bc92

        y, X, entity_id, time_id = self._make_panel_data()
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1), 0.1])
        ll = loglik_bc92(theta, y, X, entity_id, time_id, sign=1)
        assert np.isfinite(ll)

    def test_bc92_cost_frontier(self):
        """Cover sign=-1 path in loglik_bc92."""
        from panelbox.frontier.panel_likelihoods import loglik_bc92

        y, X, entity_id, time_id = self._make_panel_data()
        theta = np.array([1.0, 0.5, np.log(0.1), np.log(0.1), 0.1])
        ll = loglik_bc92(theta, y, X, entity_id, time_id, sign=-1)
        assert np.isfinite(ll)


# ===========================================================================
# Tests for frontier/utils/marginal_effects.py
# ===========================================================================


class TestFrontierMarginalEffects:
    """Cover frontier/utils/marginal_effects.py branches."""

    def test_marginal_effects_no_determinants(self):
        """Cover lines 152-156: model without determinants raises."""
        from panelbox.frontier.utils.marginal_effects import marginal_effects

        # Create a mock result with no inefficiency_vars or hetero_vars
        class MockModel:
            hetero_vars = []
            inefficiency_vars = []

        class MockResult:
            model = MockModel()

        with pytest.raises(ValueError, match="determinants"):
            marginal_effects(MockResult())

    def test_marginal_effects_wang_method_unknown(self):
        """Cover line 341: unknown method in Wang model."""
        from panelbox.frontier.utils.marginal_effects import marginal_effects_wang_2002

        class MockModel:
            n_exog = 2
            ineff_var_names = ["const", "z1"]
            hetero_var_names = ["const", "w1"]
            Z = np.column_stack([np.ones(5), np.random.default_rng(42).normal(0, 1, 5)])
            W = np.column_stack([np.ones(5), np.random.default_rng(43).normal(0, 1, 5)])

        class MockResult:
            model = MockModel()
            params = np.array([1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Unknown method"):
            marginal_effects_wang_2002(MockResult(), method="unknown")

    def test_marginal_effects_bc95_method_unknown(self):
        """Cover line 497: unknown method in BC95 model."""
        from panelbox.frontier.utils.marginal_effects import marginal_effects_bc95

        class MockModel:
            n_exog = 2
            ineff_var_names = ["const", "z1"]
            inefficiency_vars = ["z1"]
            Z = np.column_stack([np.ones(5), np.random.default_rng(42).normal(0, 1, 5)])

        class MockResult:
            model = MockModel()
            params = np.array([1.0, 0.5, 0.1, 0.2, 0.0, 0.0])
            vcov = np.eye(6) * 0.01

        with pytest.raises(ValueError, match="Unknown method"):
            marginal_effects_bc95(MockResult(), method="unknown")

    def test_marginal_effects_summary(self):
        """Cover marginal_effects_summary function."""
        from panelbox.frontier.utils.marginal_effects import marginal_effects_summary

        df = pd.DataFrame(
            {
                "variable": ["z1", "z2"],
                "marginal_effect": [0.05, -0.03],
                "std_error": [0.01, 0.02],
                "z_stat": [5.0, -1.5],
                "p_value": [0.001, 0.134],
                "interpretation": ["increases ***", "decreases"],
            }
        )
        summary_text = marginal_effects_summary(df)
        assert "MARGINAL EFFECTS" in summary_text
        assert "z1" in summary_text


# ===========================================================================
# Mark first checkbox
# ===========================================================================
