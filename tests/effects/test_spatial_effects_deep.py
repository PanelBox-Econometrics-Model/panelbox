"""
Deep coverage tests for panelbox.effects.spatial_effects.

Current coverage is 96.30% — targets uncovered lines:
36-37 (plotly ImportError branch), 260-262 (singular matrix skip),
320-322 (empty simulation fallback), 359 (delta method var not in effects),
586->585 (format loop partial), 621 (no plotly raises ImportError),
769->768 (to_latex loop partial).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.effects.spatial_effects import (
    SpatialEffectsResult,
    _compute_pvalue,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def effects_dict():
    """Pre-computed effects dictionary for SpatialEffectsResult."""
    return {
        "x1": {
            "direct": 0.5,
            "indirect": 0.2,
            "total": 0.7,
            "direct_se": 0.05,
            "indirect_se": 0.03,
            "total_se": 0.06,
            "direct_ci": (0.4, 0.6),
            "indirect_ci": (0.14, 0.26),
            "total_ci": (0.58, 0.82),
            "direct_pvalue": 0.001,
            "indirect_pvalue": 0.01,
            "total_pvalue": 0.001,
        },
        "x2": {
            "direct": -0.3,
            "indirect": -0.1,
            "total": -0.4,
            "direct_se": 0.04,
            "indirect_se": 0.02,
            "total_se": 0.05,
            "direct_ci": (-0.38, -0.22),
            "indirect_ci": (-0.14, -0.06),
            "total_ci": (-0.5, -0.3),
            "direct_pvalue": 0.001,
            "indirect_pvalue": 0.02,
            "total_pvalue": 0.001,
        },
    }


class _FakeModel:
    """Minimal model for SpatialEffectsResult."""

    spatial_model_type = "SAR"


class _FakeModelResult:
    """Minimal model result for SpatialEffectsResult."""

    def __init__(self):
        self.model = _FakeModel()


@pytest.fixture
def effects_result(effects_dict):
    """SpatialEffectsResult instance."""
    return SpatialEffectsResult(
        effects=effects_dict,
        model_result=_FakeModelResult(),
        method="simulation",
        n_simulations=1000,
        confidence_level=0.95,
    )


# ---------------------------------------------------------------------------
# SpatialEffectsResult
# ---------------------------------------------------------------------------


class TestSpatialEffectsResultDisplay:
    """Cover display and export methods."""

    def test_summary_prints(self, effects_result):
        """Cover summary method including format_dict loop (line 586)."""
        df = effects_result.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_print_summary(self, effects_result):
        """Cover print_summary method."""
        effects_result.print_summary()

    def test_plot_matplotlib(self, effects_result):
        """Cover plot with matplotlib backend."""
        fig = effects_result.plot(backend="matplotlib", show_ci=True)
        assert fig is not None

    def test_plot_invalid_backend_raises(self, effects_result):
        """Cover line 616: ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            effects_result.plot(backend="bokeh")

    def test_to_latex(self, effects_result):
        """Cover to_latex method (line 769->768 partial)."""
        latex = effects_result.to_latex()
        assert isinstance(latex, str)
        assert "x1" in latex or "tabular" in latex

    def test_to_latex_with_file(self, effects_result, tmp_path):
        """Cover to_latex saving to file."""
        filepath = str(tmp_path / "effects.tex")
        latex = effects_result.to_latex(filename=filepath)
        assert isinstance(latex, str)
        assert (tmp_path / "effects.tex").exists()

    def test_plot_plotly_no_plotly(self, effects_result):
        """Cover line 621: ImportError when plotly not available."""
        import panelbox.effects.spatial_effects as mod

        original = mod.HAS_PLOTLY
        mod.HAS_PLOTLY = False
        try:
            with pytest.raises(ImportError, match="plotly"):
                effects_result.plot(backend="plotly")
        finally:
            mod.HAS_PLOTLY = original


# ---------------------------------------------------------------------------
# _compute_pvalue
# ---------------------------------------------------------------------------


class TestComputePvalue:
    """Cover _compute_pvalue function."""

    def test_pvalue_positive_estimate(self):
        """Test p-value computation for positive estimate."""
        simulated = np.random.randn(1000)
        pval = _compute_pvalue(2.0, simulated)
        assert 0 <= pval <= 1

    def test_pvalue_zero_estimate(self):
        """Test p-value computation for estimate at zero."""
        simulated = np.random.randn(1000)
        pval = _compute_pvalue(0.0, simulated)
        assert 0 <= pval <= 1


# ---------------------------------------------------------------------------
# Effects without SE (nan fallback)
# ---------------------------------------------------------------------------


class TestEffectsWithoutSE:
    """Cover effects dict without SE keys."""

    def test_summary_no_se(self):
        """Cover effects without standard errors."""
        effects = {
            "x1": {
                "direct": 0.5,
                "indirect": 0.2,
                "total": 0.7,
            },
        }
        result = SpatialEffectsResult(
            effects=effects,
            model_result=_FakeModelResult(),
            method="analytical",
            n_simulations=None,
            confidence_level=0.95,
        )
        df = result.summary()
        assert isinstance(df, pd.DataFrame)
