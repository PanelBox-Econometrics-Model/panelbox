"""Tests for utility functions (simulation, plotting, comparison helpers)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add utils to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))

from simulation_helpers import (
    generate_card_education,
    generate_crossing_example,
    generate_financial_returns,
    generate_firm_production,
    generate_heteroskedastic,
    generate_labor_program,
    generate_location_shift,
    generate_treatment_effects,
)

# ── Simulation helpers ───────────────────────────────────────────────────────


class TestGenerateCardEducation:
    def test_returns_dataframe(self):
        df = generate_card_education(n_individuals=10, n_years=3)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = generate_card_education(n_individuals=10, n_years=3)
        assert df.shape == (30, 12)

    def test_reproducibility(self):
        df1 = generate_card_education(n_individuals=10, seed=123)
        df2 = generate_card_education(n_individuals=10, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_columns(self):
        df = generate_card_education(n_individuals=10, n_years=3)
        expected = [
            "id",
            "year",
            "lwage",
            "educ",
            "exper",
            "black",
            "south",
            "married",
            "female",
            "union",
            "hours",
            "age",
        ]
        assert list(df.columns) == expected


class TestGenerateFirmProduction:
    def test_returns_dataframe(self):
        df = generate_firm_production(n_firms=10, n_years=5)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = generate_firm_production(n_firms=10, n_years=5)
        assert df.shape == (50, 10)

    def test_sectors(self):
        df = generate_firm_production(n_firms=50, n_years=5)
        assert df["sector"].nunique() <= 5


class TestGenerateFinancialReturns:
    def test_returns_dataframe(self):
        df = generate_financial_returns(n_firms=10, n_months=12)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = generate_financial_returns(n_firms=10, n_months=12)
        assert df.shape == (120, 8)


class TestGenerateLaborProgram:
    def test_returns_dataframe(self):
        df = generate_labor_program(n_individuals=50)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = generate_labor_program(n_individuals=50)
        assert df.shape == (100, 8)

    def test_treatment_balance(self):
        df = generate_labor_program(n_individuals=1000)
        prop = df.groupby("id")["treatment"].first().mean()
        assert 0.4 < prop < 0.6


class TestGenerateCrossingExample:
    def test_shape(self):
        df = generate_crossing_example(n_units=10, n_periods=5)
        assert df.shape == (50, 5)


class TestGenerateLocationShift:
    def test_shape(self):
        df = generate_location_shift(n_units=20, n_periods=5)
        assert df.shape == (100, 6)

    def test_groups(self):
        df = generate_location_shift(n_units=20, n_periods=5)
        assert set(df["group"].unique()) == {"shift", "no_shift"}


class TestGenerateHeteroskedastic:
    def test_shape(self):
        df = generate_heteroskedastic(n_units=10, n_periods=5)
        assert df.shape == (50, 6)


class TestGenerateTreatmentEffects:
    def test_shape(self):
        df = generate_treatment_effects(n_individuals=50)
        assert df.shape == (100, 7)


# ── Plot helpers ─────────────────────────────────────────────────────────────


class TestPlotHelpers:
    def test_set_quantile_style(self):
        from plot_helpers import set_quantile_style

        set_quantile_style()  # Should not raise

    def test_plot_check_loss(self):
        import matplotlib

        matplotlib.use("Agg")
        from plot_helpers import plot_check_loss

        fig = plot_check_loss()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close("all")


# ── Comparison helpers ───────────────────────────────────────────────────────


class TestComparisonHelpers:
    def test_inter_quantile_test(self):
        from comparison_helpers import inter_quantile_test

        class MockResult:
            def __init__(self, params, ses):
                self.params = np.array(params)
                self.std_errors = np.array(ses)

        r1 = MockResult([1.0, 2.0], [0.1, 0.2])
        r2 = MockResult([1.5, 2.5], [0.1, 0.2])

        result = inter_quantile_test(r1, r2, variable=0, tau_low=0.1, tau_high=0.9)
        assert "diff" in result
        assert "p_value" in result
        assert result["diff"] == pytest.approx(0.5)

    def test_pseudo_r2_table(self):
        from comparison_helpers import pseudo_r2_table

        class MockResult:
            pseudo_r2 = 0.15

        results = {0.25: MockResult(), 0.5: MockResult(), 0.75: MockResult()}
        df = pseudo_r2_table(results)
        assert len(df) == 3
        assert "pseudo_r2" in df.columns
