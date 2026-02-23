"""
Tests for data integrity of all quantile regression tutorial datasets.

Verifies that all CSV files exist, have correct shapes, expected columns,
and reasonable value ranges.
"""

import os

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SIM_DIR = os.path.join(DATA_DIR, "simulated")


# ── Main datasets ────────────────────────────────────────────────────────────


class TestCardEducation:
    """Tests for card_education.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.path = os.path.join(DATA_DIR, "card_education.csv")
        self.df = pd.read_csv(self.path)

    def test_file_exists(self):
        assert os.path.exists(self.path)

    def test_shape(self):
        assert self.df.shape == (3000, 12)

    def test_columns(self):
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
        assert list(self.df.columns) == expected

    def test_n_individuals(self):
        assert self.df["id"].nunique() == 500

    def test_n_years(self):
        assert self.df["year"].nunique() == 6

    def test_binary_vars(self):
        for col in ["black", "south", "married", "female", "union"]:
            assert set(self.df[col].unique()).issubset({0, 1})

    def test_no_missing(self):
        assert self.df.isnull().sum().sum() == 0


class TestFirmProduction:
    """Tests for firm_production.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.path = os.path.join(DATA_DIR, "firm_production.csv")
        self.df = pd.read_csv(self.path)

    def test_file_exists(self):
        assert os.path.exists(self.path)

    def test_shape(self):
        assert self.df.shape == (5000, 10)

    def test_columns(self):
        expected = [
            "firm_id",
            "year",
            "log_output",
            "log_capital",
            "log_labor",
            "log_materials",
            "profit",
            "size",
            "sector",
            "exporter",
        ]
        assert list(self.df.columns) == expected

    def test_n_firms(self):
        assert self.df["firm_id"].nunique() == 500

    def test_n_years(self):
        assert self.df["year"].nunique() == 10

    def test_sectors(self):
        assert self.df["sector"].nunique() == 5


class TestFinancialReturns:
    """Tests for financial_returns.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.path = os.path.join(DATA_DIR, "financial_returns.csv")
        self.df = pd.read_csv(self.path)

    def test_file_exists(self):
        assert os.path.exists(self.path)

    def test_shape(self):
        assert self.df.shape == (12000, 8)

    def test_columns(self):
        expected = [
            "firm_id",
            "month",
            "returns",
            "size",
            "book_to_market",
            "momentum",
            "volatility",
            "sector",
        ]
        assert list(self.df.columns) == expected

    def test_n_firms(self):
        assert self.df["firm_id"].nunique() == 200

    def test_n_months(self):
        assert self.df["month"].nunique() == 60


class TestLaborProgram:
    """Tests for labor_program.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.path = os.path.join(DATA_DIR, "labor_program.csv")
        self.df = pd.read_csv(self.path)

    def test_file_exists(self):
        assert os.path.exists(self.path)

    def test_shape(self):
        assert self.df.shape == (2000, 8)

    def test_columns(self):
        expected = [
            "id",
            "period",
            "treatment",
            "earnings",
            "education",
            "age",
            "experience",
            "female",
        ]
        assert list(self.df.columns) == expected

    def test_n_individuals(self):
        assert self.df["id"].nunique() == 1000

    def test_periods(self):
        assert set(self.df["period"].unique()) == {0, 1}

    def test_treatment_binary(self):
        assert set(self.df["treatment"].unique()).issubset({0, 1})


# ── Simulated datasets ──────────────────────────────────────────────────────


class TestCrossingExample:
    """Tests for simulated/crossing_example.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.path = os.path.join(SIM_DIR, "crossing_example.csv")
        self.df = pd.read_csv(self.path)

    def test_file_exists(self):
        assert os.path.exists(self.path)

    def test_shape(self):
        assert self.df.shape == (2400, 5)

    def test_columns(self):
        assert list(self.df.columns) == ["id", "t", "y", "x1", "x2"]

    def test_n_units(self):
        assert self.df["id"].nunique() == 300

    def test_n_periods(self):
        assert self.df["t"].nunique() == 8


class TestLocationShift:
    """Tests for simulated/location_shift.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.path = os.path.join(SIM_DIR, "location_shift.csv")
        self.df = pd.read_csv(self.path)

    def test_file_exists(self):
        assert os.path.exists(self.path)

    def test_shape(self):
        assert self.df.shape == (4000, 6)

    def test_columns(self):
        assert list(self.df.columns) == ["id", "t", "y", "x1", "x2", "group"]

    def test_groups(self):
        assert set(self.df["group"].unique()) == {"shift", "no_shift"}


class TestHeteroskedastic:
    """Tests for simulated/heteroskedastic.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.path = os.path.join(SIM_DIR, "heteroskedastic.csv")
        self.df = pd.read_csv(self.path)

    def test_file_exists(self):
        assert os.path.exists(self.path)

    def test_shape(self):
        assert self.df.shape == (4000, 6)

    def test_columns(self):
        assert list(self.df.columns) == ["id", "t", "y", "x1", "x2", "x3"]


class TestTreatmentEffects:
    """Tests for simulated/treatment_effects.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.path = os.path.join(SIM_DIR, "treatment_effects.csv")
        self.df = pd.read_csv(self.path)

    def test_file_exists(self):
        assert os.path.exists(self.path)

    def test_shape(self):
        assert self.df.shape == (1600, 7)

    def test_columns(self):
        expected = [
            "id",
            "period",
            "treatment",
            "earnings",
            "education",
            "age",
            "female",
        ]
        assert list(self.df.columns) == expected
