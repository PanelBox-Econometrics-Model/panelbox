"""Verify datasets load and have correct structure."""

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestFirmPanel:
    """Tests for firm_panel.csv."""

    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA_DIR / "firm_panel.csv")

    def test_firm_panel_loads(self, df):
        """firm_panel.csv loads with correct columns and dimensions."""
        assert df.shape == (2000, 7)

    def test_firm_panel_columns(self, df):
        expected = {"firm_id", "year", "investment", "value", "capital", "sales", "sector"}
        assert set(df.columns) == expected

    def test_firm_panel_no_missing(self, df):
        assert df.notna().all().all()

    def test_firm_panel_entities(self, df):
        assert df["firm_id"].nunique() == 100

    def test_firm_panel_periods(self, df):
        assert df["year"].nunique() == 20


class TestBankLGD:
    """Tests for bank_lgd.csv."""

    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA_DIR / "bank_lgd.csv")

    def test_bank_lgd_loads(self, df):
        """bank_lgd.csv loads with correct columns and dimensions."""
        assert df.shape == (3000, 7)

    def test_bank_lgd_columns(self, df):
        expected = {
            "contract_id",
            "month",
            "lgd_logit",
            "saldo_real",
            "pib_growth",
            "selic",
            "collateral_ratio",
        }
        assert set(df.columns) == expected

    def test_bank_lgd_no_missing(self, df):
        assert df.notna().all().all()

    def test_bank_lgd_entities(self, df):
        assert df["contract_id"].nunique() == 200

    def test_bank_lgd_periods(self, df):
        assert df["month"].nunique() == 15


class TestMacroQuarterly:
    """Tests for macro_quarterly.csv."""

    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA_DIR / "macro_quarterly.csv")

    def test_macro_quarterly_loads(self, df):
        """macro_quarterly.csv loads with correct columns and dimensions."""
        assert df.shape == (1200, 5)

    def test_macro_quarterly_columns(self, df):
        expected = {"country", "quarter", "gdp_growth", "inflation", "interest_rate"}
        assert set(df.columns) == expected

    def test_macro_quarterly_entities(self, df):
        assert df["country"].nunique() == 30

    def test_macro_quarterly_periods(self, df):
        assert df["quarter"].nunique() == 40


class TestNewFirms:
    """Tests for new_firms.csv compatibility."""

    def test_new_firms_compatible(self):
        """new_firms.csv has same columns as firm_panel.csv."""
        firm = pd.read_csv(DATA_DIR / "firm_panel.csv")
        new = pd.read_csv(DATA_DIR / "new_firms.csv")
        assert set(new.columns) == set(firm.columns)

    def test_new_firms_dimensions(self):
        new = pd.read_csv(DATA_DIR / "new_firms.csv")
        assert new.shape == (100, 7)
        assert new["firm_id"].nunique() == 20


class TestNewBank:
    """Tests for new_bank_data.csv compatibility."""

    def test_new_bank_compatible(self):
        """new_bank_data.csv has same columns as bank_lgd.csv."""
        bank = pd.read_csv(DATA_DIR / "bank_lgd.csv")
        new = pd.read_csv(DATA_DIR / "new_bank_data.csv")
        assert set(new.columns) == set(bank.columns)

    def test_new_bank_dimensions(self):
        new = pd.read_csv(DATA_DIR / "new_bank_data.csv")
        assert new.shape == (150, 7)
        assert new["contract_id"].nunique() == 50


class TestFutureMacro:
    """Tests for future_macro.csv."""

    def test_future_macro_loads(self):
        df = pd.read_csv(DATA_DIR / "future_macro.csv")
        assert df.shape == (120, 4)

    def test_future_macro_no_dep_var(self):
        df = pd.read_csv(DATA_DIR / "future_macro.csv")
        assert "gdp_growth" not in df.columns

    def test_future_macro_quarters(self):
        df = pd.read_csv(DATA_DIR / "future_macro.csv")
        assert df["quarter"].min() == 41  # starts after main panel
