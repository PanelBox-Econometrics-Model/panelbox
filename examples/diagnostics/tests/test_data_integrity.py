"""
Test data integrity for diagnostics tutorials.

Verifies that all datasets:
- Exist and can be loaded
- Have expected dimensions
- Have no unexpected missing values
- Have correct data types
- Match codebook specifications
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


# -----------------------------------------------------------------------
# Unit Root datasets
# -----------------------------------------------------------------------


class TestPennWorldTable:
    """Test penn_world_table.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "unit_root" / "penn_world_table.csv")

    def test_exists(self):
        assert (DATA_DIR / "unit_root" / "penn_world_table.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (1500, 8), f"Expected (1500, 8), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "countrycode",
            "year",
            "rgdpna",
            "rkna",
            "emp",
            "labsh",
            "pop",
            "hc",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_n_countries(self, data):
        assert data["countrycode"].nunique() == 30

    def test_n_years(self, data):
        assert data["year"].nunique() == 50

    def test_gdp_positive(self, data):
        assert (data["rgdpna"] > 0).all()

    def test_labsh_range(self, data):
        assert data["labsh"].between(0.1, 0.95).all()


class TestPricesPanel:
    """Test prices_panel.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "unit_root" / "prices_panel.csv")

    def test_exists(self):
        assert (DATA_DIR / "unit_root" / "prices_panel.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (1200, 5), f"Expected (1200, 5), got {data.shape}"

    def test_columns(self, data):
        expected = ["region", "year", "price_index", "log_price", "inflation"]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_n_regions(self, data):
        assert data["region"].nunique() == 40

    def test_n_years(self, data):
        assert data["year"].nunique() == 30


# -----------------------------------------------------------------------
# Cointegration datasets
# -----------------------------------------------------------------------


class TestOECDMacro:
    """Test oecd_macro.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "cointegration" / "oecd_macro.csv")

    def test_exists(self):
        assert (DATA_DIR / "cointegration" / "oecd_macro.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (800, 7), f"Expected (800, 7), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "country",
            "year",
            "consumption",
            "income",
            "investment",
            "log_C",
            "log_Y",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_n_countries(self, data):
        assert data["country"].nunique() == 20


class TestPPPData:
    """Test ppp_data.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "cointegration" / "ppp_data.csv")

    def test_exists(self):
        assert (DATA_DIR / "cointegration" / "ppp_data.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (875, 7), f"Expected (875, 7), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "country",
            "year",
            "exchange_rate",
            "price_domestic",
            "price_foreign",
            "log_S",
            "log_P_ratio",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0


class TestInterestRates:
    """Test interest_rates.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "cointegration" / "interest_rates.csv")

    def test_exists(self):
        assert (DATA_DIR / "cointegration" / "interest_rates.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (450, 6), f"Expected (450, 6), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "country",
            "year",
            "domestic_rate",
            "us_rate",
            "spread",
            "forward_premium",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0


# -----------------------------------------------------------------------
# Specification datasets
# -----------------------------------------------------------------------


class TestNLSWork:
    """Test nlswork.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "specification" / "nlswork.csv")

    def test_exists(self):
        assert (DATA_DIR / "specification" / "nlswork.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (60000, 10), f"Expected (60000, 10), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "idcode",
            "year",
            "ln_wage",
            "experience",
            "tenure",
            "education",
            "union",
            "married",
            "hours",
            "industry",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_n_individuals(self, data):
        assert data["idcode"].nunique() == 4000

    def test_education_range(self, data):
        assert data["education"].between(5, 25).all()

    def test_binary_vars(self, data):
        assert set(data["union"].unique()).issubset({0, 1})
        assert set(data["married"].unique()).issubset({0, 1})


class TestFirmProductivity:
    """Test firm_productivity.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "specification" / "firm_productivity.csv")

    def test_exists(self):
        assert (DATA_DIR / "specification" / "firm_productivity.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (4000, 9), f"Expected (4000, 9), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "firm_id",
            "year",
            "log_output",
            "log_capital",
            "log_labor",
            "log_materials",
            "rd_intensity",
            "sector",
            "exporter",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_n_firms(self, data):
        assert data["firm_id"].nunique() == 200

    def test_sectors(self, data):
        assert data["sector"].nunique() == 5


class TestTradePanel:
    """Test trade_panel.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "specification" / "trade_panel.csv")

    def test_exists(self):
        assert (DATA_DIR / "specification" / "trade_panel.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (4500, 9), f"Expected (4500, 9), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "pair_id",
            "year",
            "log_exports",
            "log_gdp_i",
            "log_gdp_j",
            "log_distance",
            "tariff",
            "border",
            "language",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_n_pairs(self, data):
        assert data["pair_id"].nunique() == 300


# -----------------------------------------------------------------------
# Spatial datasets
# -----------------------------------------------------------------------


class TestUSCounties:
    """Test us_counties.csv and weight matrices."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "spatial" / "us_counties.csv")

    def test_data_exists(self):
        assert (DATA_DIR / "spatial" / "us_counties.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (2000, 8), f"Expected (2000, 8), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "county_id",
            "state",
            "year",
            "unemployment",
            "log_income",
            "log_population",
            "manufacturing_share",
            "education_pct",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_n_counties(self, data):
        assert data["county_id"].nunique() == 200

    def test_unemployment_range(self, data):
        assert data["unemployment"].between(0, 0.5).all()

    def test_w_contiguity_exists(self):
        assert (DATA_DIR / "spatial" / "W_counties.npy").exists()

    def test_w_contiguity_shape(self):
        W = np.load(DATA_DIR / "spatial" / "W_counties.npy")
        assert W.shape == (200, 200)

    def test_w_contiguity_diagonal_zero(self):
        W = np.load(DATA_DIR / "spatial" / "W_counties.npy")
        assert np.allclose(np.diag(W), 0)

    def test_w_contiguity_row_normalized(self):
        W = np.load(DATA_DIR / "spatial" / "W_counties.npy")
        row_sums = W.sum(axis=1)
        nonzero = row_sums > 0
        assert np.allclose(row_sums[nonzero], 1.0, atol=1e-10)

    def test_w_distance_exists(self):
        assert (DATA_DIR / "spatial" / "W_counties_distance.npy").exists()

    def test_w_distance_shape(self):
        W = np.load(DATA_DIR / "spatial" / "W_counties_distance.npy")
        assert W.shape == (200, 200)

    def test_coordinates_exists(self):
        assert (DATA_DIR / "spatial" / "coordinates_us.csv").exists()

    def test_coordinates_shape(self):
        c = pd.read_csv(DATA_DIR / "spatial" / "coordinates_us.csv")
        assert c.shape == (200, 3)
        assert list(c.columns) == ["county_id", "latitude", "longitude"]


class TestEURegions:
    """Test eu_regions.csv and weight matrix."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "spatial" / "eu_regions.csv")

    def test_data_exists(self):
        assert (DATA_DIR / "spatial" / "eu_regions.csv").exists()

    def test_dimensions(self, data):
        assert data.shape == (1500, 8), f"Expected (1500, 8), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "region_id",
            "country",
            "year",
            "gdp_per_capita",
            "log_gdp_pc",
            "fdi",
            "rd_expenditure",
            "infrastructure",
        ]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_n_regions(self, data):
        assert data["region_id"].nunique() == 100

    def test_w_eu_exists(self):
        assert (DATA_DIR / "spatial" / "W_eu_contiguity.npy").exists()

    def test_w_eu_shape(self):
        W = np.load(DATA_DIR / "spatial" / "W_eu_contiguity.npy")
        assert W.shape == (100, 100)

    def test_w_eu_diagonal_zero(self):
        W = np.load(DATA_DIR / "spatial" / "W_eu_contiguity.npy")
        assert np.allclose(np.diag(W), 0)

    def test_coordinates_eu_exists(self):
        assert (DATA_DIR / "spatial" / "coordinates_eu.csv").exists()

    def test_coordinates_eu_shape(self):
        c = pd.read_csv(DATA_DIR / "spatial" / "coordinates_eu.csv")
        assert c.shape == (100, 3)
        assert list(c.columns) == ["region_id", "latitude", "longitude"]
