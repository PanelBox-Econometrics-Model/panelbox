"""
Test data integrity for count models tutorials.

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

# Base path
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


class TestDatasetExistence:
    """Test that all required datasets exist."""

    def test_healthcare_visits_exists(self):
        assert (DATA_DIR / "healthcare_visits.csv").exists()

    def test_firm_patents_exists(self):
        assert (DATA_DIR / "firm_patents.csv").exists()

    def test_city_crime_exists(self):
        assert (DATA_DIR / "city_crime.csv").exists()

    def test_bilateral_trade_exists(self):
        assert (DATA_DIR / "bilateral_trade.csv").exists()

    def test_healthcare_zinb_exists(self):
        assert (DATA_DIR / "healthcare_zinb.csv").exists()

    def test_policy_impact_exists(self):
        assert (DATA_DIR / "policy_impact.csv").exists()

    def test_firm_innovation_exists(self):
        assert (DATA_DIR / "firm_innovation_full.csv").exists()


class TestHealthcareVisits:
    """Test healthcare_visits.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "healthcare_visits.csv")

    def test_dimensions(self, data):
        assert data.shape == (2000, 6), f"Expected (2000, 6), got {data.shape}"

    def test_columns(self, data):
        expected = ["individual_id", "visits", "age", "income", "insurance", "chronic"]
        assert list(data.columns) == expected

    def test_no_missing(self, data):
        assert data.isnull().sum().sum() == 0

    def test_visits_range(self, data):
        assert data["visits"].min() >= 0
        assert data["visits"].max() <= 25

    def test_age_range(self, data):
        assert data["age"].min() >= 18
        assert data["age"].max() <= 85

    def test_binary_variables(self, data):
        assert set(data["insurance"].unique()) <= {0, 1}
        assert set(data["chronic"].unique()) <= {0, 1}


class TestFirmPatents:
    """Test firm_patents.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "firm_patents.csv")

    def test_dimensions(self, data):
        assert data.shape == (7500, 7), f"Expected (7500, 7), got {data.shape}"

    def test_columns(self, data):
        expected = [
            "firm_id",
            "year",
            "patents",
            "rd_expenditure",
            "firm_size",
            "industry",
            "region",
        ]
        assert list(data.columns) == expected

    def test_panel_structure(self, data):
        # Check balanced panel
        firms_per_year = data.groupby("year")["firm_id"].nunique()
        assert (firms_per_year == 1500).all()

    def test_years(self, data):
        assert set(data["year"].unique()) == set(range(2015, 2020))

    def test_patents_nonnegative(self, data):
        assert (data["patents"] >= 0).all()


class TestCityCrime:
    """Test city_crime.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "city_crime.csv")

    def test_dimensions(self, data):
        assert data.shape == (1500, 8), f"Expected (1500, 8), got {data.shape}"

    def test_balanced_panel(self, data):
        # 150 cities × 10 years = 1500
        n_cities = data["city_id"].nunique()
        n_years = data["year"].nunique()
        assert n_cities == 150
        assert n_years == 10
        assert n_cities * n_years == 1500

    def test_crime_positive(self, data):
        assert (data["crime_count"] > 0).all()


class TestBilateralTrade:
    """Test bilateral_trade.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "bilateral_trade.csv")

    def test_dimensions_approx(self, data):
        # Should be around 10,000 observations
        assert 9000 <= data.shape[0] <= 11000

    def test_columns(self, data):
        expected = [
            "exporter",
            "importer",
            "year",
            "trade_value",
            "distance",
            "contiguous",
            "common_language",
            "gdp_exporter",
            "gdp_importer",
            "trade_agreement",
        ]
        assert list(data.columns) == expected

    def test_zeros_present(self, data):
        # Should have around 23% zeros
        zero_pct = (data["trade_value"] == 0).mean()
        assert 0.15 <= zero_pct <= 0.35, f"Zero prevalence {zero_pct:.2%} outside expected range"

    def test_trade_nonnegative(self, data):
        assert (data["trade_value"] >= 0).all()


class TestHealthcareZINB:
    """Test healthcare_zinb.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "healthcare_zinb.csv")

    def test_dimensions(self, data):
        assert data.shape == (3000, 8), f"Expected (3000, 8), got {data.shape}"

    def test_high_zero_prevalence(self, data):
        # Should have around 60% zeros
        zero_pct = (data["visits"] == 0).mean()
        assert 0.50 <= zero_pct <= 0.70, f"Zero prevalence {zero_pct:.2%} outside expected range"


class TestPolicyImpact:
    """Test policy_impact.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "policy_impact.csv")

    def test_dimensions(self, data):
        assert data.shape == (1200, 8), f"Expected (1200, 8), got {data.shape}"

    def test_treatment_binary(self, data):
        assert set(data["treatment"].unique()) <= {0, 1}

    def test_treatment_prevalence(self, data):
        # Should be around 40% treated
        treated_pct = data["treatment"].mean()
        assert 0.30 <= treated_pct <= 0.50


class TestFirmInnovation:
    """Test firm_innovation_full.csv dataset."""

    @pytest.fixture
    def data(self):
        return pd.read_csv(DATA_DIR / "firm_innovation_full.csv")

    def test_dimensions(self, data):
        assert data.shape == (4000, 11), f"Expected (4000, 11), got {data.shape}"

    def test_panel_structure(self, data):
        # 500 firms × 8 years = 4000
        n_firms = data["firm_id"].nunique()
        n_years = data["year"].nunique()
        assert n_firms == 500
        assert n_years == 8

    def test_subsidy_binary(self, data):
        assert set(data["subsidy"].unique()) <= {0, 1}


class TestCodebooks:
    """Test that codebooks exist."""

    def test_healthcare_codebook_exists(self):
        assert (DATA_DIR / "codebooks" / "healthcare_visits_codebook.txt").exists()

    def test_codebooks_directory_exists(self):
        assert (DATA_DIR / "codebooks").exists()
        assert (DATA_DIR / "codebooks").is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
