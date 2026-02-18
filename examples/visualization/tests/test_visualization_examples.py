"""
Sanity tests for the Visualization tutorial series.

These tests verify that:
1. Data generators produce correctly shaped and typed DataFrames.
2. Key output directories exist.
3. Notebook files are present (basic existence check).

Run with:
    pytest examples/visualization/tests/test_visualization_examples.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow import of utils without installing the package
# ---------------------------------------------------------------------------
VIZ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIZ_ROOT))

from utils.data_generators import (
    generate_autocorrelated_panel,
    generate_heteroskedastic_panel,
    generate_panel_data,
    generate_spatial_panel,
)

# ---------------------------------------------------------------------------
# Data generator tests
# ---------------------------------------------------------------------------


class TestGeneratePanelData:
    """Tests for generate_panel_data()."""

    def test_shape(self):
        df = generate_panel_data(n_individuals=50, n_periods=5, n_covariates=3, seed=0)
        assert df.shape == (250, 4)  # x1, x2, x3, y (entity/time in index)

    def test_index(self):
        df = generate_panel_data(n_individuals=10, n_periods=4, seed=0)
        assert df.index.names == ["entity", "time"]

    def test_columns(self):
        df = generate_panel_data(n_individuals=10, n_periods=4, n_covariates=2, seed=0)
        assert set(df.columns) == {"x1", "x2", "y"}

    def test_no_nans(self):
        df = generate_panel_data(n_individuals=100, n_periods=10, seed=42)
        assert not df.isnull().any().any()

    def test_balanced(self):
        n, t = 30, 7
        df = generate_panel_data(n_individuals=n, n_periods=t, seed=1)
        assert len(df) == n * t

    def test_reproducible(self):
        df1 = generate_panel_data(seed=99)
        df2 = generate_panel_data(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestGenerateHeteroskedasticPanel:
    """Tests for generate_heteroskedastic_panel()."""

    def test_shape(self):
        df = generate_heteroskedastic_panel(n_individuals=50, n_periods=5, seed=0)
        assert df.shape == (250, 4)  # x1, x2, sigma, y

    def test_columns(self):
        df = generate_heteroskedastic_panel(seed=0)
        assert set(df.columns) == {"x1", "x2", "sigma", "y"}

    def test_sigma_positive(self):
        df = generate_heteroskedastic_panel(seed=42)
        assert (df["sigma"] >= 0).all()

    def test_sigma_correlated_with_x1(self):
        """sigma = 0.5 * |x1|, so they should be perfectly correlated."""
        df = generate_heteroskedastic_panel(seed=42)
        np.testing.assert_allclose(df["sigma"], 0.5 * df["x1"].abs(), rtol=1e-10)

    def test_no_nans(self):
        df = generate_heteroskedastic_panel(seed=42)
        assert not df.isnull().any().any()


class TestGenerateAutocorrelatedPanel:
    """Tests for generate_autocorrelated_panel()."""

    def test_shape(self):
        df = generate_autocorrelated_panel(n_individuals=20, n_periods=8, seed=0)
        assert df.shape == (160, 4)  # x1, x2, epsilon, y

    def test_columns(self):
        df = generate_autocorrelated_panel(seed=0)
        assert set(df.columns) == {"x1", "x2", "epsilon", "y"}

    def test_no_nans(self):
        df = generate_autocorrelated_panel(seed=42)
        assert not df.isnull().any().any()

    def test_ar1_structure(self):
        """Within-entity residuals should show significant lag-1 correlation."""
        df = generate_autocorrelated_panel(n_individuals=500, n_periods=20, rho=0.8, seed=42)
        # For each entity compute lag-1 autocorrelation of epsilon
        correlations = df["epsilon"].groupby(level="entity").apply(lambda s: s.autocorr(lag=1))
        # Mean autocorrelation should be close to rho=0.8
        assert correlations.mean() > 0.5, "Expected positive autocorrelation"

    def test_rho_zero_uncorrelated(self):
        """With rho=0, errors should be approximately uncorrelated."""
        df = generate_autocorrelated_panel(n_individuals=200, n_periods=20, rho=0.0, seed=0)
        correlations = df["epsilon"].groupby(level="entity").apply(lambda s: s.autocorr(lag=1))
        assert abs(correlations.mean()) < 0.15, "Expected near-zero autocorrelation"


class TestGenerateSpatialPanel:
    """Tests for generate_spatial_panel()."""

    def test_shape(self):
        df = generate_spatial_panel(n_individuals=30, n_periods=6, seed=0)
        assert df.shape == (180, 4)  # x1, f_t, loading, y

    def test_columns(self):
        df = generate_spatial_panel(seed=0)
        assert set(df.columns) == {"x1", "f_t", "loading", "y"}

    def test_no_nans(self):
        df = generate_spatial_panel(seed=42)
        assert not df.isnull().any().any()

    def test_common_factor_constant_within_time(self):
        """All entities share the same f_t for each time period."""
        df = generate_spatial_panel(n_individuals=50, n_periods=8, seed=42)
        # For each time period, f_t should be identical across entities
        f_by_time = df.groupby(level="time")["f_t"].nunique()
        assert (f_by_time == 1).all(), "f_t should be constant within each time period"


# ---------------------------------------------------------------------------
# Directory structure tests
# ---------------------------------------------------------------------------


class TestDirectoryStructure:
    """Verify the visualization directory structure is in place."""

    @pytest.fixture(autouse=True)
    def viz_root(self):
        self.root = VIZ_ROOT

    def _check(self, *parts):
        p = self.root.joinpath(*parts)
        assert p.exists(), f"Expected path does not exist: {p}"

    def test_notebooks_dir(self):
        self._check("notebooks")

    def test_data_dir(self):
        self._check("data")

    def test_data_gitkeep(self):
        self._check("data", ".gitkeep")

    def test_outputs_charts_png(self):
        self._check("outputs", "charts", "png")

    def test_outputs_charts_svg(self):
        self._check("outputs", "charts", "svg")

    def test_outputs_charts_pdf(self):
        self._check("outputs", "charts", "pdf")

    def test_outputs_reports_html(self):
        self._check("outputs", "reports", "html")

    def test_outputs_reports_latex(self):
        self._check("outputs", "reports", "latex")

    def test_solutions_dir(self):
        self._check("solutions")

    def test_utils_init(self):
        self._check("utils", "__init__.py")

    def test_utils_data_generators(self):
        self._check("utils", "data_generators.py")

    def test_readme(self):
        self._check("README.md")

    def test_getting_started(self):
        self._check("GETTING_STARTED.md")


# ---------------------------------------------------------------------------
# Notebook existence tests
# ---------------------------------------------------------------------------


EXPECTED_NOTEBOOKS = [
    "01_visualization_introduction.ipynb",
    "02_visual_diagnostics.ipynb",
    "03_advanced_visualizations.ipynb",
    "04_automated_reports.ipynb",
]

EXPECTED_SOLUTIONS = [
    "01_visualization_introduction_solution.ipynb",
    "02_visual_diagnostics_solution.ipynb",
    "03_advanced_visualizations_solution.ipynb",
    "04_automated_reports_solution.ipynb",
]


@pytest.mark.parametrize("notebook", EXPECTED_NOTEBOOKS)
def test_notebook_exists(notebook):
    """Each tutorial notebook file must exist."""
    p = VIZ_ROOT / "notebooks" / notebook
    assert p.exists(), f"Missing notebook: {p}"


@pytest.mark.parametrize("solution", EXPECTED_SOLUTIONS)
def test_solution_exists(solution):
    """Each solution notebook file must exist."""
    p = VIZ_ROOT / "solutions" / solution
    assert p.exists(), f"Missing solution notebook: {p}"
