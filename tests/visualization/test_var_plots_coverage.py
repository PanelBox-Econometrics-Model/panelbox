"""Coverage tests for panelbox.visualization.var_plots.

Focuses on uncovered branches: unstable eigenvalues, invalid backends,
sensitivity with many coefficients, IRF impulse/response filtering,
FEVD bar charts, theme validation, and plotly backends.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from panelbox.visualization.var_plots import (
    _get_theme_config,
    plot_fevd,
    plot_instrument_sensitivity,
    plot_irf,
    plot_stability,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockIRFResult:
    """Mock IRFResult with configurable CI and var_names."""

    def __init__(self, n_vars=2, periods=10, has_ci=True):
        np.random.seed(42)
        self.var_names = [f"var{i + 1}" for i in range(n_vars)]
        self.periods = periods
        self.K = n_vars
        self.method = "cholesky"
        self.irf_matrix = np.random.randn(periods + 1, n_vars, n_vars) * 0.1

        if has_ci:
            self.ci_lower = self.irf_matrix - 0.5
            self.ci_upper = self.irf_matrix + 0.5
            self.ci_level = 0.95
        else:
            self.ci_lower = None
            self.ci_upper = None
            self.ci_level = None

    def __getitem__(self, key):
        resp, imp = key
        resp_idx = self.var_names.index(resp)
        imp_idx = self.var_names.index(imp)
        return self.irf_matrix[:, resp_idx, imp_idx]


class MockFEVDResult:
    """Mock FEVDResult with valid decomposition summing to 1."""

    def __init__(self, n_vars=2, periods=10):
        np.random.seed(42)
        raw = np.random.dirichlet([1] * n_vars, size=(periods + 1, n_vars))
        self.decomposition = raw.reshape(periods + 1, n_vars, n_vars)
        self.var_names = [f"var{i + 1}" for i in range(n_vars)]
        self.periods = periods
        self.K = n_vars
        self.method = "cholesky"
        self.ci_lower = None
        self.ci_upper = None
        self.ci_level = None


# ---------------------------------------------------------------------------
# Theme config tests
# ---------------------------------------------------------------------------


def test_get_theme_config_all_themes():
    """Test _get_theme_config returns valid dict for all valid themes."""
    for theme in ["academic", "professional", "presentation"]:
        config = _get_theme_config(theme)
        assert isinstance(config, dict)
        assert "fontsize" in config
        assert "linewidth" in config
        assert "line_color" in config
        assert "ci_color" in config
        assert "grid" in config


def test_get_theme_config_invalid():
    """Test _get_theme_config raises ValueError for unknown theme."""
    with pytest.raises(ValueError, match="Unknown theme"):
        _get_theme_config("invalid_theme")


def test_get_theme_config_professional_no_grid():
    """Test professional theme has grid=False."""
    config = _get_theme_config("professional")
    assert config["grid"] is False


# ---------------------------------------------------------------------------
# Stability plot tests
# ---------------------------------------------------------------------------


def test_plot_stability_matplotlib_stable():
    """Test stability plot with all stable eigenvalues."""
    eigenvalues = np.array([0.3 + 0.2j, 0.3 - 0.2j, 0.1, -0.4])
    fig = plot_stability(eigenvalues, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_stability_matplotlib_unstable():
    """Test stability plot with some unstable eigenvalues (modulus >= 1)."""
    eigenvalues = np.array([0.5 + 0.3j, 0.5 - 0.3j, 1.2, -0.1])
    fig = plot_stability(eigenvalues, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_stability_matplotlib_all_unstable():
    """Test stability plot when all eigenvalues are unstable."""
    eigenvalues = np.array([1.1 + 0.5j, 1.1 - 0.5j, 1.5])
    fig = plot_stability(eigenvalues, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_stability_plotly_stable():
    """Test stability plotly with all stable eigenvalues."""
    eigenvalues = np.array([0.3 + 0.2j, 0.3 - 0.2j, 0.1])
    fig = plot_stability(eigenvalues, backend="plotly", show=False)
    assert fig is not None
    assert hasattr(fig, "data")


def test_plot_stability_plotly_unstable():
    """Test stability plotly with unstable eigenvalues."""
    eigenvalues = np.array([0.5, 1.2 + 0.3j, 1.2 - 0.3j])
    fig = plot_stability(eigenvalues, backend="plotly", show=False)
    assert fig is not None


def test_plot_stability_invalid_backend():
    """Test stability plot raises ValueError for invalid backend."""
    eigenvalues = np.array([0.5, 0.3])
    with pytest.raises(ValueError, match="backend must be"):
        plot_stability(eigenvalues, backend="invalid", show=False)


def test_plot_stability_custom_title():
    """Test stability plot with custom title."""
    eigenvalues = np.array([0.5, 0.3])
    fig = plot_stability(eigenvalues, title="Custom Title", backend="matplotlib", show=False)
    assert fig is not None


# ---------------------------------------------------------------------------
# Sensitivity plot tests
# ---------------------------------------------------------------------------


def _make_sensitivity_results(n_coefs=3, with_changes=True, interpretation=""):
    """Build a mock sensitivity_results dict."""
    data = {
        "coefficients": {
            f"coef_{i}": [0.5 + 0.01 * i * j for j in range(4)] for i in range(n_coefs)
        },
        "n_instruments_actual": [10, 15, 20, 25],
    }
    if with_changes:
        data["coefficient_changes"] = {f"coef_{i}": float(i * 5.0) for i in range(n_coefs)}
    if interpretation:
        data["interpretation"] = interpretation
    return data


def test_plot_sensitivity_matplotlib_basic():
    """Test sensitivity matplotlib with few coefficients."""
    data = _make_sensitivity_results(n_coefs=3)
    fig = plot_instrument_sensitivity(data, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_sensitivity_matplotlib_many_coefs():
    """Test sensitivity matplotlib with more coefs than max_coefs_to_plot."""
    data = _make_sensitivity_results(n_coefs=10, with_changes=True)
    fig = plot_instrument_sensitivity(data, backend="matplotlib", show=False, max_coefs_to_plot=4)
    assert fig is not None


def test_plot_sensitivity_matplotlib_many_coefs_no_changes():
    """Test sensitivity matplotlib truncation without coefficient_changes."""
    data = _make_sensitivity_results(n_coefs=10, with_changes=False)
    fig = plot_instrument_sensitivity(data, backend="matplotlib", show=False, max_coefs_to_plot=4)
    assert fig is not None


def test_plot_sensitivity_matplotlib_with_interpretation():
    """Test sensitivity matplotlib with interpretation text."""
    data = _make_sensitivity_results(
        n_coefs=3, interpretation="Coefficients are stable across specs."
    )
    fig = plot_instrument_sensitivity(data, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_sensitivity_plotly_basic():
    """Test sensitivity plotly basic."""
    data = _make_sensitivity_results(n_coefs=3)
    fig = plot_instrument_sensitivity(data, backend="plotly", show=False)
    assert fig is not None
    assert hasattr(fig, "data")


def test_plot_sensitivity_plotly_many_coefs():
    """Test sensitivity plotly with many coefs exceeding max."""
    data = _make_sensitivity_results(n_coefs=10, with_changes=True)
    fig = plot_instrument_sensitivity(data, backend="plotly", show=False, max_coefs_to_plot=4)
    assert fig is not None


def test_plot_sensitivity_plotly_with_interpretation_stable():
    """Test sensitivity plotly with stable interpretation annotation."""
    data = _make_sensitivity_results(n_coefs=3, interpretation="Coefficients are stable.")
    fig = plot_instrument_sensitivity(data, backend="plotly", show=False)
    assert fig is not None


def test_plot_sensitivity_plotly_with_interpretation_unstable():
    """Test sensitivity plotly with unstable interpretation annotation."""
    data = _make_sensitivity_results(
        n_coefs=3, interpretation="Coefficients are VOLATILE across specs."
    )
    fig = plot_instrument_sensitivity(data, backend="plotly", show=False)
    assert fig is not None


def test_plot_sensitivity_invalid_backend():
    """Test sensitivity raises ValueError for invalid backend."""
    data = _make_sensitivity_results(n_coefs=2)
    with pytest.raises(ValueError, match="backend must be"):
        plot_instrument_sensitivity(data, backend="invalid", show=False)


# ---------------------------------------------------------------------------
# IRF plot tests
# ---------------------------------------------------------------------------


def test_plot_irf_matplotlib_full_grid():
    """Test IRF matplotlib full grid (all impulse-response pairs)."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_irf_matplotlib_impulse_only():
    """Test IRF matplotlib filtering by impulse (column of responses)."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, impulse="var1", backend="matplotlib", show=False)
    assert fig is not None


def test_plot_irf_matplotlib_response_only():
    """Test IRF matplotlib filtering by response (row of impulses)."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, response="var1", backend="matplotlib", show=False)
    assert fig is not None


def test_plot_irf_matplotlib_single_pair():
    """Test IRF matplotlib with both impulse and response specified."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, impulse="var1", response="var2", backend="matplotlib", show=False)
    assert fig is not None


def test_plot_irf_matplotlib_no_ci():
    """Test IRF matplotlib without confidence intervals."""
    irf = MockIRFResult(n_vars=2, has_ci=False)
    fig = plot_irf(irf, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_irf_matplotlib_professional_theme():
    """Test IRF matplotlib with professional theme (grid=False)."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, backend="matplotlib", theme="professional", show=False)
    assert fig is not None


def test_plot_irf_matplotlib_custom_figsize():
    """Test IRF matplotlib with custom figsize."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, backend="matplotlib", figsize=(12, 10), show=False)
    assert fig is not None


def test_plot_irf_matplotlib_variables_subset():
    """Test IRF matplotlib with variables subset."""
    irf = MockIRFResult(n_vars=3, has_ci=True)
    fig = plot_irf(irf, variables=["var1", "var2"], backend="matplotlib", show=False)
    assert fig is not None


def test_plot_irf_plotly_full_grid():
    """Test IRF plotly full grid."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, backend="plotly", show=False)
    assert fig is not None
    assert hasattr(fig, "data")


def test_plot_irf_plotly_impulse_only():
    """Test IRF plotly filtering by impulse."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, impulse="var1", backend="plotly", show=False)
    assert fig is not None


def test_plot_irf_plotly_response_only():
    """Test IRF plotly filtering by response."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, response="var1", backend="plotly", show=False)
    assert fig is not None


def test_plot_irf_plotly_single_pair():
    """Test IRF plotly with both impulse and response specified."""
    irf = MockIRFResult(n_vars=2, has_ci=True)
    fig = plot_irf(irf, impulse="var1", response="var2", backend="plotly", show=False)
    assert fig is not None


def test_plot_irf_plotly_no_ci():
    """Test IRF plotly without confidence intervals."""
    irf = MockIRFResult(n_vars=2, has_ci=False)
    fig = plot_irf(irf, backend="plotly", show=False)
    assert fig is not None


def test_plot_irf_invalid_backend():
    """Test IRF raises ValueError for invalid backend."""
    irf = MockIRFResult(n_vars=2)
    with pytest.raises(ValueError, match="backend must be"):
        plot_irf(irf, backend="invalid", show=False)


# ---------------------------------------------------------------------------
# FEVD plot tests
# ---------------------------------------------------------------------------


def test_plot_fevd_matplotlib_area():
    """Test FEVD matplotlib area chart (default)."""
    fevd = MockFEVDResult(n_vars=2)
    fig = plot_fevd(fevd, kind="area", backend="matplotlib", show=False)
    assert fig is not None


def test_plot_fevd_matplotlib_bar():
    """Test FEVD matplotlib bar chart."""
    fevd = MockFEVDResult(n_vars=2)
    fig = plot_fevd(fevd, kind="bar", backend="matplotlib", show=False)
    assert fig is not None


def test_plot_fevd_matplotlib_bar_custom_horizons():
    """Test FEVD matplotlib bar chart with custom horizons."""
    fevd = MockFEVDResult(n_vars=2, periods=20)
    fig = plot_fevd(
        fevd,
        kind="bar",
        horizons=[1, 5, 10],
        backend="matplotlib",
        show=False,
    )
    assert fig is not None


def test_plot_fevd_matplotlib_variables_subset():
    """Test FEVD matplotlib with variables subset."""
    fevd = MockFEVDResult(n_vars=3)
    fig = plot_fevd(
        fevd,
        variables=["var1"],
        backend="matplotlib",
        show=False,
    )
    assert fig is not None


def test_plot_fevd_matplotlib_professional_theme():
    """Test FEVD matplotlib with professional theme (no grid)."""
    fevd = MockFEVDResult(n_vars=2)
    fig = plot_fevd(fevd, backend="matplotlib", theme="professional", show=False)
    assert fig is not None


def test_plot_fevd_matplotlib_invalid_kind():
    """Test FEVD matplotlib raises ValueError for invalid kind."""
    fevd = MockFEVDResult(n_vars=2)
    with pytest.raises(ValueError, match="kind must be"):
        plot_fevd(fevd, kind="scatter", backend="matplotlib", show=False)


def test_plot_fevd_plotly_area():
    """Test FEVD plotly area chart."""
    fevd = MockFEVDResult(n_vars=2)
    fig = plot_fevd(fevd, kind="area", backend="plotly", show=False)
    assert fig is not None
    assert hasattr(fig, "data")


def test_plot_fevd_plotly_bar():
    """Test FEVD plotly bar chart."""
    fevd = MockFEVDResult(n_vars=2)
    fig = plot_fevd(fevd, kind="bar", backend="plotly", show=False)
    assert fig is not None


def test_plot_fevd_plotly_bar_custom_horizons():
    """Test FEVD plotly bar chart with custom horizons."""
    fevd = MockFEVDResult(n_vars=2, periods=20)
    fig = plot_fevd(
        fevd,
        kind="bar",
        horizons=[1, 5, 10],
        backend="plotly",
        show=False,
    )
    assert fig is not None


def test_plot_fevd_plotly_invalid_kind():
    """Test FEVD plotly raises ValueError for invalid kind."""
    fevd = MockFEVDResult(n_vars=2)
    with pytest.raises(ValueError, match="kind must be"):
        plot_fevd(fevd, kind="scatter", backend="plotly", show=False)


def test_plot_fevd_invalid_backend():
    """Test FEVD raises ValueError for invalid backend."""
    fevd = MockFEVDResult(n_vars=2)
    with pytest.raises(ValueError, match="backend must be"):
        plot_fevd(fevd, backend="invalid", show=False)


def test_plot_fevd_matplotlib_custom_figsize():
    """Test FEVD matplotlib with custom figsize."""
    fevd = MockFEVDResult(n_vars=2)
    fig = plot_fevd(fevd, backend="matplotlib", figsize=(14, 8), show=False)
    assert fig is not None
