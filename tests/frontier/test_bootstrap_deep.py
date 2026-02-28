"""Deep coverage tests for panelbox.frontier.bootstrap module.

Note: bootstrap.py is already at 100% coverage from test_bootstrap_coverage.py.
This file adds additional integration tests for robustness.
"""

import numpy as np
import pandas as pd

from panelbox.frontier import StochasticFrontier
from panelbox.frontier.bootstrap import SFABootstrap


def _make_fitted_model(seed=42):
    """Create a simple fitted SFA model for bootstrap testing."""
    np.random.seed(seed)
    n = 150
    x = np.random.normal(0, 1, n)
    v = np.random.normal(0, 0.1, n)
    u = np.abs(np.random.normal(0, 0.2, n))
    y = 1.0 + 0.5 * x + v - u

    df = pd.DataFrame({"y": y, "x": x})
    sf = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        frontier="production",
        dist="half_normal",
    )
    return sf.fit(method="mle", verbose=False)


class TestBootstrapDeep:
    """Additional integration tests for bootstrap module."""

    def test_parametric_bootstrap_runs(self):
        """Test parametric bootstrap with minimum n_boot."""
        result = _make_fitted_model()
        boot = SFABootstrap(result, n_boot=100, method="parametric", seed=42)
        params_result = boot.bootstrap_parameters()

        assert params_result is not None
        assert "mean_boot" in params_result
        assert "ci_lower" in params_result
        assert "ci_upper" in params_result

    def test_pairs_bootstrap_runs(self):
        """Test pairs bootstrap with minimum n_boot."""
        result = _make_fitted_model()
        boot = SFABootstrap(result, n_boot=100, method="pairs", seed=42)
        params_result = boot.bootstrap_parameters()

        assert params_result is not None
        assert "mean_boot" in params_result

    def test_bootstrap_ci_level(self):
        """Test bootstrap with custom confidence level."""
        result = _make_fitted_model()
        boot = SFABootstrap(result, n_boot=100, ci_level=0.90, seed=42)
        params_result = boot.bootstrap_parameters()

        assert params_result is not None
        assert "ci_lower" in params_result

    def test_bootstrap_efficiency(self):
        """Test bootstrap efficiency estimation."""
        result = _make_fitted_model()
        boot = SFABootstrap(result, n_boot=100, seed=42)
        eff_result = boot.bootstrap_efficiency()

        assert eff_result is not None
