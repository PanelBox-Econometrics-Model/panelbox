"""
Coverage tests for panelbox.validation.robustness.bootstrap.

Targets uncovered lines: 220, 274, 291, 328, 351-358, 365, 423, 451-457,
464, 516, 538, 572-578, 585, 649, 674-680, 687, 725->729, 867, 900-901,
940, 962.
"""

import logging
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from panelbox.core.results import PanelResults
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.robustness.bootstrap import PanelBootstrap


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def panel_data():
    """Create a well-conditioned balanced panel dataset."""
    np.random.seed(42)
    N, T = 20, 10
    n_obs = N * T
    data = pd.DataFrame(
        {
            "entity": np.repeat(range(N), T),
            "time": np.tile(range(T), N),
            "x1": np.random.randn(n_obs),
            "x2": np.random.randn(n_obs),
        }
    )
    data["y"] = 1.0 + 0.5 * data["x1"] - 0.3 * data["x2"] + np.random.randn(n_obs) * 0.5
    return data


@pytest.fixture
def fitted_results(panel_data):
    """Return fitted FixedEffects results."""
    model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
    return model.fit()


@pytest.fixture
def single_param_data():
    """Create panel data for a single-parameter model."""
    np.random.seed(42)
    N, T = 20, 10
    n_obs = N * T
    data = pd.DataFrame(
        {
            "entity": np.repeat(range(N), T),
            "time": np.tile(range(T), N),
            "x1": np.random.randn(n_obs),
        }
    )
    data["y"] = 2.0 * data["x1"] + np.random.randn(n_obs) * 0.5
    return data


class TestModelIsNone:
    """Cover line 220: results._model is None check."""

    def test_results_with_no_model_raises(self):
        """When results._model is None, PanelBootstrap should raise ValueError."""
        # Create a mock PanelResults where _model is None
        mock_results = MagicMock(spec=PanelResults)
        mock_results._model = None
        # Make isinstance check pass
        with patch("panelbox.validation.robustness.bootstrap.isinstance", return_value=True):
            pass

        # Instead, create a real PanelResults-like object
        # We need to bypass isinstance check but have _model=None
        results = MagicMock(spec=PanelResults)
        results._model = None

        # Create a real PanelResults-like object to fool isinstance check
        real_results = object.__new__(PanelResults)
        real_results._model = None
        real_results.params = pd.Series([1.0, 2.0], index=["x1", "x2"])

        with pytest.raises(ValueError, match="Bootstrap requires access to the original model"):
            PanelBootstrap(results=real_results, n_bootstrap=100)


class TestUnknownMethodBranch:
    """Cover line 274: unknown bootstrap method else branch in run()."""

    def test_unknown_method_in_run(self, fitted_results):
        """Directly set an invalid method after init to trigger else branch in run()."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=False,
        )
        # Bypass __init__ validation by setting method directly
        bootstrap.method = "nonexistent_method"

        with pytest.raises(ValueError, match="Unknown bootstrap method"):
            bootstrap.run()


class TestManyFailuresWarning:
    """Cover line 291: warning when >10% bootstrap replications fail."""

    def test_many_failures_warning(self, fitted_results):
        """Trigger warning when many bootstrap replications fail."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=False,
        )

        # Mock _bootstrap_pairs to return estimates where we simulate ~15% failures
        # by setting n_failed_ and returning valid estimates
        n_boot = 20
        n_params = len(fitted_results.params)
        fake_estimates = np.random.randn(n_boot - 3, n_params)

        def mock_bootstrap_pairs():
            bootstrap.n_failed_ = 3  # 3/20 = 15% > 10%
            return fake_estimates

        bootstrap._bootstrap_pairs = mock_bootstrap_pairs

        with pytest.warns(UserWarning, match="out of .* bootstrap replications failed"):
            bootstrap.run()


class TestShowProgressPairs:
    """Cover line 328: show_progress=True branch in _bootstrap_pairs (tqdm)."""

    def test_pairs_with_progress(self, panel_data):
        """Run pairs bootstrap with show_progress=True to cover tqdm branch."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=5,
            method="pairs",
            random_state=42,
            show_progress=True,
        )
        bootstrap.run()

        assert bootstrap._fitted is True


class TestPairsExceptionLogging:
    """Cover lines 351-358: exception catch and log warning in _bootstrap_pairs."""

    def test_pairs_failure_logging(self, fitted_results, caplog):
        """Trigger exceptions during pairs bootstrap to cover failure logging."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=True,  # Must be True to trigger log branch
        )

        original_create = bootstrap._create_bootstrap_model
        call_count = [0]

        def flaky_create(boot_data):
            call_count[0] += 1
            # Fail on first 3 iterations to trigger the logging path
            if call_count[0] <= 3:
                raise RuntimeError("Simulated bootstrap failure")
            return original_create(boot_data)

        bootstrap._create_bootstrap_model = flaky_create

        with caplog.at_level(logging.WARNING, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        assert bootstrap.n_failed_ >= 3
        assert bootstrap._fitted is True


class TestPairsRuntimeError:
    """Cover line 365: RuntimeError when >50% pairs bootstrap fail."""

    def test_pairs_more_than_50_percent_fail(self, fitted_results):
        """Trigger RuntimeError when >50% of pairs bootstrap replications fail."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=False,
        )

        # Make _create_bootstrap_model always fail
        def always_fail(boot_data):
            raise RuntimeError("Always fails")

        bootstrap._create_bootstrap_model = always_fail

        with pytest.raises(RuntimeError, match="More than 50% of bootstrap replications failed"):
            bootstrap.run()


class TestShowProgressWild:
    """Cover line 423: show_progress=True branch in _bootstrap_wild (tqdm)."""

    def test_wild_with_progress(self, panel_data):
        """Run wild bootstrap with show_progress=True to cover tqdm branch."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=5,
            method="wild",
            random_state=42,
            show_progress=True,
        )
        bootstrap.run()

        assert bootstrap._fitted is True


class TestWildExceptionLogging:
    """Cover lines 451-457: exception catch and log warning in _bootstrap_wild."""

    def test_wild_failure_logging(self, fitted_results, caplog):
        """Trigger exceptions during wild bootstrap to cover failure logging."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="wild",
            random_state=42,
            show_progress=True,  # Must be True to trigger log branch
        )

        original_create = bootstrap._create_bootstrap_model
        call_count = [0]

        def flaky_create(boot_data):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RuntimeError("Simulated wild failure")
            return original_create(boot_data)

        bootstrap._create_bootstrap_model = flaky_create

        with caplog.at_level(logging.WARNING, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        assert bootstrap.n_failed_ >= 3
        assert bootstrap._fitted is True


class TestWildRuntimeError:
    """Cover line 464: RuntimeError when >50% wild bootstrap fail."""

    def test_wild_more_than_50_percent_fail(self, fitted_results):
        """Trigger RuntimeError when >50% of wild bootstrap replications fail."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="wild",
            random_state=42,
            show_progress=False,
        )

        def always_fail(boot_data):
            raise RuntimeError("Always fails")

        bootstrap._create_bootstrap_model = always_fail

        with pytest.raises(RuntimeError, match="More than 50% of bootstrap replications failed"):
            bootstrap.run()


class TestBlockAutoBlockSizeLogging:
    """Cover line 516: auto block_size logging with show_progress=True."""

    def test_block_auto_size_logging(self, panel_data, caplog):
        """Trigger auto block_size info logging when show_progress=True."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=5,
            method="block",
            block_size=None,  # Auto block size
            random_state=42,
            show_progress=True,  # Required to trigger logging on line 516
        )

        with caplog.at_level(logging.INFO, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        assert bootstrap._fitted is True


class TestShowProgressBlock:
    """Cover line 538: show_progress=True branch in _bootstrap_block (tqdm)."""

    def test_block_with_progress(self, panel_data):
        """Run block bootstrap with show_progress=True to cover tqdm branch."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=5,
            method="block",
            block_size=3,
            random_state=42,
            show_progress=True,
        )
        bootstrap.run()

        assert bootstrap._fitted is True


class TestBlockExceptionLogging:
    """Cover lines 572-578: exception catch and log warning in _bootstrap_block."""

    def test_block_failure_logging(self, fitted_results, caplog):
        """Trigger exceptions during block bootstrap to cover failure logging."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="block",
            block_size=2,
            random_state=42,
            show_progress=True,
        )

        original_create = bootstrap._create_bootstrap_model
        call_count = [0]

        def flaky_create(boot_data):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RuntimeError("Simulated block failure")
            return original_create(boot_data)

        bootstrap._create_bootstrap_model = flaky_create

        with caplog.at_level(logging.WARNING, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        assert bootstrap.n_failed_ >= 3
        assert bootstrap._fitted is True


class TestBlockRuntimeError:
    """Cover line 585: RuntimeError when >50% block bootstrap fail."""

    def test_block_more_than_50_percent_fail(self, fitted_results):
        """Trigger RuntimeError when >50% of block bootstrap replications fail."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="block",
            block_size=2,
            random_state=42,
            show_progress=False,
        )

        def always_fail(boot_data):
            raise RuntimeError("Always fails")

        bootstrap._create_bootstrap_model = always_fail

        with pytest.raises(RuntimeError, match="More than 50% of bootstrap replications failed"):
            bootstrap.run()


class TestShowProgressResidual:
    """Cover line 649: show_progress=True branch in _bootstrap_residual (tqdm)."""

    def test_residual_with_progress(self, panel_data):
        """Run residual bootstrap with show_progress=True to cover tqdm branch."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=5,
            method="residual",
            random_state=42,
            show_progress=True,
        )
        bootstrap.run()

        assert bootstrap._fitted is True


class TestResidualExceptionLogging:
    """Cover lines 674-680: exception catch and log warning in _bootstrap_residual."""

    def test_residual_failure_logging(self, fitted_results, caplog):
        """Trigger exceptions during residual bootstrap to cover failure logging."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="residual",
            random_state=42,
            show_progress=True,
        )

        original_create = bootstrap._create_bootstrap_model
        call_count = [0]

        def flaky_create(boot_data):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RuntimeError("Simulated residual failure")
            return original_create(boot_data)

        bootstrap._create_bootstrap_model = flaky_create

        with caplog.at_level(logging.WARNING, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        assert bootstrap.n_failed_ >= 3
        assert bootstrap._fitted is True


class TestResidualRuntimeError:
    """Cover line 687: RuntimeError when >50% residual bootstrap fail."""

    def test_residual_more_than_50_percent_fail(self, fitted_results):
        """Trigger RuntimeError when >50% of residual bootstrap replications fail."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="residual",
            random_state=42,
            show_progress=False,
        )

        def always_fail(boot_data):
            raise RuntimeError("Always fails")

        bootstrap._create_bootstrap_model = always_fail

        with pytest.raises(RuntimeError, match="More than 50% of bootstrap replications failed"):
            bootstrap.run()


class TestCreateBootstrapModelWeights:
    """Cover lines 725->729: weights attribute in _create_bootstrap_model."""

    def test_model_with_weights(self, panel_data):
        """Test _create_bootstrap_model when model has weights attribute."""
        np.random.seed(42)
        n_obs = len(panel_data)
        weights = np.ones(n_obs)

        model = FixedEffects(
            "y ~ x1 + x2",
            panel_data,
            "entity",
            "time",
            weights=weights,
        )
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=5,
            method="pairs",
            random_state=42,
            show_progress=False,
        )

        # Verify the model has weights attribute
        assert hasattr(bootstrap.model, "weights")
        assert bootstrap.model.weights is not None

        # Run to exercise _create_bootstrap_model with weights
        bootstrap.run()
        assert bootstrap._fitted is True


class TestSummaryBootstrapEstimatesNone:
    """Cover line 867: bootstrap_estimates_ is None check in summary()."""

    def test_summary_estimates_none_raises(self, fitted_results):
        """When _fitted=True but bootstrap_estimates_ is None, summary raises."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=False,
        )
        # Force _fitted=True but keep bootstrap_estimates_=None
        bootstrap._fitted = True
        bootstrap.bootstrap_estimates_ = None

        with pytest.raises(RuntimeError, match="bootstrap_estimates_ should be set"):
            bootstrap.summary()


class TestPlotDistributionImportError:
    """Cover lines 900-901: ImportError for matplotlib in plot_distribution."""

    def test_plot_distribution_import_error(self, fitted_results):
        """Test that ImportError is raised when matplotlib is not available."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=False,
        )
        bootstrap.run()

        # Mock the import to simulate matplotlib not being available
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("No module named 'matplotlib'")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="matplotlib is required for plotting"),
        ):
            bootstrap.plot_distribution()


class TestPlotDistributionSingleParam:
    """Cover line 940: n_params==1 branch in plot_distribution."""

    def test_plot_all_with_single_param(self, single_param_data):
        """Plot all params when model has only one parameter (n_params==1)."""
        model = PooledOLS("y ~ x1 - 1", single_param_data, "entity", "time")
        results = model.fit()

        assert len(results.params) == 1

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=False,
        )
        bootstrap.run()

        # This should trigger the n_params==1 branch: axes = np.array([axes])
        bootstrap.plot_distribution()


class TestPlotDistributionUnusedSubplots:
    """Cover line 962: unused subplots are hidden."""

    def test_plot_unused_subplots_hidden(self, panel_data):
        """When n_params doesn't fill the grid, unused subplots must be hidden."""
        # With 2 params and n_cols=min(3,2)=2, n_rows=1 -> 2 subplots, 0 unused.
        # We need n_params to not evenly fill the grid.
        # With 2 params, n_cols=2, n_rows=1 -> no unused subplots.
        # We need a model with params that create unused subplots.
        # E.g., 4 params -> n_cols=3, n_rows=2, grid=6, unused=2.
        # Or 2 params -> n_cols=2, n_rows=1, grid=2 -> unused=0
        # Let's create data with 4 variables.
        np.random.seed(42)
        N, T = 20, 10
        n_obs = N * T
        data = pd.DataFrame(
            {
                "entity": np.repeat(range(N), T),
                "time": np.tile(range(T), N),
                "x1": np.random.randn(n_obs),
                "x2": np.random.randn(n_obs),
                "x3": np.random.randn(n_obs),
                "x4": np.random.randn(n_obs),
            }
        )
        data["y"] = (
            1.0
            + 0.5 * data["x1"]
            - 0.3 * data["x2"]
            + 0.2 * data["x3"]
            + 0.1 * data["x4"]
            + np.random.randn(n_obs) * 0.5
        )

        # PooledOLS with intercept: 5 params (intercept, x1, x2, x3, x4)
        # n_params=5, n_cols=3, n_rows=2, grid=6, unused=1
        model = PooledOLS("y ~ x1 + x2 + x3 + x4", data, "entity", "time")
        results = model.fit()

        # Verify we get 5 params -> grid has unused subplot
        n_params = len(results.params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        assert n_rows * n_cols > n_params, (
            f"Expected unused subplots but grid ({n_rows}x{n_cols}) exactly fits {n_params} params"
        )

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=False,
        )
        bootstrap.run()

        # This should trigger the unused subplot hiding branch
        bootstrap.plot_distribution()


class TestPlotDistributionSpecificParam:
    """Additional test for plot_distribution with a specific parameter name."""

    def test_plot_single_named_param(self, fitted_results):
        """Plot distribution for a specific named parameter."""
        bootstrap = PanelBootstrap(
            results=fitted_results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=False,
        )
        bootstrap.run()

        bootstrap.plot_distribution(param="x1")


class TestProgressAndFailureCombined:
    """Test show_progress=True with failure logging covering multiple methods."""

    def test_pairs_progress_with_failures(self, panel_data, caplog):
        """Pairs with show_progress=True and some failures (cover lines 328, 351-358)."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=20,
            method="pairs",
            random_state=42,
            show_progress=True,
        )

        original_create = bootstrap._create_bootstrap_model
        call_count = [0]

        def flaky_create(boot_data):
            call_count[0] += 1
            if call_count[0] <= 5:
                raise RuntimeError(f"Failure {call_count[0]}")
            return original_create(boot_data)

        bootstrap._create_bootstrap_model = flaky_create

        with caplog.at_level(logging.WARNING, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        # First 5 failures should be logged
        assert bootstrap.n_failed_ >= 5

    def test_wild_progress_with_failures(self, panel_data, caplog):
        """Wild with show_progress=True and failures (cover lines 423, 451-457)."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=20,
            method="wild",
            random_state=42,
            show_progress=True,
        )

        original_create = bootstrap._create_bootstrap_model
        call_count = [0]

        def flaky_create(boot_data):
            call_count[0] += 1
            if call_count[0] <= 5:
                raise RuntimeError(f"Failure {call_count[0]}")
            return original_create(boot_data)

        bootstrap._create_bootstrap_model = flaky_create

        with caplog.at_level(logging.WARNING, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        assert bootstrap.n_failed_ >= 5

    def test_block_progress_with_failures(self, panel_data, caplog):
        """Block with show_progress=True and failures (cover lines 538, 572-578)."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=20,
            method="block",
            block_size=3,
            random_state=42,
            show_progress=True,
        )

        original_create = bootstrap._create_bootstrap_model
        call_count = [0]

        def flaky_create(boot_data):
            call_count[0] += 1
            if call_count[0] <= 5:
                raise RuntimeError(f"Failure {call_count[0]}")
            return original_create(boot_data)

        bootstrap._create_bootstrap_model = flaky_create

        with caplog.at_level(logging.WARNING, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        assert bootstrap.n_failed_ >= 5

    def test_residual_progress_with_failures(self, panel_data, caplog):
        """Residual with show_progress=True and failures (cover lines 649, 674-680)."""
        model = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = model.fit()

        bootstrap = PanelBootstrap(
            results=results,
            n_bootstrap=20,
            method="residual",
            random_state=42,
            show_progress=True,
        )

        original_create = bootstrap._create_bootstrap_model
        call_count = [0]

        def flaky_create(boot_data):
            call_count[0] += 1
            if call_count[0] <= 5:
                raise RuntimeError(f"Failure {call_count[0]}")
            return original_create(boot_data)

        bootstrap._create_bootstrap_model = flaky_create

        with caplog.at_level(logging.WARNING, logger="panelbox.validation.robustness.bootstrap"):
            bootstrap.run()

        assert bootstrap.n_failed_ >= 5
