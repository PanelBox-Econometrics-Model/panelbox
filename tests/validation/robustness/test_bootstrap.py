"""
Tests for PanelBootstrap class.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.robustness.bootstrap import PanelBootstrap


class TestPanelBootstrapInitialization:
    """Tests for PanelBootstrap initialization."""

    def test_init_valid(self, balanced_panel_data):
        """Test initialization with valid inputs."""
        # Fit model
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        # Initialize bootstrap
        bootstrap = PanelBootstrap(results, n_bootstrap=100, random_state=42)

        assert bootstrap.results == results
        assert bootstrap.n_bootstrap == 100
        assert bootstrap.method == 'pairs'
        assert bootstrap.random_state == 42
        assert bootstrap._fitted is False

    def test_init_invalid_results_type(self):
        """Test initialization with invalid results type."""
        with pytest.raises(TypeError, match="results must be PanelResults"):
            PanelBootstrap("not_a_result", n_bootstrap=100)

    def test_init_small_n_bootstrap_warning(self, balanced_panel_data):
        """Test warning for small n_bootstrap."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        with pytest.warns(UserWarning, match="n_bootstrap=50 is quite small"):
            PanelBootstrap(results, n_bootstrap=50)

    def test_init_invalid_method(self, balanced_panel_data):
        """Test initialization with invalid method."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        with pytest.raises(ValueError, match="method must be one of"):
            PanelBootstrap(results, n_bootstrap=100, method='invalid')

    def test_init_parallel_warning(self, balanced_panel_data):
        """Test warning for parallel processing."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        with pytest.warns(UserWarning, match="Parallel processing not yet implemented"):
            PanelBootstrap(results, n_bootstrap=100, parallel=True)


class TestPanelBootstrapPairsMethod:
    """Tests for pairs bootstrap method."""

    def test_pairs_bootstrap_runs(self, balanced_panel_data):
        """Test that pairs bootstrap runs without error."""
        # Fit model
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        # Run bootstrap (small number for speed)
        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            method='pairs',
            random_state=42,
            show_progress=False
        )
        bootstrap.run()

        assert bootstrap._fitted is True
        assert bootstrap.bootstrap_estimates_ is not None
        assert bootstrap.bootstrap_se_ is not None
        assert bootstrap.n_failed_ >= 0

    def test_pairs_bootstrap_shape(self, balanced_panel_data):
        """Test that bootstrap estimates have correct shape."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        n_params = len(results.params)
        n_boot = 50

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=n_boot,
            method='pairs',
            random_state=42,
            show_progress=False
        )
        bootstrap.run()

        # Shape should be (n_successful_boots, n_params)
        assert bootstrap.bootstrap_estimates_.shape[1] == n_params
        assert bootstrap.bootstrap_estimates_.shape[0] <= n_boot  # May have failures

    def test_pairs_bootstrap_reproducible(self, balanced_panel_data):
        """Test that bootstrap is reproducible with random_state."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        # Run twice with same random_state
        boot1 = PanelBootstrap(
            results,
            n_bootstrap=30,
            random_state=42,
            show_progress=False
        ).run()

        boot2 = PanelBootstrap(
            results,
            n_bootstrap=30,
            random_state=42,
            show_progress=False
        ).run()

        # Should get same estimates
        np.testing.assert_array_almost_equal(
            boot1.bootstrap_estimates_,
            boot2.bootstrap_estimates_,
            decimal=10
        )

    def test_pairs_bootstrap_different_seeds(self, balanced_panel_data):
        """Test that different seeds give different results."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        boot1 = PanelBootstrap(
            results,
            n_bootstrap=30,
            random_state=42,
            show_progress=False
        ).run()

        boot2 = PanelBootstrap(
            results,
            n_bootstrap=30,
            random_state=123,
            show_progress=False
        ).run()

        # Should get different estimates
        assert not np.allclose(
            boot1.bootstrap_estimates_,
            boot2.bootstrap_estimates_
        )

    def test_bootstrap_se_positive(self, balanced_panel_data):
        """Test that bootstrap SEs are positive."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            random_state=42,
            show_progress=False
        ).run()

        assert np.all(bootstrap.bootstrap_se_ > 0)


class TestPanelBootstrapConfidenceIntervals:
    """Tests for bootstrap confidence intervals."""

    def test_conf_int_percentile(self, balanced_panel_data):
        """Test percentile confidence intervals."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=100,
            random_state=42,
            show_progress=False
        ).run()

        ci = bootstrap.conf_int(alpha=0.05, method='percentile')

        # Check shape and structure
        assert ci.shape == (len(results.params), 2)
        assert list(ci.columns) == ['lower', 'upper']
        assert all(ci.index == results.params.index)

        # Check that lower < upper
        assert np.all(ci['lower'] < ci['upper'])

    def test_conf_int_basic(self, balanced_panel_data):
        """Test basic (reflection) confidence intervals."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=100,
            random_state=42,
            show_progress=False
        ).run()

        ci = bootstrap.conf_int(alpha=0.05, method='basic')

        # Check shape
        assert ci.shape == (len(results.params), 2)
        assert np.all(ci['lower'] < ci['upper'])

    def test_conf_int_before_run_raises(self, balanced_panel_data):
        """Test that conf_int before run raises error."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(results, n_bootstrap=100)

        with pytest.raises(RuntimeError, match="Must call run\\(\\) before conf_int"):
            bootstrap.conf_int()

    def test_conf_int_invalid_method(self, balanced_panel_data):
        """Test invalid confidence interval method."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            random_state=42,
            show_progress=False
        ).run()

        with pytest.raises(ValueError, match="method must be"):
            bootstrap.conf_int(method='invalid')

    def test_conf_int_alpha_levels(self, balanced_panel_data):
        """Test different alpha levels."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=100,
            random_state=42,
            show_progress=False
        ).run()

        # 95% CI should be wider than 90% CI
        ci_95 = bootstrap.conf_int(alpha=0.05)
        ci_90 = bootstrap.conf_int(alpha=0.10)

        width_95 = (ci_95['upper'] - ci_95['lower']).values
        width_90 = (ci_90['upper'] - ci_90['lower']).values

        assert np.all(width_95 >= width_90)


class TestPanelBootstrapSummary:
    """Tests for bootstrap summary."""

    def test_summary_structure(self, balanced_panel_data):
        """Test summary table structure."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            random_state=42,
            show_progress=False
        ).run()

        summary = bootstrap.summary()

        # Check columns
        expected_cols = [
            'Original',
            'Bootstrap Mean',
            'Bootstrap Bias',
            'Original SE',
            'Bootstrap SE',
            'SE Ratio'
        ]
        assert list(summary.columns) == expected_cols

        # Check index matches params
        assert all(summary.index == results.params.index)

    def test_summary_before_run_raises(self, balanced_panel_data):
        """Test that summary before run raises error."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(results, n_bootstrap=100)

        with pytest.raises(RuntimeError, match="Must call run\\(\\) before summary"):
            bootstrap.summary()

    def test_summary_bias_small(self, balanced_panel_data):
        """Test that bootstrap bias is reasonably small."""
        # With enough bootstrap samples, bias should be small
        np.random.seed(42)

        # Generate data with known parameters
        n_entities = 20
        n_periods = 10
        beta_true = np.array([2.0, -1.5])

        entities = np.repeat(range(1, n_entities + 1), n_periods)
        times = np.tile(range(1, n_periods + 1), n_entities)

        x1 = np.random.randn(n_entities * n_periods)
        x2 = np.random.randn(n_entities * n_periods)
        y = beta_true[0] * x1 + beta_true[1] * x2 + np.random.randn(n_entities * n_periods) * 0.5

        data = pd.DataFrame({
            'entity': entities,
            'time': times,
            'y': y,
            'x1': x1,
            'x2': x2
        })

        # Fit and bootstrap
        ols = PooledOLS("y ~ x1 + x2 - 1", data, "entity", "time")
        results = ols.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=200,
            random_state=42,
            show_progress=False
        ).run()

        summary = bootstrap.summary()

        # Bootstrap bias should be small (less than 10% of SE)
        relative_bias = np.abs(summary['Bootstrap Bias'] / summary['Bootstrap SE'])
        assert np.all(relative_bias < 0.1)


class TestPanelBootstrapWildMethod:
    """Tests for wild bootstrap method."""

    def test_wild_bootstrap_runs(self, balanced_panel_data):
        """Test that wild bootstrap runs without error."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            method='wild',
            random_state=42,
            show_progress=False
        )
        bootstrap.run()

        assert bootstrap._fitted is True
        assert bootstrap.bootstrap_estimates_ is not None
        assert bootstrap.n_failed_ >= 0

    def test_wild_bootstrap_shape(self, balanced_panel_data):
        """Test that wild bootstrap estimates have correct shape."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        n_params = len(results.params)
        n_boot = 50

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=n_boot,
            method='wild',
            random_state=42,
            show_progress=False
        ).run()

        assert bootstrap.bootstrap_estimates_.shape[1] == n_params
        assert bootstrap.bootstrap_estimates_.shape[0] <= n_boot

    def test_wild_bootstrap_reproducible(self, balanced_panel_data):
        """Test that wild bootstrap is reproducible."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        boot1 = PanelBootstrap(
            results,
            n_bootstrap=30,
            method='wild',
            random_state=42,
            show_progress=False
        ).run()

        boot2 = PanelBootstrap(
            results,
            n_bootstrap=30,
            method='wild',
            random_state=42,
            show_progress=False
        ).run()

        np.testing.assert_array_almost_equal(
            boot1.bootstrap_estimates_,
            boot2.bootstrap_estimates_,
            decimal=10
        )


class TestPanelBootstrapBlockMethod:
    """Tests for block bootstrap method."""

    def test_block_bootstrap_runs(self, balanced_panel_data):
        """Test that block bootstrap runs without error."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            method='block',
            block_size=2,
            random_state=42,
            show_progress=False
        )
        bootstrap.run()

        assert bootstrap._fitted is True
        assert bootstrap.bootstrap_estimates_ is not None

    def test_block_bootstrap_auto_block_size(self, balanced_panel_data):
        """Test that block bootstrap works with automatic block size."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=30,
            method='block',
            block_size=None,  # Automatic
            random_state=42,
            show_progress=False
        ).run()

        assert bootstrap._fitted is True

    def test_block_bootstrap_reproducible(self, balanced_panel_data):
        """Test that block bootstrap is reproducible."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        boot1 = PanelBootstrap(
            results,
            n_bootstrap=30,
            method='block',
            block_size=2,
            random_state=42,
            show_progress=False
        ).run()

        boot2 = PanelBootstrap(
            results,
            n_bootstrap=30,
            method='block',
            block_size=2,
            random_state=42,
            show_progress=False
        ).run()

        np.testing.assert_array_almost_equal(
            boot1.bootstrap_estimates_,
            boot2.bootstrap_estimates_,
            decimal=10
        )


class TestPanelBootstrapResidualMethod:
    """Tests for residual bootstrap method."""

    def test_residual_bootstrap_runs(self, balanced_panel_data):
        """Test that residual bootstrap runs without error."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            method='residual',
            random_state=42,
            show_progress=False
        )
        bootstrap.run()

        assert bootstrap._fitted is True
        assert bootstrap.bootstrap_estimates_ is not None

    def test_residual_bootstrap_shape(self, balanced_panel_data):
        """Test that residual bootstrap estimates have correct shape."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        n_params = len(results.params)
        n_boot = 50

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=n_boot,
            method='residual',
            random_state=42,
            show_progress=False
        ).run()

        assert bootstrap.bootstrap_estimates_.shape[1] == n_params
        assert bootstrap.bootstrap_estimates_.shape[0] <= n_boot

    def test_residual_bootstrap_reproducible(self, balanced_panel_data):
        """Test that residual bootstrap is reproducible."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        boot1 = PanelBootstrap(
            results,
            n_bootstrap=30,
            method='residual',
            random_state=42,
            show_progress=False
        ).run()

        boot2 = PanelBootstrap(
            results,
            n_bootstrap=30,
            method='residual',
            random_state=42,
            show_progress=False
        ).run()

        np.testing.assert_array_almost_equal(
            boot1.bootstrap_estimates_,
            boot2.bootstrap_estimates_,
            decimal=10
        )


class TestPanelBootstrapMethodComparison:
    """Tests comparing different bootstrap methods."""

    def test_all_methods_run(self, balanced_panel_data):
        """Test that all bootstrap methods can run successfully."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        methods = ['pairs', 'wild', 'block', 'residual']
        for method in methods:
            bootstrap = PanelBootstrap(
                results,
                n_bootstrap=20,
                method=method,
                block_size=2 if method == 'block' else None,
                random_state=42,
                show_progress=False
            ).run()

            assert bootstrap._fitted is True, f"{method} bootstrap failed"
            assert bootstrap.bootstrap_estimates_ is not None

    def test_methods_give_different_results(self, balanced_panel_data):
        """Test that different methods give different bootstrap distributions."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        boot_pairs = PanelBootstrap(
            results,
            n_bootstrap=50,
            method='pairs',
            random_state=42,
            show_progress=False
        ).run()

        boot_wild = PanelBootstrap(
            results,
            n_bootstrap=50,
            method='wild',
            random_state=42,
            show_progress=False
        ).run()

        # Should give different estimates
        assert not np.allclose(
            boot_pairs.bootstrap_estimates_,
            boot_wild.bootstrap_estimates_
        )


class TestPanelBootstrapRepr:
    """Tests for string representation."""

    def test_repr_before_fitting(self, balanced_panel_data):
        """Test repr before fitting."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(results, n_bootstrap=100, method='pairs')
        repr_str = repr(bootstrap)

        assert "method='pairs'" in repr_str
        assert "n_bootstrap=100" in repr_str
        assert "not fitted" in repr_str

    def test_repr_after_fitting(self, balanced_panel_data):
        """Test repr after fitting."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            random_state=42,
            show_progress=False
        ).run()

        repr_str = repr(bootstrap)

        assert "method='pairs'" in repr_str
        assert "fitted with" in repr_str
        assert "successful replications" in repr_str


class TestPanelBootstrapUnbalancedPanel:
    """Tests for bootstrap with unbalanced panels."""

    def test_bootstrap_unbalanced_panel(self, unbalanced_panel_data):
        """Test that bootstrap works with unbalanced panels."""
        fe = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = fe.fit()

        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=50,
            random_state=42,
            show_progress=False
        ).run()

        # Should run successfully
        assert bootstrap._fitted is True
        assert bootstrap.bootstrap_estimates_ is not None

        # Check confidence intervals
        ci = bootstrap.conf_int()
        assert ci.shape[0] == len(results.params)


class TestPanelBootstrapEdgeCases:
    """Tests for edge cases."""

    def test_bootstrap_with_single_entity_warning(self, balanced_panel_data):
        """Test bootstrap with very few entities."""
        # Keep only 2 entities
        small_data = balanced_panel_data[balanced_panel_data['entity'] <= 2]

        fe = FixedEffects("y ~ x1 + x2", small_data, "entity", "time")
        results = fe.fit()

        # Should work but might have failures
        bootstrap = PanelBootstrap(
            results,
            n_bootstrap=20,
            random_state=42,
            show_progress=False
        )

        # May have some failures due to resampling same entity multiple times
        bootstrap.run()
        assert bootstrap._fitted is True
