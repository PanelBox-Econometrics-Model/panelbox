"""
Tests for efficiency estimation in stochastic frontier models.

Tests:
    - Efficiency bounds (0,1] for production, [1,∞) for cost
    - JLMS vs BC estimator comparison
    - Confidence intervals coverage
    - Correlation with true efficiency
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import StochasticFrontier


class TestEfficiencyEstimation:
    """Tests for efficiency estimators."""

    @pytest.fixture
    def production_with_known_efficiency(self):
        """Data with known true efficiency."""
        np.random.seed(42)
        n = 500

        beta_true = np.array([2.0, 0.6, 0.3])
        sigma_v_true = 0.05  # Small noise
        sigma_u_true = 0.2

        const = np.ones(n)
        log_labor = np.random.uniform(0, 3, n)
        log_capital = np.random.uniform(0, 3, n)
        X = np.column_stack([const, log_labor, log_capital])

        v = np.random.normal(0, sigma_v_true, n)
        u = np.abs(np.random.normal(0, sigma_u_true, n))

        log_output = X @ beta_true + v - u

        # True efficiency
        true_efficiency = np.exp(-u)

        df = pd.DataFrame(
            {
                "log_output": log_output,
                "log_labor": log_labor,
                "log_capital": log_capital,
                "true_efficiency": true_efficiency,
            }
        )

        return df

    def test_efficiency_bounds_production(self, production_with_known_efficiency):
        """Efficiency should be in (0, 1] for production frontier."""
        df = production_with_known_efficiency

        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        # Test both estimators
        for estimator in ["jlms", "bc"]:
            eff = result.efficiency(estimator=estimator)

            # All efficiencies should be in (0, 1]
            assert np.all(eff["efficiency"] > 0), f"{estimator}: Some efficiencies <= 0"
            assert np.all(eff["efficiency"] <= 1), f"{estimator}: Some efficiencies > 1"

    def test_efficiency_bounds_cost(self):
        """Efficiency should be >= 1 for cost frontier."""
        np.random.seed(123)
        n = 500

        beta_true = np.array([1.0, 0.4, 0.5])
        sigma_v_true = 0.1
        sigma_u_true = 0.15

        const = np.ones(n)
        log_labor = np.random.uniform(0, 2, n)
        log_capital = np.random.uniform(0, 2, n)
        X = np.column_stack([const, log_labor, log_capital])

        v = np.random.normal(0, sigma_v_true, n)
        u = np.abs(np.random.normal(0, sigma_u_true, n))

        log_cost = X @ beta_true + v + u

        df = pd.DataFrame(
            {"log_cost": log_cost, "log_labor": log_labor, "log_capital": log_capital}
        )

        sf = StochasticFrontier(
            data=df,
            depvar="log_cost",
            exog=["log_labor", "log_capital"],
            frontier="cost",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        eff = result.efficiency(estimator="bc")

        # Cost efficiency should be >= 1
        assert np.all(eff["efficiency"] >= 1), "Some cost efficiencies < 1"

    def test_jlms_vs_bc_correlation(self, production_with_known_efficiency):
        """JLMS and BC estimators should be highly correlated."""
        df = production_with_known_efficiency

        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        eff_jlms = result.efficiency(estimator="jlms")
        eff_bc = result.efficiency(estimator="bc")

        # Should be highly correlated
        correlation = np.corrcoef(eff_jlms["efficiency"], eff_bc["efficiency"])[0, 1]

        assert correlation > 0.95, f"JLMS-BC correlation too low: {correlation}"

    def test_correlation_with_true_efficiency(self, production_with_known_efficiency):
        """Estimated efficiency should correlate with true efficiency."""
        df = production_with_known_efficiency

        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        eff = result.efficiency(estimator="bc")

        # Correlation with true efficiency
        correlation = np.corrcoef(df["true_efficiency"], eff["efficiency"])[0, 1]

        # Should be positively correlated (exact recovery not expected due to noise)
        assert correlation > 0.5, f"Correlation with true efficiency too low: {correlation}"

    def test_confidence_intervals_coverage(self):
        """Test that confidence intervals have proper coverage."""
        np.random.seed(999)

        # Run Monte Carlo
        n_sim = 100
        coverage_count = 0

        for _ in range(n_sim):
            n = 200
            beta_true = np.array([1.0, 0.5])
            sigma_v_true = 0.1
            sigma_u_true = 0.2

            const = np.ones(n)
            x = np.random.normal(0, 1, n)
            X = np.column_stack([const, x])

            v = np.random.normal(0, sigma_v_true, n)
            u = np.abs(np.random.normal(0, sigma_u_true, n))

            y = X @ beta_true + v - u
            true_eff = np.exp(-u)

            df = pd.DataFrame({"y": y, "x": x})

            sf = StochasticFrontier(
                data=df, depvar="y", exog=["x"], frontier="production", dist="half_normal"
            )

            try:
                result = sf.fit(method="mle", verbose=False, maxiter=500)

                if result.converged:
                    eff = result.efficiency(estimator="bc", ci_level=0.95)

                    # Check if true efficiency is within CI
                    within_ci = (true_eff >= eff["ci_lower"]) & (true_eff <= eff["ci_upper"])

                    # Count proportion within CI
                    coverage_count += within_ci.mean()

            except Exception:
                continue

        # Average coverage should be close to 0.95 (allow some variation)
        avg_coverage = coverage_count / n_sim
        # Note: This test may be flaky due to Monte Carlo variation
        # We use a generous bound
        assert avg_coverage > 0.7, f"Coverage too low: {avg_coverage}"

    def test_mean_efficiency_property(self, production_with_known_efficiency):
        """Test mean_efficiency property."""
        df = production_with_known_efficiency

        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        # mean_efficiency should equal mean of BC estimator
        mean_eff = result.mean_efficiency
        eff = result.efficiency(estimator="bc")

        assert abs(mean_eff - eff["efficiency"].mean()) < 1e-6

    def test_efficiency_dataframe_structure(self, production_with_known_efficiency):
        """Test that efficiency DataFrame has correct structure."""
        df = production_with_known_efficiency

        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        eff = result.efficiency(estimator="bc", ci_level=0.95)

        # Check columns
        assert "inefficiency" in eff.columns
        assert "efficiency" in eff.columns
        assert "ci_lower" in eff.columns
        assert "ci_upper" in eff.columns

        # Check length
        assert len(eff) == len(df)

        # Check CI ordering
        assert np.all(eff["ci_lower"] <= eff["efficiency"])
        assert np.all(eff["efficiency"] <= eff["ci_upper"])

    def test_mode_estimator(self, production_with_known_efficiency):
        """Test mode estimator."""
        df = production_with_known_efficiency

        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        eff_mode = result.efficiency(estimator="mode")

        # Basic checks
        assert np.all(eff_mode["efficiency"] > 0)
        assert np.all(eff_mode["efficiency"] <= 1)

        # Mode should be similar to other estimators
        eff_bc = result.efficiency(estimator="bc")
        correlation = np.corrcoef(eff_mode["efficiency"], eff_bc["efficiency"])[0, 1]

        assert correlation > 0.8


class TestEfficiencyExponential:
    """Test efficiency estimation with exponential distribution."""

    def test_efficiency_exponential(self):
        """Test efficiency estimation with exponential inefficiency."""
        np.random.seed(42)
        n = 500

        beta_true = np.array([1.0, 0.5])
        sigma_v_true = 0.1
        sigma_u_true = 0.2

        const = np.ones(n)
        x = np.random.normal(0, 1, n)
        X = np.column_stack([const, x])

        v = np.random.normal(0, sigma_v_true, n)
        u = np.random.exponential(sigma_u_true, n)

        y = X @ beta_true + v - u

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df, depvar="y", exog=["x"], frontier="production", dist="exponential"
        )

        result = sf.fit(method="mle", verbose=False)

        # Test efficiency estimation
        eff_jlms = result.efficiency(estimator="jlms")
        eff_bc = result.efficiency(estimator="bc")

        # Bounds check
        assert np.all(eff_bc["efficiency"] > 0)
        assert np.all(eff_bc["efficiency"] <= 1)

        # Correlation
        correlation = np.corrcoef(eff_jlms["efficiency"], eff_bc["efficiency"])[0, 1]

        assert correlation > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
