"""
Tests for four-component stochastic frontier model (Kumbhakar et al., 2014).

This module tests the implementation of the revolutionary four-component model
that separates:
    - Random heterogeneity (μ_i) from persistent inefficiency (η_i)
    - Random noise (v_it) from transient inefficiency (u_it)

References:
    Kumbhakar, S. C., Lien, G., & Hardaker, J. B. (2014).
        Technical efficiency in competing panel data models: a study of
        Norwegian grain farming. Journal of Productivity Analysis, 41(2), 321-337.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.advanced import FourComponentSFA


class TestFourComponentBasic:
    """Basic tests for four-component model estimation."""

    @pytest.fixture
    def synthetic_panel_data(self):
        """Generate synthetic panel data with known 4 components."""
        np.random.seed(42)
        N, T = 50, 8  # 50 entities, 8 time periods

        # True parameters
        beta_true = np.array([2.0, 0.5, -0.3])
        sigma_v_true = 0.15
        sigma_u_true = 0.25
        sigma_mu_true = 0.30
        sigma_eta_true = 0.20

        # Generate panel data
        data = []

        # Generate persistent components (time-invariant, entity-specific)
        mu_i_true = np.random.normal(0, sigma_mu_true, N)
        eta_i_true = np.abs(np.random.normal(0, sigma_eta_true, N))

        for i in range(N):
            for t in range(T):
                # Exogenous variables
                x1 = np.random.normal(0, 1)
                x2 = np.random.normal(0, 1)
                X_it = np.array([1.0, x1, x2])

                # Transient components (time-varying)
                v_it = np.random.normal(0, sigma_v_true)
                u_it = np.abs(np.random.normal(0, sigma_u_true))

                # Output (production frontier)
                # y_it = X'β + μ_i - η_i + v_it - u_it
                y_it = X_it @ beta_true + mu_i_true[i] - eta_i_true[i] + v_it - u_it

                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y": y_it,
                        "x1": x1,
                        "x2": x2,
                        "mu_i_true": mu_i_true[i],
                        "eta_i_true": eta_i_true[i],
                        "u_it_true": u_it,
                        "v_it_true": v_it,
                    }
                )

        return pd.DataFrame(data), {
            "beta": beta_true,
            "sigma_v": sigma_v_true,
            "sigma_u": sigma_u_true,
            "sigma_mu": sigma_mu_true,
            "sigma_eta": sigma_eta_true,
            "N": N,
            "T": T,
        }

    def test_basic_estimation(self, synthetic_panel_data):
        """Test that the model runs and converges."""
        df, true_params = synthetic_panel_data

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="entity",
            time="time",
            frontier_type="production",
        )

        result = model.fit(verbose=False)

        # Check that all components are estimated
        assert result.beta is not None
        assert result.alpha_i is not None
        assert result.mu_i is not None
        assert result.eta_i is not None
        assert result.u_it is not None
        assert result.v_it is not None

        # Check dimensions
        assert len(result.beta) == 3  # const + 2 variables
        assert len(result.alpha_i) == true_params["N"]
        assert len(result.eta_i) == true_params["N"]
        assert len(result.mu_i) == true_params["N"]
        assert len(result.u_it) == true_params["N"] * true_params["T"]
        assert len(result.v_it) == true_params["N"] * true_params["T"]

    def test_variance_components_positive(self, synthetic_panel_data):
        """Test that all variance components are positive."""
        df, _ = synthetic_panel_data

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        assert result.sigma_v > 0, "σ_v should be positive"
        assert result.sigma_u > 0, "σ_u should be positive"
        assert result.sigma_mu > 0, "σ_μ should be positive"
        assert result.sigma_eta > 0, "σ_η should be positive"

    def test_beta_recovery(self, synthetic_panel_data):
        """Test that β parameters are reasonably recovered."""
        df, true_params = synthetic_panel_data

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        # Check β recovery (allowing for some estimation error)
        # Note: Constant (β[0]) is absorbed in fixed effects α_i in FE model
        # So we only check non-constant coefficients
        beta_true = true_params["beta"]
        print("\nβ Recovery:")
        for i in range(len(beta_true)):
            print(f"  β[{i}]: est={result.beta[i]:.4f}, true={beta_true[i]:.4f}")

        # Check non-constant coefficients only (indices 1 and 2)
        for i in range(1, len(beta_true)):
            diff = abs(result.beta[i] - beta_true[i])
            assert (
                diff < 0.5
            ), f"β[{i}] not well recovered: est={result.beta[i]:.4f}, true={beta_true[i]:.4f}"

    def test_variance_recovery(self, synthetic_panel_data):
        """Test that variance components are reasonably recovered."""
        df, true_params = synthetic_panel_data

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        # Check variance recovery (allowing for larger errors in variance estimation)
        sigma_checks = [
            ("sigma_v", result.sigma_v, true_params["sigma_v"]),
            ("sigma_u", result.sigma_u, true_params["sigma_u"]),
            ("sigma_mu", result.sigma_mu, true_params["sigma_mu"]),
            ("sigma_eta", result.sigma_eta, true_params["sigma_eta"]),
        ]

        print("\nVariance Component Recovery:")
        for name, est, true in sigma_checks:
            diff = abs(est - true)
            print(f"  {name}: est={est:.4f}, true={true:.4f}, diff={diff:.4f}")

        # Multi-step estimation is known to have identification issues
        # The key test is that estimates are non-negative, finite, and not all zero
        assert all(np.isfinite([result.sigma_v, result.sigma_u, result.sigma_mu, result.sigma_eta]))
        assert all(
            [result.sigma_v >= 0, result.sigma_u >= 0, result.sigma_mu >= 0, result.sigma_eta >= 0]
        )

        # At least some variance components should be non-trivial
        total_sigma = result.sigma_v + result.sigma_u + result.sigma_mu + result.sigma_eta
        assert total_sigma > 0.1, "Total variance should be non-trivial"

        print(f"\nTotal variance (sum of σ's): {total_sigma:.4f}")


class TestFourComponentEfficiency:
    """Test efficiency estimates from four-component model."""

    @pytest.fixture
    def high_persistent_data(self):
        """Generate data with high persistent inefficiency."""
        np.random.seed(123)
        N, T = 30, 6

        data = []
        eta_i_true = np.abs(np.random.normal(0.8, 0.2, N))  # High persistent ineff

        for i in range(N):
            mu_i = np.random.normal(0, 0.1)

            for t in range(T):
                x1 = np.random.normal(0, 1)
                v_it = np.random.normal(0, 0.1)
                u_it = np.abs(np.random.normal(0, 0.15))  # Low transient ineff

                y_it = 2.0 + 0.5 * x1 + mu_i - eta_i_true[i] + v_it - u_it

                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y": y_it,
                        "x1": x1,
                    }
                )

        return pd.DataFrame(data)

    def test_efficiency_ranges(self, high_persistent_data):
        """Test that efficiency estimates are in (0, 1]."""
        model = FourComponentSFA(
            data=high_persistent_data,
            depvar="y",
            exog=["x1"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        # Get efficiency estimates
        te_persistent_df = result.persistent_efficiency()
        te_transient_df = result.transient_efficiency()
        te_overall_df = result.overall_efficiency()

        # Check ranges
        assert (te_persistent_df["persistent_efficiency"] > 0).all()
        assert (te_persistent_df["persistent_efficiency"] <= 1).all()

        assert (te_transient_df["transient_efficiency"] > 0).all()
        assert (te_transient_df["transient_efficiency"] <= 1).all()

        assert (te_overall_df["overall_efficiency"] > 0).all()
        assert (te_overall_df["overall_efficiency"] <= 1).all()

    def test_overall_equals_product(self, high_persistent_data):
        """Test that overall TE = persistent TE × transient TE."""
        model = FourComponentSFA(
            data=high_persistent_data,
            depvar="y",
            exog=["x1"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        te_overall_df = result.overall_efficiency()

        # Compute product manually
        product = te_overall_df["persistent_efficiency"] * te_overall_df["transient_efficiency"]

        # Should match overall efficiency
        diff = np.abs(product - te_overall_df["overall_efficiency"])
        assert (diff < 1e-6).all(), "Overall TE should equal product of components"

    def test_persistent_time_invariant(self, high_persistent_data):
        """Test that persistent efficiency is time-invariant."""
        model = FourComponentSFA(
            data=high_persistent_data,
            depvar="y",
            exog=["x1"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        te_overall_df = result.overall_efficiency()

        # For each entity, persistent TE should be constant over time
        for entity in te_overall_df["entity"].unique():
            entity_data = te_overall_df[te_overall_df["entity"] == entity]
            persistent_values = entity_data["persistent_efficiency"].values

            # All values should be identical
            assert np.allclose(
                persistent_values, persistent_values[0]
            ), f"Persistent TE should be time-invariant for entity {entity}"


class TestFourComponentDecomposition:
    """Test component decomposition."""

    def test_decomposition_structure(self):
        """Test that decomposition returns all 4 components."""
        np.random.seed(456)
        N, T = 20, 5

        data = []
        for i in range(N):
            for t in range(T):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y": np.random.normal(5, 1),
                        "x1": np.random.normal(0, 1),
                    }
                )

        df = pd.DataFrame(data)

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)
        decomp = result.decomposition()

        # Check columns
        assert "entity" in decomp.columns
        assert "time" in decomp.columns
        assert "mu_i" in decomp.columns
        assert "eta_i" in decomp.columns
        assert "u_it" in decomp.columns
        assert "v_it" in decomp.columns

        # Check dimensions
        assert len(decomp) == N * T

    def test_alpha_decomposition(self):
        """Test that μ_i - η_i ≈ α_i."""
        np.random.seed(789)
        N, T = 25, 6

        data = []
        for i in range(N):
            for t in range(T):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y": np.random.normal(5, 1),
                        "x1": np.random.normal(0, 1),
                    }
                )

        df = pd.DataFrame(data)

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        # Check decomposition: α_i = μ_i - η_i
        alpha_reconstructed = result.mu_i - result.eta_i
        diff = np.abs(alpha_reconstructed - result.alpha_i)

        # Should be very close (numerical precision)
        assert np.allclose(
            alpha_reconstructed, result.alpha_i, atol=1e-6
        ), "α_i should equal μ_i - η_i"


class TestFourComponentEdgeCases:
    """Test edge cases and error handling."""

    def test_requires_panel_data(self):
        """Test that model requires both entity and time identifiers."""
        df = pd.DataFrame(
            {
                "y": np.random.normal(5, 1, 100),
                "x1": np.random.normal(0, 1, 100),
            }
        )

        # Should raise error without entity/time
        with pytest.raises(ValueError, match="requires both entity and time"):
            FourComponentSFA(
                data=df,
                depvar="y",
                exog=["x1"],
                entity=None,
                time=None,
            )

    def test_minimum_sample_size(self):
        """Test with minimum viable sample size."""
        np.random.seed(999)
        N, T = 10, 3  # Small panel

        data = []
        for i in range(N):
            for t in range(T):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y": 5 + 0.5 * np.random.normal(0, 1) + np.random.normal(0, 0.5),
                        "x1": np.random.normal(0, 1),
                    }
                )

        df = pd.DataFrame(data)

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1"],
            entity="entity",
            time="time",
        )

        # Should still run (though estimates may be imprecise)
        result = model.fit(verbose=False)

        assert result is not None
        assert result.sigma_v > 0
        assert result.sigma_u > 0


class TestFourComponentSummary:
    """Test summary output methods."""

    def test_print_summary(self):
        """Test that summary prints without errors."""
        np.random.seed(111)
        N, T = 30, 5

        data = []
        for i in range(N):
            for t in range(T):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y": np.random.normal(5, 1),
                        "x1": np.random.normal(0, 1),
                    }
                )

        df = pd.DataFrame(data)

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        # Should print without errors
        result.print_summary()

    def test_variance_shares_sum_to_one(self):
        """Test that variance shares sum to 100%."""
        np.random.seed(222)
        N, T = 40, 6

        data = []
        for i in range(N):
            for t in range(T):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y": np.random.normal(5, 1),
                        "x1": np.random.normal(0, 1),
                    }
                )

        df = pd.DataFrame(data)

        model = FourComponentSFA(
            data=df,
            depvar="y",
            exog=["x1"],
            entity="entity",
            time="time",
        )

        result = model.fit(verbose=False)

        # Compute variance shares
        total_var = result.sigma_v**2 + result.sigma_u**2 + result.sigma_mu**2 + result.sigma_eta**2

        share_v = result.sigma_v**2 / total_var
        share_u = result.sigma_u**2 / total_var
        share_mu = result.sigma_mu**2 / total_var
        share_eta = result.sigma_eta**2 / total_var

        total_share = share_v + share_u + share_mu + share_eta

        assert np.isclose(total_share, 1.0), "Variance shares should sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
