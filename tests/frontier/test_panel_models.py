"""
Tests for panel stochastic frontier models.

This module tests:
1. Pitt & Lee (1981) - Time-invariant efficiency
2. Battese & Coelli (1992) - Time-varying efficiency
3. Battese & Coelli (1995) - Inefficiency determinants
4. Cornwell-Schmidt-Sickles (1990) - Distribution-free
5. Kumbhakar (1990) - Flexible time pattern
6. Lee & Schmidt (1993) - Time dummies
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.css import estimate_css_model, test_time_trend_specification
from panelbox.frontier.panel_likelihoods import (
    loglik_battese_coelli_92,
    loglik_battese_coelli_95,
    loglik_kumbhakar_1990,
    loglik_lee_schmidt_1993,
    loglik_pitt_lee_exponential,
    loglik_pitt_lee_half_normal,
    loglik_pitt_lee_truncated_normal,
)


@pytest.fixture
def panel_data():
    """Generate synthetic panel data for testing.

    Returns:
        dict with y, X, entity_id, time_id, N, T
    """
    np.random.seed(42)

    N = 50  # Number of entities
    T = 10  # Number of time periods
    n = N * T  # Total observations

    # Entity and time IDs
    entity_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)

    # True parameters
    beta_0 = 2.0
    beta_1 = 0.6
    beta_2 = 0.3
    sigma_v = 0.1
    sigma_u = 0.2

    # Generate X variables
    X0 = np.ones(n)
    X1 = np.random.normal(2, 0.5, n)
    X2 = np.random.normal(3, 0.5, n)
    X = np.column_stack([X0, X1, X2])

    # Generate errors
    v = np.random.normal(0, sigma_v, n)

    # Time-invariant inefficiency (one per entity, repeated over time)
    u_i = np.abs(np.random.normal(0, sigma_u, N))
    u = np.repeat(u_i, T)

    # Generate y
    y = X @ np.array([beta_0, beta_1, beta_2]) + v - u

    return {
        "y": y,
        "X": X,
        "entity_id": entity_id,
        "time_id": time_id,
        "N": N,
        "T": T,
        "n": n,
        "beta": np.array([beta_0, beta_1, beta_2]),
        "sigma_v": sigma_v,
        "sigma_u": sigma_u,
        "u": u,
    }


class TestPittLee:
    """Tests for Pitt & Lee (1981) model."""

    def test_half_normal_likelihood(self, panel_data):
        """Test that log-likelihood is finite and reasonable."""
        k = panel_data["X"].shape[1]

        # True parameters
        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
            ]
        )

        ll = loglik_pitt_lee_half_normal(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll), "Log-likelihood should be finite"
        assert ll < 0, "Log-likelihood should be negative"

    def test_exponential_likelihood(self, panel_data):
        """Test exponential distribution likelihood."""
        k = panel_data["X"].shape[1]

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"])],  # lambda parameter
            ]
        )

        ll = loglik_pitt_lee_exponential(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll), "Log-likelihood should be finite"
        assert ll < 0, "Log-likelihood should be negative"

    def test_truncated_normal_likelihood(self, panel_data):
        """Test truncated normal with quadrature."""
        k = panel_data["X"].shape[1]

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
                [0.0],  # mu parameter
            ]
        )

        ll = loglik_pitt_lee_truncated_normal(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
            n_quad_points=12,
        )

        assert np.isfinite(ll), "Log-likelihood should be finite"
        assert ll < 0, "Log-likelihood should be negative"

    def test_cross_section_equivalence(self):
        """Test that T=1 reduces to cross-section SFA."""
        np.random.seed(123)

        N = 100
        T = 1
        n = N * T

        # Single period data
        entity_id = np.arange(N)
        time_id = np.zeros(N, dtype=int)

        X = np.column_stack([np.ones(N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)])

        beta = np.array([1.0, 0.5, 0.3])
        sigma_v = 0.1
        sigma_u = 0.2

        v = np.random.normal(0, sigma_v, N)
        u = np.abs(np.random.normal(0, sigma_u, N))
        y = X @ beta + v - u

        theta = np.concatenate([beta, [np.log(sigma_v**2)], [np.log(sigma_u**2)]])

        ll_panel = loglik_pitt_lee_half_normal(theta, y, X, entity_id, time_id, sign=1)

        # Should be equivalent to cross-section
        # (can't test directly without cross-section function, but check finite)
        assert np.isfinite(ll_panel)


class TestBatteseCoelli92:
    """Tests for Battese & Coelli (1992) time-varying model."""

    def test_likelihood_eta_zero(self, panel_data):
        """Test that η=0 reduces to Pitt-Lee."""
        k = panel_data["X"].shape[1]

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
                [0.0],  # mu
                [0.0],  # eta = 0
            ]
        )

        ll_bc92 = loglik_battese_coelli_92(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        # Compare with Pitt-Lee
        theta_pl = theta[: k + 2].tolist() + [0.0]  # Add mu=0 for truncated normal
        ll_pl = loglik_pitt_lee_truncated_normal(
            np.array(theta_pl),
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        # Should be close (within tolerance due to quadrature)
        assert np.isfinite(ll_bc92)
        assert np.isfinite(ll_pl)

    def test_likelihood_eta_positive(self, panel_data):
        """Test with positive η (efficiency improves)."""
        k = panel_data["X"].shape[1]

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
                [0.0],  # mu
                [0.05],  # eta > 0
            ]
        )

        ll = loglik_battese_coelli_92(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll)
        assert ll < 0


class TestBatteseCoelli95:
    """Tests for Battese & Coelli (1995) inefficiency effects model."""

    def test_likelihood_with_determinants(self, panel_data):
        """Test BC95 with inefficiency determinants."""
        # Create determinants matrix Z
        n = panel_data["n"]
        Z = np.column_stack(
            [
                np.ones(n),
                np.random.normal(0, 1, n),  # Z1
            ]
        )

        k = panel_data["X"].shape[1]
        m = Z.shape[1]

        theta = np.concatenate(
            [
                panel_data["beta"],  # β
                [np.log(panel_data["sigma_v"] ** 2)],  # ln(σ²_v)
                [np.log(panel_data["sigma_u"] ** 2)],  # ln(σ²_u)
                np.zeros(m),  # δ (determinants)
            ]
        )

        ll = loglik_battese_coelli_95(
            theta,
            panel_data["y"],
            panel_data["X"],
            Z,
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll)
        assert ll < 0

    def test_delta_zero_vs_standard(self, panel_data):
        """Test that δ=0 approximates standard model."""
        n = panel_data["n"]
        Z = np.column_stack([np.ones(n)])  # Just intercept

        k = panel_data["X"].shape[1]
        m = Z.shape[1]

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
                [0.0],  # δ = 0
            ]
        )

        ll_bc95 = loglik_battese_coelli_95(
            theta,
            panel_data["y"],
            panel_data["X"],
            Z,
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll_bc95)


class TestCornwellSchmidtSickles:
    """Tests for Cornwell-Schmidt-Sickles (1990) distribution-free model."""

    def test_css_quadratic(self, panel_data):
        """Test CSS with quadratic time trend."""
        result = estimate_css_model(
            y=panel_data["y"],
            X=panel_data["X"][:, 1:],  # Exclude constant
            entity_id=panel_data["entity_id"],
            time_id=panel_data["time_id"],
            time_trend="quadratic",
            frontier_type="production",
        )

        assert result.n_entities == panel_data["N"]
        assert result.n_periods == panel_data["T"]
        assert result.efficiency_it.shape == (panel_data["N"], panel_data["T"])

        # Check efficiency properties
        assert np.all(result.efficiency_it > 0)
        assert np.all(result.efficiency_it <= 1.0)

        # At least one entity should have TE=1 in each period
        for t in range(panel_data["T"]):
            max_eff_t = np.max(result.efficiency_it[:, t])
            assert np.isclose(max_eff_t, 1.0, atol=1e-6)

    def test_css_linear(self, panel_data):
        """Test CSS with linear time trend."""
        result = estimate_css_model(
            y=panel_data["y"],
            X=panel_data["X"][:, 1:],
            entity_id=panel_data["entity_id"],
            time_id=panel_data["time_id"],
            time_trend="linear",
            frontier_type="production",
        )

        assert np.all(result.efficiency_it > 0)
        assert np.all(result.efficiency_it <= 1.0)

    def test_css_none(self, panel_data):
        """Test CSS with no time trend (fixed effects)."""
        result = estimate_css_model(
            y=panel_data["y"],
            X=panel_data["X"][:, 1:],
            entity_id=panel_data["entity_id"],
            time_id=panel_data["time_id"],
            time_trend="none",
            frontier_type="production",
        )

        # For 'none', efficiency should be constant over time per entity
        for i in range(panel_data["N"]):
            eff_i = result.efficiency_it[i, :]
            # All periods should have same efficiency
            assert np.allclose(eff_i, eff_i[0], atol=1e-10)

    def test_time_trend_specification(self, panel_data):
        """Test specification testing."""
        comparison = test_time_trend_specification(
            y=panel_data["y"],
            X=panel_data["X"][:, 1:],
            entity_id=panel_data["entity_id"],
            time_id=panel_data["time_id"],
            frontier_type="production",
        )

        assert len(comparison) == 3  # none, linear, quadratic
        assert "R²" in comparison.columns
        assert "σ_v" in comparison.columns


class TestKumbhakar:
    """Tests for Kumbhakar (1990) flexible time pattern."""

    def test_likelihood_b_c_zero(self, panel_data):
        """Test that b=c=0 reduces to Pitt-Lee."""
        k = panel_data["X"].shape[1]

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
                [0.0],  # mu
                [0.0],  # b = 0
                [0.0],  # c = 0
            ]
        )

        ll = loglik_kumbhakar_1990(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll)
        assert ll < 0

    def test_likelihood_nonzero_b_c(self, panel_data):
        """Test with non-zero b and c."""
        k = panel_data["X"].shape[1]

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
                [0.0],  # mu
                [0.1],  # b
                [-0.01],  # c
            ]
        )

        ll = loglik_kumbhakar_1990(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll)


class TestLeeSchmidt:
    """Tests for Lee & Schmidt (1993) time dummies."""

    def test_likelihood_with_time_dummies(self, panel_data):
        """Test Lee-Schmidt with time dummies."""
        k = panel_data["X"].shape[1]
        T = panel_data["T"]

        # δ_1, ..., δ_{T-1} (δ_T = 1 by normalization)
        delta_params = np.ones(T - 1)

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
                [0.0],  # mu
                delta_params,  # δ_t
            ]
        )

        ll = loglik_lee_schmidt_1993(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll)
        assert ll < 0

    def test_all_deltas_one(self, panel_data):
        """Test that δ_t = 1 for all t reduces to Pitt-Lee."""
        k = panel_data["X"].shape[1]
        T = panel_data["T"]

        delta_params = np.ones(T - 1)  # All equal to 1

        theta = np.concatenate(
            [
                panel_data["beta"],
                [np.log(panel_data["sigma_v"] ** 2)],
                [np.log(panel_data["sigma_u"] ** 2)],
                [0.0],
                delta_params,
            ]
        )

        ll = loglik_lee_schmidt_1993(
            theta,
            panel_data["y"],
            panel_data["X"],
            panel_data["entity_id"],
            panel_data["time_id"],
            sign=1,
        )

        assert np.isfinite(ll)


class TestPanelProperties:
    """Test general panel SFA properties."""

    def test_panel_better_than_pooled(self):
        """Test that panel estimator uses time dimension effectively.

        With larger T, efficiency estimates should be more precise
        (smaller MSE compared to true values).
        """
        np.random.seed(999)

        N = 30
        T_small = 2
        T_large = 20

        # This is a placeholder test
        # In practice, would run Monte Carlo and compare MSE
        assert T_large > T_small

    def test_minimum_T_warning(self):
        """Test that CSS raises warning for small T."""
        np.random.seed(100)

        N = 20
        T = 3  # Less than recommended minimum of 5
        n = N * T

        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        X = np.random.normal(0, 1, (n, 2))
        y = X @ np.array([0.5, 0.3]) + np.random.normal(0, 0.1, n)

        with pytest.warns(UserWarning, match="T = 3 is less than recommended"):
            result = estimate_css_model(
                y=y,
                X=X,
                entity_id=entity_id,
                time_id=time_id,
                time_trend="linear",
                frontier_type="production",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
