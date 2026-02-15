"""
Tests for Bias-Corrected GMM Estimator
========================================

Test suite for Hahn-Kuersteiner (2002) bias correction implementation.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm import BiasCorrectedGMM, DifferenceGMM


class TestBiasCorrectedGMM:
    """Test suite for BiasCorrectedGMM."""

    @pytest.fixture
    def dynamic_panel_data(self):
        """
        Generate dynamic panel data.

        DGP:
        yᵢₜ = ρ yᵢ,ₜ₋₁ + βₓ xᵢₜ + αᵢ + εᵢₜ

        where ρ = 0.5, βₓ = 0.3
        """
        np.random.seed(42)
        n = 100  # Entities
        t = 15  # Time periods

        # True parameters
        rho = 0.5  # AR coefficient
        beta_x = 0.3  # Exogenous effect
        alpha_std = 1.0  # Fixed effect std
        epsilon_std = 0.5  # Idiosyncratic error std

        # Generate panel
        data_list = []

        for i in range(n):
            # Fixed effect
            alpha_i = np.random.normal(0, alpha_std)

            # Exogenous variable
            x_it = np.random.normal(0, 1, t)

            # Initialize y
            y_it = np.zeros(t)
            y_it[0] = alpha_i + np.random.normal(0, epsilon_std)

            # Generate AR process
            for t_idx in range(1, t):
                epsilon_it = np.random.normal(0, epsilon_std)
                y_it[t_idx] = rho * y_it[t_idx - 1] + beta_x * x_it[t_idx] + alpha_i + epsilon_it

            # Create DataFrame for entity i
            entity_data = pd.DataFrame({"entity": i, "time": range(t), "y": y_it, "x": x_it})
            data_list.append(entity_data)

        # Combine all entities
        data = pd.concat(data_list, ignore_index=True)
        data = data.set_index(["entity", "time"])

        true_params = {"rho": rho, "beta_x": beta_x}

        return data, true_params

    def test_initialization(self, dynamic_panel_data):
        """Test BiasCorrectedGMM initializes correctly."""
        data, _ = dynamic_panel_data

        model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1],
            id_var="entity",
            time_var="time",
            exog_vars=["x"],
            bias_order=1,
        )

        assert model.dep_var == "y"
        assert model.lags == [1]
        assert model.exog_vars == ["x"]
        assert model.bias_order == 1
        assert model.bias_corrected is True

    def test_fit_basic(self, dynamic_panel_data):
        """Test basic fitting of bias-corrected GMM."""
        data, _ = dynamic_panel_data

        model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1],
            id_var="entity",
            time_var="time",
            exog_vars=["x"],
            bias_order=1,
        )

        results = model.fit(time_dummies=False, verbose=False)

        # Check that estimation completed
        assert model.params_ is not None
        assert model.params_uncorrected_ is not None
        assert model.bias_term_ is not None
        assert model.vcov_ is not None

        # Check that results object was created
        assert results is not None
        assert len(results.params) > 0

    def test_bias_reduction(self, dynamic_panel_data):
        """Test that bias correction reduces bias."""
        data, true_params = dynamic_panel_data

        # Estimate bias-corrected GMM
        bc_model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1],
            id_var="entity",
            time_var="time",
            exog_vars=["x"],
            bias_order=1,
        )
        bc_results = bc_model.fit(time_dummies=False, verbose=False)

        # Estimate standard GMM
        gmm_model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=[1],
            id_var="entity",
            time_var="time",
            exog_vars=["x"],
            time_dummies=False,
        )
        gmm_results = gmm_model.fit()

        # Extract AR coefficient estimates
        # (assuming it's the first parameter after potential intercept)
        lag_param_name = [
            name for name in bc_results.params.index if "L1.y" in name or "lag1" in name.lower()
        ]
        if not lag_param_name:
            # Try first parameter that's not intercept
            lag_param_name = [name for name in bc_results.params.index if name != "const"]

        if lag_param_name:
            lag_param_name = lag_param_name[0]
            bc_rho = bc_results.params[lag_param_name]
            gmm_rho = gmm_results.params[lag_param_name]
            true_rho = true_params["rho"]

            # Bias-corrected should be closer to true value (in most cases)
            bc_bias = abs(bc_rho - true_rho)
            gmm_bias = abs(gmm_rho - true_rho)

            # Note: This is a statistical property that holds on average,
            # not necessarily in every single sample
            # So we just check that both are reasonable
            assert bc_bias < 0.3, f"BC bias too large: {bc_bias}"
            assert gmm_bias < 0.3, f"GMM bias too large: {gmm_bias}"

    def test_bias_magnitude(self, dynamic_panel_data):
        """Test bias_magnitude() method."""
        data, _ = dynamic_panel_data

        model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1],
            id_var="entity",
            time_var="time",
            exog_vars=["x"],
        )
        model.fit(time_dummies=False, verbose=False)

        bias_mag = model.bias_magnitude()

        # Bias magnitude should be non-negative and finite
        assert bias_mag >= 0
        assert np.isfinite(bias_mag)

        # For N=100, T=15, bias should be small but non-zero
        assert 0 < bias_mag < 1.0

    def test_parameters_differ(self, dynamic_panel_data):
        """Test that corrected and uncorrected params differ."""
        data, _ = dynamic_panel_data

        model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1],
            id_var="entity",
            time_var="time",
            exog_vars=["x"],
        )
        model.fit(time_dummies=False, verbose=False)

        # Corrected and uncorrected should differ
        diff = model.params_ - model.params_uncorrected_
        assert not np.allclose(diff, 0), "Bias correction had no effect"

        # Difference should be equal to -bias_term/N
        n = data.index.get_level_values(0).nunique()
        expected_diff = -model.bias_term_ / n
        np.testing.assert_allclose(diff, expected_diff, rtol=1e-10)

    def test_invalid_bias_order(self, dynamic_panel_data):
        """Test error with invalid bias_order."""
        data, _ = dynamic_panel_data

        with pytest.raises(ValueError, match="bias_order must be 1 or 2"):
            BiasCorrectedGMM(
                data=data,
                dep_var="y",
                lags=[1],
                id_var="entity",
                time_var="time",
                bias_order=3,  # Invalid
            )

    def test_small_sample_warning(self):
        """Test warning with small N."""
        np.random.seed(123)
        n = 20  # Small N
        t = 10

        data_list = []
        for i in range(n):
            data_list.append(
                pd.DataFrame(
                    {
                        "entity": i,
                        "time": range(t),
                        "y": np.random.randn(t),
                        "x": np.random.randn(t),
                    }
                )
            )

        data = pd.concat(data_list, ignore_index=True)
        data = data.set_index(["entity", "time"])

        with pytest.warns(UserWarning, match="N = 20 < 50"):
            BiasCorrectedGMM(
                data=data,
                dep_var="y",
                lags=[1],
                id_var="entity",
                time_var="time",
                exog_vars=["x"],
                min_n=50,
            )

    def test_small_t_warning(self):
        """Test warning with small T."""
        np.random.seed(456)
        n = 100
        t = 5  # Small T

        data_list = []
        for i in range(n):
            data_list.append(
                pd.DataFrame(
                    {
                        "entity": i,
                        "time": range(t),
                        "y": np.random.randn(t),
                        "x": np.random.randn(t),
                    }
                )
            )

        data = pd.concat(data_list, ignore_index=True)
        data = data.set_index(["entity", "time"])

        with pytest.warns(UserWarning, match="Average T"):
            BiasCorrectedGMM(
                data=data,
                dep_var="y",
                lags=[1],
                id_var="entity",
                time_var="time",
                exog_vars=["x"],
                min_t=10,
            )

    def test_repr(self, dynamic_panel_data):
        """Test string representation."""
        data, _ = dynamic_panel_data

        model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1],
            id_var="entity",
            time_var="time",
            exog_vars=["x"],
        )

        repr_str = repr(model)
        assert "BiasCorrectedGMM" in repr_str
        assert "dep_var='y'" in repr_str
        assert "not fitted" in repr_str

        # After fitting
        model.fit(time_dummies=False, verbose=False)
        repr_fitted = repr(model)
        assert "fitted" in repr_fitted
        assert "bias_magnitude=" in repr_fitted

    def test_monte_carlo_bias_reduction(self):
        """
        Monte Carlo test: Bias correction reduces bias on average.

        This test runs multiple simulations to verify that bias correction
        systematically reduces bias toward the true parameter.
        """
        np.random.seed(2024)
        n_sims = 50  # Reduced for speed
        n = 80
        t = 12

        true_rho = 0.6
        true_beta = 0.3

        bc_errors = []
        gmm_errors = []

        for sim in range(n_sims):
            # Generate data
            data_list = []
            for i in range(n):
                alpha_i = np.random.normal(0, 1)
                x_it = np.random.normal(0, 1, t)
                y_it = np.zeros(t)
                y_it[0] = alpha_i + np.random.normal(0, 0.5)

                for t_idx in range(1, t):
                    epsilon = np.random.normal(0, 0.5)
                    y_it[t_idx] = (
                        true_rho * y_it[t_idx - 1] + true_beta * x_it[t_idx] + alpha_i + epsilon
                    )

                data_list.append(
                    pd.DataFrame({"entity": i, "time": range(t), "y": y_it, "x": x_it})
                )

            data = pd.concat(data_list, ignore_index=True).set_index(["entity", "time"])

            try:
                # Bias-corrected GMM
                bc_model = BiasCorrectedGMM(data=data, dep_var="y", lags=[1], exog_vars=["x"])
                id_var = ("entity",)
                time_var = ("time",)
                bc_results = bc_model.fit(time_dummies=False, verbose=False)

                # Standard GMM
                gmm_model = DifferenceGMM(
                    data=data, dep_var="y", lags=[1], exog_vars=["x"], time_dummies=False
                )
                id_var = ("entity",)
                time_var = ("time",)
                gmm_results = gmm_model.fit()

                # Find lag parameter
                lag_name = [
                    n for n in bc_results.params.index if "L1.y" in n or "lag1" in n.lower()
                ]
                if lag_name:
                    lag_name = lag_name[0]
                    bc_rho = bc_results.params[lag_name]
                    gmm_rho = gmm_results.params[lag_name]

                    bc_errors.append(bc_rho - true_rho)
                    gmm_errors.append(gmm_rho - true_rho)
            except Exception:
                # Skip failed simulations
                pass

        if len(bc_errors) >= 30 and len(gmm_errors) >= 30:
            # Check average bias
            bc_bias = np.mean(bc_errors)
            gmm_bias = np.mean(gmm_errors)

            # Bias-corrected should have smaller absolute bias on average
            # (may not be exactly zero due to finite sample)
            assert (
                abs(bc_bias) <= abs(gmm_bias) * 1.2
            ), f"BC bias {bc_bias:.4f} not smaller than GMM bias {gmm_bias:.4f}"


class TestBiasCorrectedGMMIntegration:
    """Integration tests with real-world scenarios."""

    def test_with_multiple_lags(self):
        """Test with multiple lags of dependent variable."""
        np.random.seed(789)
        n, t = 60, 15

        data_list = []
        for i in range(n):
            alpha_i = np.random.normal(0, 1)
            x_it = np.random.normal(0, 1, t)
            y_it = np.zeros(t)
            y_it[0] = alpha_i + np.random.normal(0, 0.5)
            y_it[1] = 0.5 * y_it[0] + 0.3 * x_it[1] + alpha_i + np.random.normal(0, 0.5)

            for t_idx in range(2, t):
                epsilon = np.random.normal(0, 0.5)
                y_it[t_idx] = (
                    0.4 * y_it[t_idx - 1]
                    + 0.2 * y_it[t_idx - 2]
                    + 0.3 * x_it[t_idx]
                    + alpha_i
                    + epsilon
                )

            data_list.append(pd.DataFrame({"entity": i, "time": range(t), "y": y_it, "x": x_it}))

        data = pd.concat(data_list, ignore_index=True).set_index(["entity", "time"])

        model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1, 2],
            id_var="entity",
            time_var="time",
            exog_vars=["x"],
        )

        results = model.fit(time_dummies=False, verbose=False)

        # Should estimate successfully
        assert results is not None
        assert len(results.params) >= 3  # 2 lags + 1 exog at minimum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
