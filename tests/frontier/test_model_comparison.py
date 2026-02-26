"""
Tests for model comparison tools for four-component SFA.

This module tests the model comparison functions that compare the
four-component SFA model with simpler alternatives (Pitt-Lee, True FE/RE).
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.advanced import (
    FourComponentSFA,
    ModelComparisonResult,
    compare_all_models,
    compare_with_pitt_lee,
    compare_with_true_effects,
)


@pytest.fixture(scope="module")
def four_comp_result():
    """Fit a FourComponentSFA model on small synthetic data for reuse across tests.

    Generates a balanced panel with N=20 entities and T=8 periods,
    fits the four-component model, and returns the result.
    """
    np.random.seed(42)
    N, T = 20, 8

    # True parameters
    beta_true = np.array([2.0, 0.5, -0.3])
    sigma_v_true = 0.15
    sigma_u_true = 0.25
    sigma_mu_true = 0.30
    sigma_eta_true = 0.20

    data = []
    mu_i_true = np.random.normal(0, sigma_mu_true, N)
    eta_i_true = np.abs(np.random.normal(0, sigma_eta_true, N))

    for i in range(N):
        for t in range(T):
            x1 = np.random.normal(0, 1)
            x2 = np.random.normal(0, 1)
            X_it = np.array([1.0, x1, x2])

            v_it = np.random.normal(0, sigma_v_true)
            u_it = np.abs(np.random.normal(0, sigma_u_true))

            y_it = X_it @ beta_true + mu_i_true[i] - eta_i_true[i] + v_it - u_it

            data.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x1": x1,
                    "x2": x2,
                }
            )

    df = pd.DataFrame(data)

    model = FourComponentSFA(
        data=df,
        depvar="y",
        exog=["x1", "x2"],
        entity="entity",
        time="time",
        frontier_type="production",
    )

    return model.fit(verbose=False)


class TestCompareWithPittLee:
    """Tests for compare_with_pitt_lee function."""

    def test_returns_model_comparison_result(self, four_comp_result):
        """compare_with_pitt_lee should return a ModelComparisonResult."""
        result = compare_with_pitt_lee(four_comp_result)
        assert isinstance(result, ModelComparisonResult)

    def test_model_names(self, four_comp_result):
        """Result should contain Four-Component and Pitt-Lee model names."""
        result = compare_with_pitt_lee(four_comp_result)
        assert result.model_names == ["Four-Component", "Pitt-Lee"]

    def test_log_likelihoods_are_finite(self, four_comp_result):
        """All log-likelihoods should be finite numbers."""
        result = compare_with_pitt_lee(four_comp_result)
        for model_name, ll in result.log_likelihoods.items():
            assert np.isfinite(ll), f"Log-likelihood for {model_name} is not finite: {ll}"

    def test_aic_bic_are_finite(self, four_comp_result):
        """AIC and BIC values should be finite for all models."""
        result = compare_with_pitt_lee(four_comp_result)
        for model_name in result.model_names:
            assert np.isfinite(result.aics[model_name]), f"AIC for {model_name} is not finite"
            assert np.isfinite(result.bics[model_name]), f"BIC for {model_name} is not finite"

    def test_variance_shares_sum_to_100(self, four_comp_result):
        """Variance shares for each model should sum to approximately 100%."""
        result = compare_with_pitt_lee(four_comp_result)
        for model_name, shares in result.variance_shares.items():
            total = sum(shares.values())
            assert abs(total - 100.0) < 1e-6, (
                f"Variance shares for {model_name} sum to {total}, expected ~100%"
            )

    def test_efficiency_correlation_is_dataframe(self, four_comp_result):
        """Efficiency correlation should be a valid DataFrame."""
        result = compare_with_pitt_lee(four_comp_result)
        assert isinstance(result.efficiency_correlation, pd.DataFrame)
        assert result.efficiency_correlation.shape[0] > 0
        assert result.efficiency_correlation.shape[1] > 0

    def test_efficiency_correlation_diagonal_is_one(self, four_comp_result):
        """Diagonal of correlation matrix should be 1.0."""
        result = compare_with_pitt_lee(four_comp_result)
        corr = result.efficiency_correlation
        for idx in corr.index:
            if idx in corr.columns:
                assert abs(corr.loc[idx, idx] - 1.0) < 1e-6, (
                    f"Diagonal element for {idx} is {corr.loc[idx, idx]}, expected 1.0"
                )

    def test_four_component_result_preserved(self, four_comp_result):
        """The original four_component result should be preserved in output."""
        result = compare_with_pitt_lee(four_comp_result)
        assert result.four_component is four_comp_result

    def test_variance_shares_keys(self, four_comp_result):
        """Four-Component model should have 4 variance share components."""
        result = compare_with_pitt_lee(four_comp_result)
        fc_shares = result.variance_shares["Four-Component"]
        assert len(fc_shares) == 4

    def test_pitt_lee_has_two_variance_components(self, four_comp_result):
        """Pitt-Lee model should have 2 variance share components."""
        result = compare_with_pitt_lee(four_comp_result)
        pl_shares = result.variance_shares["Pitt-Lee"]
        assert len(pl_shares) == 2


class TestCompareWithTrueEffects:
    """Tests for compare_with_true_effects function."""

    def test_returns_model_comparison_result(self, four_comp_result):
        """compare_with_true_effects should return a ModelComparisonResult."""
        result = compare_with_true_effects(four_comp_result)
        assert isinstance(result, ModelComparisonResult)

    def test_model_names(self, four_comp_result):
        """Result should contain Four-Component, True FE, and True RE."""
        result = compare_with_true_effects(four_comp_result)
        assert result.model_names == ["Four-Component", "True FE", "True RE"]

    def test_log_likelihoods_are_finite(self, four_comp_result):
        """All log-likelihoods should be finite."""
        result = compare_with_true_effects(four_comp_result)
        for model_name, ll in result.log_likelihoods.items():
            assert np.isfinite(ll), f"Log-likelihood for {model_name} is not finite: {ll}"

    def test_aic_bic_are_finite(self, four_comp_result):
        """AIC and BIC values should be finite."""
        result = compare_with_true_effects(four_comp_result)
        for model_name in result.model_names:
            assert np.isfinite(result.aics[model_name]), f"AIC for {model_name} is not finite"
            assert np.isfinite(result.bics[model_name]), f"BIC for {model_name} is not finite"

    def test_variance_shares_sum_to_100(self, four_comp_result):
        """Variance shares for each model should sum to approximately 100%."""
        result = compare_with_true_effects(four_comp_result)
        for model_name, shares in result.variance_shares.items():
            total = sum(shares.values())
            assert abs(total - 100.0) < 1e-6, (
                f"Variance shares for {model_name} sum to {total}, expected ~100%"
            )

    def test_efficiency_correlation_is_dataframe(self, four_comp_result):
        """Efficiency correlation should be a valid DataFrame."""
        result = compare_with_true_effects(four_comp_result)
        assert isinstance(result.efficiency_correlation, pd.DataFrame)

    def test_true_fe_re_have_same_ll(self, four_comp_result):
        """True FE and True RE should have the same log-likelihood
        (both use same residual variance in approximation)."""
        result = compare_with_true_effects(four_comp_result)
        assert abs(result.log_likelihoods["True FE"] - result.log_likelihoods["True RE"]) < 1e-6

    def test_true_fe_re_correlation_with_four_component(self, four_comp_result):
        """True FE/RE assume perfect efficiency so correlation with 4C should be 0."""
        result = compare_with_true_effects(four_comp_result)
        corr = result.efficiency_correlation
        # True FE and True RE should have 0 correlation with four-component efficiencies
        assert corr.loc["True FE", "Overall (4C)"] == 0.0
        assert corr.loc["True RE", "Overall (4C)"] == 0.0

    def test_four_component_result_preserved(self, four_comp_result):
        """The original four_component result should be preserved."""
        result = compare_with_true_effects(four_comp_result)
        assert result.four_component is four_comp_result


class TestCompareAllModels:
    """Tests for compare_all_models function."""

    def test_returns_model_comparison_result(self, four_comp_result):
        """compare_all_models should return a ModelComparisonResult."""
        result = compare_all_models(four_comp_result)
        assert isinstance(result, ModelComparisonResult)

    def test_model_names_has_four_models(self, four_comp_result):
        """Result should contain all 4 model names."""
        result = compare_all_models(four_comp_result)
        assert len(result.model_names) == 4
        assert result.model_names == ["Four-Component", "Pitt-Lee", "True FE", "True RE"]

    def test_log_likelihoods_are_finite(self, four_comp_result):
        """All log-likelihoods should be finite."""
        result = compare_all_models(four_comp_result)
        assert len(result.log_likelihoods) == 4
        for model_name, ll in result.log_likelihoods.items():
            assert np.isfinite(ll), f"Log-likelihood for {model_name} is not finite: {ll}"

    def test_aic_bic_are_finite(self, four_comp_result):
        """All AIC and BIC values should be finite."""
        result = compare_all_models(four_comp_result)
        assert len(result.aics) == 4
        assert len(result.bics) == 4
        for model_name in result.model_names:
            assert np.isfinite(result.aics[model_name]), f"AIC for {model_name} is not finite"
            assert np.isfinite(result.bics[model_name]), f"BIC for {model_name} is not finite"

    def test_variance_shares_sum_to_100(self, four_comp_result):
        """Variance shares for each model should sum to approximately 100%."""
        result = compare_all_models(four_comp_result)
        for model_name, shares in result.variance_shares.items():
            total = sum(shares.values())
            assert abs(total - 100.0) < 1e-6, (
                f"Variance shares for {model_name} sum to {total}, expected ~100%"
            )

    def test_efficiency_correlation_is_dataframe(self, four_comp_result):
        """Efficiency correlation should be a valid DataFrame."""
        result = compare_all_models(four_comp_result)
        assert isinstance(result.efficiency_correlation, pd.DataFrame)
        # The correlation matrix from compare_all_models uses corrcoef with 4 series
        assert result.efficiency_correlation.shape == (4, 4)

    def test_efficiency_correlation_diagonal_is_one(self, four_comp_result):
        """Diagonal of correlation matrix should be 1.0."""
        result = compare_all_models(four_comp_result)
        corr = result.efficiency_correlation
        for i in range(corr.shape[0]):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-6

    def test_efficiency_correlation_symmetric(self, four_comp_result):
        """Correlation matrix should be symmetric."""
        result = compare_all_models(four_comp_result)
        corr = result.efficiency_correlation.values
        assert np.allclose(corr, corr.T, atol=1e-10)

    def test_four_component_has_most_variance_components(self, four_comp_result):
        """Four-Component model should have the most variance components."""
        result = compare_all_models(four_comp_result)
        fc_n = len(result.variance_shares["Four-Component"])
        for model_name in ["Pitt-Lee", "True FE", "True RE"]:
            other_n = len(result.variance_shares[model_name])
            assert fc_n >= other_n, (
                f"Four-Component has {fc_n} components, but {model_name} has {other_n}"
            )

    def test_four_component_result_preserved(self, four_comp_result):
        """The original four_component result should be preserved."""
        result = compare_all_models(four_comp_result)
        assert result.four_component is four_comp_result


class TestPrintSummary:
    """Tests for ModelComparisonResult.print_summary() method."""

    def test_print_summary_pitt_lee(self, four_comp_result, capsys):
        """print_summary should produce output for Pitt-Lee comparison."""
        result = compare_with_pitt_lee(four_comp_result)
        result.print_summary()
        captured = capsys.readouterr()
        assert "MODEL COMPARISON RESULTS" in captured.out
        assert "Pitt-Lee" in captured.out
        assert "Four-Component" in captured.out

    def test_print_summary_true_effects(self, four_comp_result, capsys):
        """print_summary should produce output for True Effects comparison."""
        result = compare_with_true_effects(four_comp_result)
        result.print_summary()
        captured = capsys.readouterr()
        assert "MODEL COMPARISON RESULTS" in captured.out
        assert "True FE" in captured.out
        assert "True RE" in captured.out

    def test_print_summary_all_models(self, four_comp_result, capsys):
        """print_summary should produce output for all-models comparison."""
        result = compare_all_models(four_comp_result)
        result.print_summary()
        captured = capsys.readouterr()
        assert "MODEL COMPARISON RESULTS" in captured.out
        assert "Variance Decomposition" in captured.out
        assert "Efficiency Correlation Matrix" in captured.out
        assert "Model Fit Statistics" in captured.out

    def test_print_summary_contains_numeric_values(self, four_comp_result, capsys):
        """print_summary output should contain numeric AIC/BIC/LL values."""
        result = compare_all_models(four_comp_result)
        result.print_summary()
        captured = capsys.readouterr()
        # The output should contain formatted numbers (e.g. "-123.45")
        # At minimum, the variance decomposition percentages should appear
        assert "%" in captured.out


class TestModelComparisonResultDataclass:
    """Tests for ModelComparisonResult dataclass structure."""

    def test_dataclass_fields(self, four_comp_result):
        """ModelComparisonResult should have all expected fields."""
        result = compare_with_pitt_lee(four_comp_result)
        assert hasattr(result, "four_component")
        assert hasattr(result, "model_names")
        assert hasattr(result, "log_likelihoods")
        assert hasattr(result, "aics")
        assert hasattr(result, "bics")
        assert hasattr(result, "variance_shares")
        assert hasattr(result, "efficiency_correlation")

    def test_log_likelihoods_keys_match_model_names(self, four_comp_result):
        """Log-likelihood dict keys should correspond to model_names."""
        result = compare_all_models(four_comp_result)
        for name in result.model_names:
            assert name in result.log_likelihoods
            assert name in result.aics
            assert name in result.bics
            assert name in result.variance_shares

    def test_aic_bic_relationship(self, four_comp_result):
        """BIC should generally be >= AIC for sufficient n (BIC penalty is larger)."""
        result = compare_all_models(four_comp_result)
        # For n > e^2 ~ 7.39, BIC penalty (log(n)) > AIC penalty (2)
        # Our n = 20*8 = 160, so log(160) ~ 5.07 > 2
        for model_name in result.model_names:
            assert result.bics[model_name] >= result.aics[model_name], (
                f"BIC should be >= AIC for {model_name} with n=160"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
