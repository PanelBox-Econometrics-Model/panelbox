"""
Tests for Granger causality in Panel VAR.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality import (
    GrangerCausalityResult,
    construct_granger_restriction_matrix,
    granger_causality_matrix,
    granger_causality_wald,
)


def test_construct_granger_restriction_matrix():
    """Test restriction matrix construction."""
    exog_names = ["L1.x", "L1.y", "L2.x", "L2.y", "L3.x", "L3.y", "const"]
    causing_var = "x"
    lags = 3

    R = construct_granger_restriction_matrix(exog_names, causing_var, lags)

    # Should be 3x7 matrix
    assert R.shape == (3, 7)

    # Should have 1s at positions corresponding to L1.x, L2.x, L3.x
    assert R[0, 0] == 1  # L1.x
    assert R[1, 2] == 1  # L2.x
    assert R[2, 4] == 1  # L3.x

    # All other entries should be 0
    assert np.sum(R) == 3


def test_construct_granger_restriction_matrix_missing_lag():
    """Test error when lag is missing."""
    exog_names = ["L1.x", "L1.y", "const"]
    causing_var = "x"
    lags = 2  # But only L1.x is present

    with pytest.raises(ValueError, match="Lag 2 of variable"):
        construct_granger_restriction_matrix(exog_names, causing_var, lags)


def test_granger_causality_result_dataclass():
    """Test GrangerCausalityResult dataclass."""
    result = GrangerCausalityResult(
        cause="x",
        effect="y",
        wald_stat=15.5,
        f_stat=5.17,
        df=3,
        p_value=0.001,
        p_value_f=0.002,
        lags_tested=3,
    )

    assert result.cause == "x"
    assert result.effect == "y"
    assert result.wald_stat == 15.5
    assert result.df == 3
    assert result.p_value == 0.001

    # Check automatic hypothesis generation
    assert "does not Granger-cause" in result.hypothesis

    # Check automatic conclusion generation
    assert "Rejects H0" in result.conclusion
    assert "***" in result.conclusion  # p < 0.01

    # Test summary method
    summary = result.summary()
    assert "Granger Causality Test" in summary
    assert "x" in summary
    assert "y" in summary


def test_granger_causality_result_significance_levels():
    """Test significance level interpretation in conclusions."""
    # p < 0.01
    r1 = GrangerCausalityResult("x", "y", 20, 10, 2, 0.005)
    assert "***" in r1.conclusion

    # 0.01 <= p < 0.05
    r2 = GrangerCausalityResult("x", "y", 8, 4, 2, 0.02)
    assert "**" in r2.conclusion

    # 0.05 <= p < 0.10
    r3 = GrangerCausalityResult("x", "y", 4, 2, 2, 0.07)
    assert "*" in r3.conclusion

    # p >= 0.10
    r4 = GrangerCausalityResult("x", "y", 2, 1, 2, 0.15)
    assert "Fails to reject" in r4.conclusion


def test_granger_causality_wald_basic():
    """Test basic Granger causality Wald test."""
    # Create simple coefficient vector and covariance
    # Equation: y = 0.5 L1.x + 0.3 L2.x + 0.4 L1.y + 0.2 L2.y + const
    params = np.array([0.1, 0.5, 0.4, 0.3, 0.2])  # [const, L1.x, L1.y, L2.x, L2.y]
    cov_params = np.eye(5) * 0.01  # Simple diagonal covariance

    exog_names = ["const", "L1.x", "L1.y", "L2.x", "L2.y"]

    result = granger_causality_wald(
        params=params,
        cov_params=cov_params,
        exog_names=exog_names,
        causing_var="x",
        caused_var="y",
        lags=2,
        n_obs=100,
    )

    # Should test L1.x and L2.x jointly
    assert result.cause == "x"
    assert result.effect == "y"
    assert result.df == 2
    assert result.lags_tested == 2

    # With non-zero coefficients (0.5, 0.3) and small SE (0.1), should reject
    assert result.p_value < 0.05

    # F-statistic should be Wald/df
    assert abs(result.f_stat - result.wald_stat / 2) < 1e-10


def test_granger_causality_wald_no_causality():
    """Test Granger test when there is no causality."""
    # Coefficients of x lags are zero
    params = np.array([0.1, 0.0, 0.4, 0.0, 0.2])  # [const, L1.x, L1.y, L2.x, L2.y]
    cov_params = np.eye(5) * 0.01

    exog_names = ["const", "L1.x", "L1.y", "L2.x", "L2.y"]

    result = granger_causality_wald(
        params=params,
        cov_params=cov_params,
        exog_names=exog_names,
        causing_var="x",
        caused_var="y",
        lags=2,
        n_obs=100,
    )

    # With zero coefficients, should NOT reject (high p-value)
    # Note: With exactly zero and small SE, p-value will be very close to 1
    assert result.p_value > 0.10


def test_granger_causality_matrix_structure():
    """Test granger causality matrix structure."""

    # Create mock result object
    class MockResult:
        def __init__(self):
            self.K = 3
            self.endog_names = ["x", "y", "z"]

        def granger_causality(self, cause, effect):
            # Mock result: return low p-value if x causes y, high otherwise
            class MockGCResult:
                def __init__(self, p):
                    self.p_value = p

            if cause == "x" and effect == "y":
                return MockGCResult(0.01)
            elif cause == "y" and effect == "z":
                return MockGCResult(0.03)
            else:
                return MockGCResult(0.50)

    result = MockResult()
    matrix = granger_causality_matrix(result, significance_level=0.05)

    # Should be 3x3
    assert matrix.shape == (3, 3)

    # Diagonal should be NaN
    assert pd.isna(matrix.loc["x", "x"])
    assert pd.isna(matrix.loc["y", "y"])
    assert pd.isna(matrix.loc["z", "z"])

    # Check specific values
    assert matrix.loc["x", "y"] == 0.01  # x causes y
    assert matrix.loc["y", "z"] == 0.03  # y causes z
    assert matrix.loc["y", "x"] == 0.50  # y does not cause x


class TestGrangerCausalityDGP:
    """Test Granger causality with simulated DGP."""

    def generate_var_data(self, N=50, T=100, rho_x_to_y=0.5, rho_y_to_x=0.0, seed=42):
        """
        Generate VAR(1) data with known Granger causality structure.

        Parameters
        ----------
        N : int
            Number of entities
        T : int
            Time periods
        rho_x_to_y : float
            Coefficient of x_{t-1} in y_t equation (Granger causality strength)
        rho_y_to_x : float
            Coefficient of y_{t-1} in x_t equation
        seed : int
            Random seed

        Returns
        -------
        data : pd.DataFrame
            Panel data with columns ['entity', 'time', 'x', 'y']
        """
        np.random.seed(seed)

        data_list = []

        for i in range(N):
            x = np.zeros(T)
            y = np.zeros(T)

            # Initial values
            x[0] = np.random.normal(0, 1)
            y[0] = np.random.normal(0, 1)

            # VAR(1) dynamics
            for t in range(1, T):
                x[t] = 0.3 * x[t - 1] + rho_y_to_x * y[t - 1] + np.random.normal(0, 0.5)
                y[t] = 0.3 * y[t - 1] + rho_x_to_y * x[t - 1] + np.random.normal(0, 0.5)

            # Create DataFrame for this entity
            entity_df = pd.DataFrame({"entity": i, "time": np.arange(T), "x": x, "y": y})

            data_list.append(entity_df)

        return pd.concat(data_list, ignore_index=True)

    @pytest.fixture
    def var_result_x_causes_y(self):
        """VAR result where x causes y but not vice versa."""
        from panelbox.var.data import PanelVARData
        from panelbox.var.model import PanelVAR

        df = self.generate_var_data(N=30, T=80, rho_x_to_y=0.5, rho_y_to_x=0.0)

        data = PanelVARData(
            data=df, endog_vars=["x", "y"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data=data)
        result = model.fit(method="ols", cov_type="clustered")
        return result

    @pytest.fixture
    def var_result_no_causality(self):
        """VAR result with no Granger causality."""
        from panelbox.var.data import PanelVARData
        from panelbox.var.model import PanelVAR

        df = self.generate_var_data(N=30, T=80, rho_x_to_y=0.0, rho_y_to_x=0.0)

        data = PanelVARData(
            data=df, endog_vars=["x", "y"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data=data)
        result = model.fit(method="ols", cov_type="clustered")
        return result

    def test_granger_causality_detects_x_to_y(self, var_result_x_causes_y):
        """Test that Granger test detects x→y causality."""
        result = var_result_x_causes_y

        # Test x → y (should reject H0)
        gc_x_to_y = result.granger_causality("x", "y")
        assert gc_x_to_y.p_value < 0.05, "Should detect x→y causality"

        # Test y → x (should NOT reject H0)
        gc_y_to_x = result.granger_causality("y", "x")
        assert gc_y_to_x.p_value > 0.05, "Should NOT detect y→x causality"

    def test_granger_causality_no_causality(self, var_result_no_causality):
        """Test that Granger test does not detect causality when there is none."""
        result = var_result_no_causality

        # Both tests should NOT reject
        gc_x_to_y = result.granger_causality("x", "y")
        gc_y_to_x = result.granger_causality("y", "x")

        assert gc_x_to_y.p_value > 0.05
        assert gc_y_to_x.p_value > 0.05

    def test_granger_causality_matrix_dgp(self, var_result_x_causes_y):
        """Test Granger causality matrix with DGP."""
        result = var_result_x_causes_y

        matrix = result.granger_causality_matrix()

        # Check diagonal is NaN
        assert pd.isna(matrix.loc["x", "x"])
        assert pd.isna(matrix.loc["y", "y"])

        # Check x→y is significant
        assert matrix.loc["x", "y"] < 0.05

        # Check y→x is not significant
        assert matrix.loc["y", "x"] > 0.05


def test_granger_causality_matrix_errors():
    """Test error handling in granger_causality_matrix."""

    class MockResultWithErrors:
        def __init__(self):
            self.K = 2
            self.endog_names = ["x", "y"]

        def granger_causality(self, cause, effect):
            # Always raise an error
            raise ValueError("Test error")

    result = MockResultWithErrors()
    matrix = granger_causality_matrix(result)

    # Should have NaN for all off-diagonal elements due to errors
    assert pd.isna(matrix.loc["x", "y"])
    assert pd.isna(matrix.loc["y", "x"])
