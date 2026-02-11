"""
Tests for compare_transforms function - comparing FOD vs FD transformations.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.diagnostics import compare_transforms


class TestCompareTransforms:
    """Test FOD vs FD transformation comparison."""

    @pytest.fixture
    def balanced_panel_data(self):
        """Create balanced panel data for testing."""
        np.random.seed(42)

        N = 30  # entities
        T = 15  # time periods
        K = 2  # variables

        data = []
        for i in range(N):
            for t in range(T):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn() + 0.5 * (i % 3),  # Some entity FE
                        "y2": np.random.randn() + 0.3 * (i % 2),
                    }
                )

        return pd.DataFrame(data)

    @pytest.fixture
    def unbalanced_panel_data(self):
        """Create unbalanced panel data for testing."""
        np.random.seed(123)

        data = []
        # Entity 0: Full time series
        for t in range(20):
            data.append({"entity": 0, "time": t, "y1": np.random.randn(), "y2": np.random.randn()})

        # Entity 1: Missing some periods
        for t in [0, 1, 2, 3, 5, 7, 9, 12, 15, 18]:
            data.append({"entity": 1, "time": t, "y1": np.random.randn(), "y2": np.random.randn()})

        # Entity 2: Short series
        for t in range(8):
            data.append({"entity": 2, "time": t, "y1": np.random.randn(), "y2": np.random.randn()})

        # Add more entities with varying lengths
        for i in range(3, 25):
            T_i = np.random.randint(10, 18)
            for t in range(T_i):
                data.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )

        return pd.DataFrame(data)

    def test_compare_transforms_balanced_panel(self, balanced_panel_data):
        """Test FOD vs FD on balanced panel - should be similar."""
        result = compare_transforms(
            data=balanced_panel_data,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=6,
        )

        # Check all keys present
        assert "fod_result" in result
        assert "fd_result" in result
        assert "n_obs_fod" in result
        assert "n_obs_fd" in result
        assert "coef_diff_max" in result
        assert "coef_diff_mean" in result
        assert "interpretation" in result
        assert "summary" in result

        # Results should exist
        assert result["fod_result"] is not None
        assert result["fd_result"] is not None

        # Observations should be close (balanced panel)
        obs_diff = abs(result["n_obs_fod"] - result["n_obs_fd"])
        total_obs = max(result["n_obs_fod"], result["n_obs_fd"])
        obs_diff_pct = obs_diff / total_obs * 100

        # For balanced panel, FOD and FD should have same or very similar n_obs
        assert obs_diff_pct < 10.0, f"Observation difference too large: {obs_diff_pct:.1f}%"

        # Coefficients should not be wildly different
        assert result["coef_diff_max"] < 1.0, "Coefficient differences too large"

        # Summary should be formatted
        assert len(result["summary"]) > 0
        assert "FOD vs FD" in result["summary"]

    def test_compare_transforms_unbalanced_panel(self, unbalanced_panel_data):
        """Test FOD vs FD on unbalanced panel - FOD should preserve more obs."""
        result = compare_transforms(
            data=unbalanced_panel_data,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=6,
        )

        # Results should exist
        assert result["fod_result"] is not None
        assert result["fd_result"] is not None

        # FOD should have more or equal observations than FD
        # (This is the key advantage of FOD in unbalanced panels)
        assert result["n_obs_fod"] >= result["n_obs_fd"]

        # If there's a significant difference, interpretation should mention it
        obs_diff = result["n_obs_fod"] - result["n_obs_fd"]
        if obs_diff > 5:
            # Should be mentioned in summary or recommendation
            summary_lower = result["summary"].lower()
            assert "observation" in summary_lower or "obs" in summary_lower

    def test_compare_transforms_interpretation_levels(self):
        """Test interpretation thresholds for different levels of divergence."""
        np.random.seed(456)

        # Create simple data
        data = []
        for i in range(20):
            for t in range(12):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn() * 0.5,
                        "y2": np.random.randn() * 0.5,
                    }
                )
        df = pd.DataFrame(data)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="one-step",  # Faster for testing
            instrument_type="collapsed",
            max_instruments=4,
        )

        # Should have an interpretation
        assert "interpretation" in result
        assert len(result["interpretation"]) > 0

        # Interpretation should be one of the expected levels
        interp = result["interpretation"]
        valid_markers = ["EXCELLENT", "GOOD", "MODERATE", "WARNING"]
        assert any(marker in interp for marker in valid_markers)

    def test_compare_transforms_with_different_gmm_steps(self):
        """Test comparison works with both one-step and two-step GMM."""
        np.random.seed(789)

        data = []
        for i in range(15):
            for t in range(10):
                data.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )
        df = pd.DataFrame(data)

        # Test with one-step
        result_1step = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="one-step",
            instrument_type="collapsed",
        )
        assert result_1step["fod_result"] is not None
        assert result_1step["fd_result"] is not None

        # Test with two-step
        result_2step = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="two-step",
            instrument_type="collapsed",
        )
        assert result_2step["fod_result"] is not None
        assert result_2step["fd_result"] is not None

    def test_compare_transforms_coefficient_differences(self):
        """Test coefficient difference calculations."""
        np.random.seed(321)

        data = []
        for i in range(25):
            for t in range(15):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": 0.6 * np.random.randn(),
                        "y2": 0.4 * np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=6,
        )

        # Check coefficient difference metrics
        assert "coef_diff_max" in result
        assert "coef_diff_mean" in result
        assert "coef_diff_pct_max" in result
        assert "coef_diff_pct_mean" in result

        # Max should be >= mean
        assert result["coef_diff_max"] >= result["coef_diff_mean"]
        assert result["coef_diff_pct_max"] >= result["coef_diff_pct_mean"]

        # All should be non-negative
        assert result["coef_diff_max"] >= 0
        assert result["coef_diff_mean"] >= 0
        assert result["coef_diff_pct_max"] >= 0
        assert result["coef_diff_pct_mean"] >= 0

    def test_compare_transforms_summary_format(self):
        """Test summary output is well-formatted."""
        np.random.seed(654)

        data = []
        for i in range(20):
            for t in range(12):
                data.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )
        df = pd.DataFrame(data)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="two-step",
            instrument_type="all",
            max_instruments=8,
        )

        summary = result["summary"]

        # Check summary contains key information
        assert "FOD vs FD" in summary
        assert "Observations:" in summary
        assert "Instruments:" in summary
        assert "Coefficient Differences:" in summary
        assert "INTERPRETATION:" in summary
        assert "RECOMMENDATION:" in summary

        # Check observation counts are shown
        assert str(result["n_obs_fod"]) in summary
        assert str(result["n_obs_fd"]) in summary

        # Check instrument counts are shown
        assert str(result["n_instruments_fod"]) in summary
        assert str(result["n_instruments_fd"]) in summary

    def test_compare_transforms_with_var2(self):
        """Test comparison with VAR(2) model."""
        np.random.seed(987)

        data = []
        for i in range(30):
            for t in range(18):
                data.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )
        df = pd.DataFrame(data)

        result = compare_transforms(
            data=df,
            var_lags=2,  # VAR(2)
            value_cols=["y1", "y2"],
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=10,
        )

        # Should work with VAR(2)
        assert result["fod_result"] is not None
        assert result["fd_result"] is not None

        # Both should have fewer observations due to 2 lags
        assert result["n_obs_fod"] > 0
        assert result["n_obs_fd"] > 0

    def test_compare_transforms_error_handling(self):
        """Test error handling in compare_transforms."""
        # Test with insufficient data
        small_data = pd.DataFrame(
            {
                "entity": [0, 0, 1, 1],
                "time": [0, 1, 0, 1],
                "y1": [1.0, 2.0, 3.0, 4.0],
                "y2": [5.0, 6.0, 7.0, 8.0],
            }
        )

        # This might fail due to insufficient observations
        # We just check it doesn't crash catastrophically
        result = compare_transforms(
            data=small_data,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="one-step",
            instrument_type="collapsed",
            max_instruments=2,
        )

        # Should either succeed or return error
        if "error" in result:
            assert result["fod_result"] is None or result["fd_result"] is None
        else:
            # If it succeeded, check basic structure
            assert "fod_result" in result
            assert "fd_result" in result

    def test_compare_transforms_recommendation_logic(self):
        """Test that recommendation varies with divergence level."""
        np.random.seed(111)

        data = []
        for i in range(20):
            for t in range(12):
                data.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )
        df = pd.DataFrame(data)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="two-step",
            instrument_type="collapsed",
        )

        # Should have a recommendation
        assert "recommendation" in result
        assert len(result["recommendation"]) > 0

        # Recommendation should provide actionable guidance
        rec = result["recommendation"].lower()

        # Should mention either FOD, FD, or diagnostic actions
        relevant_keywords = ["fod", "fd", "transformation", "hansen", "diagnostic", "appropriate"]
        assert any(kw in rec for kw in relevant_keywords)

    def test_compare_transforms_prints_summary(self, capsys):
        """Test that summary can be printed without error."""
        np.random.seed(222)

        data = []
        for i in range(15):
            for t in range(10):
                data.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )
        df = pd.DataFrame(data)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="one-step",
            instrument_type="collapsed",
        )

        # Print summary - should not crash
        print(result["summary"])

        captured = capsys.readouterr()
        assert "FOD vs FD" in captured.out
