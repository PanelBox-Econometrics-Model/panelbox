"""
Test GMM invariance to variable reordering.

Per FASE_2.md line 105: GMM estimates should be invariant to variable reordering.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.gmm import estimate_panel_var_gmm


class TestGMMInvariance:
    """Test GMM invariance properties."""

    def test_invariance_to_variable_reordering(self):
        """
        ACCEPTANCE TEST (FASE_2.md line 105):
        GMM estimates must be invariant to variable reordering.

        If we estimate VAR([y1, y2]) and VAR([y2, y1]), the coefficient
        matrices should be consistent when accounting for reordering.
        """
        np.random.seed(123)

        # Generate data
        A1_true = np.array([[0.5, 0.2], [0.1, 0.6]])
        n_entities = 50
        n_periods = 12

        data_list = []
        for entity in range(1, n_entities + 1):
            y_prev = np.random.randn(2) * 0.5
            for t in range(1, n_periods + 1):
                epsilon = np.random.randn(2) * 0.3
                y = A1_true @ y_prev + epsilon
                data_list.append({"entity": entity, "time": t, "y1": y[0], "y2": y[1]})
                y_prev = y.copy()

        df = pd.DataFrame(data_list)

        # Estimate with original order [y1, y2]
        result_original = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            transform="fod",
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Estimate with reversed order [y2, y1]
        result_reordered = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y2", "y1"],  # Reversed
            transform="fod",
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Extract coefficient matrices
        # Original: A1[i,j] = effect of y_j on y_i
        # Reordered: A1_reord[i,j] = effect of y_j_reord on y_i_reord
        # where y_reord = [y2, y1]
        #
        # Mapping:
        # A1_orig[0,0] = y1 <- y1 = A1_reord[1,1] (y1 <- y1)
        # A1_orig[0,1] = y1 <- y2 = A1_reord[1,0] (y1 <- y2)
        # A1_orig[1,0] = y2 <- y1 = A1_reord[0,1] (y2 <- y1)
        # A1_orig[1,1] = y2 <- y2 = A1_reord[0,0] (y2 <- y2)

        A1_orig = result_original.coefficients
        A1_reord = result_reordered.coefficients

        # Reconstruct A1_orig from A1_reord by reversing indices
        A1_reconstructed = np.array(
            [
                [A1_reord[1, 1], A1_reord[1, 0]],  # y1 <- [y1, y2]
                [A1_reord[0, 1], A1_reord[0, 0]],  # y2 <- [y1, y2]
            ]
        )

        # Check invariance: A1_orig should equal A1_reconstructed
        max_diff = np.max(np.abs(A1_orig - A1_reconstructed))

        # Tolerance: coefficients should be nearly identical
        # Allow small numerical error (1e-6)
        assert max_diff < 1e-6, (
            f"GMM NOT INVARIANT to variable reordering: max_diff={max_diff:.8f}\n"
            f"Original order [y1, y2]:\n{A1_orig}\n"
            f"Reordered [y2, y1] mapped back:\n{A1_reconstructed}\n"
            f"Difference:\n{A1_orig - A1_reconstructed}"
        )

        print("\n" + "=" * 70)
        print("INVARIANCE TEST RESULT (FASE 2 - US-2.1 line 105)")
        print("=" * 70)
        print(f"Original order [y1, y2]:\n{A1_orig}")
        print(f"\nReordered [y2, y1] mapped back:\n{A1_reconstructed}")
        print(f"\nMax difference: {max_diff:.8f}")
        print(f"Status: {'PASS ✓' if max_diff < 1e-6 else 'FAIL ✗'}")
        print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
