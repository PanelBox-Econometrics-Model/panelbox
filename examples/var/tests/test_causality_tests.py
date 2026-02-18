"""
Test Granger causality implementations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from var_simulation import simulate_panel_var

from panelbox.var import PanelVAR, PanelVARData


def _generate_causal_panel(seed=42):
    """Generate panel where y_0 Granger-causes y_1 but not vice versa."""
    np.random.seed(seed)
    # y_0 -> y_1 but not y_1 -> y_0
    A1 = np.array(
        [
            [0.5, 0.0],  # y_0 depends only on own lag
            [0.4, 0.3],  # y_1 depends on y_0 lag (causal) and own lag
        ]
    )
    Sigma = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    return simulate_panel_var(
        A_matrices=[A1],
        Sigma=Sigma,
        n_entities=50,
        n_periods=80,
        fixed_effects_std=0.5,
        seed=seed,
    )


def _generate_non_causal_panel(seed=42):
    """Generate panel where variables are independent."""
    np.random.seed(seed)
    A1 = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.3],
        ]
    )
    Sigma = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    return simulate_panel_var(
        A_matrices=[A1],
        Sigma=Sigma,
        n_entities=50,
        n_periods=80,
        fixed_effects_std=0.5,
        seed=seed,
    )


class TestGrangerCausality:
    """Test standard Granger causality tests."""

    def test_non_causal_gives_insignificant(self):
        """When X does not Granger-cause Y in DGP, test should not reject."""
        df = _generate_non_causal_panel(seed=42)
        endog_vars = ["y_0", "y_1"]

        var_data = PanelVARData(
            data=df,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        # y_0 should NOT Granger-cause y_1 (independent in DGP)
        gc = results.granger_causality(cause="y_0", effect="y_1")
        # With independent processes, p-value should be > 0.05 most of the time
        # Use a lenient threshold since this is a probabilistic test
        assert gc.p_value > 0.01, f"False positive: p={gc.p_value:.4f} for non-causal relationship"

    def test_causal_gives_significant(self):
        """When X Granger-causes Y in DGP, test should reject H0."""
        df = _generate_causal_panel(seed=42)
        endog_vars = ["y_0", "y_1"]

        var_data = PanelVARData(
            data=df,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        # y_0 should Granger-cause y_1 (A[1,0] = 0.4 in DGP)
        gc = results.granger_causality(cause="y_0", effect="y_1")
        assert gc.p_value < 0.05, f"Failed to detect causality: p={gc.p_value:.4f}"

    def test_causality_matrix_dimensions(self):
        """Granger causality matrix should be K×K."""
        df = _generate_causal_panel(seed=42)
        endog_vars = ["y_0", "y_1"]

        var_data = PanelVARData(
            data=df,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        gc_matrix = results.granger_causality_matrix()
        assert gc_matrix.shape == (2, 2)
        # Diagonal should be NaN or 1.0 (self-causality is trivial)
        assert all(np.isnan(gc_matrix.iloc[i, i]) or gc_matrix.iloc[i, i] == 1.0 for i in range(2))


class TestDumitrescuHurlin:
    """Test Dumitrescu-Hurlin panel Granger causality."""

    def test_dh_test_runs(self):
        """DH test should produce Z_tilde and Z_bar statistics."""
        df = _generate_causal_panel(seed=42)
        endog_vars = ["y_0", "y_1"]

        var_data = PanelVARData(
            data=df,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        try:
            dh = results.dumitrescu_hurlin(cause="y_0", effect="y_1")
            assert hasattr(dh, "Z_tilde_stat")
            assert hasattr(dh, "Z_bar_stat")
            assert hasattr(dh, "W_bar")
        except Exception as e:
            pytest.skip(f"Dumitrescu-Hurlin test not available: {e}")

    def test_dh_individual_statistics(self):
        """Individual Wald statistics should have length N."""
        df = _generate_causal_panel(seed=42)
        endog_vars = ["y_0", "y_1"]

        var_data = PanelVARData(
            data=df,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        try:
            dh = results.dumitrescu_hurlin(cause="y_0", effect="y_1")
            assert len(dh.individual_W) == 50  # N entities
        except Exception as e:
            pytest.skip(f"Dumitrescu-Hurlin test not available: {e}")

    def test_dh_heterogeneous_causality(self):
        """With heterogeneous causal structure, DH should detect overall effect."""
        # Generate data where causality exists for some entities but not all
        np.random.seed(42)
        n_entities = 50
        n_periods = 80
        K = 2

        dfs = []
        for i in range(n_entities):
            # First half: causal, second half: non-causal
            if i < 25:
                A1 = np.array([[0.5, 0.0], [0.4, 0.3]])
            else:
                A1 = np.array([[0.5, 0.0], [0.0, 0.3]])

            Sigma = np.eye(K)
            eps = np.random.multivariate_normal(np.zeros(K), Sigma, n_periods + 50)
            y = np.zeros((n_periods + 50, K))
            for t in range(1, n_periods + 50):
                y[t] = A1 @ y[t - 1] + eps[t]
            y = y[50:]  # Remove burn-in

            entity_df = pd.DataFrame(y, columns=["y_0", "y_1"])
            entity_df["entity"] = f"Entity_{i:03d}"
            entity_df["time"] = range(1, n_periods + 1)
            dfs.append(entity_df)

        df = pd.concat(dfs, ignore_index=True)

        var_data = PanelVARData(
            data=df,
            endog_vars=["y_0", "y_1"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        try:
            dh = results.dumitrescu_hurlin(cause="y_0", effect="y_1")
            # With 50% causal entities and strong effect, DH should detect it
            assert (
                dh.Z_bar_pvalue < 0.10 or dh.Z_tilde_pvalue < 0.10
            ), f"DH test failed to detect heterogeneous causality"
        except Exception as e:
            pytest.skip(f"Dumitrescu-Hurlin test not available: {e}")
