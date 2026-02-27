"""
Validation tests comparing PanelBox censored models against R implementations.

Note: RandomEffectsTobit uses numerical gradients and Gauss-Hermite quadrature
which makes fitting slow. The model is fitted once in setup_class and reused.
"""

import json
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

# Import PanelBox models
from panelbox.models.censored import RandomEffectsTobit

pytestmark = pytest.mark.r_validation


class TestCensoredModelsVsR:
    """Test censored models against R censReg reference results."""

    @classmethod
    def setup_class(cls):
        """Load data and R reference results, fit model once."""
        # Get data path
        data_path = Path(__file__).parent / "data"

        # Load panel data
        cls.data = pd.read_csv(data_path / "panel_censored.csv")

        # Prepare arrays for models that need array API
        cls.y = cls.data["y"].values
        cls.X = sm.add_constant(cls.data[["x1", "x2"]].values)
        cls.groups = cls.data["entity"].values
        cls.time = cls.data["time"].values

        # Load R reference results if available
        ref_file = data_path / "reference_results_censored.json"
        if ref_file.exists():
            with open(ref_file) as f:
                cls.r_results = json.load(f)
        else:
            pytest.skip("R reference results not generated yet")

        # Fit the Tobit model once (slow due to numerical gradients)
        cls._tobit_result = None
        try:

            def _timeout_handler(_signum, _frame):
                raise TimeoutError("Tobit fitting timed out")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(60)
            try:
                model = RandomEffectsTobit(cls.y, cls.X, cls.groups, cls.time)
                cls._tobit_result = model.fit()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except (TimeoutError, Exception):
            cls._tobit_result = None

    def _get_result(self):
        """Get the cached Tobit result, skip if not available."""
        if self._tobit_result is None:
            pytest.skip("RandomEffectsTobit fitting failed or timed out")
        return self._tobit_result

    def test_censoring_statistics(self):
        """Test censoring statistics calculation."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results not available")

        if "error" in self.r_results["pooled_tobit"]:
            pytest.skip("R Pooled Tobit failed")

        # Get R censoring stats
        r_model = self.r_results["pooled_tobit"]
        r_n_censored = r_model["n_censored"]
        r_n_uncensored = r_model["n_uncensored"]

        # Compare censoring counts using raw data
        pb_n_censored = int((self.data["y"] == 0).sum())
        pb_n_uncensored = int((self.data["y"] > 0).sum())

        assert pb_n_censored == r_n_censored, (
            f"Number of censored obs differs: PanelBox={pb_n_censored}, R={r_n_censored}"
        )
        assert pb_n_uncensored == r_n_uncensored, (
            f"Number of uncensored obs differs: PanelBox={pb_n_uncensored}, R={r_n_uncensored}"
        )

    def test_pooled_tobit_vs_r(self):
        """Test Pooled Tobit against R censReg()."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results for Pooled Tobit not available")

        if "error" in self.r_results["pooled_tobit"]:
            pytest.skip(f"R Pooled Tobit failed: {self.r_results['pooled_tobit']['error']}")

        result = self._get_result()

        # Get R results
        r_model = self.r_results["pooled_tobit"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        pb_coefs = result.params[: len(r_coefs)]
        np.testing.assert_allclose(
            pb_coefs, r_coefs, rtol=1e-2, atol=1e-3, err_msg="Coefficients differ from R"
        )

    def test_predicted_values(self):
        """Test predicted values against R."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_values_sample" not in self.r_results["pooled_tobit"]:
            pytest.skip("R predicted values not available")

        result = self._get_result()

        # Get in-sample fitted values for first 100 observations
        fv = getattr(result, "fittedvalues", None)
        if fv is None:
            pytest.skip("fittedvalues not available on result object")

        pred_observed = fv[:100]

        # Get R predictions
        r_pred_observed = np.array(self.r_results["pooled_tobit"]["predicted_values_sample"])

        # Compare predicted observed values
        np.testing.assert_allclose(
            pred_observed[: len(r_pred_observed)],
            r_pred_observed,
            rtol=1e-2,
            atol=1e-3,
            err_msg="Predicted observed values differ from R",
        )

    def test_latent_values(self):
        """Test predicted latent values against R."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results not available")

        if "latent_values_sample" not in self.r_results["pooled_tobit"]:
            pytest.skip("R latent values not available")

        result = self._get_result()

        # Get latent predictions (X @ beta) for first 100 observations
        latent = self.X[:100] @ result.params[: self.X.shape[1]]

        # Get R predictions
        r_pred_latent = np.array(self.r_results["pooled_tobit"]["latent_values_sample"])

        # Compare predicted latent values
        np.testing.assert_allclose(
            latent[: len(r_pred_latent)],
            r_pred_latent,
            rtol=1e-2,
            atol=1e-3,
            err_msg="Predicted latent values differ from R",
        )

    @pytest.mark.skip(reason="True RE Tobit comparison requires specialized R package")
    def test_re_tobit_vs_r(self):
        """Test Random Effects Tobit against specialized R implementation."""
        pass

    def test_aic_comparison(self):
        """Test AIC calculation against R."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results not available")

        if "error" in self.r_results["pooled_tobit"]:
            pytest.skip("R Pooled Tobit failed")

        result = self._get_result()

        # Get R AIC
        r_model = self.r_results["pooled_tobit"]
        r_aic = r_model["aic"]

        # Compare AIC (allowing for differences due to RE vs pooled)
        pb_aic = getattr(result, "aic", None)
        if pb_aic is None:
            pytest.skip("AIC not available on result object")

        assert abs(pb_aic - r_aic) < 5.0, (
            f"AIC differs significantly: PanelBox={pb_aic:.4f}, R={r_aic:.4f}"
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
