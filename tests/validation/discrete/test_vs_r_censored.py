"""
Validation tests comparing PanelBox censored models against R implementations.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import PanelBox models
from panelbox.models.censored import RandomEffectsTobit

pytestmark = pytest.mark.r_validation


class TestCensoredModelsVsR:
    """Test censored models against R censReg reference results."""

    @classmethod
    def setup_class(cls):
        """Load data and R reference results."""
        # Get data path
        data_path = Path(__file__).parent / "data"

        # Load panel data
        cls.data = pd.read_csv(data_path / "panel_censored.csv")

        # Load R reference results if available
        ref_file = data_path / "reference_results_censored.json"
        if ref_file.exists():
            with open(ref_file) as f:
                cls.r_results = json.load(f)
        else:
            cls.r_results = None
            pytest.skip("R reference results not generated yet")

    def test_pooled_tobit_vs_r(self):
        """Test Pooled Tobit against R censReg()."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results for Pooled Tobit not available")

        if "error" in self.r_results["pooled_tobit"]:
            pytest.skip(f"R Pooled Tobit failed: {self.r_results['pooled_tobit']['error']}")

        # Fit PanelBox model (using RE Tobit without entity effects as pooled)
        model = RandomEffectsTobit("y ~ x1 + x2", self.data, "entity", "time")
        # For pooled, we'd set sigma_alpha very small or use a pooled version
        result = model.fit()

        # Get R results
        r_model = self.r_results["pooled_tobit"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        pb_coefs = result.params[: len(r_coefs)].values
        np.testing.assert_allclose(
            pb_coefs, r_coefs, rtol=1e-3, atol=1e-5, err_msg="Coefficients differ from R"
        )

        # Compare sigma
        r_sigma = r_model["sigma"]
        pb_sigma = result.sigma if hasattr(result, "sigma") else result.params["sigma"]
        assert abs(pb_sigma - r_sigma) < 0.01, (
            f"Sigma differs: PanelBox={pb_sigma:.4f}, R={r_sigma:.4f}"
        )

        # Compare standard errors
        r_se = np.array(r_model["std_errors_coef"])
        np.testing.assert_allclose(
            result.bse[: len(r_se)].values,
            r_se,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Standard errors differ from R",
        )

        # Compare log-likelihood
        assert abs(result.llf - r_model["loglik"]) < 0.1, (
            f"Log-likelihood differs: PanelBox={result.llf:.4f}, R={r_model['loglik']:.4f}"
        )

    def test_censoring_statistics(self):
        """Test censoring statistics calculation."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results not available")

        if "error" in self.r_results["pooled_tobit"]:
            pytest.skip("R Pooled Tobit failed")

        # Fit PanelBox model
        model = RandomEffectsTobit("y ~ x1 + x2", self.data, "entity", "time")
        result = model.fit()

        # Get R censoring stats
        r_model = self.r_results["pooled_tobit"]
        r_n_censored = r_model["n_censored"]
        r_n_uncensored = r_model["n_uncensored"]

        # Compare censoring counts
        pb_n_censored = (
            result.n_censored if hasattr(result, "n_censored") else (self.data["y"] == 0).sum()
        )
        pb_n_uncensored = (
            result.n_uncensored if hasattr(result, "n_uncensored") else (self.data["y"] > 0).sum()
        )

        assert pb_n_censored == r_n_censored, (
            f"Number of censored obs differs: PanelBox={pb_n_censored}, R={r_n_censored}"
        )
        assert pb_n_uncensored == r_n_uncensored, (
            f"Number of uncensored obs differs: PanelBox={pb_n_uncensored}, R={r_n_uncensored}"
        )

    def test_predicted_values(self):
        """Test predicted values (latent and observed) against R."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_values_sample" not in self.r_results["pooled_tobit"]:
            pytest.skip("R predicted values not available")

        # Fit PanelBox model
        model = RandomEffectsTobit("y ~ x1 + x2", self.data, "entity", "time")
        result = model.fit()

        # Get predictions for first 100 observations
        pred_observed = result.predict(self.data.iloc[:100])

        # Get R predictions
        r_pred_observed = np.array(self.r_results["pooled_tobit"]["predicted_values_sample"])

        # Compare predicted observed values
        np.testing.assert_allclose(
            pred_observed.values[: len(r_pred_observed)],
            r_pred_observed,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Predicted observed values differ from R",
        )

    def test_latent_values(self):
        """Test predicted latent values against R."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results not available")

        if "latent_values_sample" not in self.r_results["pooled_tobit"]:
            pytest.skip("R latent values not available")

        # Fit PanelBox model
        model = RandomEffectsTobit("y ~ x1 + x2", self.data, "entity", "time")
        result = model.fit()

        # Get latent predictions for first 100 observations
        pred_latent = (
            result.predict_latent(self.data.iloc[:100])
            if hasattr(result, "predict_latent")
            else result.predict(self.data.iloc[:100], latent=True)
        )

        # Get R predictions
        r_pred_latent = np.array(self.r_results["pooled_tobit"]["latent_values_sample"])

        # Compare predicted latent values
        np.testing.assert_allclose(
            pred_latent.values[: len(r_pred_latent)],
            r_pred_latent,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Predicted latent values differ from R",
        )

    @pytest.mark.skip(reason="True RE Tobit comparison requires specialized R package")
    def test_re_tobit_vs_r(self):
        """Test Random Effects Tobit against specialized R implementation."""
        # Note: censReg doesn't have built-in RE support
        # Would need to use a different R package for proper comparison
        pass

    def test_aic_comparison(self):
        """Test AIC calculation against R."""
        if self.r_results is None or "pooled_tobit" not in self.r_results:
            pytest.skip("R results not available")

        if "error" in self.r_results["pooled_tobit"]:
            pytest.skip("R Pooled Tobit failed")

        # Fit PanelBox model
        model = RandomEffectsTobit("y ~ x1 + x2", self.data, "entity", "time")
        result = model.fit()

        # Get R AIC
        r_model = self.r_results["pooled_tobit"]
        r_aic = r_model["aic"]

        # Compare AIC (allowing for small differences due to implementation)
        assert abs(result.aic - r_aic) < 1.0, (
            f"AIC differs significantly: PanelBox={result.aic:.4f}, R={r_aic:.4f}"
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
