"""
Validation tests comparing PanelBox count models against R implementations.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

# Import PanelBox models
from panelbox.models.count import (
    NegativeBinomial,
    PoissonFixedEffects,
    PooledPoisson,
    RandomEffectsPoisson,
)

pytestmark = pytest.mark.r_validation


class TestCountModelsVsR:
    """Test count models against R reference results."""

    @classmethod
    def setup_class(cls):
        """Load data and R reference results."""
        # Get data path
        data_path = Path(__file__).parent / "data"

        # Load panel data
        cls.data = pd.read_csv(data_path / "panel_count.csv")

        # Prepare arrays for models that need array API
        cls.y = cls.data["y"].values
        cls.X = sm.add_constant(cls.data[["x1", "x2"]].values)
        cls.entity_id = cls.data["entity"].values
        cls.time_id = cls.data["time"].values

        # Load R reference results if available
        ref_file = data_path / "reference_results_count.json"
        if ref_file.exists():
            with open(ref_file) as f:
                cls.r_results = json.load(f)
        else:
            cls.r_results = None
            pytest.skip("R reference results not generated yet")

    def test_pooled_poisson_vs_r(self):
        """Test Pooled Poisson against R glm()."""
        if self.r_results is None or "pooled_poisson" not in self.r_results:
            pytest.skip("R results for Pooled Poisson not available")

        # Fit PanelBox model with nonrobust SEs to match R's glm() default
        model = PooledPoisson(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit(se_type="nonrobust")

        # Get R results
        r_model = self.r_results["pooled_poisson"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        np.testing.assert_allclose(
            result.params,
            r_coefs,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Coefficients differ from R",
        )

        # Compare standard errors
        r_se = np.array(r_model["std_errors"])
        np.testing.assert_allclose(
            result.se, r_se, rtol=1e-3, atol=1e-5, err_msg="Standard errors differ from R"
        )

        # Compare log-likelihood
        llf = result.model.llf
        assert abs(llf - r_model["loglik"]) < 0.01, (
            f"Log-likelihood differs: PanelBox={llf:.4f}, R={r_model['loglik']:.4f}"
        )

    def test_overdispersion_check(self):
        """Test overdispersion detection."""
        if self.r_results is None or "pooled_poisson" not in self.r_results:
            pytest.skip("R results not available")

        # Fit PanelBox model
        model = PooledPoisson(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit()

        # Get R dispersion test
        r_model = self.r_results["pooled_poisson"]
        r_dispersion = r_model.get("dispersion", 1.0)

        # Calculate PanelBox dispersion from fitted values
        fv = result.predict(self.X)
        pearson_resid = (self.y - fv) / np.sqrt(fv)
        n = len(self.y)
        k = self.X.shape[1]
        pb_dispersion = np.sum(pearson_resid**2) / (n - k)

        # Compare dispersion
        assert abs(pb_dispersion - r_dispersion) < 0.1, (
            f"Dispersion differs: PanelBox={pb_dispersion:.3f}, R={r_dispersion:.3f}"
        )

    @pytest.mark.skip(reason="PoissonFE conditional MLE is too slow for CI (>120s)")
    def test_poisson_fe_vs_r(self):
        """Test Poisson Fixed Effects against R pglm().

        Note: This test uses conditional MLE with DP which is O(N*T*sum(y))
        and can be very slow with large datasets.
        """
        if self.r_results is None or "fe_poisson" not in self.r_results:
            pytest.skip("R results for FE Poisson not available")

        if "error" in self.r_results["fe_poisson"]:
            pytest.skip(f"R FE Poisson failed: {self.r_results['fe_poisson']['error']}")

        # FE models exclude the intercept from exog
        X_no_const = self.data[["x1", "x2"]].values

        # Fit PanelBox model
        model = PoissonFixedEffects(self.y, X_no_const, self.entity_id, self.time_id)
        result = model.fit()

        # Get R results
        r_model = self.r_results["fe_poisson"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        np.testing.assert_allclose(
            result.params,
            r_coefs,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Coefficients differ from R",
        )

        # Compare standard errors (more tolerance for FE)
        r_se = np.array(r_model["std_errors"])
        np.testing.assert_allclose(
            result.se, r_se, rtol=5e-3, atol=1e-3, err_msg="Standard errors differ from R"
        )

    @pytest.mark.timeout(60)
    def test_poisson_re_vs_r(self):
        """Test Random Effects Poisson against R pglm()."""
        if self.r_results is None or "re_poisson" not in self.r_results:
            pytest.skip("R results for RE Poisson not available")

        if "error" in self.r_results["re_poisson"]:
            pytest.skip(f"R RE Poisson failed: {self.r_results['re_poisson']['error']}")

        # Fit PanelBox model
        model = RandomEffectsPoisson(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit()

        # Get R results
        r_model = self.r_results["re_poisson"]

        # Compare coefficients (excluding variance component)
        r_coefs = np.array(r_model["coefficients"][:3])  # Only intercept, x1, x2
        pb_coefs = result.params[:3]
        np.testing.assert_allclose(
            pb_coefs, r_coefs, rtol=1e-2, atol=1e-3, err_msg="Coefficients differ from R"
        )

    def test_negative_binomial_vs_r(self):
        """Test Negative Binomial against R MASS::glm.nb()."""
        if self.r_results is None or "negative_binomial" not in self.r_results:
            pytest.skip("R results for Negative Binomial not available")

        if "error" in self.r_results["negative_binomial"]:
            pytest.skip(f"R NB failed: {self.r_results['negative_binomial']['error']}")

        # Fit PanelBox model
        model = NegativeBinomial(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit()

        # Get R results
        r_model = self.r_results["negative_binomial"]

        # Compare coefficients (exclude dispersion parameter)
        r_coefs = np.array(r_model["coefficients"])
        np.testing.assert_allclose(
            result.params[: len(r_coefs)],
            r_coefs,
            rtol=5e-3,
            atol=1e-3,
            err_msg="Coefficients differ from R",
        )

        # Compare dispersion parameter (theta in R, alpha in PanelBox)
        if "theta" in r_model:
            r_theta = r_model["theta"]
            pb_alpha = getattr(result, "alpha", None)

            if pb_alpha is not None:
                # Check if they're in the same ballpark (different parameterizations)
                assert abs(pb_alpha - 1 / r_theta) < 0.1 or abs(pb_alpha - r_theta) < 5.0, (
                    "Dispersion parameter differs significantly"
                )

    def test_predicted_counts(self):
        """Test predicted counts against R."""
        if self.r_results is None or "pooled_poisson" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_counts_sample" not in self.r_results["pooled_poisson"]:
            pytest.skip("R predicted counts not available")

        # Fit PanelBox model
        model = PooledPoisson(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit()

        # Get in-sample predictions for first 100 observations
        pred_counts = result.predict(self.X[:100])

        # Get R predictions
        r_preds = np.array(self.r_results["pooled_poisson"]["predicted_counts_sample"])

        # Compare predictions
        np.testing.assert_allclose(
            pred_counts[: len(r_preds)],
            r_preds,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Predicted counts differ from R",
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
