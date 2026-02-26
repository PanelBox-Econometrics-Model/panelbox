"""
Validation tests against R implementations for count models.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import models from panelbox
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
        cls.data = cls.data.set_index(["entity", "time"])

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

        # Fit PanelBox model
        model = PooledPoisson.from_formula("y ~ x1 + x2 + x3", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["pooled_poisson"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        np.testing.assert_allclose(
            result.params.values,
            r_coefs,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Coefficients differ from R",
        )

        # Compare standard errors
        r_se = np.array(r_model["std_errors"])
        np.testing.assert_allclose(
            result.bse.values, r_se, rtol=1e-3, atol=1e-5, err_msg="Standard errors differ from R"
        )

        # Compare log-likelihood
        assert abs(result.llf - r_model["loglik"]) < 0.01, (
            f"Log-likelihood differs: PanelBox={result.llf:.4f}, R={r_model['loglik']:.4f}"
        )

        # Compare deviance
        if "deviance" in r_model:
            assert abs(result.deviance - r_model["deviance"]) < 0.1, (
                f"Deviance differs: PanelBox={result.deviance:.4f}, R={r_model['deviance']:.4f}"
            )

    def test_negative_binomial_vs_r(self):
        """Test Negative Binomial against R MASS::glm.nb()."""
        if self.r_results is None or "negative_binomial" not in self.r_results:
            pytest.skip("R results for Negative Binomial not available")

        # Fit PanelBox model
        model = NegativeBinomial.from_formula("y ~ x1 + x2 + x3", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["negative_binomial"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        np.testing.assert_allclose(
            result.params.values[:-1],
            r_coefs,  # Exclude dispersion parameter
            rtol=1e-3,
            atol=1e-4,
            err_msg="Coefficients differ from R",
        )

        # Compare dispersion parameter (theta)
        if "theta" in r_model:
            assert abs(result.alpha - (1 / r_model["theta"])) < 0.01, (
                f"Dispersion differs: PanelBox alpha={result.alpha:.4f}, R theta={r_model['theta']:.4f}"
            )

        # Compare log-likelihood
        assert abs(result.llf - r_model["loglik"]) < 1.0, (
            f"Log-likelihood differs: PanelBox={result.llf:.4f}, R={r_model['loglik']:.4f}"
        )

    def test_poisson_fe_vs_r(self):
        """Test Poisson Fixed Effects against R pglm()."""
        if self.r_results is None or "fe_poisson" not in self.r_results:
            pytest.skip("R results for FE Poisson not available")

        if "error" in self.r_results["fe_poisson"]:
            pytest.skip(f"R FE Poisson failed: {self.r_results['fe_poisson']['error']}")

        # Fit PanelBox model
        model = PoissonFixedEffects.from_formula("y ~ x1 + x2 + x3", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["fe_poisson"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        np.testing.assert_allclose(
            result.params.values,
            r_coefs,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Coefficients differ from R",
        )

        # Compare standard errors
        r_se = np.array(r_model["std_errors"])
        np.testing.assert_allclose(
            result.bse.values, r_se, rtol=5e-3, atol=1e-3, err_msg="Standard errors differ from R"
        )

    def test_re_poisson_vs_r(self):
        """Test Random Effects Poisson against R pglm()."""
        if self.r_results is None or "re_poisson" not in self.r_results:
            pytest.skip("R results for RE Poisson not available")

        if "error" in self.r_results["re_poisson"]:
            pytest.skip(f"R RE Poisson failed: {self.r_results['re_poisson']['error']}")

        # Fit PanelBox model
        model = RandomEffectsPoisson.from_formula("y ~ x1 + x2 + x3", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["re_poisson"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        pb_coefs = result.params[result.params.index != "sigma_alpha"].values
        np.testing.assert_allclose(
            pb_coefs, r_coefs, rtol=1e-2, atol=1e-3, err_msg="Coefficients differ from R"
        )

        # Compare variance component
        if "sigma" in r_model and hasattr(result, "sigma_alpha"):
            assert abs(result.sigma_alpha - r_model["sigma"]) < 0.1, (
                f"Sigma differs: PanelBox={result.sigma_alpha:.4f}, R={r_model['sigma']:.4f}"
            )

    def test_overdispersion_test(self):
        """Test overdispersion diagnostics against R."""
        if self.r_results is None or "overdispersion" not in self.r_results:
            pytest.skip("R overdispersion test not available")

        # Fit PanelBox model
        model = PooledPoisson.from_formula("y ~ x1 + x2 + x3", data=self.data)
        result = model.fit()

        # Test overdispersion
        od_test = result.overdispersion_test()

        # Get R results
        r_od = self.r_results["overdispersion"]

        # Compare overdispersion statistic
        assert abs(od_test["statistic"] - r_od["statistic"]) < 0.1, (
            f"Overdispersion stat differs: PanelBox={od_test['statistic']:.4f}, R={r_od['statistic']:.4f}"
        )

    def test_marginal_effects_poisson(self):
        """Test marginal effects for Poisson model."""
        if self.r_results is None or "marginal_effects_poisson" not in self.r_results:
            pytest.skip("R marginal effects not available")

        # Fit PanelBox model
        model = PooledPoisson.from_formula("y ~ x1 + x2 + x3", data=self.data)
        result = model.fit()

        # Calculate marginal effects
        me = result.marginal_effects(kind="average")

        # Get R results
        r_me = self.r_results["marginal_effects_poisson"]

        # Compare effects
        r_effects = np.array(r_me["effects"])
        # Exclude intercept from comparison
        pb_effects = me.effects.iloc[1:].values
        np.testing.assert_allclose(
            np.sort(pb_effects),
            np.sort(r_effects),
            rtol=5e-3,
            atol=1e-3,
            err_msg="Marginal effects differ from R",
        )

    def test_predicted_counts(self):
        """Test predicted counts against R."""
        if self.r_results is None or "pooled_poisson" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_counts_sample" not in self.r_results["pooled_poisson"]:
            pytest.skip("R predicted counts not available")

        # Fit PanelBox model
        model = PooledPoisson.from_formula("y ~ x1 + x2 + x3", data=self.data)
        result = model.fit()

        # Get predictions for first 100 observations
        pred_counts = result.predict(self.data.iloc[:100])

        # Get R predictions
        r_preds = np.array(self.r_results["pooled_poisson"]["predicted_counts_sample"])

        # Compare predictions
        np.testing.assert_allclose(
            pred_counts.values[: len(r_preds)],
            r_preds,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Predicted counts differ from R",
        )

    def test_goodness_of_fit(self):
        """Test goodness of fit statistics."""
        if self.r_results is None or "goodness_of_fit" not in self.r_results:
            pytest.skip("R goodness of fit not available")

        # Fit PanelBox model
        model = PooledPoisson.from_formula("y ~ x1 + x2 + x3", data=self.data)
        result = model.fit()

        # Get goodness of fit
        gof = result.goodness_of_fit()

        # Get R results
        r_gof = self.r_results["goodness_of_fit"]

        # Compare chi-squared statistic (with tolerance)
        if "chi_squared" in r_gof:
            assert abs(gof["chi_squared"] - r_gof["chi_squared"]) < 5.0, (
                f"Chi-squared differs: PanelBox={gof['chi_squared']:.2f}, R={r_gof['chi_squared']:.2f}"
            )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
