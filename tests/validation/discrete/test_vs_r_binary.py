"""
Validation tests against R implementations for binary choice models.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import models from panelbox
from panelbox.models.discrete import (
    FixedEffectsLogit,
    PooledLogit,
    PooledProbit,
    RandomEffectsProbit,
)

pytestmark = pytest.mark.r_validation


class TestBinaryModelsVsR:
    """Test binary choice models against R reference results."""

    @classmethod
    def setup_class(cls):
        """Load data and R reference results."""
        # Get data path
        data_path = Path(__file__).parent / "data"

        # Load panel data
        cls.data = pd.read_csv(data_path / "panel_binary.csv")
        cls.data = cls.data.set_index(["entity", "time"])

        # Load R reference results if available
        ref_file = data_path / "reference_results_binary.json"
        if ref_file.exists():
            with open(ref_file) as f:
                cls.r_results = json.load(f)
        else:
            cls.r_results = None
            pytest.skip("R reference results not generated yet")

    def test_pooled_logit_vs_r(self):
        """Test Pooled Logit against R glm()."""
        if self.r_results is None or "pooled_logit" not in self.r_results:
            pytest.skip("R results for Pooled Logit not available")

        # Fit PanelBox model
        model = PooledLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["pooled_logit"]

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

        # Compare AIC
        assert abs(result.aic - r_model["aic"]) < 0.02, (
            f"AIC differs: PanelBox={result.aic:.4f}, R={r_model['aic']:.4f}"
        )

    def test_pooled_probit_vs_r(self):
        """Test Pooled Probit against R glm()."""
        if self.r_results is None or "pooled_probit" not in self.r_results:
            pytest.skip("R results for Pooled Probit not available")

        # Fit PanelBox model
        model = PooledProbit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["pooled_probit"]

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

    def test_fe_logit_vs_r(self):
        """Test Fixed Effects Logit against R pglm()."""
        if self.r_results is None or "fe_logit" not in self.r_results:
            pytest.skip("R results for FE Logit not available")

        if "error" in self.r_results["fe_logit"]:
            pytest.skip(f"R FE Logit failed: {self.r_results['fe_logit']['error']}")

        # Fit PanelBox model
        model = FixedEffectsLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["fe_logit"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        np.testing.assert_allclose(
            result.params.values,
            r_coefs,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Coefficients differ from R",
        )

        # Compare standard errors (more tolerance for FE)
        r_se = np.array(r_model["std_errors"])
        np.testing.assert_allclose(
            result.bse.values, r_se, rtol=5e-3, atol=1e-3, err_msg="Standard errors differ from R"
        )

    def test_re_probit_vs_r(self):
        """Test Random Effects Probit against R pglm()."""
        if self.r_results is None or "re_probit" not in self.r_results:
            pytest.skip("R results for RE Probit not available")

        if "error" in self.r_results["re_probit"]:
            pytest.skip(f"R RE Probit failed: {self.r_results['re_probit']['error']}")

        # Fit PanelBox model
        model = RandomEffectsProbit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["re_probit"]

        # Compare coefficients (excluding variance component)
        r_coefs = np.array(r_model["coefficients"])
        # RE models may order coefficients differently
        pb_coefs = result.params[result.params.index != "sigma_alpha"].values
        np.testing.assert_allclose(
            pb_coefs, r_coefs, rtol=1e-2, atol=1e-3, err_msg="Coefficients differ from R"
        )

        # Compare variance component if available
        if "sigma" in r_model and hasattr(result, "sigma_alpha"):
            assert abs(result.sigma_alpha - r_model["sigma"]) < 0.05, (
                f"Sigma_alpha differs: PanelBox={result.sigma_alpha:.4f}, R={r_model['sigma']:.4f}"
            )

    def test_marginal_effects_vs_r(self):
        """Test Average Marginal Effects against R margins."""
        if self.r_results is None or "ame_logit" not in self.r_results:
            pytest.skip("R results for AME not available")

        if "error" in self.r_results["ame_logit"]:
            pytest.skip(f"R AME failed: {self.r_results['ame_logit']['error']}")

        # Fit PanelBox model
        model = PooledLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Calculate AME
        ame = result.marginal_effects(kind="average")

        # Get R results
        r_ame = self.r_results["ame_logit"]

        # Compare marginal effects
        r_effects = np.array(r_ame["marginal_effects"])
        # Note: Order might differ between R and Python
        np.testing.assert_allclose(
            np.sort(ame.effects.values),
            np.sort(r_effects),
            rtol=1e-3,
            atol=1e-4,
            err_msg="Marginal effects differ from R",
        )

    def test_predicted_probabilities(self):
        """Test predicted probabilities against R."""
        if self.r_results is None or "pooled_logit" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_probs_sample" not in self.r_results["pooled_logit"]:
            pytest.skip("R predicted probabilities not available")

        # Fit PanelBox model
        model = PooledLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get predictions for first 100 observations
        pred_probs = result.predict(self.data.iloc[:100])

        # Get R predictions
        r_preds = np.array(self.r_results["pooled_logit"]["predicted_probs_sample"])

        # Compare predictions
        np.testing.assert_allclose(
            pred_probs.values[: len(r_preds)],
            r_preds,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Predicted probabilities differ from R",
        )

    def test_model_fit_statistics(self):
        """Test model fit statistics against R."""
        if self.r_results is None or "pooled_logit" not in self.r_results:
            pytest.skip("R results not available")

        # Fit PanelBox model
        model = PooledLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        r_model = self.r_results["pooled_logit"]

        # Test McFadden R-squared if available
        if "mcfadden_r2" in r_model:
            pb_mcfadden = result.pseudo_r2("mcfadden")
            assert abs(pb_mcfadden - r_model["mcfadden_r2"]) < 0.001, (
                f"McFadden R² differs: PanelBox={pb_mcfadden:.4f}, R={r_model['mcfadden_r2']:.4f}"
            )

        # Test classification accuracy if available
        if "accuracy" in r_model:
            pb_accuracy = result.accuracy()
            assert abs(pb_accuracy - r_model["accuracy"]) < 0.01, (
                f"Accuracy differs: PanelBox={pb_accuracy:.4f}, R={r_model['accuracy']:.4f}"
            )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
