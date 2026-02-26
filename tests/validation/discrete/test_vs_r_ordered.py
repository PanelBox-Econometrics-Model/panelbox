"""
Validation tests comparing PanelBox ordered choice models against R implementations.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import PanelBox models
from panelbox.models.discrete import OrderedLogit, OrderedProbit

pytestmark = pytest.mark.r_validation


class TestOrderedModelsVsR:
    """Test ordered choice models against R MASS::polr reference results."""

    @classmethod
    def setup_class(cls):
        """Load data and R reference results."""
        # Get data path
        data_path = Path(__file__).parent / "data"

        # Load panel data
        cls.data = pd.read_csv(data_path / "panel_ordered.csv")
        cls.data = cls.data.set_index(["entity", "time"])

        # Load R reference results if available
        ref_file = data_path / "reference_results_ordered.json"
        if ref_file.exists():
            with open(ref_file) as f:
                cls.r_results = json.load(f)
        else:
            cls.r_results = None
            pytest.skip("R reference results not generated yet")

    def test_ordered_logit_vs_r(self):
        """Test Ordered Logit against R MASS::polr()."""
        if self.r_results is None or "ordered_logit" not in self.r_results:
            pytest.skip("R results for Ordered Logit not available")

        if "error" in self.r_results["ordered_logit"]:
            pytest.skip(f"R Ordered Logit failed: {self.r_results['ordered_logit']['error']}")

        # Fit PanelBox model
        model = OrderedLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["ordered_logit"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        # Note: polr uses negative coefficients by convention
        pb_coefs = result.params[: len(r_coefs)].values
        np.testing.assert_allclose(
            pb_coefs,
            -r_coefs,  # Note the sign difference
            rtol=1e-4,
            atol=1e-6,
            err_msg="Coefficients differ from R (accounting for sign convention)",
        )

        # Compare thresholds
        r_thresholds = np.array(r_model["thresholds"])
        pb_thresholds = (
            result.thresholds
            if hasattr(result, "thresholds")
            else result.params[len(r_coefs) :].values
        )

        np.testing.assert_allclose(
            pb_thresholds, r_thresholds, rtol=1e-3, atol=1e-4, err_msg="Thresholds differ from R"
        )

        # Compare standard errors
        r_se_coef = np.array(r_model["std_errors_coef"])
        np.testing.assert_allclose(
            result.bse[: len(r_se_coef)].values,
            r_se_coef,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Standard errors for coefficients differ from R",
        )

        # Compare log-likelihood
        assert abs(result.llf - r_model["loglik"]) < 0.01, (
            f"Log-likelihood differs: PanelBox={result.llf:.4f}, R={r_model['loglik']:.4f}"
        )

    def test_ordered_probit_vs_r(self):
        """Test Ordered Probit against R MASS::polr()."""
        if self.r_results is None or "ordered_probit" not in self.r_results:
            pytest.skip("R results for Ordered Probit not available")

        if "error" in self.r_results["ordered_probit"]:
            pytest.skip(f"R Ordered Probit failed: {self.r_results['ordered_probit']['error']}")

        # Fit PanelBox model
        model = OrderedProbit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get R results
        r_model = self.r_results["ordered_probit"]

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        # Note: polr uses negative coefficients by convention
        pb_coefs = result.params[: len(r_coefs)].values
        np.testing.assert_allclose(
            pb_coefs,
            -r_coefs,  # Note the sign difference
            rtol=1e-4,
            atol=1e-6,
            err_msg="Coefficients differ from R (accounting for sign convention)",
        )

        # Compare thresholds
        r_thresholds = np.array(r_model["thresholds"])
        pb_thresholds = (
            result.thresholds
            if hasattr(result, "thresholds")
            else result.params[len(r_coefs) :].values
        )

        np.testing.assert_allclose(
            pb_thresholds, r_thresholds, rtol=1e-3, atol=1e-4, err_msg="Thresholds differ from R"
        )

        # Compare standard errors
        r_se_coef = np.array(r_model["std_errors_coef"])
        np.testing.assert_allclose(
            result.bse[: len(r_se_coef)].values,
            r_se_coef,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Standard errors for coefficients differ from R",
        )

        # Compare log-likelihood
        assert abs(result.llf - r_model["loglik"]) < 0.01, (
            f"Log-likelihood differs: PanelBox={result.llf:.4f}, R={r_model['loglik']:.4f}"
        )

    def test_predicted_probabilities(self):
        """Test predicted probabilities against R."""
        if self.r_results is None or "ordered_logit" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_probs_sample" not in self.r_results["ordered_logit"]:
            pytest.skip("R predicted probabilities not available")

        # Fit PanelBox model
        model = OrderedLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get predictions for first 50 observations
        pred_probs = result.predict_proba(self.data.iloc[:50])

        # Get R predictions
        r_preds = np.array(self.r_results["ordered_logit"]["predicted_probs_sample"])

        # Compare predictions (shape should match)
        assert pred_probs.shape == r_preds.shape, (
            f"Prediction shape differs: PanelBox={pred_probs.shape}, R={r_preds.shape}"
        )

        # Compare actual values
        np.testing.assert_allclose(
            pred_probs,
            r_preds,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Predicted probabilities differ from R",
        )

    def test_predicted_classes(self):
        """Test predicted classes against R."""
        if self.r_results is None or "ordered_logit" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_class_sample" not in self.r_results["ordered_logit"]:
            pytest.skip("R predicted classes not available")

        # Fit PanelBox model
        model = OrderedLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get predictions for first 50 observations
        pred_class = result.predict(self.data.iloc[:50])

        # Get R predictions
        r_class = self.r_results["ordered_logit"]["predicted_class_sample"]

        # Compare predicted classes
        # Convert to same type for comparison
        pb_class = pred_class.astype(str).tolist()
        assert pb_class == r_class, "Predicted classes differ from R"

    def test_aic_comparison(self):
        """Test AIC calculation against R."""
        if self.r_results is None or "ordered_logit" not in self.r_results:
            pytest.skip("R results not available")

        if "error" in self.r_results["ordered_logit"]:
            pytest.skip("R Ordered Logit failed")

        # Fit PanelBox model
        model = OrderedLogit.from_formula("y ~ x1 + x2", data=self.data)
        result = model.fit()

        # Get R AIC
        r_model = self.r_results["ordered_logit"]
        r_aic = r_model["aic"]

        # Compare AIC
        assert abs(result.aic - r_aic) < 0.02, (
            f"AIC differs: PanelBox={result.aic:.4f}, R={r_aic:.4f}"
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
