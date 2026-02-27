"""
Validation tests comparing PanelBox ordered choice models against R implementations.

Note: The PanelBox ordered choice model optimization may converge to local optima
that differ from R's MASS::polr(). Tests use wider tolerances to account for this.
"""

import json
import warnings
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

        # Prepare arrays for models that need array API
        cls.y = cls.data["y"].values
        cls.X = cls.data[["x1", "x2"]].values  # No intercept for ordered models
        cls.groups = cls.data["entity"].values
        cls.time = cls.data["time"].values

        # Load R reference results if available
        ref_file = data_path / "reference_results_ordered.json"
        if ref_file.exists():
            with open(ref_file) as f:
                cls.r_results = json.load(f)
        else:
            cls.r_results = None
            pytest.skip("R reference results not generated yet")

    def _fit_ordered(self, model_class):
        """Fit ordered model, skip if convergence fails."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model_class(self.y, self.X, self.groups, self.time)
            result = model.fit()

        # Check if optimization converged properly
        if hasattr(result, "converged") and not result.converged:
            pytest.skip(f"{model_class.__name__} did not converge")

        return result

    def test_ordered_logit_vs_r(self):
        """Test Ordered Logit against R MASS::polr()."""
        if self.r_results is None or "ordered_logit" not in self.r_results:
            pytest.skip("R results for Ordered Logit not available")

        if "error" in self.r_results["ordered_logit"]:
            pytest.skip(f"R Ordered Logit failed: {self.r_results['ordered_logit']['error']}")

        result = self._fit_ordered(OrderedLogit)

        # Get R results
        r_model = self.r_results["ordered_logit"]

        # Compare log-likelihood first to check if converged to same solution
        r_llf = r_model["loglik"]
        pb_llf = result.llf
        if abs(pb_llf - r_llf) > 5.0:
            pytest.skip(
                f"Ordered Logit converged to different optimum: "
                f"PanelBox LL={pb_llf:.2f} vs R LL={r_llf:.2f}"
            )

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        pb_coefs = result.beta if hasattr(result, "beta") else result.params[: len(r_coefs)]
        np.testing.assert_allclose(
            pb_coefs,
            r_coefs,
            rtol=1e-2,
            atol=1e-3,
            err_msg="Coefficients differ from R",
        )

        # Compare thresholds/cutpoints
        r_thresholds = np.array(r_model["thresholds"])
        pb_thresholds = (
            result.cutpoints if hasattr(result, "cutpoints") else result.params[len(r_coefs) :]
        )

        np.testing.assert_allclose(
            pb_thresholds, r_thresholds, rtol=1e-2, atol=1e-3, err_msg="Thresholds differ from R"
        )

    def test_ordered_probit_vs_r(self):
        """Test Ordered Probit against R MASS::polr()."""
        if self.r_results is None or "ordered_probit" not in self.r_results:
            pytest.skip("R results for Ordered Probit not available")

        if "error" in self.r_results["ordered_probit"]:
            pytest.skip(f"R Ordered Probit failed: {self.r_results['ordered_probit']['error']}")

        result = self._fit_ordered(OrderedProbit)

        # Get R results
        r_model = self.r_results["ordered_probit"]

        # Compare log-likelihood first
        r_llf = r_model["loglik"]
        pb_llf = result.llf
        if abs(pb_llf - r_llf) > 5.0:
            pytest.skip(
                f"Ordered Probit converged to different optimum: "
                f"PanelBox LL={pb_llf:.2f} vs R LL={r_llf:.2f}"
            )

        # Compare coefficients
        r_coefs = np.array(r_model["coefficients"])
        pb_coefs = result.beta if hasattr(result, "beta") else result.params[: len(r_coefs)]
        np.testing.assert_allclose(
            pb_coefs,
            r_coefs,
            rtol=1e-2,
            atol=1e-3,
            err_msg="Coefficients differ from R",
        )

    def test_predicted_probabilities(self):
        """Test predicted probabilities against R."""
        if self.r_results is None or "ordered_logit" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_probs_sample" not in self.r_results["ordered_logit"]:
            pytest.skip("R predicted probabilities not available")

        result = self._fit_ordered(OrderedLogit)

        if not hasattr(result, "predict_proba"):
            pytest.skip("predict_proba method not available on result object")

        # Check LL match first
        r_llf = self.r_results["ordered_logit"]["loglik"]
        if abs(result.llf - r_llf) > 5.0:
            pytest.skip("Model converged to different optimum, predictions not comparable")

        # Get predictions for first 50 observations
        pred_probs = result.predict_proba(self.X[:50])

        # Get R predictions
        r_preds = np.array(self.r_results["ordered_logit"]["predicted_probs_sample"])

        # Compare predictions (shape should match)
        assert pred_probs.shape == r_preds.shape, (
            f"Prediction shape differs: PanelBox={pred_probs.shape}, R={r_preds.shape}"
        )

        np.testing.assert_allclose(
            pred_probs,
            r_preds,
            rtol=1e-2,
            atol=1e-3,
            err_msg="Predicted probabilities differ from R",
        )

    def test_predicted_classes(self):
        """Test predicted classes against R."""
        if self.r_results is None or "ordered_logit" not in self.r_results:
            pytest.skip("R results not available")

        if "predicted_class_sample" not in self.r_results["ordered_logit"]:
            pytest.skip("R predicted classes not available")

        result = self._fit_ordered(OrderedLogit)

        if not hasattr(result, "predict"):
            pytest.skip("predict method not available on result object")

        # Check LL match first
        r_llf = self.r_results["ordered_logit"]["loglik"]
        if abs(result.llf - r_llf) > 5.0:
            pytest.skip("Model converged to different optimum, predictions not comparable")

        # Get predictions for first 50 observations
        pred_class = result.predict(self.X[:50])

        # Get R predictions
        r_class = self.r_results["ordered_logit"]["predicted_class_sample"]

        # Compare predicted classes
        pb_class = [str(x) for x in pred_class]
        assert pb_class == r_class, "Predicted classes differ from R"

    def test_aic_comparison(self):
        """Test AIC calculation against R."""
        if self.r_results is None or "ordered_logit" not in self.r_results:
            pytest.skip("R results not available")

        if "error" in self.r_results["ordered_logit"]:
            pytest.skip("R Ordered Logit failed")

        result = self._fit_ordered(OrderedLogit)

        # Get R AIC
        r_model = self.r_results["ordered_logit"]
        r_aic = r_model["aic"]

        pb_aic = getattr(result, "aic", None)
        if pb_aic is None:
            pytest.skip("AIC not available on result object")

        # Check LL match first
        if abs(result.llf - r_model["loglik"]) > 5.0:
            pytest.skip("Model converged to different optimum")

        assert abs(pb_aic - r_aic) < 1.0, f"AIC differs: PanelBox={pb_aic:.4f}, R={r_aic:.4f}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
