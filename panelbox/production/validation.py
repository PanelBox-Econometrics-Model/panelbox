"""ModelValidator -- Pre-deployment validation for panel models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates a fitted model before production deployment.

    Runs a suite of checks: parameter validity, prediction sanity,
    diagnostic tests (for GMM), and data compatibility.

    Parameters
    ----------
    results : object
        Fitted model results
    training_data : pd.DataFrame, optional
        Original training data for comparison

    Examples
    --------
    >>> validator = ModelValidator(results, training_data=df_train)
    >>> report = validator.run_all()
    >>> print(report["passed"])
    >>> print(report["summary"])
    """

    def __init__(self, results: Any, training_data: pd.DataFrame | None = None):
        self.results = results
        self.training_data = training_data

    def check_params(self) -> dict[str, Any]:
        """Check parameter validity (no NaN, no Inf, reasonable magnitudes)."""
        params = self.results.params
        return {
            "name": "parameter_validity",
            "no_nan": not np.any(np.isnan(params.values)),
            "no_inf": not np.any(np.isinf(params.values)),
            "max_abs": float(np.max(np.abs(params.values))),
            "reasonable_magnitude": float(np.max(np.abs(params.values))) < 1000,
        }

    def check_predict_sanity(self, test_data: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Check that predict() runs without error and returns reasonable values.

        Parameters
        ----------
        test_data : pd.DataFrame, optional
            Data to test predict on. Uses training_data if not provided.
        """
        data = test_data if test_data is not None else self.training_data
        if data is None:
            return {
                "name": "predict_sanity",
                "skipped": True,
                "reason": "No test data",
            }

        try:
            preds = self.results.predict(data)
            return {
                "name": "predict_sanity",
                "passed": True,
                "n_predictions": len(preds),
                "n_nan": int(np.sum(np.isnan(preds))),
                "n_inf": int(np.sum(np.isinf(preds))),
                "mean": float(np.nanmean(preds)),
                "std": float(np.nanstd(preds)),
            }
        except Exception as e:
            return {
                "name": "predict_sanity",
                "passed": False,
                "error": str(e),
            }

    def check_gmm_diagnostics(self) -> dict[str, Any] | None:
        """Check GMM-specific diagnostics (Hansen J, AR(2))."""
        if not hasattr(self.results, "hansen_j"):
            return None

        return {
            "name": "gmm_diagnostics",
            "hansen_j_pvalue": float(self.results.hansen_j.pvalue),
            "hansen_j_ok": self.results.hansen_j.pvalue > 0.10,
            "ar2_pvalue": float(self.results.ar2_test.pvalue),
            "ar2_ok": self.results.ar2_test.pvalue > 0.10,
            "instrument_ratio": float(self.results.instrument_ratio),
            "instrument_ratio_ok": self.results.instrument_ratio <= 1.0,
        }

    def run_all(self) -> dict[str, Any]:
        """Run all validation checks and return report."""
        checks = []

        checks.append(self.check_params())
        checks.append(self.check_predict_sanity())

        gmm_check = self.check_gmm_diagnostics()
        if gmm_check:
            checks.append(gmm_check)

        # Determine overall pass
        passed = all(c.get("passed", True) and not c.get("skipped", False) for c in checks)

        return {
            "passed": passed,
            "checks": checks,
            "summary": self._format_summary(checks),
        }

    def _format_summary(self, checks: list[dict]) -> str:
        lines = ["Model Validation Report", "=" * 40]
        for check in checks:
            name = check.get("name", "unknown")
            if check.get("skipped"):
                lines.append(f"  {name}: SKIPPED ({check.get('reason', '')})")
            elif check.get("passed", True):
                lines.append(f"  {name}: PASSED")
            else:
                lines.append(f"  {name}: FAILED")
        return "\n".join(lines)
