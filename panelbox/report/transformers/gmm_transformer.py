"""
GMM Result Transformer.

Converts GMM estimation results into template-ready dictionaries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GMMTransformer:
    """
    Transform GMM results into template-ready data.

    Parameters
    ----------
    data : dict
        GMM result data dictionary.
    """

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def transform(self) -> dict[str, Any]:
        """
        Transform GMM data into template context.

        Returns
        -------
        dict
            Template-ready dictionary.
        """
        return {
            "model_info": self._transform_model_info(),
            "coefficients": self._transform_coefficients(),
            "diagnostics": self._transform_diagnostics(),
            "summary": self._compute_summary(),
        }

    def _transform_model_info(self) -> dict[str, Any]:
        n_groups = self.data.get("n_groups", "—")
        n_instruments = self.data.get("n_instruments", "—")
        return {
            "estimator": self.data.get("model_type", self.data.get("estimator", "GMM")),
            "nobs": self.data.get("nobs", "—"),
            "n_groups": n_groups,
            "n_instruments": n_instruments,
            "two_step": self.data.get("two_step", False),
            "instrument_ratio": (
                f"{n_instruments}/{n_groups}"
                if isinstance(n_instruments, (int, float)) and isinstance(n_groups, (int, float))
                else "—"
            ),
        }

    def _transform_coefficients(self) -> list[dict[str, Any]]:
        coefficients = self.data.get("coefficients", [])
        result = []
        for coef in coefficients:
            pval = coef.get("pvalue", coef.get("pval", 1.0))
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            result.append(
                {
                    "name": coef.get("name", ""),
                    "coef": coef.get("coef", coef.get("coefficient", 0)),
                    "se": coef.get("se", coef.get("std_error", 0)),
                    "tstat": coef.get("tstat", coef.get("t_stat", 0)),
                    "pvalue": pval,
                    "stars": stars,
                }
            )
        return result

    def _transform_diagnostics(self) -> dict[str, Any]:
        hansen = self.data.get("hansen_test", self.data.get("hansen_j", {}))
        ar_tests = self.data.get("ar_tests", {})
        hansen_p = hansen.get("pvalue", hansen.get("p_value", 0))
        ar2_p = ar_tests.get("ar2", {}).get("pvalue", 0)
        return {
            "hansen": {
                "statistic": hansen.get("statistic", "—"),
                "pvalue": hansen_p if hansen_p != 0 else "—",
                "df": hansen.get("df", "—"),
                "status": "pass"
                if isinstance(hansen_p, (int, float)) and hansen_p > 0.05
                else "fail",
            },
            "ar1": {
                "statistic": ar_tests.get("ar1", {}).get("statistic", "—"),
                "pvalue": ar_tests.get("ar1", {}).get("pvalue", "—"),
            },
            "ar2": {
                "statistic": ar_tests.get("ar2", {}).get("statistic", "—"),
                "pvalue": ar2_p if ar2_p != 0 else "—",
                "status": "pass" if isinstance(ar2_p, (int, float)) and ar2_p > 0.05 else "fail",
            },
        }

    def _compute_summary(self) -> dict[str, Any]:
        diagnostics = self._transform_diagnostics()
        hansen_ok = diagnostics["hansen"]["status"] == "pass"
        ar2_ok = diagnostics["ar2"]["status"] == "pass"
        return {
            "overall_status": "good" if hansen_ok and ar2_ok else "warning",
            "hansen_ok": hansen_ok,
            "ar2_ok": ar2_ok,
        }
