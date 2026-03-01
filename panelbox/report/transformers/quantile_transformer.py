"""
Quantile Regression Diagnostic Result Transformer.

Converts DiagnosticReport objects into template-ready dictionaries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class QuantileTransformer:
    """
    Transform quantile diagnostic results into template-ready data.

    Parameters
    ----------
    report : DiagnosticReport
        Quantile diagnostic report object.
    """

    def __init__(self, report: Any):
        self.report = report

    def transform(self) -> dict[str, Any]:
        """
        Transform quantile diagnostic report into template context.

        Returns
        -------
        dict
            Template-ready dictionary.
        """
        return {
            "health": self._transform_health(),
            "tests": self._transform_tests(),
            "recommendations": self._transform_recommendations(),
        }

    def _transform_health(self) -> dict[str, Any]:
        r = self.report
        score = getattr(r, "health_score", 0)
        status = getattr(r, "health_status", "poor")

        # Color based on status
        color_map = {
            "good": "#10b981",
            "fair": "#f59e0b",
            "poor": "#ef4444",
        }
        return {
            "score": score,
            "score_pct": f"{score * 100:.1f}%",
            "status": status,
            "color": color_map.get(status, "#6b7280"),
        }

    def _transform_tests(self) -> list[dict[str, Any]]:
        r = self.report
        diagnostics = getattr(r, "diagnostics", [])

        tests = []
        status_icons = {
            "pass": "&#10003;",
            "warning": "&#9888;",
            "fail": "&#10007;",
        }
        status_classes = {
            "pass": "text-success",
            "warning": "text-warning",
            "fail": "text-danger",
        }
        for d in diagnostics:
            status = getattr(d, "status", "fail")
            tests.append(
                {
                    "name": getattr(d, "test_name", ""),
                    "statistic": getattr(d, "statistic", 0),
                    "pvalue": getattr(d, "p_value", None),
                    "status": status,
                    "status_icon": status_icons.get(status, "?"),
                    "status_class": status_classes.get(status, ""),
                    "message": getattr(d, "message", ""),
                }
            )
        return tests

    def _transform_recommendations(self) -> list[str]:
        r = self.report
        diagnostics = getattr(r, "diagnostics", [])
        recs = []
        for d in diagnostics:
            rec = getattr(d, "recommendation", None)
            if rec:
                recs.append(rec)
        return recs
