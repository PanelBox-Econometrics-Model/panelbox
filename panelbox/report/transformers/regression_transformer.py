"""
Regression Result Transformer.

Converts panel regression results into template-ready dictionaries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RegressionTransformer:
    """
    Transform regression results into template-ready data.

    Parameters
    ----------
    data : dict
        Regression result data dictionary.
    """

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def transform(self) -> dict[str, Any]:
        """
        Transform regression data into template context.

        Returns
        -------
        dict
            Template-ready dictionary.
        """
        return {
            "model_info": self._transform_model_info(),
            "coefficients": self._transform_coefficients(),
            "fit_statistics": self._transform_fit_statistics(),
        }

    def _transform_model_info(self) -> dict[str, Any]:
        return {
            "estimator": self.data.get("model_type", self.data.get("estimator", "Panel OLS")),
            "formula": self.data.get("formula", "—"),
            "nobs": self.data.get("nobs", "—"),
            "n_entities": self.data.get("n_entities", "—"),
            "n_periods": self.data.get("n_periods", "—"),
            "se_type": self.data.get("se_type", self.data.get("cov_type", "—")),
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
                    "ci_lower": coef.get("ci_lower", ""),
                    "ci_upper": coef.get("ci_upper", ""),
                }
            )
        return result

    def _transform_fit_statistics(self) -> dict[str, Any]:
        return {
            "r_squared": self.data.get("r_squared", "—"),
            "adj_r_squared": self.data.get("adj_r_squared", "—"),
            "f_statistic": self.data.get("f_statistic", "—"),
            "f_pvalue": self.data.get("f_pvalue", "—"),
            "aic": self.data.get("aic", "—"),
            "bic": self.data.get("bic", "—"),
        }
