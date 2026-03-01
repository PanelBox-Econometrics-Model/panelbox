"""
Discrete/MLE Result Transformer.

Converts nonlinear panel model results into template-ready dictionaries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DiscreteTransformer:
    """
    Transform discrete/MLE results into template-ready data.

    Parameters
    ----------
    data : dict
        Discrete model result data dictionary.
    """

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def transform(self) -> dict[str, Any]:
        """
        Transform discrete model data into template context.

        Returns
        -------
        dict
            Template-ready dictionary.
        """
        return {
            "model_info": self._transform_model_info(),
            "coefficients": self._transform_coefficients(),
            "fit_statistics": self._transform_fit_statistics(),
            "classification": self._transform_classification(),
        }

    def _transform_model_info(self) -> dict[str, Any]:
        return {
            "model_type": self.data.get("model_type_full", self.data.get("model_type", "MLE")),
            "distribution": self.data.get("distribution", "—"),
            "nobs": self.data.get("nobs", "—"),
            "n_entities": self.data.get("n_entities", "—"),
            "n_periods": self.data.get("n_periods", "—"),
            "converged": self.data.get("converged", False),
            "n_iter": self.data.get("n_iter", self.data.get("iterations", "—")),
            "se_type": self.data.get("se_type", "—"),
        }

    def _transform_coefficients(self) -> list[dict[str, Any]]:
        coefficients = self.data.get("coefficients", [])
        if isinstance(coefficients, list) and len(coefficients) > 0:
            result = []
            for coef in coefficients:
                pval = coef.get("pvalue", 1.0)
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                result.append(
                    {
                        "name": coef.get("name", ""),
                        "coef": coef.get("coef", 0),
                        "se": coef.get("se", 0),
                        "zstat": coef.get("tstat", coef.get("zstat", 0)),
                        "pvalue": pval,
                        "stars": stars,
                        "ci_lower": coef.get("ci_lower", ""),
                        "ci_upper": coef.get("ci_upper", ""),
                    }
                )
            return result
        return []

    def _transform_fit_statistics(self) -> dict[str, Any]:
        return {
            "loglikelihood": self.data.get("loglikelihood", self.data.get("loglik", "—")),
            "aic": self.data.get("aic", "—"),
            "bic": self.data.get("bic", "—"),
            "pseudo_r_squared": self.data.get("pseudo_r_squared", self.data.get("pseudo_r2", "—")),
        }

    def _transform_classification(self) -> dict[str, Any] | None:
        metrics = self.data.get("classification_metrics")
        if not metrics:
            return None
        return {
            "accuracy": metrics.get("accuracy", "—"),
            "precision": metrics.get("precision", "—"),
            "recall": metrics.get("recall", "—"),
            "f1_score": metrics.get("f1_score", "—"),
        }
