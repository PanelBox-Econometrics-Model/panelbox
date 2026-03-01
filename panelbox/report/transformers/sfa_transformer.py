"""
SFA (Stochastic Frontier Analysis) Result Transformer.

Converts SFResult objects into template-ready dictionaries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SFATransformer:
    """
    Transform SFA results into template-ready data.

    Parameters
    ----------
    result : SFResult
        SFA result object.
    """

    def __init__(self, result: Any):
        self.result = result

    def transform(self) -> dict[str, Any]:
        """
        Transform SFA result into template context.

        Returns
        -------
        dict
            Template-ready dictionary.
        """
        return {
            "model_info": self._transform_model_info(),
            "coefficients": self._transform_coefficients(),
            "variance_components": self._transform_variance_components(),
            "efficiency": self._transform_efficiency(),
            "fit_statistics": self._transform_fit_statistics(),
        }

    def _transform_model_info(self) -> dict[str, Any]:
        r = self.result
        frontier_type = getattr(r, "frontier_type", getattr(r, "frontier", "production"))
        distribution = getattr(r, "distribution", getattr(r, "dist", "half-normal"))
        return {
            "frontier_type": frontier_type,
            "distribution": distribution,
            "nobs": getattr(r, "nobs", "—"),
            "n_entities": getattr(r, "n_entities", None),
            "n_periods": getattr(r, "n_periods", None),
            "converged": getattr(r, "converged", True),
            "nparams": getattr(r, "nparams", "—"),
        }

    def _transform_coefficients(self) -> list[dict[str, Any]]:
        r = self.result
        params = getattr(r, "params", None)
        se = getattr(r, "se", None)
        tvalues = getattr(r, "tvalues", None)
        pvalues = getattr(r, "pvalues", None)

        if params is None:
            return []

        result = []
        # Filter out variance parameters (they go in variance_components)
        variance_params = {"sigma_v", "sigma_u", "sigma", "lambda", "mu", "eta", "gamma"}

        for name in params.index:
            if name.lower() in variance_params:
                continue
            pval = float(pvalues[name]) if pvalues is not None and name in pvalues.index else 1.0
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            result.append(
                {
                    "name": name,
                    "coef": float(params[name]),
                    "se": float(se[name]) if se is not None and name in se.index else 0,
                    "tstat": (
                        float(tvalues[name]) if tvalues is not None and name in tvalues.index else 0
                    ),
                    "pvalue": pval,
                    "stars": stars,
                }
            )
        return result

    def _transform_variance_components(self) -> dict[str, Any]:
        r = self.result
        return {
            "sigma_v": _safe_float(getattr(r, "sigma_v", None)),
            "sigma_u": _safe_float(getattr(r, "sigma_u", None)),
            "sigma": _safe_float(getattr(r, "sigma", None)),
            "sigma_sq": _safe_float(getattr(r, "sigma_sq", None)),
            "lambda_param": _safe_float(getattr(r, "lambda_param", None)),
            "gamma": _safe_float(getattr(r, "gamma", None)),
        }

    def _transform_efficiency(self) -> dict[str, Any] | None:
        r = self.result
        try:
            eff = r.efficiency_scores
            if eff is None or len(eff) == 0:
                return None
            return {
                "mean": float(eff.mean()) if hasattr(eff, "mean") else None,
                "median": float(eff.median()) if hasattr(eff, "median") else None,
                "std": float(eff.std()) if hasattr(eff, "std") else None,
                "min": float(eff.min()),
                "max": float(eff.max()),
                "count": len(eff),
            }
        except Exception:
            return None

    def _transform_fit_statistics(self) -> dict[str, Any]:
        r = self.result
        return {
            "loglikelihood": _safe_float(getattr(r, "loglik", None)),
            "aic": _safe_float(getattr(r, "aic", None)),
            "bic": _safe_float(getattr(r, "bic", None)),
        }


def _safe_float(value: Any) -> Any:
    """Safely convert to float, return '—' if not possible."""
    if value is None:
        return "—"
    try:
        return float(value)
    except (ValueError, TypeError):
        return "—"
