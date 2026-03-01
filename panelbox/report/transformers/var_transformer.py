"""
VAR (Vector Autoregression) Result Transformer.

Converts PanelVARResult objects into template-ready dictionaries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VARTransformer:
    """
    Transform VAR results into template-ready data.

    Parameters
    ----------
    result : PanelVARResult
        Panel VAR result object.
    """

    def __init__(self, result: Any):
        self.result = result

    def transform(self) -> dict[str, Any]:
        """
        Transform VAR result into template context.

        Returns
        -------
        dict
            Template-ready dictionary.
        """
        return {
            "model_info": self._transform_model_info(),
            "equations": self._transform_equations(),
            "diagnostics": self._transform_diagnostics(),
            "stability": self._transform_stability(),
        }

    def _transform_model_info(self) -> dict[str, Any]:
        r = self.result
        return {
            "K": getattr(r, "K", "—"),
            "p": getattr(r, "p", "—"),
            "N": getattr(r, "N", "—"),
            "n_obs": getattr(r, "n_obs", "—"),
            "method": getattr(r, "method", "ols"),
            "cov_type": getattr(r, "cov_type", "—"),
            "endog_names": getattr(r, "endog_names", []),
        }

    def _transform_equations(self) -> list[dict[str, Any]]:
        r = self.result
        K = getattr(r, "K", 0)
        endog_names = getattr(r, "endog_names", [f"y{k + 1}" for k in range(K)])
        exog_names = getattr(r, "exog_names", [])
        params_by_eq = getattr(r, "params_by_eq", [])
        std_errors_by_eq = getattr(r, "std_errors_by_eq", [])

        equations = []
        for k in range(K):
            if k >= len(params_by_eq):
                break
            params = params_by_eq[k]
            se = std_errors_by_eq[k] if k < len(std_errors_by_eq) else None

            coeffs = []
            for j, name in enumerate(exog_names):
                if j >= len(params):
                    break
                coef_val = float(params[j])
                se_val = float(se[j]) if se is not None and j < len(se) else 0.0
                tstat = coef_val / se_val if se_val != 0 else 0.0

                # Approximate p-value from t-stat (two-tailed)
                try:
                    pval = 2 * (1 - _normal_cdf(abs(tstat)))
                except Exception:
                    pval = 1.0

                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                coeffs.append(
                    {
                        "name": name,
                        "coef": coef_val,
                        "se": se_val,
                        "tstat": tstat,
                        "pvalue": pval,
                        "stars": stars,
                    }
                )

            equations.append(
                {
                    "name": endog_names[k],
                    "coefficients": coeffs,
                }
            )

        return equations

    def _transform_diagnostics(self) -> dict[str, Any]:
        r = self.result
        return {
            "aic": _safe_float(getattr(r, "aic", None)),
            "bic": _safe_float(getattr(r, "bic", None)),
            "hqic": _safe_float(getattr(r, "hqic", None)),
            "loglik": _safe_float(getattr(r, "loglik", None)),
        }

    def _transform_stability(self) -> dict[str, Any]:
        r = self.result
        max_mod = _safe_float(getattr(r, "max_eigenvalue_modulus", None))
        margin = _safe_float(getattr(r, "stability_margin", None))
        is_stable = isinstance(max_mod, float) and max_mod < 1.0
        return {
            "is_stable": is_stable,
            "max_eigenvalue_modulus": max_mod,
            "stability_margin": margin,
        }


def _safe_float(value: Any) -> Any:
    """Safely convert to float."""
    if value is None:
        return "—"
    try:
        return float(value)
    except (ValueError, TypeError):
        return "—"


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    import math

    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
