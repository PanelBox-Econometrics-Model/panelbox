"""Model evaluator: estimate panel models and auto-select standard errors."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Model type aliases -> (class attribute name on pb module, model_type string in results)
_MODEL_MAP = {
    # --- Linear static ---
    "pooled_ols": "PooledOLS",
    "pooled": "PooledOLS",
    "fe": "FixedEffects",
    "fixed_effects": "FixedEffects",
    "re": "RandomEffects",
    "random_effects": "RandomEffects",
    "fd": "FirstDifferenceEstimator",
    "first_difference": "FirstDifferenceEstimator",
    "between": "BetweenEstimator",
    "be": "BetweenEstimator",
    # --- Heterogeneous slopes ---
    "mean_group": "MeanGroupEstimator",
    "mg": "MeanGroupEstimator",
    "pooled_mean_group": "PooledMeanGroupEstimator",
    "pmg": "PooledMeanGroupEstimator",
    # --- System ---
    "sur": "SUR",
    "sur_homogeneous": "SUR",
    # --- Dynamic (bias-corrected FE) ---
    "lsdvc": "LSDVC",
    "lsdvc_ah": "LSDVC",
    "lsdvc_bb": "LSDVC",
    # --- Discrete choice (binary/ordered depvar) ---
    "pooled_logit": "PooledLogit",
    "logit": "PooledLogit",
    "pooled_probit": "PooledProbit",
    "probit": "PooledProbit",
    "fe_logit": "FixedEffectsLogit",
    "re_probit": "RandomEffectsProbit",
}

# Canonical keys (used to normalise user input)
_CANONICAL = {
    # --- Linear static ---
    "pooled_ols": "pooled_ols",
    "pooled": "pooled_ols",
    "fe": "fe",
    "fixed_effects": "fe",
    "re": "re",
    "random_effects": "re",
    "fd": "fd",
    "first_difference": "fd",
    "between": "between",
    "be": "between",
    # --- Heterogeneous slopes ---
    "mean_group": "mean_group",
    "mg": "mean_group",
    "pooled_mean_group": "pooled_mean_group",
    "pmg": "pooled_mean_group",
    # --- System ---
    "sur": "sur",
    "sur_homogeneous": "sur_homogeneous",
    # --- Dynamic ---
    "lsdvc": "lsdvc",
    "lsdvc_ah": "lsdvc_ah",
    "lsdvc_bb": "lsdvc_bb",
    # --- Discrete choice ---
    "pooled_logit": "pooled_logit",
    "logit": "pooled_logit",
    "pooled_probit": "pooled_probit",
    "probit": "pooled_probit",
    "fe_logit": "fe_logit",
    "re_probit": "re_probit",
}

# Models that need special handling in _fit_model
_SUR_MODELS = {"sur", "sur_homogeneous"}
_DISCRETE_MODELS = {"pooled_logit", "pooled_probit", "fe_logit", "re_probit"}
_DYNAMIC_MODELS = {"lsdvc"}


def _compute_ic(results: Any, criterion: str) -> float:
    """Compute BIC or AIC from a PanelResults object."""
    n = results.nobs
    k = len(results.params)
    ssr = float(np.sum(results.resid**2))
    if ssr <= 0:
        return np.inf
    log_lik_term = n * np.log(ssr / n)
    if criterion == "bic":
        return log_lik_term + k * np.log(n)
    else:
        return log_lik_term + 2 * k


class ModelEvaluator:
    """Estimate a panel model, run validation tests, and auto-select standard errors.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    entity_col : str
        Column identifying cross-sectional units.
    time_col : str
        Column identifying time periods.
    """

    def __init__(self, data: pd.DataFrame, entity_col: str, time_col: str):
        self.data = data
        self.entity_col = entity_col
        self.time_col = time_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        formula: str,
        model_type: str,
        data: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> dict:
        """Estimate a model, run diagnostics, and auto-select standard errors.

        Parameters
        ----------
        formula : str
            Model formula (e.g. ``"invest ~ value + capital"``).
        model_type : str
            One of ``pooled_ols``, ``fe``, ``re``, ``fd`` (or long-form aliases).
        data : pd.DataFrame, optional
            If provided, overrides the data passed at init.
        alpha : float, default 0.05
            Significance level for diagnostic tests.

        Returns
        -------
        dict
            Keys: ``results``, ``model_type``, ``formula``, ``cov_type``,
            ``metrics``, ``validation``, ``classification``, ``failed_tests``.
        """
        from panelbox.autoexperiment.specification_filter import SpecificationFilter

        df = data if data is not None else self.data
        canonical = _CANONICAL.get(model_type.lower())
        if canonical is None:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Valid types: {list(_CANONICAL.keys())}"
            )

        # Dynamic models (LSDVC) use bootstrap — no cov_type re-estimation
        is_dynamic = canonical in ("lsdvc", "lsdvc_ah", "lsdvc_bb")

        # 1. Initial estimation
        if is_dynamic:
            results = self._fit_model(formula, model_type, df)
            auto_cov = "bootstrap"
        else:
            results = self._fit_model(formula, model_type, df, cov_type="nonrobust")

        # 2. Run validation suite
        validation_report = self._run_validation(results, alpha=alpha)

        # 3. Auto-select cov_type (skip for dynamic models)
        if not is_dynamic:
            auto_cov = self._auto_select_cov_type(validation_report)

        # 4. Re-estimate with correct SEs if needed
        if not is_dynamic and auto_cov != "nonrobust":
            try:
                results = self._fit_model(formula, model_type, df, cov_type=auto_cov)
                # Re-run validation on the re-estimated model
                validation_report = self._run_validation(results, alpha=alpha)
            except Exception:
                logger.warning(
                    "Re-estimation with cov_type='%s' failed; keeping nonrobust.",
                    auto_cov,
                )
                auto_cov = "nonrobust"

        # 5. Classify the specification
        spec_filter = SpecificationFilter(alpha=alpha)
        classification_info = spec_filter.classify(canonical, validation_report, alpha=alpha)

        # 6. Compute metrics
        metrics = self._compute_metrics(results)

        return {
            "results": results,
            "model_type": canonical,
            "formula": formula,
            "cov_type": auto_cov,
            "metrics": metrics,
            "validation": validation_report,
            "classification": classification_info["classification"],
            "failed_tests": classification_info["failed_tests"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_model(
        self,
        formula: str,
        model_type: str,
        data: pd.DataFrame,
        cov_type: str = "nonrobust",
    ) -> Any:
        """Instantiate and fit a panel model."""
        import panelbox as pb

        canonical = _CANONICAL.get(model_type.lower(), model_type.lower())
        cls_name = _MODEL_MAP.get(model_type.lower())
        if cls_name is None:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # LSDVC: dynamic model with bootstrap inference (no cov_type)
        if canonical in ("lsdvc", "lsdvc_ah", "lsdvc_bb"):
            from panelbox.models.dynamic.lsdvc import LSDVC

            initial_map = {
                "lsdvc": "ab",
                "lsdvc_ah": "ah",
                "lsdvc_bb": "bb",
            }
            model = LSDVC(
                formula,
                data,
                self.entity_col,
                self.time_col,
                lags=1,
                initial_estimator=initial_map[canonical],
                bias_order=2,
            )
            return model.fit(n_bootstrap=100, seed=42)

        # Resolve class from panelbox namespace (with fallback for submodules)
        cls = getattr(pb, cls_name, None)
        if cls is None:
            # Try submodule imports for classes not in top-level __init__
            _submodule_map = {
                "MeanGroupEstimator": "panelbox.models.static.mean_group",
                "PooledMeanGroupEstimator": "panelbox.models.static.mean_group",
                "SUR": "panelbox.models.static.sur",
            }
            mod_path = _submodule_map.get(cls_name)
            if mod_path:
                import importlib

                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name, None)
        if cls is None:
            raise ValueError(f"Cannot find class '{cls_name}' in panelbox")

        # SUR needs special handling for the homogeneous flag
        if canonical == "sur_homogeneous":
            model = cls(formula, data, self.entity_col, self.time_col, homogeneous=True)
            return model.fit(cov_type="sur")
        elif canonical == "sur":
            model = cls(formula, data, self.entity_col, self.time_col, homogeneous=False)
            return model.fit(cov_type="sur")

        model = cls(formula, data, self.entity_col, self.time_col)

        # Discrete models may not support all cov_types
        if canonical in _DISCRETE_MODELS:
            try:
                return model.fit(cov_type=cov_type)
            except (TypeError, ValueError):
                return model.fit()

        return model.fit(cov_type=cov_type)

    @staticmethod
    def _run_validation(results: Any, alpha: float = 0.05) -> Any:
        """Run the full ValidationSuite on fitted results."""
        from panelbox.validation.validation_suite import ValidationSuite

        suite = ValidationSuite(results)
        return suite.run(tests="all", alpha=alpha)

    @staticmethod
    def _auto_select_cov_type(validation_report: Any) -> str:
        """Choose a covariance estimator based on diagnostic test outcomes.

        Priority order (most severe first):
        1. Cross-sectional dependence (Pesaran CD rejects) -> driscoll_kraay
        2. Heteroskedasticity (Modified Wald or Breusch-Pagan rejects) -> robust
        3. Serial correlation (Wooldridge AR rejects) -> clustered
        4. Nothing rejects -> nonrobust
        """
        # Check cross-sectional dependence
        cd_tests = validation_report.cd_tests or {}
        if "Pesaran CD" in cd_tests and cd_tests["Pesaran CD"].reject_null:
            return "driscoll_kraay"

        # Check heteroskedasticity
        het_tests = validation_report.het_tests or {}
        het_reject = False
        for name in ("Modified Wald", "Breusch-Pagan", "White"):
            if name in het_tests and het_tests[name].reject_null:
                het_reject = True
                break
        if het_reject:
            return "robust"

        # Check serial correlation
        serial_tests = validation_report.serial_tests or {}
        for name in ("Wooldridge", "Breusch-Godfrey"):
            if name in serial_tests and serial_tests[name].reject_null:
                return "clustered"

        return "nonrobust"

    @staticmethod
    def _compute_metrics(results: Any) -> dict:
        """Extract key goodness-of-fit metrics from PanelResults."""
        n = results.nobs
        resid = results.resid
        ssr = float(np.sum(resid**2))
        rmse = float(np.sqrt(ssr / n)) if n > 0 else np.nan

        rsq = getattr(results, "rsquared", None)
        if rsq is None:
            rsq = getattr(results, "rsquared_within", None)

        rsq_adj = getattr(results, "rsquared_adj", None)

        return {
            "rsq": rsq,
            "rsq_adj": rsq_adj,
            "bic": _compute_ic(results, "bic"),
            "aic": _compute_ic(results, "aic"),
            "rmse": rmse,
            "nobs": n,
        }
