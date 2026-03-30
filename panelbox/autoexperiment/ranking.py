"""Composite ranking and automatic standard error selection for AutoExperiment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default component weights for composite scoring
_DEFAULT_WEIGHTS = {
    "criterion": 0.50,
    "tests": 0.30,
    "signs": 0.15,
    "parsimony": 0.05,
}


class ModelRanker:
    """Rank models by a weighted composite score.

    Parameters
    ----------
    weights : dict, optional
        Weights for each component of the score.
        Default: ``{'criterion': 0.50, 'tests': 0.30, 'signs': 0.15, 'parsimony': 0.05}``.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(weights)

        # Normalise weights so they sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def rank(self, model_evaluations: list[dict]) -> pd.DataFrame:
        """Calculate composite score and return a ranked DataFrame.

        Each element in *model_evaluations* is a dict with keys:

        - ``model_type`` (str)
        - ``formula`` (str)
        - ``metrics`` (dict with ``bic``, ``aic``, ``rsq_adj``)
        - ``classification`` (str: ``VALID``, ``WARNING``, ``INVALID``)
        - ``cov_type`` (str)
        - ``failed_tests`` (list[str])
        - ``validation`` (ValidationReport, optional)
        - ``sign_violations`` (int, optional) – number of violated sign constraints
        - ``sign_total`` (int, optional) – total sign constraints checked

        Parameters
        ----------
        model_evaluations : list[dict]
            One dict per model evaluation.

        Returns
        -------
        pd.DataFrame
            Ranked table with columns: ``rank``, ``model_type``, ``formula``,
            ``score``, ``bic``, ``aic``, ``rsq_adj``, ``classification``,
            ``n_vars``, ``cov_type``, ``tests_passed``, ``signs_ok``.
        """
        if not model_evaluations:
            return pd.DataFrame(
                columns=[
                    "rank",
                    "model_type",
                    "formula",
                    "score",
                    "bic",
                    "aic",
                    "rsq_adj",
                    "classification",
                    "n_vars",
                    "cov_type",
                    "tests_passed",
                    "signs_ok",
                ]
            )

        # ── Collect raw values ─────────────────────────────────────────
        rows: list[dict] = []
        for ev in model_evaluations:
            m = ev.get("metrics", {})
            n_vars = self._count_vars(ev.get("formula", ""))
            failed = ev.get("failed_tests", [])
            validation = ev.get("validation")

            # Count total tests run and passed
            n_total_tests = self._count_total_tests(validation, failed)
            n_passed = n_total_tests - len(failed)

            # Sign constraints
            sign_violations = ev.get("sign_violations", 0)
            sign_total = ev.get("sign_total", 0)
            signs_ok = sign_total - sign_violations

            rows.append(
                {
                    "model_type": ev.get("model_type", ""),
                    "formula": ev.get("formula", ""),
                    "bic": m.get("bic", np.inf),
                    "aic": m.get("aic", np.inf),
                    "rsq_adj": m.get("rsq_adj"),
                    "classification": ev.get("classification", "INVALID"),
                    "n_vars": n_vars,
                    "cov_type": ev.get("cov_type", ""),
                    "tests_passed": f"{n_passed}/{n_total_tests}",
                    "signs_ok": f"{signs_ok}/{sign_total}" if sign_total > 0 else "n/a",
                    # Raw values for scoring
                    "_bic": m.get("bic", np.inf),
                    "_aic": m.get("aic", np.inf),
                    "_n_passed": n_passed,
                    "_n_total_tests": n_total_tests,
                    "_signs_ok": signs_ok,
                    "_sign_total": sign_total,
                    "_n_vars": n_vars,
                }
            )

        df = pd.DataFrame(rows)

        # ── Component scores [0, 1] ───────────────────────────────────
        # 1. Criterion score (using BIC; higher = better)
        df["_criterion_score"] = self._normalize_criterion(df["_bic"].values, prefer_lower=True)

        # 2. Tests score: n_passed / n_total
        df["_tests_score"] = np.where(
            df["_n_total_tests"] > 0,
            df["_n_passed"] / df["_n_total_tests"],
            0.0,
        )

        # 3. Signs score: n_compliant / n_constrained
        df["_signs_score"] = np.where(
            df["_sign_total"] > 0,
            df["_signs_ok"] / df["_sign_total"],
            1.0,  # No constraints → full score
        )

        # 4. Parsimony score: 1 - (n_vars / max_vars)
        max_vars = df["_n_vars"].max()
        if max_vars > 0:
            df["_parsimony_score"] = 1.0 - (df["_n_vars"] / max_vars)
        else:
            df["_parsimony_score"] = 1.0

        # ── Composite score ────────────────────────────────────────────
        w = self.weights
        df["score"] = (
            w.get("criterion", 0) * df["_criterion_score"]
            + w.get("tests", 0) * df["_tests_score"]
            + w.get("signs", 0) * df["_signs_score"]
            + w.get("parsimony", 0) * df["_parsimony_score"]
        )

        # ── INVALID models go last ────────────────────────────────────
        cls_order = {"VALID": 0, "WARNING": 1, "INVALID": 2}
        df["_cls_order"] = df["classification"].map(cls_order).fillna(2).astype(int)

        df = df.sort_values(["_cls_order", "score"], ascending=[True, False]).reset_index(drop=True)

        df["rank"] = range(1, len(df) + 1)

        # ── Clean up internal columns ──────────────────────────────────
        keep = [
            "rank",
            "model_type",
            "formula",
            "score",
            "bic",
            "aic",
            "rsq_adj",
            "classification",
            "n_vars",
            "cov_type",
            "tests_passed",
            "signs_ok",
        ]
        return df[keep]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_criterion(values: np.ndarray, prefer_lower: bool = True) -> np.ndarray:
        """Min-max normalisation to [0, 1].

        Parameters
        ----------
        values : np.ndarray
            Raw criterion values (e.g. BIC).
        prefer_lower : bool
            If ``True``, the lowest value maps to 1.0 (best).

        Returns
        -------
        np.ndarray
            Normalised scores in [0, 1].
        """
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.zeros_like(values, dtype=float)

        finite_vals = values[finite]
        vmin = finite_vals.min()
        vmax = finite_vals.max()

        if vmax == vmin:
            # All values identical → everyone gets 1.0
            result = np.ones_like(values, dtype=float)
            result[~finite] = 0.0
            return result

        normalised = (values - vmin) / (vmax - vmin)
        if prefer_lower:
            normalised = 1.0 - normalised

        # Non-finite values get 0 score
        normalised[~finite] = 0.0
        return normalised

    @staticmethod
    def _count_vars(formula: str) -> int:
        """Count the number of RHS variables in a formula string."""
        if "~" not in formula:
            return 0
        rhs = formula.split("~", 1)[1].strip()
        if not rhs:
            return 0
        return len([v.strip() for v in rhs.split("+") if v.strip()])

    @staticmethod
    def _count_total_tests(validation: Any, failed_tests: list[str]) -> int:
        """Count total tests run from a validation report."""
        if validation is None:
            return max(len(failed_tests), 1)

        total = 0
        for category in (
            getattr(validation, "specification_tests", None),
            getattr(validation, "serial_tests", None),
            getattr(validation, "het_tests", None),
            getattr(validation, "cd_tests", None),
        ):
            if category:
                total += len(category)
        return max(total, 1)


class AutoCovTypeSelector:
    """Select the most appropriate standard error type based on diagnostic tests.

    Decision hierarchy:

    1. Pesaran CD rejects (cross-sectional dependence) → ``'driscoll_kraay'``
    2. Wooldridge AR rejects (serial correlation) → ``'newey_west'``
    3. Modified Wald / Breusch-Pagan rejects (heteroskedasticity) → ``'robust'``
    4. Nothing rejects → ``'clustered'`` (conservative default for panels)
       or ``'nonrobust'`` if *n_entities* > 30.
    """

    def select(self, validation_report: Any, n_entities: int = 0) -> str:
        """Return the recommended ``cov_type``.

        Parameters
        ----------
        validation_report : ValidationReport
            Output of ``ValidationSuite.run()``.
        n_entities : int
            Number of cross-sectional units in the panel.

        Returns
        -------
        str
            One of ``'driscoll_kraay'``, ``'newey_west'``, ``'robust'``,
            ``'clustered'``, ``'nonrobust'``.
        """
        cd_tests = getattr(validation_report, "cd_tests", None) or {}
        serial_tests = getattr(validation_report, "serial_tests", None) or {}
        het_tests = getattr(validation_report, "het_tests", None) or {}

        # 1. Cross-sectional dependence
        pesaran = cd_tests.get("Pesaran CD")
        if pesaran is not None and pesaran.reject_null:
            return "driscoll_kraay"

        # 2. Serial correlation
        for name in ("Wooldridge", "Breusch-Godfrey"):
            t = serial_tests.get(name)
            if t is not None and t.reject_null:
                return "newey_west"

        # 3. Heteroskedasticity
        for name in ("Modified Wald", "Breusch-Pagan", "White"):
            t = het_tests.get(name)
            if t is not None and t.reject_null:
                return "robust"

        # 4. Nothing rejects
        if n_entities > 30:
            return "nonrobust"
        return "clustered"

    def explain(self, validation_report: Any, n_entities: int = 0) -> str:
        """Return a human-readable explanation for the selected ``cov_type``.

        Parameters
        ----------
        validation_report : ValidationReport
            Output of ``ValidationSuite.run()``.
        n_entities : int
            Number of cross-sectional units.

        Returns
        -------
        str
            Textual justification for the chosen standard error type.
        """
        cd_tests = getattr(validation_report, "cd_tests", None) or {}
        serial_tests = getattr(validation_report, "serial_tests", None) or {}
        het_tests = getattr(validation_report, "het_tests", None) or {}

        # 1. Cross-sectional dependence
        pesaran = cd_tests.get("Pesaran CD")
        if pesaran is not None and pesaran.reject_null:
            return (
                "Pesaran CD test rejects the null of no cross-sectional dependence "
                f"(p={pesaran.pvalue:.4f}). Using Driscoll-Kraay standard errors, "
                "which are robust to cross-sectional dependence, heteroskedasticity, "
                "and serial correlation."
            )

        # 2. Serial correlation
        for name in ("Wooldridge", "Breusch-Godfrey"):
            t = serial_tests.get(name)
            if t is not None and t.reject_null:
                return (
                    f"{name} test rejects the null of no serial correlation "
                    f"(p={t.pvalue:.4f}). Using Newey-West (HAC) standard errors, "
                    "which are robust to heteroskedasticity and autocorrelation."
                )

        # 3. Heteroskedasticity
        for name in ("Modified Wald", "Breusch-Pagan", "White"):
            t = het_tests.get(name)
            if t is not None and t.reject_null:
                return (
                    f"{name} test rejects the null of homoskedasticity "
                    f"(p={t.pvalue:.4f}). Using heteroskedasticity-robust (HC1) "
                    "standard errors."
                )

        # 4. Nothing rejects
        if n_entities > 30:
            return (
                "No diagnostic test detected violations. With N > 30 entities, "
                "using conventional (nonrobust) standard errors."
            )
        return (
            "No diagnostic test detected violations. Using clustered standard "
            "errors as a conservative default for panel data."
        )
