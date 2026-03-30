"""Specification filter: classify models as VALID / WARNING / INVALID."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SpecificationFilter:
    """Classify panel model specifications based on diagnostic test results.

    Parameters
    ----------
    required_tests : list[str], optional
        Override the default per-model-type test list.
    alpha : float, default 0.05
        Significance level used when interpreting test results.
    """

    # Tests that should be checked for each model type.
    # Keys in the validation report use their canonical names:
    #   specification_tests: "RESET", "Mundlak"
    #   het_tests: "Breusch-Pagan", "Modified Wald", "White"
    #   serial_tests: "Wooldridge", "Breusch-Godfrey", "Baltagi-Wu"
    #   cd_tests: "Pesaran CD", "Breusch-Pagan LM", "Frees"
    DEFAULT_TESTS: dict[str, list[str]] = {
        # Linear static
        "pooled_ols": ["Breusch-Pagan", "RESET"],
        "fe": ["Pesaran CD", "Wooldridge", "Modified Wald", "RESET"],
        "re": ["Mundlak", "Breusch-Pagan", "RESET"],
        "fd": ["Breusch-Pagan", "RESET"],
        "between": ["Breusch-Pagan", "RESET"],
        # Heterogeneous slopes
        "mean_group": ["Breusch-Pagan", "RESET"],
        "pooled_mean_group": ["Breusch-Pagan", "RESET"],
        # System
        "sur": ["Breusch-Pagan", "RESET"],
        "sur_homogeneous": ["Breusch-Pagan", "RESET"],
        # Dynamic
        "lsdvc": ["Breusch-Pagan", "RESET"],
        "lsdvc_ah": ["Breusch-Pagan", "RESET"],
        "lsdvc_bb": ["Breusch-Pagan", "RESET"],
        # Discrete choice
        "pooled_logit": ["RESET"],
        "pooled_probit": ["RESET"],
        "fe_logit": ["RESET"],
        "re_probit": ["RESET"],
    }

    # Failing a critical test makes the specification INVALID.
    CRITICAL_TESTS: list[str] = ["RESET", "Mundlak"]

    def __init__(
        self,
        required_tests: list[str] | None = None,
        alpha: float = 0.05,
    ):
        self._required_tests_override = required_tests
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        model_type: str,
        validation_report: Any,
        alpha: float | None = None,
    ) -> dict:
        """Classify a specification as VALID, WARNING, or INVALID.

        Parameters
        ----------
        model_type : str
            Canonical model key (``pooled_ols``, ``fe``, ``re``, ``fd``).
        validation_report : ValidationReport
            Output of ``ValidationSuite.run()``.
        alpha : float, optional
            Override significance level (default: use the instance level).

        Returns
        -------
        dict
            ``classification``, ``passed_tests``, ``failed_tests``,
            ``critical_failures``, ``auto_cov_type``.
        """
        if alpha is None:
            alpha = self.alpha

        required = (
            self._required_tests_override
            if self._required_tests_override is not None
            else self.DEFAULT_TESTS.get(model_type, [])
        )

        # Collect every test result from the report into a flat dict
        all_tests = self._flatten_tests(validation_report)

        passed: list[str] = []
        failed: list[str] = []
        critical_failures: list[str] = []

        for test_name in required:
            result = all_tests.get(test_name)
            if result is None:
                # Test was not run (e.g. model type doesn't support it) – skip
                continue
            if result.reject_null:
                failed.append(test_name)
                if test_name in self.CRITICAL_TESTS:
                    critical_failures.append(test_name)
            else:
                passed.append(test_name)

        # Determine classification
        if critical_failures:
            classification = "INVALID"
        elif failed:
            classification = "WARNING"
        else:
            classification = "VALID"

        # Auto cov_type recommendation
        auto_cov = self._recommend_cov_type(all_tests)

        return {
            "classification": classification,
            "passed_tests": passed,
            "failed_tests": failed,
            "critical_failures": critical_failures,
            "auto_cov_type": auto_cov,
        }

    def hausman_decision(
        self,
        data: Any,
        formula: str,
        entity_col: str,
        time_col: str,
        alpha: float | None = None,
    ) -> str:
        """Run the Hausman test and return ``'fe'`` or ``'re'``.

        Parameters
        ----------
        data : pd.DataFrame
            Panel dataset.
        formula : str
            Model formula.
        entity_col, time_col : str
            Panel identifiers.
        alpha : float, optional
            Significance level (default: instance level).

        Returns
        -------
        str
            ``'fe'`` if the null is rejected, ``'re'`` otherwise.
        """
        import panelbox as pb
        from panelbox.validation.specification.hausman import HausmanTest

        if alpha is None:
            alpha = self.alpha

        fe_model = pb.FixedEffects(formula, data, entity_col, time_col)
        re_model = pb.RandomEffects(formula, data, entity_col, time_col)

        fe_results = fe_model.fit(cov_type="nonrobust")
        re_results = re_model.fit(cov_type="nonrobust")

        hausman = HausmanTest(fe_results, re_results, alpha=alpha)

        if hausman.reject_null:
            return "fe"
        return "re"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_tests(validation_report: Any) -> dict:
        """Merge all test categories into a single ``{name: result}`` dict."""
        flat: dict[str, Any] = {}
        for category in (
            validation_report.specification_tests,
            validation_report.serial_tests,
            validation_report.het_tests,
            validation_report.cd_tests,
        ):
            if category:
                flat.update(category)
        return flat

    @staticmethod
    def _recommend_cov_type(all_tests: dict) -> str:
        """Recommend a covariance type based on which tests reject H0."""
        # Cross-sectional dependence → driscoll_kraay
        pesaran = all_tests.get("Pesaran CD")
        if pesaran is not None and pesaran.reject_null:
            return "driscoll_kraay"

        # Heteroskedasticity → robust
        for name in ("Modified Wald", "Breusch-Pagan", "White"):
            t = all_tests.get(name)
            if t is not None and t.reject_null:
                return "robust"

        # Serial correlation → clustered
        for name in ("Wooldridge", "Breusch-Godfrey"):
            t = all_tests.get(name)
            if t is not None and t.reject_null:
                return "clustered"

        return "nonrobust"
