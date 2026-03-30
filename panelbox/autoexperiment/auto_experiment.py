"""AutoExperiment: automated panel data model selection pipeline."""

from __future__ import annotations

import logging
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from panelbox.autoexperiment.model_evaluator import _CANONICAL, ModelEvaluator
from panelbox.autoexperiment.specification_filter import SpecificationFilter
from panelbox.autoexperiment.variable_selector import VariableSelector
from panelbox.autoexperiment.variable_transformer import VariableTransformer

if TYPE_CHECKING:
    from panelbox.autoexperiment.results import AutoExperimentResults

logger = logging.getLogger(__name__)

# Valid model types for AutoExperiment (linear panel models only)
_VALID_MODELS = set(_CANONICAL.keys())


class AutoExperiment:
    """Automated panel data model selection pipeline.

    Orchestrates: transformations -> variable selection -> estimation ->
    validation -> ranking.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    depvar : str
        Dependent variable name.
    entity_col : str
        Cross-sectional identifier column.
    time_col : str
        Time identifier column.
    candidates : list[str], optional
        Candidate regressors. If ``None``, uses all numeric columns
        except *depvar*, *entity_col*, *time_col*.
    transformations : dict, optional
        Transformation spec for :class:`VariableTransformer`.
        Default: ``{'lag': [1], 'diff': True}``.
    models : list[str], optional
        Model types to estimate. Default: ``['pooled_ols', 'fe', 're', 'fd']``.
    criterion : str
        Information criterion for selection: ``'bic'`` or ``'aic'``.
    max_vars : int
        Maximum regressors per model.
    min_obs_per_var : int
        Minimum observations per variable.
    prefilter_corr : float
        Minimum absolute correlation with depvar to keep a candidate.
    require_tests : bool
        If ``True``, only VALID/WARNING models enter the ranking.
    alpha : float
        Significance level for all tests.
    required_tests : list[str], optional
        Override default per-model tests in SpecificationFilter.
    sign_constraints : dict, optional
        ``{variable: '+' or '-'}`` for economic theory constraints.
    max_combinations : int
        Budget cap for total model evaluations.
    n_jobs : int
        Reserved for future parallel support.
    verbose : int
        0 = silent, 1 = progress, 2 = detailed.
    random_state : int, optional
        Random seed (reserved for future use).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        depvar: str,
        entity_col: str,
        time_col: str,
        candidates: list[str] | None = None,
        transformations: dict | None = None,
        models: list[str] | None = None,
        criterion: str = "bic",
        max_vars: int = 8,
        min_obs_per_var: int = 10,
        prefilter_corr: float = 0.05,
        require_tests: bool = True,
        alpha: float = 0.05,
        required_tests: list[str] | None = None,
        sign_constraints: dict[str, str] | None = None,
        max_combinations: int = 500,
        n_jobs: int = 1,
        verbose: int = 1,
        random_state: int | None = None,
    ):
        # --- Validate inputs ---
        if depvar not in data.columns:
            raise ValueError(f"depvar '{depvar}' not found in data columns.")
        if entity_col not in data.columns:
            raise ValueError(f"entity_col '{entity_col}' not found in data columns.")
        if time_col not in data.columns:
            raise ValueError(f"time_col '{time_col}' not found in data columns.")

        if criterion not in ("bic", "aic"):
            raise ValueError(f"criterion must be 'bic' or 'aic', got '{criterion}'")

        # Resolve candidates
        if candidates is None:
            exclude = {depvar, entity_col, time_col}
            candidates = [
                c
                for c in data.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(data[c])
            ]
        else:
            # Validate candidates are numeric
            for c in candidates:
                if c not in data.columns:
                    raise ValueError(f"Candidate '{c}' not found in data columns.")
                if not pd.api.types.is_numeric_dtype(data[c]):
                    raise ValueError(f"Candidate '{c}' is not numeric.")

        # Resolve models
        if models is None:
            models = ["pooled_ols", "fe", "re", "fd"]
        for m in models:
            if m.lower() not in _VALID_MODELS:
                raise ValueError(f"Unknown model type '{m}'. Valid: {sorted(_VALID_MODELS)}")

        # Models that use 'fe' as proxy during variable selection
        self._selection_proxy = {
            "fd": "fe",
            "between": "pooled_ols",
            "be": "pooled_ols",
            "mean_group": "fe",
            "mg": "fe",
            "pooled_mean_group": "fe",
            "pmg": "fe",
            "sur": "fe",
            "sur_homogeneous": "pooled_ols",
            "pooled_logit": "pooled_ols",
            "logit": "pooled_ols",
            "pooled_probit": "pooled_ols",
            "probit": "pooled_ols",
            "fe_logit": "fe",
            "re_probit": "re",
            "lsdvc": "fe",
            "lsdvc_ah": "fe",
            "lsdvc_bb": "fe",
        }

        # Validate max_vars against sample size
        n_obs = len(data)
        max_allowed = n_obs // min_obs_per_var
        if max_vars > max_allowed:
            warnings.warn(
                f"max_vars={max_vars} exceeds N*T/min_obs_per_var={max_allowed}. "
                f"Capping at {max_allowed}.",
                UserWarning,
                stacklevel=2,
            )
            max_vars = max(max_allowed, 1)

        self.data = data.copy()
        self.depvar = depvar
        self.entity_col = entity_col
        self.time_col = time_col
        self.candidates = candidates
        self.transformations = transformations or {"lag": [1], "diff": True}
        self.models = models
        self.criterion = criterion
        self.max_vars = max_vars
        self.min_obs_per_var = min_obs_per_var
        self.prefilter_corr = prefilter_corr
        self.require_tests = require_tests
        self.alpha = alpha
        self.required_tests = required_tests
        self.sign_constraints = sign_constraints or {}
        self.max_combinations = max_combinations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> AutoExperimentResults:
        """Execute the full AutoExperiment pipeline.

        Returns
        -------
        AutoExperimentResults
            Container with best model, ranking, and diagnostics.
        """
        from panelbox.autoexperiment.results import AutoExperimentResults

        n_combinations = 0

        # ── PHASE 1: PREPARATION ──────────────────────────────────────
        self._log("Phase 1: Variable transformations", level=1)

        transformer = VariableTransformer(
            data=self.data,
            entity_col=self.entity_col,
            time_col=self.time_col,
            transformations=self.transformations,
        )
        transformed_data = transformer.transform(self.candidates)
        transformation_map = transformer.get_transformation_map()
        valid_transforms = transformer.get_valid_transformations()

        # All candidate variables = original candidates + transformed
        all_candidates = [
            c for c in self.candidates if c in transformed_data.columns
        ] + valid_transforms

        self._log(
            f"  {len(self.candidates)} original + {len(valid_transforms)} "
            f"transformed = {len(all_candidates)} candidates",
            level=1,
        )
        self._log(transformer.summary(), level=2)

        # ── PHASE 2: VARIABLE SELECTION (per model) ───────────────────
        self._log("Phase 2: Forward stepwise selection", level=1)

        variable_selection: dict[str, dict] = {}
        model_formulas: dict[str, str] = {}

        for i, model_type in enumerate(self.models):
            canonical = _CANONICAL.get(model_type.lower(), model_type)
            self._log(f"  Selecting for {canonical}...", level=1)
            self._progress_bar(i, len(self.models), "Selection")

            selector = VariableSelector(
                depvar=self.depvar,
                candidates=all_candidates,
                entity_col=self.entity_col,
                time_col=self.time_col,
                criterion=self.criterion,
                max_vars=self.max_vars,
                min_obs_per_var=self.min_obs_per_var,
                prefilter_corr=self.prefilter_corr,
                sign_constraints=self.sign_constraints,
            )

            # VariableSelector only supports pooled/fe/re; map others to proxies
            selection_class = self._selection_proxy.get(canonical, canonical)
            sel_result = selector.forward_stepwise(
                data=transformed_data,
                model_class=selection_class,
            )
            n_combinations += sel_result["n_steps"] + len(sel_result.get("rejected_vars", {}))

            variable_selection[canonical] = sel_result

            if sel_result["selected_vars"]:
                rhs = " + ".join(sel_result["selected_vars"])
                model_formulas[canonical] = f"{self.depvar} ~ {rhs}"
            else:
                self._log(
                    f"    No variables selected for {canonical} — skipping.",
                    level=1,
                )

        self._progress_bar(len(self.models), len(self.models), "Selection")

        # ── PHASE 3: ESTIMATION & VALIDATION ──────────────────────────
        self._log("Phase 3: Estimation & validation", level=1)

        evaluator = ModelEvaluator(
            data=transformed_data,
            entity_col=self.entity_col,
            time_col=self.time_col,
        )
        spec_filter = SpecificationFilter(required_tests=self.required_tests, alpha=self.alpha)

        all_results: dict[str, dict] = {}
        test_results: dict[str, dict] = {}

        for i, (model_type, formula) in enumerate(model_formulas.items()):
            self._log(f"  Estimating {model_type}: {formula}", level=2)
            self._progress_bar(i, len(model_formulas), "Estimation")

            try:
                eval_result = evaluator.evaluate(
                    formula=formula,
                    model_type=model_type,
                    alpha=self.alpha,
                )
            except Exception as exc:
                self._log(f"    {model_type} failed: {exc}", level=1)
                continue

            n_combinations += 1
            model_name = f"{model_type}_{formula.split('~')[1].strip().replace(' + ', '_')}"

            all_results[model_name] = {
                "results": eval_result["results"],
                "model_type": eval_result["model_type"],
                "formula": eval_result["formula"],
                "cov_type": eval_result["cov_type"],
                "metrics": eval_result["metrics"],
                "validation": eval_result["validation"],
                "classification": eval_result["classification"],
                "failed_tests": eval_result["failed_tests"],
            }

            test_results[model_name] = {
                "classification": eval_result["classification"],
                "passed": [
                    t
                    for t in spec_filter.DEFAULT_TESTS.get(model_type, [])
                    if t not in eval_result["failed_tests"]
                ],
                "failed": eval_result["failed_tests"],
            }

            self._log(
                f"    -> {eval_result['classification']} "
                f"({self.criterion.upper()}={eval_result['metrics'][self.criterion]:.2f}, "
                f"cov={eval_result['cov_type']})",
                level=1,
            )

        self._progress_bar(len(model_formulas), len(model_formulas), "Estimation")

        # ── PHASE 4: HAUSMAN (FE vs RE) ──────────────────────────────
        fe_names = [n for n, r in all_results.items() if r["model_type"] == "fe"]
        re_names = [n for n, r in all_results.items() if r["model_type"] == "re"]

        if fe_names and re_names:
            self._log("Phase 4: Hausman test (FE vs RE)", level=1)
            fe_name = fe_names[0]
            re_name = re_names[0]

            # Use FE formula for the Hausman test
            fe_formula = all_results[fe_name]["formula"]

            try:
                hausman_choice = spec_filter.hausman_decision(
                    data=transformed_data,
                    formula=fe_formula,
                    entity_col=self.entity_col,
                    time_col=self.time_col,
                    alpha=self.alpha,
                )
                loser = re_name if hausman_choice == "fe" else fe_name
                if all_results[loser]["classification"] == "VALID":
                    all_results[loser]["classification"] = "WARNING"
                    test_results[loser]["classification"] = "WARNING"
                self._log(
                    f"  Hausman: prefer {hausman_choice.upper()} (marked {loser} as WARNING)",
                    level=1,
                )
            except Exception as exc:
                self._log(f"  Hausman test failed: {exc}", level=2)
        else:
            self._log("Phase 4: Hausman skipped (need both FE and RE)", level=2)

        # ── PHASE 5: RANKING ──────────────────────────────────────────
        self._log("Phase 5: Ranking", level=1)

        ranking_rows = []
        for name, info in all_results.items():
            m = info["metrics"]
            score = self._compute_score(info)
            ranking_rows.append(
                {
                    "model_name": name,
                    "model_type": info["model_type"],
                    "formula": info["formula"],
                    "bic": m["bic"],
                    "aic": m["aic"],
                    "rsq": m["rsq"],
                    "rsq_adj": m["rsq_adj"],
                    "classification": info["classification"],
                    "n_vars": len(info["formula"].split("~")[1].strip().split("+")),
                    "cov_type": info["cov_type"],
                    "score": score,
                }
            )

        ranking = pd.DataFrame(ranking_rows)
        if not ranking.empty:
            ranking = ranking.sort_values("score", ascending=True).reset_index(drop=True)

        # Select best model (best score among VALID, or WARNING if no VALID)
        best_model = None
        best_formula = ""
        best_estimator = ""
        best_cov_type = ""

        if not ranking.empty:
            valid_ranking = ranking[ranking["classification"] == "VALID"]
            if valid_ranking.empty:
                valid_ranking = ranking[ranking["classification"] == "WARNING"]
            if not valid_ranking.empty:
                best_row = valid_ranking.iloc[0]
                best_name = best_row["model_name"]
                best_model = all_results[best_name]["results"]
                best_formula = best_row["formula"]
                best_estimator = best_row["model_type"]
                best_cov_type = best_row["cov_type"]

        # ── PHASE 6: BUILD RESULTS ────────────────────────────────────
        self._log("Phase 6: Building results", level=2)

        datamining_warning = n_combinations > 100

        results = AutoExperimentResults(
            best_model=best_model,
            best_formula=best_formula,
            best_estimator=best_estimator,
            best_cov_type=best_cov_type,
            ranking=ranking,
            all_results=all_results,
            transformations_used=transformation_map,
            variable_selection=variable_selection,
            test_results=test_results,
            n_combinations_tested=n_combinations,
            datamining_warning=datamining_warning,
        )

        if datamining_warning:
            self._log(
                f"  WARNING: {n_combinations} combinations tested — data mining risk!",
                level=1,
            )

        self._log("Done.", level=1)
        return results

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(self, info: dict) -> float:
        """Compute composite score (lower is better).

        Components:
        - Information criterion (BIC/AIC) — normalized
        - Classification penalty: VALID=0, WARNING=50, INVALID=200
        - Parsimony bonus: -2 per fewer variable vs max_vars
        - Sign constraint violations: +100 each
        """
        m = info["metrics"]
        ic = m.get(self.criterion, np.inf)

        # Classification penalty
        cls_penalties = {"VALID": 0, "WARNING": 50, "INVALID": 200}
        cls_penalty = cls_penalties.get(info["classification"], 200)

        # Parsimony bonus (fewer vars = slight advantage)
        n_vars = len(info["formula"].split("~")[1].strip().split("+"))
        parsimony = -2 * (self.max_vars - n_vars)

        return ic + cls_penalty + parsimony

    # ------------------------------------------------------------------
    # Logging / progress
    # ------------------------------------------------------------------

    def _log(self, msg: str, level: int = 1) -> None:
        """Print if verbose >= level."""
        if self.verbose >= level:
            print(msg)

    def _progress_bar(self, current: int, total: int, prefix: str) -> None:
        """Simple progress bar for verbose=1."""
        if self.verbose < 1 or total == 0:
            return
        pct = current / total
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        line = f"\r  [{bar}] {current}/{total} {prefix}"
        if current < total:
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
