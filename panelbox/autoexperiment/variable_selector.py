"""Variable selector: forward stepwise selection with BIC/AIC for panel data."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Prefixes used by VariableTransformer
_TRANSFORM_PREFIXES = [
    (r"^L(\d+)_(.+)$", "lag"),
    (r"^D_(.+)$", "diff"),
    (r"^log_(.+)$", "log"),
    (r"^growth_(.+)$", "growth"),
    (r"^sq_(.+)$", "sq"),
    (r"^acum(\d+)_(.+)$", "acum"),
]

# Transformations that may invert the sign relative to the original variable
_SIGN_INVERTING = {"diff", "growth"}


def _get_source_variable(var_name: str) -> str | None:
    """Extract the original (untransformed) variable name, or None if not transformed."""
    for pattern, _ in _TRANSFORM_PREFIXES:
        m = re.match(pattern, var_name)
        if m:
            # The source variable is always the last capture group
            last = m.lastindex
            if last is not None:
                return m.group(last)
    return None


def _get_transform_type(var_name: str) -> str | None:
    """Return the transformation type for a transformed variable name."""
    for pattern, ttype in _TRANSFORM_PREFIXES:
        if re.match(pattern, var_name):
            return ttype
    return None


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
    else:  # aic
        return log_lik_term + 2 * k


def _resolve_model_class(model_class: str | type) -> type:
    """Resolve a model alias string to the actual class."""
    if isinstance(model_class, type):
        return model_class

    import panelbox as pb

    alias_map = {
        "pooled": pb.PooledOLS,
        "pooled_ols": pb.PooledOLS,
        "fe": pb.FixedEffects,
        "fixed_effects": pb.FixedEffects,
        "re": pb.RandomEffects,
        "random_effects": pb.RandomEffects,
    }
    key = model_class.lower()
    if key not in alias_map:
        raise ValueError(f"Unknown model_class '{model_class}'. Valid: {list(alias_map.keys())}")
    return alias_map[key]


def _fit_model(
    model_cls: type,
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    **fit_kwargs: Any,
) -> Any:
    """Instantiate and fit a panel model, returning PanelResults."""
    model = model_cls(
        formula=formula,
        data=data,
        entity_col=entity_col,
        time_col=time_col,
    )
    return model.fit(**fit_kwargs)


class VariableSelector:
    """
    Forward stepwise variable selection for panel data models.

    Parameters
    ----------
    depvar : str
        Dependent variable name.
    candidates : list[str]
        Candidate regressor names (may include transformed variables).
    entity_col : str
        Entity (cross-section) identifier column.
    time_col : str
        Time identifier column.
    criterion : str
        Information criterion: ``'bic'`` or ``'aic'``. Default ``'bic'``.
    max_vars : int
        Maximum number of regressors to select. Default 8.
    min_obs_per_var : int
        Minimum observations per variable (used to cap max_vars
        based on available sample size). Default 10.
    prefilter_corr : float
        Minimum absolute correlation with *depvar* required to keep a
        candidate in the pool. Default 0.05.
    sign_constraints : dict or None
        Mapping ``{variable_name: '+' or '-'}`` encoding expected sign
        from economic theory. Transformed variables inherit the
        constraint of their source variable.
    """

    def __init__(
        self,
        depvar: str,
        candidates: list[str],
        entity_col: str,
        time_col: str,
        criterion: str = "bic",
        max_vars: int = 8,
        min_obs_per_var: int = 10,
        prefilter_corr: float = 0.05,
        sign_constraints: dict[str, str] | None = None,
    ):
        if criterion not in ("bic", "aic"):
            raise ValueError(f"criterion must be 'bic' or 'aic', got '{criterion}'")

        self.depvar = depvar
        self.candidates = list(candidates)
        self.entity_col = entity_col
        self.time_col = time_col
        self.criterion = criterion
        self.max_vars = max_vars
        self.min_obs_per_var = min_obs_per_var
        self.prefilter_corr = prefilter_corr
        self.sign_constraints = sign_constraints or {}

    # ------------------------------------------------------------------
    # Pre-filter
    # ------------------------------------------------------------------

    def prefilter(self, data: pd.DataFrame) -> list[str]:
        """Eliminate candidates whose absolute correlation with *depvar* is below ``prefilter_corr``.

        Returns
        -------
        list[str]
            Candidates that pass the correlation filter.
        """
        surviving: list[str] = []
        y = data[self.depvar]

        for var in self.candidates:
            if var not in data.columns:
                logger.warning("Candidate '%s' not found in data — skipped.", var)
                continue
            # Use pairwise-complete observations
            mask = y.notna() & data[var].notna()
            if mask.sum() < 3:
                logger.info("Prefilter: '%s' dropped (fewer than 3 valid observations).", var)
                continue
            corr = y[mask].corr(data[var][mask])
            if np.isnan(corr) or abs(corr) < self.prefilter_corr:
                logger.info(
                    "Prefilter: '%s' dropped (|corr| = %.4f < %.4f).",
                    var,
                    abs(corr) if not np.isnan(corr) else 0.0,
                    self.prefilter_corr,
                )
            else:
                surviving.append(var)

        n_dropped = len(self.candidates) - len(surviving)
        if n_dropped:
            logger.info(
                "Prefilter: %d / %d candidates eliminated (|corr| < %.4f).",
                n_dropped,
                len(self.candidates),
                self.prefilter_corr,
            )
        return surviving

    # ------------------------------------------------------------------
    # Sign constraints
    # ------------------------------------------------------------------

    def _check_sign_constraint(
        self, var_name: str, coef_value: float, constraints: dict[str, str]
    ) -> bool:
        """
        Check whether *coef_value* respects the sign constraint for *var_name*.

        For transformed variables the constraint is inherited from the source
        variable.  ``diff`` and ``growth`` transformations can invert the
        economic sign, so they are **not** checked (always pass).

        Returns ``True`` if the constraint is satisfied or no constraint
        applies.
        """
        if not constraints:
            return True

        # Direct constraint lookup
        constraint = constraints.get(var_name)
        source_var = None

        if constraint is None:
            # Try inheriting from source variable
            source_var = _get_source_variable(var_name)
            if source_var is not None:
                constraint = constraints.get(source_var)

        if constraint is None:
            return True  # no constraint for this variable

        # diff / growth can flip the sign — skip enforcement
        ttype = _get_transform_type(var_name)
        if ttype in _SIGN_INVERTING:
            return True

        if constraint == "+":
            return coef_value > 0
        elif constraint == "-":
            return coef_value < 0
        return True

    # ------------------------------------------------------------------
    # Forward stepwise
    # ------------------------------------------------------------------

    def forward_stepwise(
        self,
        data: pd.DataFrame,
        model_class: str | type = "fe",
        **fit_kwargs: Any,
    ) -> dict:
        """
        Forward stepwise variable selection by information criterion.

        Algorithm
        ---------
        1. Start with intercept-only model.
        2. At each step, try adding each remaining candidate.
        3. Keep the candidate that yields the lowest IC **and** satisfies
           the sign constraint.
        4. Stop when no addition improves IC, ``max_vars`` is reached, or
           no candidates remain.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data in long format.
        model_class : str or type
            Model alias (``'fe'``, ``'pooled'``, …) or class.
        **fit_kwargs
            Passed to ``model.fit()``.

        Returns
        -------
        dict
            ``selected_vars``, ``bic_path``, ``rejected_vars``, ``n_steps``.
        """
        model_cls = _resolve_model_class(model_class)

        # Pre-filter candidates
        pool = self.prefilter(data)

        # Cap max_vars by sample size
        clean = data.dropna(subset=[self.depvar, *pool])
        effective_max = min(self.max_vars, len(clean) // self.min_obs_per_var)
        if effective_max < 1:
            effective_max = 1

        selected: list[str] = []
        ic_path: list[float] = []
        rejected: dict[str, str] = {}

        # Baseline IC (intercept-only uses the mean as fitted value)
        try:
            base_results = _fit_model(
                model_cls,
                f"{self.depvar} ~ 1",
                data,
                self.entity_col,
                self.time_col,
                **fit_kwargs,
            )
            best_ic = _compute_ic(base_results, self.criterion)
        except Exception:
            # If intercept-only fails (e.g. FE with entity effects),
            # set a very large baseline so the first variable always enters.
            best_ic = np.inf

        ic_path.append(best_ic)
        remaining = list(pool)

        for step in range(effective_max):
            best_candidate = None
            best_candidate_ic = best_ic

            for var in remaining:
                rhs = " + ".join([*selected, var])
                formula = f"{self.depvar} ~ {rhs}"

                try:
                    res = _fit_model(
                        model_cls,
                        formula,
                        data,
                        self.entity_col,
                        self.time_col,
                        **fit_kwargs,
                    )
                except Exception as exc:
                    rejected[var] = f"fit error: {exc}"
                    continue

                # Sign constraint check
                if var in res.params.index or var in res.params.values:
                    coef = res.params[var]
                else:
                    coef = None

                if coef is not None and not self._check_sign_constraint(
                    var, coef, self.sign_constraints
                ):
                    constraint = self.sign_constraints.get(
                        var,
                        self.sign_constraints.get(_get_source_variable(var) or "", ""),
                    )
                    rejected[var] = (
                        f"sign constraint violated: expected '{constraint}', got coef={coef:.6f}"
                    )
                    continue

                ic = _compute_ic(res, self.criterion)
                if ic < best_candidate_ic:
                    best_candidate_ic = ic
                    best_candidate = var

            if best_candidate is None or best_candidate_ic >= best_ic:
                # No improvement — stop
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_ic = best_candidate_ic
            ic_path.append(best_ic)
            logger.info(
                "Step %d: added '%s' (%s=%.4f)",
                step + 1,
                best_candidate,
                self.criterion.upper(),
                best_ic,
            )

        # Record remaining pool members that were never selected
        for var in remaining:
            if var not in rejected:
                rejected[var] = "not selected (did not improve criterion)"

        return {
            "selected_vars": selected,
            "bic_path": ic_path,
            "rejected_vars": rejected,
            "n_steps": len(selected),
        }

    # ------------------------------------------------------------------
    # Backward elimination
    # ------------------------------------------------------------------

    def backward_elimination(
        self,
        data: pd.DataFrame,
        model_class: str | type,
        selected_vars: list[str],
        alpha: float = 0.05,
        **fit_kwargs: Any,
    ) -> dict:
        """Remove insignificant variables (p > *alpha*) if removal doesn't worsen the IC.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        model_class : str or type
            Model alias or class.
        selected_vars : list[str]
            Variables selected by ``forward_stepwise``.
        alpha : float
            Significance level. Default 0.05.
        **fit_kwargs
            Passed to ``model.fit()``.

        Returns
        -------
        dict
            ``selected_vars``, ``removed_vars``, ``bic_path``.
        """
        model_cls = _resolve_model_class(model_class)
        current = list(selected_vars)
        removed: dict[str, str] = {}
        ic_path: list[float] = []

        # Full-model IC
        formula = f"{self.depvar} ~ {' + '.join(current)}"
        full_res = _fit_model(
            model_cls, formula, data, self.entity_col, self.time_col, **fit_kwargs
        )
        best_ic = _compute_ic(full_res, self.criterion)
        ic_path.append(best_ic)

        changed = True
        while changed and len(current) > 1:
            changed = False

            # Find the least significant variable
            formula = f"{self.depvar} ~ {' + '.join(current)}"
            res = _fit_model(model_cls, formula, data, self.entity_col, self.time_col, **fit_kwargs)

            # Among insignificant vars, try removing the one with highest p-value
            pvals = res.pvalues
            insignificant = [v for v in current if v in pvals.index and pvals[v] > alpha]
            if not insignificant:
                break

            # Sort by p-value descending — try removing the worst first
            insignificant.sort(key=lambda v: pvals[v], reverse=True)

            for var in insignificant:
                reduced = [v for v in current if v != var]
                formula_r = f"{self.depvar} ~ {' + '.join(reduced)}"
                try:
                    res_r = _fit_model(
                        model_cls,
                        formula_r,
                        data,
                        self.entity_col,
                        self.time_col,
                        **fit_kwargs,
                    )
                except Exception:  # noqa: S112
                    continue

                ic_r = _compute_ic(res_r, self.criterion)
                if ic_r <= best_ic:
                    removed[var] = f"p={pvals[var]:.4f} > {alpha}"
                    current = reduced
                    best_ic = ic_r
                    ic_path.append(best_ic)
                    changed = True
                    logger.info("Backward: removed '%s' (p=%.4f)", var, pvals[var])
                    break

        return {
            "selected_vars": current,
            "removed_vars": removed,
            "bic_path": ic_path,
        }
