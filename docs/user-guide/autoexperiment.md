---
title: "AutoExperiment"
description: "Automated panel data model selection — when to use, limitations, and how to interpret results"
---

# AutoExperiment User Guide

AutoExperiment is PanelBox's automated model selection pipeline for panel data. It combines variable transformation, forward stepwise selection, multi-model estimation, and econometric validation into a single `run()` call.

!!! warning "Data Mining Risk"
    AutoExperiment tests many variable/model combinations. This is a form of **data mining** — results may overfit your particular sample. Always validate findings on holdout data or with economic theory. AutoExperiment flags this risk automatically when > 100 combinations are tested.

## When to Use AutoExperiment

AutoExperiment is most useful when:

- You have **many candidate regressors** and want a systematic way to select variables
- You want to **compare multiple estimators** (Pooled OLS, FE, RE, FD) on the same data
- You need **automatic diagnostic validation** (Hausman, Pesaran CD, RESET, etc.)
- You want **reproducible model selection** with a clear audit trail

AutoExperiment is **not** a replacement for economic reasoning. It is a tool to accelerate the modeling workflow while enforcing econometric discipline.

## When NOT to Use AutoExperiment

- **You already know your model**: If economic theory dictates your specification, use the model directly
- **Non-linear models**: AutoExperiment only supports linear panel models (Pooled OLS, FE, RE, FD)
- **GMM / Spatial / Quantile**: Use specialized estimators for these model families
- **Small datasets**: With very few observations, automated selection may overfit

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.autoexperiment import AutoExperiment

data = load_grunfeld()

auto = AutoExperiment(
    data=data,
    depvar="invest",
    entity_col="firm",
    time_col="year",
    candidates=["value", "capital"],
    transformations={"lag": [1], "diff": True, "log": True},
    sign_constraints={"value": "+", "capital": "+"},
)
results = auto.run()
print(results.summary())
```

---

## Configuring Sign Constraints

Sign constraints encode your **economic theory** about the expected direction of each coefficient. This is one of AutoExperiment's most important features — it prevents the algorithm from selecting specifications that contradict theory.

```python
sign_constraints = {
    "value": "+",      # Market value should increase investment
    "capital": "+",    # Capital stock should increase investment
    "cost": "-",       # Higher costs should decrease investment
}
```

### How Sign Constraints Work

1. During variable selection, candidates that violate their sign constraint are rejected
2. Sign constraints on source variables are **inherited by transformations**: constraining `value` to `+` also constrains `L1_value`, `log_value`, `sq_value`
3. Differenced/growth transforms may invert the expected sign — the selector accounts for this
4. In the composite ranking, the fraction of satisfied sign constraints contributes 15% of the score (by default)

!!! tip "When to Use Sign Constraints"
    Always use sign constraints when economic theory provides clear predictions. This dramatically reduces the data mining problem by anchoring results in theory.

---

## Interpreting the Report

### Classification System

AutoExperiment classifies each specification into three categories:

| Classification | Meaning | Action |
|----------------|---------|--------|
| **VALID** | All diagnostic tests pass | Safe to use |
| **WARNING** | Some non-critical tests fail | Usable with caution — check which tests failed |
| **INVALID** | A critical test fails (RESET or Mundlak) | Do not use — specification is misspecified |

### Critical Tests

- **RESET**: Tests for functional form misspecification (omitted nonlinear terms). If this fails, your model is missing important structure.
- **Mundlak**: Tests whether random effects are correlated with regressors. If this fails for an RE model, FE is preferred.

### Diagnostic Tests per Model Type

| Model | Tests Run |
|-------|-----------|
| Pooled OLS | Breusch-Pagan (heteroskedasticity), RESET |
| Fixed Effects | Pesaran CD (cross-sectional dependence), Wooldridge (serial correlation), Modified Wald (heteroskedasticity), RESET |
| Random Effects | Mundlak (correlated effects), Breusch-Pagan, RESET |
| First Difference | Breusch-Pagan, RESET |

### Automatic Standard Error Selection

AutoExperiment automatically selects the most appropriate standard error type based on diagnostic results:

| Priority | Diagnostic Finding | Selected SE Type |
|----------|-------------------|------------------|
| 1st | Cross-sectional dependence (Pesaran CD) | Driscoll-Kraay |
| 2nd | Serial correlation (Wooldridge) | Newey-West (HAC) |
| 3rd | Heteroskedasticity (Modified Wald / BP) | Robust (HC1) |
| 4th | No violations detected | Clustered (conservative default) |

### Composite Score

The ranking combines four components (higher = better):

| Component | Default Weight | What It Measures |
|-----------|---------------|------------------|
| Information criterion | 50% | Statistical fit (BIC or AIC) |
| Diagnostic tests | 30% | Econometric validity |
| Sign constraints | 15% | Consistency with theory |
| Parsimony | 5% | Preference for simpler models |

---

## Adjusting Ranking Weights

You can customize the composite score weights by passing a custom `ModelRanker`:

```python
from panelbox.autoexperiment import ModelRanker

# Emphasize diagnostic validity over fit
ranker = ModelRanker(weights={
    "criterion": 0.30,   # Less weight on BIC
    "tests": 0.50,       # More weight on passing tests
    "signs": 0.15,       # Keep sign constraint weight
    "parsimony": 0.05,   # Keep parsimony weight
})

ranking = ranker.rank(model_evaluations)
```

!!! info "Weight Guidelines"
    - **Exploratory research**: Increase `criterion` weight to find the best-fitting model
    - **Confirmatory research**: Increase `tests` and `signs` weights to prioritize theory consistency
    - **Large candidate sets**: Increase `parsimony` weight to penalize overly complex models

---

## Comparing with Manual Analysis

AutoExperiment is a complement to manual analysis, not a substitute. A recommended workflow:

1. **Start with theory**: Define your candidate variables and sign constraints based on economic reasoning
2. **Run AutoExperiment**: Let the algorithm explore the specification space
3. **Review the ranking**: Check which models are VALID, what variables were selected, what tests failed
4. **Validate manually**: Re-estimate the best model, inspect residuals, run additional tests
5. **Check robustness**: Compare AutoExperiment's choice with your theory-driven specification

```python
# AutoExperiment's choice
print(results.best_formula)
print(results.best_estimator)
print(results.best_cov_type)

# Compare with manual specification
from panelbox import FixedEffects
manual = FixedEffects("invest ~ value + capital", data, "firm", "year")
manual_results = manual.fit(cov_type="clustered")

# Side-by-side
print("Auto BIC:", results.ranking.iloc[0]["bic"])
print("Manual BIC:", manual_results.bic)
```

---

## Variable Transformations

AutoExperiment can automatically generate transformed variables before selection:

```python
transformations = {
    "lag": [1, 2, 3],    # L1_var, L2_var, L3_var
    "diff": True,         # D_var (first difference)
    "log": True,          # log_var (natural log, skips x <= 0)
    "acum": [3, 6],       # acum3_var, acum6_var (rolling mean)
    "growth": True,        # growth_var (percentage growth rate)
    "sq": True,           # sq_var (squared values)
}
```

### Data Quality Controls

- **NaN threshold**: Transformed variables with > 30% missing values are discarded (configurable via `nan_threshold`)
- **Multicollinearity**: Highly correlated transformed variables (> 0.95 correlation) are automatically removed
- **Pre-filtering**: Variables with very low correlation with the dependent variable (< `prefilter_corr`) are excluded early

---

## Limitations and Caveats

!!! danger "Data Mining Warning"
    Testing many variable combinations on the same dataset inflates the risk of finding spurious relationships. AutoExperiment raises a warning when > 100 combinations are tested. **Always validate results on out-of-sample data.**

### Known Limitations

1. **Linear models only**: AutoExperiment currently supports Pooled OLS, Fixed Effects, Random Effects, and First Difference. For GMM, Spatial, Quantile, or other model families, use the dedicated estimators.

2. **No cross-validation**: Variable selection uses in-sample BIC/AIC. There is no built-in out-of-sample validation — you should do this manually.

3. **Stepwise selection bias**: Forward stepwise selection can miss the globally optimal subset. The selected variables may not be the true best combination.

4. **Multiple testing**: Running many diagnostic tests inflates the chance of false positives (detecting a problem that doesn't exist). Consider this when interpreting WARNING classifications.

5. **Sample size requirements**: The `min_obs_per_var` parameter (default: 10) caps the maximum number of variables. With small panels, this may limit the search space significantly.

---

## FAQ

**Q: Can I use AutoExperiment with unbalanced panels?**
A: Yes. AutoExperiment works with unbalanced panels. The `nan_threshold` parameter handles the additional missing values created by transformations on unbalanced data.

**Q: How does AutoExperiment handle the Hausman test?**
A: If both FE and RE models are estimated, AutoExperiment automatically runs the Hausman test. The losing estimator is downgraded to WARNING classification.

**Q: Can I add custom diagnostic tests?**
A: Yes, use the `required_tests` parameter to override the default test list. The test names must match the canonical names used by PanelBox's ValidationSuite.

**Q: What if no model passes all tests?**
A: AutoExperiment selects the best WARNING model if no VALID model exists. If all models are INVALID, the `best_model` attribute will be `None`.

**Q: How do I reduce data mining risk?**
A: Use sign constraints (anchor results in theory), limit `max_vars`, reduce the number of transformations, and validate on holdout data.

---

## See Also

- [AutoExperiment API Reference](../api/autoexperiment.md) — full parameter documentation
- [AutoExperiment Quickstart](../tutorials/autoexperiment-quickstart.md) — from zero to results in 10 lines
- [Experiment Pattern](../visualization/experiment/index.md) — manual model comparison
- [Validation & Diagnostics](../diagnostics/index.md) — diagnostic tests used by AutoExperiment
