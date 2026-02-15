# PanelBox Codebase Exploration - Complete

**Date**: February 14, 2026
**Repository**: `/home/guhaase/projetos/panelbox`

## Overview

This document summarizes the comprehensive exploration of the PanelBox econometric library codebase. Three detailed reference documents have been generated to facilitate implementation of censored and ordered models.

## Generated Documentation

### 1. CODEBASE_ARCHITECTURE.md (732 lines)
**Comprehensive technical reference** covering:
- All implemented models (discrete, count, censored)
- Complete class hierarchies and inheritance patterns
- NonlinearPanelModel base class interface
- Marginal effects computation infrastructure
- Gauss-Hermite quadrature utilities (2-50 nodes)
- Diagnostic tools (convergence checks, specification tests)
- Covariance matrix estimation (nonrobust, robust, cluster-robust)
- Data structures (PanelData, FormulaParser)
- Architecture patterns and design principles
- Implementation guidelines for new models
- File organization and locations

**Use this for**: Understanding overall architecture, theoretical foundations, design patterns

### 2. ARCHITECTURE_QUICK_REFERENCE.md
**Implementation quick start guide** containing:
- Core class hierarchy at a glance
- Essential files and their purposes
- Key utility imports and functions
- Implementation template for binary models
- Implementation template for censored models
- Covariance matrix computation patterns
- Marginal effects integration pattern
- Testing checklist
- Common pitfalls and how to avoid them
- Absolute file paths to all key modules

**Use this for**: Quick lookup during coding, templates for new models, error debugging

### 3. KEY_CODE_PATTERNS.md
**Concrete code examples** with 10 detailed patterns:
1. Logit log-likelihood (numerically stable form)
2. Sandwich covariance computation
3. Probit log-likelihood
4. Fixed Effects Logit conditional likelihood
5. Random Effects Probit with Gauss-Hermite quadrature
6. Ordered logit cutpoint handling
7. Poisson likelihood and score/Hessian
8. Classification metrics computation
9. Hosmer-Lemeshow goodness-of-fit test
10. Results object creation and population

**Use this for**: Copy-paste patterns, understanding numerical implementation details, debugging specific computations

## Key Findings

### Already Implemented Models

**Binary Choice**: PooledLogit, PooledProbit, FixedEffectsLogit, RandomEffectsProbit
**Ordered**: OrderedLogit, OrderedProbit, RandomEffectsOrderedLogit
**Count Data**: PooledPoisson, NegativeBinomial, various FE/RE variants
**Censored**: RandomEffectsTobit, HonoreModel

### Core Strengths

1. **Solid MLE Framework**
   - Flexible optimization (BFGS, Newton, Trust-Region)
   - Multiple starting values to avoid local minima
   - Built-in convergence diagnostics
   - Numerical gradient/Hessian fallback

2. **Panel Data Handling**
   - Entity-level clustering for standard errors
   - Fixed effects elimination (Chamberlain conditional MLE)
   - Support for unbalanced panels
   - Formula-based specification (R-style)

3. **Marginal Effects Infrastructure**
   - Delta method for standard errors
   - Infrastructure for AME, MEM, MER
   - Category-specific effects for ordered models

4. **Numerical Integration**
   - Gauss-Hermite quadrature (2-50 nodes)
   - Quadrature caching for efficiency
   - Support for bivariate integration

5. **Diagnostic Tools**
   - Convergence checking (gradient norm, Hessian definiteness, condition number)
   - Specification tests (Hosmer-Lemeshow, Information Matrix, Link Test)
   - Classification metrics (accuracy, precision, recall, F1, AUC-ROC)
   - Model fit measures (LL, AIC, BIC, pseudo-R²)

### Implementation Patterns

All models follow consistent patterns:

```
Model Class
├── __init__()
│   └── Initialize data, formula, quadrature if needed
├── _log_likelihood(params)
│   └── MUST override - core computation
├── _score(params)
│   └── Optional - analytical gradient (numerical if not provided)
├── _hessian(params)
│   └── Optional - analytical Hessian (numerical if not provided)
├── _create_results(params, var_names, y, X)
│   └── MUST override - create PanelResults object
├── fit()
│   └── Inherited from NonlinearPanelModel - handles optimization
└── Additional methods (predict, marginal_effects, diagnostics)
```

## How to Use This Documentation

### For Implementing a New Binary Model
1. Start with `ARCHITECTURE_QUICK_REFERENCE.md` - template for binary model
2. Copy from `KEY_CODE_PATTERNS.md` - patterns 1-2 (logit likelihood and covariance)
3. Review `CODEBASE_ARCHITECTURE.md` section 2.3 - NonlinearPanelModel interface
4. Check `panelbox/models/discrete/binary.py` - PooledLogit for full reference

### For Implementing a Censored Model
1. Check `ARCHITECTURE_QUICK_REFERENCE.md` - template for censored model
2. Review `KEY_CODE_PATTERNS.md` - pattern 5 (quadrature integration pattern)
3. Study `CODEBASE_ARCHITECTURE.md` section 4 - Quadrature utilities in detail
4. Reference `/home/guhaase/projetos/panelbox/panelbox/models/censored/tobit.py`

### For Implementing an Ordered Model
1. Use `ARCHITECTURE_QUICK_REFERENCE.md` - focus on cutpoint handling
2. Review `KEY_CODE_PATTERNS.md` - pattern 6 (cutpoint transformation)
3. Read `CODEBASE_ARCHITECTURE.md` sections 2.2 and 3 - cutpoint math
4. Check `/home/guhaase/projetos/panelbox/panelbox/models/discrete/ordered.py`

### For Adding Marginal Effects
1. Review `CODEBASE_ARCHITECTURE.md` section 3 - ME infrastructure
2. Check `panelbox/marginal_effects/delta_method.py` - delta method implementation
3. Model existing ME computations in `marginal_effects/*.py`
4. Implement following patterns in `KEY_CODE_PATTERNS.md`

## Critical Implementation Details

### Numerical Stability
- Use `np.log1p(np.exp(x))` for logit likelihood (not `np.log(1+np.exp(x))`)
- Clip probabilities to `[1e-10, 1-1e-10]` before taking log
- Use `scipy.special.gammaln()` for log-factorials in count models
- Handle singular Hessians with `np.linalg.pinv()` fallback

### Quadrature Pattern (for RE models)
```python
# Init once
nodes, weights = gauss_hermite_quadrature(n_points)

# In likelihood loop
alpha = np.sqrt(2) * sigma * node  # √2 factor is critical
prob = weight * compute_contribution(alpha)
entity_sum += prob

# Final
llf += log(entity_sum)
```

### Covariance Estimation
- Default: `-H^{-1}` (classical)
- Robust: `H^{-1} * S * H^{-1}` where `S = Σ s_i s_i'`
- Cluster: Same as robust but with cluster-level aggregation

### Always Return
- Log-likelihood: scalar float (not array)
- Standard errors: positive values
- Covariance: positive definite symmetric matrix

## File Locations (Absolute Paths)

```
/home/guhaase/projetos/panelbox/

├── panelbox/models/
│   ├── discrete/
│   │   ├── base.py              (NonlinearPanelModel)
│   │   ├── binary.py            (Logit, Probit, FE Logit, RE Probit)
│   │   └── ordered.py           (Ordered choice models)
│   ├── count/
│   │   ├── poisson.py           (Poisson variants)
│   │   └── negbin.py            (Negative Binomial)
│   └── censored/
│       ├── tobit.py             (Tobit models)
│       └── honore.py            (Semi-parametric)
├── panelbox/marginal_effects/
│   ├── discrete_me.py           (Binary model ME)
│   ├── ordered_me.py            (Ordered model ME)
│   ├── count_me.py              (Count model ME)
│   └── delta_method.py          (SE computation)
├── panelbox/optimization/
│   ├── quadrature.py            (Gauss-Hermite)
│   └── numerical_grad.py        (Finite differences)
├── panelbox/standard_errors/
│   └── mle.py                   (cluster_robust_mle)
└── panelbox/utils/
    ├── data.py                  (check_panel_data)
    └── statistics.py            (covariance functions)
```

## Testing Checklist for New Models

- [ ] Log-likelihood returns scalar float
- [ ] Log-likelihood matches numerical gradient
- [ ] Score/Hessian have correct signs
- [ ] Convergence diagnostics work (gradient norm, Hessian eigenvalues)
- [ ] Covariance matrix is positive definite
- [ ] Predictions match fitted values in-sample
- [ ] Results object has all required attributes
- [ ] Panel structure respected (entity/time indices)
- [ ] Handles edge cases (singular data, no variation, etc.)
- [ ] Marginal effects (if implemented) sum to correct value
- [ ] Documentation updated with examples

## Next Steps for Development

To implement new models using this documentation:

1. Choose model type (binary/count/censored/ordered)
2. Select appropriate template from `ARCHITECTURE_QUICK_REFERENCE.md`
3. Copy code patterns from `KEY_CODE_PATTERNS.md`
4. Study similar model in actual codebase
5. Implement `_log_likelihood()` with numerical stability checks
6. Implement or inherit `_score()` and `_hessian()`
7. Implement `_create_results()` following PanelResults pattern
8. Add marginal effects if applicable
9. Write tests for convergence and accuracy
10. Add to `__init__.py` exports

## Summary

The PanelBox codebase is well-architected with:
- Clear inheritance hierarchy
- Consistent patterns across all models
- Robust numerical infrastructure
- Comprehensive utilities for panel data
- Extensible design for new models

The three generated documentation files provide everything needed to understand and extend the codebase effectively.

---

**Documentation Generated**: February 14, 2026
**Total Lines of Documentation**: >1500
**Coverage**: Complete codebase exploration including all 50 key files
