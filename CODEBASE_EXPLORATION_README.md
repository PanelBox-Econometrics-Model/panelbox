# PanelBox Codebase Exploration - Master Index

**Completed**: February 14, 2026
**Status**: Comprehensive exploration with 4 detailed reference documents

## Quick Navigation

This document serves as the master index for all codebase exploration materials. Start here to find the right document for your needs.

## The Four Documentation Files

### 1. **CODEBASE_ARCHITECTURE.md** (732 lines)
**What**: Complete technical reference of the entire codebase architecture
**Best for**: Understanding overall design, learning theory, understanding patterns
**Key sections**:
- Models already implemented (binary, ordered, count, censored)
- Base classes (PanelModel, NonlinearPanelModel, PanelResults)
- Marginal effects computation (delta method, formulas)
- Quadrature integration (Gauss-Hermite, 2-50 nodes)
- Diagnostic tools (convergence, specification tests, metrics)
- Covariance estimation (nonrobust, robust, cluster)
- Data structures and utilities

**Start here if**: You want to understand the "big picture" architecture

---

### 2. **ARCHITECTURE_QUICK_REFERENCE.md** (267 lines)
**What**: Quick lookup guide and implementation templates
**Best for**: During coding, quick lookups, copy-paste templates
**Key sections**:
- Class hierarchy at a glance
- Essential file locations (absolute paths)
- Key utility imports
- Implementation template for binary models
- Implementation template for censored models
- Covariance computation patterns
- Testing checklist
- Common pitfalls and fixes

**Start here if**: You're implementing a new model and need a quick template

---

### 3. **KEY_CODE_PATTERNS.md** (537 lines)
**What**: 10 concrete, working code examples from actual implementation
**Best for**: Copy-paste patterns, numerical details, debugging
**Includes**:
1. Logit log-likelihood (numerically stable)
2. Sandwich covariance (with clustering)
3. Probit log-likelihood
4. Fixed Effects Logit conditional likelihood
5. Random Effects Probit with quadrature
6. Ordered logit cutpoint handling
7. Poisson score and Hessian
8. Classification metrics
9. Hosmer-Lemeshow test
10. Results object creation

**Start here if**: You need working code to understand/copy from

---

### 4. **EXPLORATION_COMPLETE.md** (251 lines)
**What**: Summary of exploration with usage guide
**Best for**: Understanding what was explored, how to use the docs
**Key sections**:
- Overview of what was explored
- Key findings summary
- How to use documentation by task
- Critical implementation details
- Testing checklist
- Next steps for development

**Start here if**: You're new to this exploration and want orientation

---

## How to Use These Documents

### Scenario 1: "I need to implement a Binary Logit variant"
1. Read: **ARCHITECTURE_QUICK_REFERENCE.md** - "Implementation Template" section
2. Copy patterns from: **KEY_CODE_PATTERNS.md** - patterns 1-2 (likelihood and covariance)
3. Reference: **CODEBASE_ARCHITECTURE.md** - section 2.3 (NonlinearPanelModel)
4. Check actual code: `panelbox/models/discrete/binary.py`

**Time**: 30-45 minutes to understand and implement

### Scenario 2: "I need to implement a Censored/Tobit Model"
1. Template: **ARCHITECTURE_QUICK_REFERENCE.md** - "For a Censored Model"
2. Understand quadrature: **CODEBASE_ARCHITECTURE.md** - section 4
3. Code pattern: **KEY_CODE_PATTERNS.md** - pattern 5 (quadrature)
4. Deep dive: `/home/guhaase/projetos/panelbox/panelbox/models/censored/tobit.py`

**Time**: 60-90 minutes to understand quadrature and implement

### Scenario 3: "I need to implement Marginal Effects"
1. Infrastructure: **CODEBASE_ARCHITECTURE.md** - section 3
2. Code examples: **KEY_CODE_PATTERNS.md** - patterns throughout
3. Reference: `panelbox/marginal_effects/discrete_me.py`
4. Delta method: `panelbox/marginal_effects/delta_method.py`

**Time**: 45-75 minutes depending on complexity

### Scenario 4: "I'm debugging a numerical issue"
1. Quick tips: **ARCHITECTURE_QUICK_REFERENCE.md** - "Common Pitfalls"
2. Code patterns: **KEY_CODE_PATTERNS.md** - relevant pattern
3. Deep reference: **CODEBASE_ARCHITECTURE.md** - relevant section
4. Actual code: Check the pattern implementation in `/home/guhaase/projetos/panelbox/panelbox/`

**Time**: 15-30 minutes to find and fix issue

---

## Key Facts From Exploration

### Models Implemented
- **Binary**: PooledLogit, PooledProbit, FixedEffectsLogit, RandomEffectsProbit
- **Ordered**: OrderedLogit, OrderedProbit, RandomEffectsOrderedLogit
- **Count**: PooledPoisson, NegativeBinomial, variants with FE/RE
- **Censored**: RandomEffectsTobit, HonoreModel

### Core Technologies
- **Optimization**: BFGS, Newton, Trust-Region (scipy.optimize)
- **Integration**: Gauss-Hermite quadrature (2-50 nodes)
- **SEs**: Nonrobust, Robust (sandwich), Cluster-robust
- **Inference**: Specification tests, Classification metrics, Model fit measures

### Architecture Pattern
All models inherit from `NonlinearPanelModel` and override:
1. `_log_likelihood(params)` - REQUIRED
2. `_score(params)` - Optional (numerical gradient if not provided)
3. `_hessian(params)` - Optional (numerical Hessian if not provided)
4. `_create_results()` - REQUIRED
5. `marginal_effects()` - If applicable
6. `predict()` - If applicable

---

## File Organization in Repository

```
/home/guhaase/projetos/panelbox/

├── EXPLORATION DOCUMENTS (NEW):
│   ├── CODEBASE_ARCHITECTURE.md          (← Read first for overview)
│   ├── ARCHITECTURE_QUICK_REFERENCE.md   (← Use during coding)
│   ├── KEY_CODE_PATTERNS.md              (← Copy patterns from here)
│   ├── EXPLORATION_COMPLETE.md           (← Summary and guide)
│   └── CODEBASE_EXPLORATION_README.md    (← You are here)
│
├── panelbox/
│   ├── models/
│   │   ├── discrete/base.py              (NonlinearPanelModel)
│   │   ├── discrete/binary.py            (Logit, Probit, FE, RE)
│   │   ├── discrete/ordered.py           (Ordered models)
│   │   ├── count/poisson.py              (Poisson variants)
│   │   └── censored/tobit.py             (Censored models)
│   ├── marginal_effects/
│   │   ├── discrete_me.py                (Binary model ME)
│   │   ├── ordered_me.py                 (Ordered model ME)
│   │   ├── delta_method.py               (SE computation)
│   │   └── count_me.py                   (Count model ME)
│   ├── optimization/
│   │   ├── quadrature.py                 (Gauss-Hermite)
│   │   └── numerical_grad.py             (Finite differences)
│   ├── standard_errors/
│   │   └── mle.py                        (cluster_robust_mle)
│   └── utils/
│       ├── data.py                       (check_panel_data)
│       └── statistics.py                 (covariance functions)
```

---

## Critical Implementation Details

### Three Most Important Code Patterns

**Pattern 1: Numerically Stable Logit Likelihood**
```python
eta = X @ params
llf = np.sum(y * eta - np.log1p(np.exp(eta)))
return float(llf)
```
(Don't use `np.log(1 + np.exp(eta))` - overflow risk!)

**Pattern 2: Quadrature Integration (for RE models)**
```python
alpha = np.sqrt(2) * sigma * node  # √2 is critical
prob = weight * compute_contribution(alpha)
entity_sum += prob
llf += np.log(entity_sum)
```

**Pattern 3: Sandwich Covariance**
```python
H_inv = np.linalg.inv(H)
scores = (y - fitted)[:, np.newaxis] * X
S = scores.T @ scores
vcov = H_inv @ S @ H_inv
```

---

## Testing Your Implementation

Always verify:
- Log-likelihood returns scalar `float`, not array
- Log-likelihood matches numerical gradient
- Gradient norm < 1e-3 at convergence
- Hessian negative definite (eigenvalues < -1e-10)
- Covariance matrix positive definite
- Predictions match fitted values in-sample
- Panel structure respected (entities/time)
- Edge cases handled (no variation, singular data, etc.)

---

## Getting Help From the Docs

| Question | Document | Section |
|----------|----------|---------|
| What models exist? | CODEBASE_ARCHITECTURE | 1. Models Already Implemented |
| How do base classes work? | CODEBASE_ARCHITECTURE | 2. Base Classes |
| How to implement a new model? | ARCHITECTURE_QUICK_REFERENCE | Implementation Template |
| Show me working code | KEY_CODE_PATTERNS | Any pattern (1-10) |
| What are common mistakes? | ARCHITECTURE_QUICK_REFERENCE | Common Pitfalls |
| What about marginal effects? | CODEBASE_ARCHITECTURE | 3. Marginal Effects |
| How does quadrature work? | CODEBASE_ARCHITECTURE | 4. Quadrature |
| File locations? | ARCHITECTURE_QUICK_REFERENCE | File Locations |
| Testing checklist? | EXPLORATION_COMPLETE | Testing Checklist |
| How do I start? | EXPLORATION_COMPLETE | How to Use Documentation |

---

## Statistics

- **Total Documentation**: >1500 lines
- **Code Patterns Provided**: 10 detailed examples
- **Files Analyzed**: 50+ Python files
- **Models Documented**: 15+ concrete implementations
- **Utilities Documented**: 20+ key functions
- **Implementation Templates**: 3 (binary, censored, ordered)

---

## Summary

You now have complete documentation covering:
1. Overall architecture and design
2. All implemented models and utilities
3. Concrete code patterns ready to copy
4. Implementation templates for new models
5. Testing and debugging guides

**Next step**: Pick a document based on your needs and start reading!

---

**Explorer**: Claude Code (Anthropic)
**Date**: February 14, 2026
**Repository**: `/home/guhaase/projetos/panelbox`
**Status**: Complete
