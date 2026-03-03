* ==============================================================================
* Validation Script: Marginal Effects (Stata)
* ==============================================================================
* Purpose: Reproduce PanelBox marginal effects results
* Notebooks: 01_me_fundamentals.ipynb, 02_discrete_me_complete.ipynb
* Models: OLS (Grunfeld), Logit/Probit (Mroz), Poisson (patents)
* NOTE: This script is for reference only. It has NOT been executed.
* ==============================================================================

clear all
set more off

* ==============================================================================
* PART 1: OLS MARGINAL EFFECTS (Grunfeld data)
* ==============================================================================
display "============================================================"
display "PART 1: OLS MARGINAL EFFECTS (Grunfeld data)"
display "============================================================"

import delimited "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv", clear

describe
summarize invest value capital

* Pooled OLS
regress invest value capital

* AME for OLS (trivial - coefficients are the marginal effects)
margins, dydx(*)
* Note: For linear models, margins output = coefficients

* MEM (marginal effects at means)
margins, dydx(*) atmeans

display "Note: For OLS, AME = MEM = coefficients"

* ==============================================================================
* PART 2: LOGIT MARGINAL EFFECTS (Mroz data)
* ==============================================================================
display "============================================================"
display "PART 2: LOGIT MARGINAL EFFECTS (Mroz labor participation)"
display "============================================================"

import delimited "/home/guhaase/projetos/panelbox/examples/marginal_effects/data/mroz.csv", clear

describe
summarize inlf educ age kidslt6 kidsge6 nwifeinc

* Binary logit model
logit inlf educ age kidslt6 kidsge6 nwifeinc

* Average Marginal Effects (AME)
* margins calculates AME by default (average over all observations)
margins, dydx(*)
display "Logit AME: These are the average of individual marginal effects"

* Marginal Effects at Means (MEM)
margins, dydx(*) atmeans
display "Logit MEM: These are evaluated at the sample means"

* Discrete change for binary variables (kidslt6 can be treated as count)
* For truly binary variables, margins calculates P(Y=1|x=1) - P(Y=1|x=0)
* margins, dydx(kidslt6) predict(pr)

* Display comparison: coefficients vs AME
display "Comparison: Logit Coefficients vs Marginal Effects"
display "  In logit, coefficients != marginal effects"
display "  AME = mean[Lambda(Xb)*(1-Lambda(Xb))] * beta_k"
display "  where Lambda() is the logistic CDF"

* ==============================================================================
* PART 3: PROBIT MARGINAL EFFECTS (Mroz data)
* ==============================================================================
display "============================================================"
display "PART 3: PROBIT MARGINAL EFFECTS (Mroz labor participation)"
display "============================================================"

* Binary probit model
probit inlf educ age kidslt6 kidsge6 nwifeinc

* AME
margins, dydx(*)
display "Probit AME"

* MEM
margins, dydx(*) atmeans
display "Probit MEM"

* Note: Probit AME formula: AME = mean[phi(Xb)] * beta_k
* where phi() is the standard normal PDF

* ==============================================================================
* PART 4: POISSON MARGINAL EFFECTS (Patents data)
* ==============================================================================
display "============================================================"
display "PART 4: POISSON MARGINAL EFFECTS (Patents data)"
display "============================================================"

import delimited "/home/guhaase/projetos/panelbox/examples/marginal_effects/data/patents.csv", clear

describe
summarize patents log_rnd log_sales log_capital

* Poisson regression
poisson patents log_rnd log_sales log_capital

* AME
margins, dydx(*)
display "Poisson AME: ME = mean[exp(Xb)] * beta_k"

* MEM
margins, dydx(*) atmeans
display "Poisson MEM"

* Incidence Rate Ratios
poisson patents log_rnd log_sales log_capital, irr
display "IRR = exp(beta): multiplicative effect on expected count"

* ==============================================================================
* PART 5: Summary and Export
* ==============================================================================
display "============================================================"
display "SUMMARY"
display "============================================================"

display "Models estimated:"
display "  1. OLS (Grunfeld): AME = coefficients"
display "  2. Logit (Mroz):   AME = mean[Lambda(Xb)*(1-Lambda(Xb))] * b"
display "  3. Probit (Mroz):  AME = mean[phi(Xb)] * b"
display "  4. Poisson (Patents): AME = mean[exp(Xb)] * b"
display ""
display "Key Insight: In nonlinear models, coefficients != marginal effects"
display "  because the link function makes the relationship between X and Y"
display "  depend on the level of X (and all other covariates)."

* To export margins results to a file:
* After each margins command, use:
*   matrix list r(table)
*   or
*   margins, dydx(*) post
*   outreg2 using results_me.xls, replace

display "============================================================"
display "MARGINAL EFFECTS VALIDATION (STATA) COMPLETE"
display "============================================================"

* ==============================================================================
* NOTES ON COMPARISON WITH R:
* ==============================================================================
* 1. Stata margins, dydx(*) = R margins::margins() = AME
*    Both compute numerical derivatives (finite differences) by default.
*
* 2. Stata margins, dydx(*) atmeans = R margins() with at=list(x=mean(x))
*    Evaluates at the mean of each covariate.
*
* 3. For binary variables, both Stata and R compute discrete changes:
*    P(Y=1|x=1) - P(Y=1|x=0)
*
* 4. Small numerical differences may arise due to:
*    - Different finite difference step sizes
*    - Different convergence criteria in MLE
*    - Stata and R may use slightly different optimization algorithms
*
* 5. Stata and R should produce very similar (but not necessarily identical)
*    AME values for the same model specification and data.
* ==============================================================================
