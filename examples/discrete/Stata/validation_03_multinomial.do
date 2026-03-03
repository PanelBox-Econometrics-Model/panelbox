/*===========================================================================
  Validation 03 - Multinomial Logit

  Replicates PanelBox notebook: 06_multinomial_logit.ipynb
  Dataset: career_choice.csv

  Models:
    1. Multinomial Logit (base): career ~ educ + exper + age + female
    2. Multinomial Logit (extended): career ~ educ + exper + age + female + urban

  career categories: 0=Manual, 1=Technical, 2=Managerial
  Base category: 0 (Manual)
===========================================================================*/

clear all
set more off

import delimited "/home/guhaase/projetos/panelbox/examples/discrete/data/career_choice.csv", clear

* Summary statistics
summarize career educ exper age female income urban
tabulate career

* --- Model 1: Base Multinomial Logit ------------------------------------------
display "=== Model 1: Multinomial Logit (base specification) ==="

* Base outcome = 0 (Manual)
mlogit career educ exper age female, baseoutcome(0)
estimates store mlogit_base

* Display results
display "Log-likelihood: " e(ll)
display "AIC: " -2*e(ll) + 2*e(k)
display "N: " e(N)

* --- Model 2: Extended Multinomial Logit --------------------------------------
display "=== Model 2: Extended Multinomial Logit ==="

mlogit career educ exper age female urban, baseoutcome(0)
estimates store mlogit_extended

display "Log-likelihood: " e(ll)
display "AIC: " -2*e(ll) + 2*e(k)
display "N: " e(N)

* --- Model comparison ---------------------------------------------------------
estimates table mlogit_base mlogit_extended, ///
    stats(N ll aic bic) b(%9.6f) se(%9.6f)

* --- Marginal effects (average) -----------------------------------------------
display "=== Average Marginal Effects (Base Model) ==="
estimates restore mlogit_base
margins, dydx(*) predict(outcome(0))
margins, dydx(*) predict(outcome(1))
margins, dydx(*) predict(outcome(2))

* --- Hausman-McFadden IIA test ------------------------------------------------
display "=== IIA Test ==="
* Estimate restricted model excluding alternative 2
mlogit career educ exper age female if career != 2, baseoutcome(0)
estimates store mlogit_restricted

hausman mlogit_restricted mlogit_base, alleqs constant

display "=== Done ==="
