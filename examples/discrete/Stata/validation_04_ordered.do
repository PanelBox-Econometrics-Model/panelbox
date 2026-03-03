/*===========================================================================
  Validation 04 - Ordered Logit and Ordered Probit

  Replicates PanelBox notebook: 07_ordered_models.ipynb
  Dataset: credit_rating.csv

  Models:
    1. Ordered Logit:  rating ~ income + debt_ratio + age + size + profitability
    2. Ordered Probit: rating ~ income + debt_ratio + age + size + profitability

  rating categories: 0=Poor, 1=Fair, 2=Good, 3=Excellent (ordinal)
===========================================================================*/

clear all
set more off

import delimited "/home/guhaase/projetos/panelbox/examples/discrete/data/credit_rating.csv", clear

* Declare panel structure
xtset id year

* Summary statistics
summarize rating income debt_ratio age size profitability
tabulate rating

* --- Model 1: Ordered Logit ---------------------------------------------------
display "=== Model 1: Ordered Logit ==="

ologit rating income debt_ratio age size profitability
estimates store ologit1

display "Log-likelihood: " e(ll)
display "AIC: " -2*e(ll) + 2*e(k)
display "BIC: " -2*e(ll) + e(k)*ln(e(N))

* Display cutpoints
display "Cutpoint 1 (Poor|Fair): " _b[/cut1]
display "Cutpoint 2 (Fair|Good): " _b[/cut2]
display "Cutpoint 3 (Good|Excellent): " _b[/cut3]

* --- Model 2: Ordered Probit --------------------------------------------------
display "=== Model 2: Ordered Probit ==="

oprobit rating income debt_ratio age size profitability
estimates store oprobit1

display "Log-likelihood: " e(ll)
display "AIC: " -2*e(ll) + 2*e(k)
display "BIC: " -2*e(ll) + e(k)*ln(e(N))

display "Cutpoint 1 (Poor|Fair): " _b[/cut1]
display "Cutpoint 2 (Fair|Good): " _b[/cut2]
display "Cutpoint 3 (Good|Excellent): " _b[/cut3]

* --- Model comparison ---------------------------------------------------------
estimates table ologit1 oprobit1, ///
    stats(N ll aic bic) b(%9.6f) se(%9.6f)

* --- Marginal effects ---------------------------------------------------------
display "=== Average Marginal Effects (Ordered Logit) ==="
estimates restore ologit1
margins, dydx(*) predict(outcome(0))
margins, dydx(*) predict(outcome(1))
margins, dydx(*) predict(outcome(2))
margins, dydx(*) predict(outcome(3))

* --- Brant test for proportional odds -----------------------------------------
display "=== Brant Test ==="
* Requires user-written command: ssc install brant
* brant, detail

display "=== Done ==="
