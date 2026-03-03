/*===========================================================================
  Validation 01 - Binary Choice Models (Pooled Logit and Pooled Probit)

  Replicates PanelBox notebook: 01_binary_choice_introduction.ipynb
  Dataset: labor_participation.csv

  Models:
    1. Pooled Logit:  lfp ~ age + educ + kids + married
    2. Pooled Probit: lfp ~ age + educ + kids + married
    3. Full Logit:    lfp ~ age + age^2 + educ + kids + married + exper
===========================================================================*/

clear all
set more off

* --- Load data ---------------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/discrete/data/labor_participation.csv", clear

* Declare panel structure
xtset id year

* Summary statistics
summarize lfp age educ kids married exper
tabulate lfp

* Generate age squared
generate age2 = age^2

* --- Model 1: Pooled Logit ---------------------------------------------------
display "=== Model 1: Pooled Logit ==="
logit lfp age educ kids married, vce(cluster id)
estimates store logit_base

* --- Model 2: Pooled Probit --------------------------------------------------
display "=== Model 2: Pooled Probit ==="
probit lfp age educ kids married, vce(cluster id)
estimates store probit_base

* --- Model 3: Full Logit (with age^2 and exper) ------------------------------
display "=== Model 3: Full Logit (quadratic age + exper) ==="
logit lfp age age2 educ kids married exper, vce(cluster id)
estimates store logit_full

* --- Model comparison table ---------------------------------------------------
estimates table logit_base probit_base logit_full, ///
    stats(N ll aic bic) b(%9.6f) se(%9.6f)

* --- Export results to CSV (optional, requires estout or manual approach) ------
* If estout is available:
* ssc install estout
* esttab logit_base probit_base logit_full using "results_01_binary_choice.csv", ///
*     cells(b(fmt(6)) se(fmt(6)) z(fmt(4)) p(fmt(4))) ///
*     stats(N ll aic bic) csv replace

display "=== Done ==="
