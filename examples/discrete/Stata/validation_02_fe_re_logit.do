/*===========================================================================
  Validation 02 - Fixed Effects and Random Effects Logit/Probit

  Replicates PanelBox notebooks:
    02_fixed_effects_logit.ipynb
    03_random_effects.ipynb

  Models:
    1. FE Logit on job_training: employed ~ training + age + prior_wage
    2. FE Logit on labor_participation: lfp ~ age + educ + kids + married
    3. RE Probit on labor_participation: lfp ~ age + educ + kids + married
    4. RE Logit on labor_participation: lfp ~ age + educ + kids + married
    5. Pooled Probit baseline
===========================================================================*/

clear all
set more off

* ==============================================================================
* Part A: FE Logit on Job Training data
* ==============================================================================

display "=== Part A: FE Logit - Job Training ==="

import delimited "/home/guhaase/projetos/panelbox/examples/discrete/data/job_training.csv", clear
xtset id year

summarize employed training age prior_wage education

* FE Logit (conditional logit)
xtlogit employed training age prior_wage, fe
estimates store fe_logit_jt

* Pooled Logit for comparison
logit employed training age prior_wage, vce(cluster id)
estimates store pooled_logit_jt

estimates table fe_logit_jt pooled_logit_jt, stats(N ll) b(%9.6f) se(%9.6f)

* ==============================================================================
* Part B: FE/RE models on Labor Participation data
* ==============================================================================

display "=== Part B: FE/RE - Labor Participation ==="

import delimited "/home/guhaase/projetos/panelbox/examples/discrete/data/labor_participation.csv", clear
xtset id year

summarize lfp age educ kids married exper

* --- Model 2: FE Logit -------------------------------------------------------
display "--- FE Logit ---"
xtlogit lfp age educ kids married, fe
estimates store fe_logit_lp

* --- Model 3: RE Probit -------------------------------------------------------
display "--- RE Probit ---"
xtprobit lfp age educ kids married, re intpoints(12)
estimates store re_probit

* Display sigma_u and rho
display "sigma_u = " e(sigma_u)
display "rho = " e(rho)

* --- Model 4: RE Logit --------------------------------------------------------
display "--- RE Logit ---"
xtlogit lfp age educ kids married, re intpoints(12)
estimates store re_logit

display "sigma_u = " e(sigma_u)
display "rho = " e(rho)

* --- Model 5: Pooled Probit (baseline) ----------------------------------------
display "--- Pooled Probit ---"
probit lfp age educ kids married
estimates store pooled_probit

* --- Model comparison ---------------------------------------------------------
estimates table fe_logit_lp re_probit re_logit pooled_probit, ///
    stats(N ll) b(%9.6f) se(%9.6f)

display "=== Done ==="
