* =============================================================================
* Validation Script 03: Tobit Panel (Random Effects)
* PanelBox Validation - Stata
*
* Dataset: health_expenditure_panel.csv
* Model: expenditure ~ income + age + chronic + insurance + female + bmi
* Panel: id (individual), time (period)
* Censoring: left-censored at 0
* =============================================================================

clear all
set more off

* --- Load data ---------------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/censored/data/health_expenditure_panel.csv", clear

* --- Summary statistics ------------------------------------------------------
describe
summarize expenditure income age chronic insurance female bmi

* Count censored observations
count if expenditure == 0
local n_censored = r(N)
count if expenditure > 0
local n_uncensored = r(N)
display "N censored: `n_censored'"
display "N uncensored: `n_uncensored'"

* --- Set panel structure -----------------------------------------------------
xtset id time

* --- Model 1: Pooled Tobit (ignoring panel structure) -----------------------
display _newline "=== Pooled Tobit (ignoring panel structure) ==="
tobit expenditure income age chronic insurance female bmi, ll(0)
estimates store tobit_pooled

display "Sigma: " e(sigma)
display "Log-likelihood: " e(ll)

* --- Model 2: Random Effects Tobit (xttobit) --------------------------------
display _newline "=== Random Effects Tobit (xttobit) ==="
xttobit expenditure income age chronic insurance female bmi, ll(0)
estimates store tobit_re

* Display variance components
display "sigma_u: " e(sigma_u)
display "sigma_e: " e(sigma_e)
display "rho (ICC): " e(rho)
display "Log-likelihood: " e(ll)

* --- LR test for random effects ----------------------------------------------
display _newline "=== LR Test for Random Effects (sigma_u = 0) ==="
* Stata's xttobit automatically reports this test
* The chibar2 test at the bottom of xttobit output is the boundary test

* --- Marginal effects for RE Tobit -------------------------------------------
display _newline "=== Marginal Effects (RE Tobit) ==="
margins, dydx(*) predict(ystar(0,.))

* --- Comparison table --------------------------------------------------------
display _newline "=== Comparison: Pooled Tobit vs RE Tobit ==="
estimates table tobit_pooled tobit_re, b(%12.6f) se(%12.6f) stats(N ll)

* --- Export results ----------------------------------------------------------
* Note: In practice, use estout/esttab to export results to CSV
* esttab tobit_pooled tobit_re using "results_tobit_panel_stata.csv", ///
*   csv replace ///
*   cells(b(fmt(6)) se(fmt(6)) z(fmt(4)) p(fmt(4))) ///
*   stats(N ll sigma_u sigma_e rho, fmt(0 4 6 6 6))

display _newline "Done."
