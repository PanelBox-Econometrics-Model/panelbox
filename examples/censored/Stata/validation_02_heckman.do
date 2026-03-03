* =============================================================================
* Validation Script 02: Heckman Selection Models (Two-Step and MLE)
* PanelBox Validation - Stata
*
* Dataset: mroz_1987.csv
* Outcome equation: wage ~ education + experience + experience_sq
* Selection equation: lfp ~ education + experience + age + children_lt6 +
*                           children_6_18 + husband_income
* =============================================================================

clear all
set more off

* --- Load data ---------------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/censored/data/mroz_1987.csv", clear

* --- Summary statistics ------------------------------------------------------
describe
summarize wage education experience experience_sq age children_lt6 children_6_18 husband_income

* Participation rate
tabulate lfp
display "Participation rate: " round(100 * r(N) / _N, 0.01) "%"

* --- Model 1: OLS on working women only (biased baseline) -------------------
display _newline "=== OLS on Working Women Only (biased baseline) ==="
regress wage education experience experience_sq if lfp == 1
estimates store ols_working

* --- Model 2: Heckman Two-Step -----------------------------------------------
display _newline "=== Heckman Two-Step ==="
heckman wage education experience experience_sq, ///
    select(lfp = education experience age children_lt6 children_6_18 husband_income) ///
    twostep
estimates store heckman_2step

* Display key parameters
display "rho: " e(rho)
display "sigma: " e(sigma)
display "lambda (IMR): " e(lambda)

* --- Model 3: Heckman MLE ---------------------------------------------------
display _newline "=== Heckman MLE ==="
heckman wage education experience experience_sq, ///
    select(lfp = education experience age children_lt6 children_6_18 husband_income)
estimates store heckman_mle

* Display key parameters
display "rho: " e(rho)
display "sigma: " e(sigma)
display "lambda (IMR): " e(rho) * e(sigma)
display "Log-likelihood: " e(ll)

* --- Test for selection bias -------------------------------------------------
display _newline "=== Test for Selection Bias ==="
* In MLE, test rho = 0
test [athrho]_cons = 0

* --- Comparison table --------------------------------------------------------
display _newline "=== Comparison: OLS vs Heckman Two-Step vs Heckman MLE ==="
estimates table ols_working heckman_2step heckman_mle, b(%12.6f) se(%12.6f) stats(N ll)

* --- Export results ----------------------------------------------------------
* Note: In practice, use estout/esttab to export results to CSV
* esttab ols_working heckman_2step heckman_mle using "results_heckman_stata.csv", ///
*   csv replace ///
*   cells(b(fmt(6)) se(fmt(6)) z(fmt(4)) p(fmt(4))) ///
*   stats(N ll rho sigma lambda, fmt(0 4 6 6 6))

display _newline "Done."
