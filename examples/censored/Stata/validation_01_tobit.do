* =============================================================================
* Validation Script 01: Tobit Pooled Model
* PanelBox Validation - Stata
*
* Dataset: labor_supply.csv
* Model: hours ~ wage + education + experience + experience_sq + age +
*                children + married + non_labor_income
* Censoring: left-censored at 0
* =============================================================================

clear all
set more off

* --- Load data ---------------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/censored/data/labor_supply.csv", clear

* --- Summary statistics ------------------------------------------------------
describe
summarize hours wage education experience experience_sq age children married non_labor_income

* Count censored observations
count if hours == 0
local n_censored = r(N)
count if hours > 0
local n_uncensored = r(N)
display "N censored: `n_censored'"
display "N uncensored: `n_uncensored'"

* --- Model 1: OLS (biased baseline) -----------------------------------------
display _newline "=== OLS Estimation (biased baseline) ==="
regress hours wage education experience experience_sq age children married non_labor_income
estimates store ols

* --- Model 2: Tobit (left-censored at 0) ------------------------------------
display _newline "=== Tobit Model (left-censored at 0) ==="
tobit hours wage education experience experience_sq age children married non_labor_income, ll(0)
estimates store tobit_model

* Display sigma
display "Sigma (SE of regression): " e(sigma)
display "Log-likelihood: " e(ll)

* --- Marginal effects --------------------------------------------------------
display _newline "=== Marginal Effects ==="

* Unconditional marginal effects (E[y])
margins, dydx(*) predict(ystar(0,.))

* Conditional marginal effects (E[y|y>0])
margins, dydx(*) predict(e(0,.))

* Probability of being uncensored
margins, dydx(*) predict(pr(0,.))

* --- Comparison table --------------------------------------------------------
display _newline "=== Comparison: OLS vs Tobit ==="
estimates table ols tobit_model, b(%12.6f) se(%12.6f) stats(N ll)

* --- Export results ----------------------------------------------------------
* Note: In practice, use estout/esttab to export results to CSV
* ssc install estout  (if not already installed)
* esttab ols tobit_model using "results_tobit_stata.csv", csv replace ///
*   cells(b(fmt(6)) se(fmt(6)) z(fmt(4)) p(fmt(4))) ///
*   stats(N ll sigma, fmt(0 4 6))

display _newline "Done."
