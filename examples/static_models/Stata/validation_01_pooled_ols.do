* =============================================================================
* Validation Script 01: Pooled OLS
* PanelBox vs Stata comparison
* Dataset: Grunfeld (N=10, T=20, 200 obs)
* Model: invest ~ value + capital
* =============================================================================

clear all
set more off

* --- Load data ---------------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv", clear

* Inspect data
describe
summarize invest value capital
list in 1/5

* Declare panel structure
xtset firm year
xtdescribe

* --- Model 1: Pooled OLS (non-robust SE) ------------------------------------
* Standard OLS ignoring panel structure
reg invest value capital
estimates store pooled_nonrobust

* --- Model 2: Pooled OLS with HC1 robust SE ----------------------------------
reg invest value capital, robust
estimates store pooled_robust

* --- Model 3: Pooled OLS with clustered SE (by firm) -------------------------
* This is the correct specification for panel data
reg invest value capital, cluster(firm)
estimates store pooled_clustered

* --- Model 4: Pooled OLS with two-way clustered SE ---------------------------
* Requires Stata 17+ or user-written commands
* Two-way clustering by firm and year
* Using cgmreg if available, otherwise approximate with cluster(firm)
capture program drop
reg invest value capital, cluster(firm)
* Note: For exact two-way clustering, use:
* cgmreg invest value capital, cluster(firm year)
* or in Stata 17+:
* reghdfe invest value capital, cluster(firm year) noabsorb

* --- Compare SE across specifications ----------------------------------------
estimates table pooled_nonrobust pooled_robust pooled_clustered, ///
    b(%9.4f) se(%9.4f) stats(N r2)

* --- Export results -----------------------------------------------------------
* Save coefficients and SE to a matrix for export
matrix b = e(b)
matrix V = e(V)
matrix se = J(1, colsof(V), 0)
forvalues i = 1/`=colsof(V)' {
    matrix se[1,`i'] = sqrt(V[`i',`i'])
}

* Display final results
di "=== Pooled OLS Results ==="
di "value coefficient: " _b[value]
di "value SE (clustered): " _se[value]
di "capital coefficient: " _b[capital]
di "capital SE (clustered): " _se[capital]
di "constant: " _b[_cons]
di "R-squared: " e(r2)
di "N: " e(N)

log close, replace
