* =============================================================================
* Validation Script 04: First Difference, Between, and IV Estimators
* PanelBox vs Stata comparison
* Dataset: Grunfeld (N=10, T=20, 200 obs)
* Model: invest ~ value + capital
* =============================================================================

clear all
set more off

* --- Load data ---------------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv", clear

* Declare panel structure
xtset firm year

* --- Model 1: First Difference ------------------------------------------------
* FD estimator: differences out entity fixed effects
* Note: Stata's xtreg does not have a built-in FD option
* Method 1: Manual differencing with reg
gen d_invest = D.invest
gen d_value = D.value
gen d_capital = D.capital

reg d_invest d_value d_capital, noconstant
estimates store fd_noconstant

* Method 2: With constant (tests for time trend in FE)
reg d_invest d_value d_capital
estimates store fd_constant

* Method 3: FD with robust SE
reg d_invest d_value d_capital, cluster(firm)
estimates store fd_robust

* Display results
di "=== First Difference Results ==="
di "d_value coefficient: " _b[d_value]
di "d_capital coefficient: " _b[d_capital]
di "N: " e(N)
di "R-squared: " e(r2)

* --- Model 2: Between Estimator -----------------------------------------------
* Uses group means (cross-sectional regression on N group averages)
xtreg invest value capital, be
estimates store between

di "=== Between Estimator Results ==="
di "value coefficient: " _b[value]
di "capital coefficient: " _b[capital]
di "constant: " _b[_cons]
di "R-squared: " e(r2_b)
di "N groups: " e(N_g)

* --- Model 3: IV-Pooled (2SLS) -----------------------------------------------
* Instrument: lag(value) for value
* First stage: value ~ capital + L.value
* Second stage: invest ~ capital + value_hat
gen lag_value = L.value

* Pooled IV (ignoring panel structure)
ivregress 2sls invest capital (value = lag_value)
estimates store iv_pooled

* First-stage diagnostics
estat firststage

di "=== IV-Pooled Results ==="
di "value coefficient: " _b[value]
di "capital coefficient: " _b[capital]

* --- Model 4: IV-FE (2SLS + Fixed Effects) ------------------------------------
* Panel IV with entity fixed effects
xtivreg invest capital (value = lag_value), fe
estimates store iv_fe

di "=== IV-FE Results ==="
di "value coefficient: " _b[value]
di "capital coefficient: " _b[capital]
di "Within R-squared: " e(r2_w)

* --- Model 5: IV-RE (2SLS + Random Effects) -----------------------------------
xtivreg invest capital (value = lag_value), re
estimates store iv_re

di "=== IV-RE Results ==="
di "value coefficient: " _b[value]
di "capital coefficient: " _b[capital]

* --- Hausman test for IV-FE vs IV-RE ------------------------------------------
quietly xtivreg invest capital (value = lag_value), fe
estimates store iv_fe2

quietly xtivreg invest capital (value = lag_value), re
estimates store iv_re2

hausman iv_fe2 iv_re2

* --- Comparison of All Estimators ---------------------------------------------
* Retrieve all stored estimates
estimates table fd_constant between iv_pooled iv_fe iv_re, ///
    b(%9.4f) se(%9.4f) stats(N r2)

* --- Full comparison including standard estimators ----------------------------
quietly reg invest value capital, cluster(firm)
estimates store pooled

quietly xtreg invest value capital, fe
estimates store fe

quietly xtreg invest value capital, re
estimates store re

estimates table pooled fe re fd_constant between iv_pooled iv_fe iv_re, ///
    b(%9.4f) se(%9.4f)

di "=== All Estimators Compared ==="
di "See table above for full comparison"

log close, replace
