*****************************************************************************
* Validation 01 - Robust Standard Errors (HC0, HC1, HC2, HC3)
*
* Estimates Pooled OLS on Grunfeld data and computes White heteroskedasticity-
* consistent standard errors (HC0-HC3).
*
* Dataset: Grunfeld (10 firms, 20 years)
* Model:   invest ~ value + capital
*****************************************************************************

clear all
set more off

* --- Data Loading -----------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv", clear

describe
summarize invest value capital

* --- Declare panel structure ------------------------------------------------
xtset firm year

* --- Classical (Non-robust) OLS ---------------------------------------------
* Standard OLS with default (classical) standard errors
regress invest value capital
estimates store ols_classical

* --- HC1 (Stata default 'robust') ------------------------------------------
* HC1 = n/(n-k) * HC0, which is Stata's default robust option
regress invest value capital, vce(robust)
estimates store ols_hc1

* --- HC0 (White 1980, no degrees-of-freedom correction) --------------------
* In Stata, HC0 can be obtained using vce(hc2) with manual adjustment
* or via the user-written 'asd' package. The simplest approach:
regress invest value capital, vce(hc2)
estimates store ols_hc2

* --- HC3 (Davidson-MacKinnon) -----------------------------------------------
regress invest value capital, vce(hc3)
estimates store ols_hc3

* --- For exact HC0 without DF correction ------------------------------------
* Stata 17+ supports: regress invest value capital, vce(hc0)
* For older versions, use:
* ssc install rhausman (or manually compute)
* Here we document the HC0 approach:
regress invest value capital
* Manually extract HC0: multiply robust SE by sqrt((n-k)/n) to undo DF correction
* where n = e(N) and k = e(df_m) + 1
local n = e(N)
local k = e(df_m) + 1
local adj = sqrt((`n' - `k') / `n')
display "HC0 adjustment factor: " `adj'
display "HC0 SE for value:   " _se[value] * `adj'
display "HC0 SE for capital: " _se[capital] * `adj'

* --- Compare all estimates --------------------------------------------------
estimates table ols_classical ols_hc1 ols_hc2 ols_hc3, ///
    se stats(N r2) title("SE Comparison: Classical vs HC1 vs HC2 vs HC3")

* --- Export results to CSV --------------------------------------------------
* Use postfile to create a CSV-compatible dataset
tempname memhold
tempfile results
postfile `memhold' str20 model_name str20 se_type str20 variable ///
    coefficient std_error t_statistic p_value ///
    using `results', replace

* Classical
estimates restore ols_classical
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    post `memhold' ("pooled_ols") ("classical") ("`var'") (`b') (`se') (`t') (`p')
}

* HC1 (robust)
estimates restore ols_hc1
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    post `memhold' ("pooled_ols") ("HC1") ("`var'") (`b') (`se') (`t') (`p')
}

* HC2
estimates restore ols_hc2
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    post `memhold' ("pooled_ols") ("HC2") ("`var'") (`b') (`se') (`t') (`p')
}

* HC3
estimates restore ols_hc3
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    post `memhold' ("pooled_ols") ("HC3") ("`var'") (`b') (`se') (`t') (`p')
}

postclose `memhold'

* Load and display results
use `results', clear
list, clean
export delimited using "/home/guhaase/projetos/panelbox/examples/standard_errors/Stata/results_robust_hc.csv", replace

display "Done."
