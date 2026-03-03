*****************************************************************************
* Validation 03 - HAC (Newey-West) Standard Errors
*
* Estimates OLS on Grunfeld data and computes Newey-West HAC standard errors
* with different lag specifications (1, 2, 3, 4).
* Uses Stata's built-in 'newey' command.
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

* --- Declare time series structure ------------------------------------------
* For newey, we need tsset. Since Grunfeld is a panel, we use xtset
* and then run newey on the pooled data (treating it as a long time series)
xtset firm year

* --- Standard OLS (baseline) ------------------------------------------------
regress invest value capital
estimates store ols_classical

* --- Newey-West HAC with lag=1 ----------------------------------------------
* The 'newey' command computes Newey-West HAC standard errors
* force option allows panel data (pooled)
newey invest value capital, lag(1) force
estimates store hac_lag1

* --- Newey-West HAC with lag=2 ----------------------------------------------
newey invest value capital, lag(2) force
estimates store hac_lag2

* --- Newey-West HAC with lag=3 ----------------------------------------------
newey invest value capital, lag(3) force
estimates store hac_lag3

* --- Newey-West HAC with lag=4 ----------------------------------------------
newey invest value capital, lag(4) force
estimates store hac_lag4

* --- Compare all estimates --------------------------------------------------
estimates table ols_classical hac_lag1 hac_lag2 hac_lag3 hac_lag4, ///
    se stats(N) title("SE Comparison: Classical vs Newey-West HAC")

* --- Automatic lag selection (rule of thumb) --------------------------------
* Common rule: L = floor(4 * (T/100)^(2/9))
* For Grunfeld pooled: T=200, L = floor(4 * (200/100)^(2/9)) = floor(4*1.166) = 4
local T = _N
local auto_lag = floor(4 * (`T'/100)^(2/9))
display "Automatic lag selection: L = " `auto_lag'

newey invest value capital, lag(`auto_lag') force
estimates store hac_auto

* --- Panel-specific HAC: xtscc (Driscoll-Kraay) ----------------------------
* Driscoll-Kraay SE are robust to cross-sectional dependence and
* heteroskedasticity. Requires user-written command:
* ssc install xtscc
*
* xtscc invest value capital, lag(2)
* estimates store driscoll_kraay

* --- Export results to CSV --------------------------------------------------
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

* HAC lag=1
estimates restore hac_lag1
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * normal(-abs(`t'))
    post `memhold' ("pooled_ols") ("HAC_lag1") ("`var'") (`b') (`se') (`t') (`p')
}

* HAC lag=2
estimates restore hac_lag2
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * normal(-abs(`t'))
    post `memhold' ("pooled_ols") ("HAC_lag2") ("`var'") (`b') (`se') (`t') (`p')
}

* HAC lag=3
estimates restore hac_lag3
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * normal(-abs(`t'))
    post `memhold' ("pooled_ols") ("HAC_lag3") ("`var'") (`b') (`se') (`t') (`p')
}

* HAC lag=4
estimates restore hac_lag4
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * normal(-abs(`t'))
    post `memhold' ("pooled_ols") ("HAC_lag4") ("`var'") (`b') (`se') (`t') (`p')
}

* HAC auto
estimates restore hac_auto
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * normal(-abs(`t'))
    post `memhold' ("pooled_ols") ("HAC_auto") ("`var'") (`b') (`se') (`t') (`p')
}

postclose `memhold'

use `results', clear
list, clean
export delimited using "/home/guhaase/projetos/panelbox/examples/standard_errors/Stata/results_hac.csv", replace

display "Done."
