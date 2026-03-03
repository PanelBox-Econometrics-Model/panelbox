* =============================================================================
* Validation 01: Pooled Quantile Regression
* Replicates PanelBox Notebook 01 results using Stata
*
* Model: lwage = b0 + b1*educ + b2*exper + b3*exper^2 + u
* Dataset: card_education.csv
* Quantiles: tau = 0.1, 0.25, 0.5, 0.75, 0.9
* =============================================================================

clear all
set more off

* ---------------------------------------------------------------------------
* 1. Load data
* ---------------------------------------------------------------------------
import delimited using "/home/guhaase/projetos/panelbox/examples/quantile/data/card_education.csv", clear

describe
summarize lwage educ exper

* Create squared experience term
gen exper_sq = exper^2

* Set panel structure (for reference, though pooled QR ignores it)
xtset id year

* ---------------------------------------------------------------------------
* 2. OLS baseline
* ---------------------------------------------------------------------------
display "============================================================"
display "OLS BASELINE"
display "============================================================"

regress lwage educ exper exper_sq
estimates store ols_baseline

* ---------------------------------------------------------------------------
* 3. Quantile regression at individual quantiles
* ---------------------------------------------------------------------------

* --- tau = 0.10 ---
display "============================================================"
display "QUANTILE REGRESSION: tau = 0.10"
display "============================================================"

qreg lwage educ exper exper_sq, quantile(.10)
estimates store qr_10

* --- tau = 0.25 ---
display "============================================================"
display "QUANTILE REGRESSION: tau = 0.25"
display "============================================================"

qreg lwage educ exper exper_sq, quantile(.25)
estimates store qr_25

* --- tau = 0.50 (median) ---
display "============================================================"
display "QUANTILE REGRESSION: tau = 0.50 (median)"
display "============================================================"

qreg lwage educ exper exper_sq, quantile(.50)
estimates store qr_50

* --- tau = 0.75 ---
display "============================================================"
display "QUANTILE REGRESSION: tau = 0.75"
display "============================================================"

qreg lwage educ exper exper_sq, quantile(.75)
estimates store qr_75

* --- tau = 0.90 ---
display "============================================================"
display "QUANTILE REGRESSION: tau = 0.90"
display "============================================================"

qreg lwage educ exper exper_sq, quantile(.90)
estimates store qr_90

* ---------------------------------------------------------------------------
* 4. Comparison table
* ---------------------------------------------------------------------------
display "============================================================"
display "COMPARISON OF EDUCATION COEFFICIENTS ACROSS QUANTILES"
display "============================================================"

estimates table ols_baseline qr_10 qr_25 qr_50 qr_75 qr_90, ///
    b(%9.6f) se(%9.6f) stats(N r2)

* ---------------------------------------------------------------------------
* 5. Export results to CSV
* ---------------------------------------------------------------------------
* Note: Stata does not have a direct CSV export from estimates.
* The following uses postfile to create a results dataset.

tempname memhold
tempfile results_file
postfile `memhold' str32 model_name float quantile str20 variable ///
    double(coefficient std_error t_statistic p_value) long n_obs ///
    using "`results_file'", replace

* OLS
local n_obs = e(N)
estimates restore ols_baseline
matrix b = e(b)
matrix V = e(V)
local vars : colnames b
local k = colsof(b)
forvalues j = 1/`k' {
    local vname : word `j' of `vars'
    local coef = b[1, `j']
    local se = sqrt(V[`j', `j'])
    local tval = `coef' / `se'
    local pval = 2 * ttail(e(df_r), abs(`tval'))
    post `memhold' ("ols") (.) ("`vname'") (`coef') (`se') (`tval') (`pval') (`n_obs')
}

* Quantile regressions
foreach tau in 10 25 50 75 90 {
    local tau_decimal = `tau' / 100
    estimates restore qr_`tau'
    local n_obs = e(N)
    matrix b = e(b)
    matrix V = e(V)
    local vars : colnames b
    local k = colsof(b)
    forvalues j = 1/`k' {
        local vname : word `j' of `vars'
        local coef = b[1, `j']
        local se = sqrt(V[`j', `j'])
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `memhold' ("pooled_qr_tau`tau_decimal'") (`tau_decimal') ("`vname'") ///
            (`coef') (`se') (`tval') (`pval') (`n_obs')
    }
}

postclose `memhold'

* Load and display results
use "`results_file'", clear
list, separator(4) noobs

* Export to CSV
export delimited using "/home/guhaase/projetos/panelbox/examples/quantile/Stata/results_01_quantile_pooled.csv", replace

display "Results saved to results_01_quantile_pooled.csv"
display "Done."
