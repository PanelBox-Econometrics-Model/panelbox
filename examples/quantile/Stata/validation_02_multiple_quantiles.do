* =============================================================================
* Validation 02: Multiple Quantiles and Quantile Process
* Replicates PanelBox Notebook 02 results using Stata
*
* Model: lwage = b0 + b1*female + b2*educ + b3*exper + b4*exper^2 + u
* Dataset: card_education.csv
* Uses sqreg for simultaneous quantile regression
* =============================================================================

clear all
set more off

* ---------------------------------------------------------------------------
* 1. Load data
* ---------------------------------------------------------------------------
import delimited using "/home/guhaase/projetos/panelbox/examples/quantile/data/card_education.csv", clear

describe
summarize lwage female educ exper

* Create squared experience term
gen exper_sq = exper^2

* Set panel structure
xtset id year

* ---------------------------------------------------------------------------
* 2. OLS baseline (with female)
* ---------------------------------------------------------------------------
display "============================================================"
display "OLS BASELINE"
display "============================================================"

regress lwage female educ exper exper_sq
estimates store ols_gender

* ---------------------------------------------------------------------------
* 3. Simultaneous quantile regression (sqreg)
* sqreg estimates multiple quantiles simultaneously and provides
* a proper variance-covariance matrix for inter-quantile tests
* ---------------------------------------------------------------------------
display "============================================================"
display "SIMULTANEOUS QUANTILE REGRESSION"
display "quantiles: 0.10, 0.25, 0.50, 0.75, 0.90"
display "============================================================"

sqreg lwage female educ exper exper_sq, ///
    quantiles(.10 .25 .50 .75 .90) reps(100)
estimates store sqr_all

* ---------------------------------------------------------------------------
* 4. Inter-quantile tests for the female coefficient
* Test H0: beta_female(tau1) = beta_female(tau2)
* ---------------------------------------------------------------------------
display "============================================================"
display "INTER-QUANTILE TESTS FOR GENDER WAGE GAP (female)"
display "============================================================"

* Test tau=0.10 vs tau=0.50
display "H0: beta_female(0.10) = beta_female(0.50)"
test [q10]female = [q50]female

* Test tau=0.50 vs tau=0.90
display "H0: beta_female(0.50) = beta_female(0.90)"
test [q50]female = [q90]female

* Test tau=0.10 vs tau=0.90
display "H0: beta_female(0.10) = beta_female(0.90)"
test [q10]female = [q90]female

* Test tau=0.25 vs tau=0.75
display "H0: beta_female(0.25) = beta_female(0.75)"
test [q25]female = [q75]female

* Joint test: beta_female is constant across all quantiles
display "============================================================"
display "JOINT TEST: female coefficient equal across all quantiles"
display "============================================================"
test [q10]female = [q25]female = [q50]female = [q75]female = [q90]female

* ---------------------------------------------------------------------------
* 5. Individual qreg at finer grid (0.05 to 0.95 by 0.05)
* ---------------------------------------------------------------------------
display "============================================================"
display "FULL QUANTILE PROCESS (tau = 0.05 to 0.95 by 0.05)"
display "============================================================"

* Store results using postfile
tempname memhold
tempfile results_file
postfile `memhold' str32 model_name float quantile str20 variable ///
    double(coefficient std_error t_statistic p_value) long n_obs ///
    using "`results_file'", replace

* OLS results
estimates restore ols_gender
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
    post `memhold' ("ols") (.) ("`vname'") (`coef') (`se') (`tval') (`pval') (`n_obs')
}

* Quantile regressions at fine grid
forvalues tau_int = 5(5)95 {
    local tau = `tau_int' / 100

    quietly qreg lwage female educ exper exper_sq, quantile(`tau')
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
        post `memhold' ("qr_process_tau`tau'") (`tau') ("`vname'") ///
            (`coef') (`se') (`tval') (`pval') (`n_obs')
    }

    display "  tau = `tau' done"
}

postclose `memhold'

* Load and display results
use "`results_file'", clear
list if variable == "female", separator(0) noobs

* Export to CSV
export delimited using "/home/guhaase/projetos/panelbox/examples/quantile/Stata/results_02_multiple_quantiles.csv", replace

display "Results saved to results_02_multiple_quantiles.csv"
display "Done."
