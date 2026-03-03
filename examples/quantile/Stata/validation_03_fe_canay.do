* =============================================================================
* Validation 03: Fixed Effects Quantile Regression - Canay (2011) Two-Step
* Replicates PanelBox Notebook 03 results using Stata
*
* Dataset: firm_production.csv (panel: 500 firms x 10+ years)
* Model: log_output ~ log_capital + log_labor + log_materials
* Method: Canay (2011) two-step:
*   Step 1: Estimate FE by within (xtreg, fe) estimator
*   Step 2: Subtract FE from Y, run pooled qreg on demeaned Y
* =============================================================================

clear all
set more off

* ---------------------------------------------------------------------------
* 1. Load data
* ---------------------------------------------------------------------------
import delimited using "/home/guhaase/projetos/panelbox/examples/quantile/data/firm_production.csv", clear

describe
summarize log_output log_capital log_labor log_materials

* Set panel structure
xtset firm_id year

* ---------------------------------------------------------------------------
* 2. Step 1: Fixed Effects OLS (within estimator)
* ---------------------------------------------------------------------------
display "============================================================"
display "STEP 1: FIXED EFFECTS OLS (WITHIN ESTIMATOR)"
display "============================================================"

xtreg log_output log_capital log_labor log_materials, fe
estimates store fe_ols

* Display coefficients
display "FE-OLS Coefficients:"
display "  log_capital:   " _b[log_capital]
display "  log_labor:     " _b[log_labor]
display "  log_materials: " _b[log_materials]

* Returns to scale
local rts = _b[log_capital] + _b[log_labor] + _b[log_materials]
display "  Returns to scale: `rts'"

* ---------------------------------------------------------------------------
* 3. Recover fixed effects and transform Y
* ---------------------------------------------------------------------------
display "============================================================"
display "RECOVERING FIXED EFFECTS AND TRANSFORMING Y"
display "============================================================"

* Predict fixed effects (alpha_hat_i)
predict alpha_hat, u

* Transform Y: y_tilde = log_output - alpha_hat
gen y_tilde = log_output - alpha_hat

summarize alpha_hat y_tilde

* ---------------------------------------------------------------------------
* 4. Step 2: Pooled QR on transformed data (Canay estimator)
* ---------------------------------------------------------------------------
display "============================================================"
display "STEP 2: CANAY TWO-STEP FE QUANTILE REGRESSION"
display "============================================================"

* tau = 0.10
display "--- Canay FE-QR: tau = 0.10 ---"
qreg y_tilde log_capital log_labor log_materials, quantile(.10)
estimates store canay_10

* tau = 0.25
display "--- Canay FE-QR: tau = 0.25 ---"
qreg y_tilde log_capital log_labor log_materials, quantile(.25)
estimates store canay_25

* tau = 0.50
display "--- Canay FE-QR: tau = 0.50 ---"
qreg y_tilde log_capital log_labor log_materials, quantile(.50)
estimates store canay_50

* tau = 0.75
display "--- Canay FE-QR: tau = 0.75 ---"
qreg y_tilde log_capital log_labor log_materials, quantile(.75)
estimates store canay_75

* tau = 0.90
display "--- Canay FE-QR: tau = 0.90 ---"
qreg y_tilde log_capital log_labor log_materials, quantile(.90)
estimates store canay_90

* ---------------------------------------------------------------------------
* 5. Pooled QR (no fixed effects) for comparison
* ---------------------------------------------------------------------------
display "============================================================"
display "POOLED QR (NO FIXED EFFECTS) FOR COMPARISON"
display "============================================================"

* tau = 0.10
display "--- Pooled QR: tau = 0.10 ---"
qreg log_output log_capital log_labor log_materials, quantile(.10)
estimates store pooled_10

* tau = 0.25
display "--- Pooled QR: tau = 0.25 ---"
qreg log_output log_capital log_labor log_materials, quantile(.25)
estimates store pooled_25

* tau = 0.50
display "--- Pooled QR: tau = 0.50 ---"
qreg log_output log_capital log_labor log_materials, quantile(.50)
estimates store pooled_50

* tau = 0.75
display "--- Pooled QR: tau = 0.75 ---"
qreg log_output log_capital log_labor log_materials, quantile(.75)
estimates store pooled_75

* tau = 0.90
display "--- Pooled QR: tau = 0.90 ---"
qreg log_output log_capital log_labor log_materials, quantile(.90)
estimates store pooled_90

* ---------------------------------------------------------------------------
* 6. Comparison table
* ---------------------------------------------------------------------------
display "============================================================"
display "COMPARISON: CANAY FE-QR vs POOLED QR vs FE-OLS"
display "============================================================"

estimates table canay_10 canay_25 canay_50 canay_75 canay_90, ///
    b(%9.6f) se(%9.6f) title("Canay FE-QR")

estimates table pooled_10 pooled_25 pooled_50 pooled_75 pooled_90, ///
    b(%9.6f) se(%9.6f) title("Pooled QR")

estimates table fe_ols, b(%9.6f) se(%9.6f) title("FE-OLS")

* ---------------------------------------------------------------------------
* 7. Export results to CSV using postfile
* ---------------------------------------------------------------------------
tempname memhold
tempfile results_file
postfile `memhold' str32 model_name float quantile str20 variable ///
    double(coefficient std_error t_statistic p_value) long n_obs ///
    using "`results_file'", replace

* Canay results
foreach tau in 10 25 50 75 90 {
    local tau_decimal = `tau' / 100
    estimates restore canay_`tau'
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
        post `memhold' ("canay_fe_qr_tau`tau_decimal'") (`tau_decimal') ("`vname'") ///
            (`coef') (`se') (`tval') (`pval') (`n_obs')
    }
}

* Pooled QR results
foreach tau in 10 25 50 75 90 {
    local tau_decimal = `tau' / 100
    estimates restore pooled_`tau'
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

* FE-OLS results
estimates restore fe_ols
local n_obs = e(N)
foreach var in log_capital log_labor log_materials {
    local coef = _b[`var']
    local se = _se[`var']
    local tval = `coef' / `se'
    local pval = 2 * ttail(e(df_r), abs(`tval'))
    post `memhold' ("fe_ols") (.) ("`var'") (`coef') (`se') (`tval') (`pval') (`n_obs')
}

postclose `memhold'

* Load and display results
use "`results_file'", clear
list, separator(4) noobs

* Export to CSV
export delimited using "/home/guhaase/projetos/panelbox/examples/quantile/Stata/results_03_fe_canay.csv", replace

* ---------------------------------------------------------------------------
* 8. Returns to scale across quantiles
* ---------------------------------------------------------------------------
display "============================================================"
display "RETURNS TO SCALE ACROSS QUANTILES (CANAY)"
display "============================================================"

foreach tau in 10 25 50 75 90 {
    local tau_decimal = `tau' / 100
    estimates restore canay_`tau'
    local rts = _b[log_capital] + _b[log_labor] + _b[log_materials]
    display "  tau = `tau_decimal': RTS = `rts'"
}

estimates restore fe_ols
local rts = _b[log_capital] + _b[log_labor] + _b[log_materials]
display "  FE-OLS:    RTS = `rts'"

display "Done."
