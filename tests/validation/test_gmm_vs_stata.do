* Validation script for Panel VAR GMM
* Compares with PanelBox implementation
* Reference: Abrigo & Love (2016)

clear all
set more off

* Load data
use "/tmp/pvar_gmm_test_data.dta", clear

* Declare panel structure
xtset entity time

* Install pvar if needed (comment out if already installed)
* ssc install pvar

* Estimate Panel VAR using GMM with Forward Orthogonal Deviations
* Following Abrigo & Love (2016)
* pvar uses forward orthogonal deviations by default
* instlag(1/5) means use lags 1 through 5 as instruments

pvar y1 y2, lags(1) fod instlag(1/5)

* Extract results
matrix B = e(b)
matrix V = e(V)

* Standard errors from variance-covariance matrix
mata: V = st_matrix("V")
mata: se = sqrt(diagonal(V))
mata: st_matrix("SE", se)

* Get test statistics
scalar hansen_j = e(j)
scalar hansen_j_p = e(jp)

* Get number of instruments
scalar n_instruments = e(ninstr)

* Export results to JSON
capture file close results_file
file open results_file using "/tmp/pvar_gmm_stata_results.json", write replace

file write results_file "{" _n
file write results_file `"  "method": "pvar_fod_gmm","' _n
file write results_file `"  "lags": 1,"' _n
file write results_file `"  "n_instruments": "' %9.0f (n_instruments) `","' _n
file write results_file `"  "hansen_j": "' %12.6f (hansen_j) `","' _n
file write results_file `"  "hansen_j_p": "' %12.6f (hansen_j_p) `","' _n

* Extract coefficients
local ncol = colsof(B)
file write results_file `"  "coefficients": ["' _n
forval i = 1/`ncol' {
    local val = B[1, `i']
    file write results_file `"    "' %12.8f (`val')
    if `i' < `ncol' {
        file write results_file `","'
    }
    file write results_file _n
}
file write results_file `"  ],"' _n

* Extract standard errors
local nrow = rowsof(SE)
file write results_file `"  "std_errors": ["' _n
forval i = 1/`nrow' {
    mata: st_numscalar("se_val", st_matrix("SE")[`i', 1])
    local val = se_val
    file write results_file `"    "' %12.8f (`val')
    if `i' < `nrow' {
        file write results_file `","'
    }
    file write results_file _n
}
file write results_file `"  ]"' _n

file write results_file "}" _n
file close results_file

* Display results for debugging
display "Hansen J: " hansen_j " (p=" hansen_j_p ")"
display "N instruments: " n_instruments
display "Coefficients:"
matrix list B
display "Standard errors:"
matrix list SE

exit, clear
