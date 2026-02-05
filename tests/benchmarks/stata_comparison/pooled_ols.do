*==============================================================================
* Stata Reference Script: Pooled OLS
* For benchmark comparison with PanelBox
*==============================================================================

clear all
set more off

* Load Grunfeld data
use "https://www.stata-press.com/data/r18/grunfeld.dta", clear

* Describe data
describe
summarize

* Pooled OLS: invest = b0 + b1*value + b2*capital + e
regress invest value capital

* Store results
matrix coef = e(b)
matrix se = e(V)

* Display detailed results
display "===================="
display "Pooled OLS Results"
display "===================="
display "Coefficients:"
matrix list coef
display ""
display "Variance-Covariance Matrix:"
matrix list se
display ""
display "R-squared: " e(r2)
display "Adj R-squared: " e(r2_a)
display "RMSE: " e(rmse)
display "N: " e(N)
display "df_m: " e(df_m)
display "df_r: " e(df_r)

* Test coefficient significance
test value
test capital

* Joint test
test value capital

* Residual diagnostics
predict residuals, residuals
predict fitted, xb
summarize residuals fitted

* Export results to JSON format (requires user-written command)
* Or manually copy results for comparison

* Alternative: Export to CSV for easier parsing
preserve
keep company year invest value capital
export delimited using "grunfeld_data.csv", replace
restore

display ""
display "===================="
display "Benchmark completed!"
display "===================="
