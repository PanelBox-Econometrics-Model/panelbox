*==============================================================================
* Stata Reference Script: Fixed Effects (Within Estimator)
* For benchmark comparison with PanelBox
*==============================================================================

clear all
set more off

* Load Grunfeld data
use "https://www.stata-press.com/data/r18/grunfeld.dta", clear

* Declare panel structure
xtset company year

* Fixed Effects: invest = alpha_i + b1*value + b2*capital + e
xtreg invest value capital, fe

* Store results
matrix coef = e(b)
matrix se = e(V)

* Display detailed results
display "========================"
display "Fixed Effects Results"
display "========================"
display "Coefficients:"
matrix list coef
display ""
display "Variance-Covariance Matrix:"
matrix list se
display ""
display "R-squared (within): " e(r2_w)
display "R-squared (between): " e(r2_b)
display "R-squared (overall): " e(r2_o)
display "N: " e(N)
display "Groups: " e(N_g)
display "Obs per group (min): " e(g_min)
display "Obs per group (avg): " e(g_avg)
display "Obs per group (max): " e(g_max)
display "F-statistic: " e(F)
display "Prob > F: " e(p)
display "rho (fraction of variance due to u_i): " e(rho)
display "sigma_u: " e(sigma_u)
display "sigma_e: " e(sigma_e)

* Test coefficient significance
test value
test capital

* Joint test
test value capital

* Extract fixed effects
predict fixed_effects, u
summarize fixed_effects

* Predict residuals and fitted values
predict residuals, e
predict fitted, xb
summarize residuals fitted

* F-test for fixed effects
xtreg invest value capital, fe
estimates store fe_model
xtreg invest value capital
estimates store pooled_model
* hausman fe_model pooled_model

display ""
display "========================"
display "Benchmark completed!"
display "========================"
