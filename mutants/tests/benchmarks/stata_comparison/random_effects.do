*==============================================================================
* Stata Reference Script: Random Effects (GLS Estimator)
* For benchmark comparison with PanelBox
*==============================================================================

clear all
set more off

* Load Grunfeld data
use "https://www.stata-press.com/data/r18/grunfeld.dta", clear

* Declare panel structure
xtset company year

* Random Effects: invest = alpha + b1*value + b2*capital + u_i + e_it
xtreg invest value capital, re

* Store results
matrix coef = e(b)
matrix se = e(V)

* Display detailed results
display "========================="
display "Random Effects Results"
display "========================="
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
display "Wald chi2: " e(chi2)
display "Prob > chi2: " e(p)
display "rho (fraction of variance due to u_i): " e(rho)
display "sigma_u: " e(sigma_u)
display "sigma_e: " e(sigma_e)
display "theta: " e(theta)

* Test coefficient significance
test value
test capital

* Joint test
test value capital

* Predict random effects
predict random_effects, u
summarize random_effects

* Predict residuals and fitted values
predict residuals, e
predict fitted, xb
summarize residuals fitted

* Hausman test (FE vs RE)
quietly xtreg invest value capital, fe
estimates store fe_model
quietly xtreg invest value capital, re
estimates store re_model
hausman fe_model re_model

display ""
display "========================="
display "Benchmark completed!"
display "========================="
