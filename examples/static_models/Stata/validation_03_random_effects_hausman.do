* =============================================================================
* Validation Script 03: Random Effects & Hausman Test
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

* --- Model 1: Random Effects (GLS, default Swamy-Arora) ----------------------
xtreg invest value capital, re
estimates store re_model

* Display variance components
di "=== Variance Components ==="
di "sigma_u (entity std dev): " e(sigma_u)
di "sigma_e (idiosyncratic std dev): " e(sigma_e)
di "rho (fraction of variance due to u_i): " e(rho)
di "theta: " e(theta)

* Display R-squared
di "Within R-squared: " e(r2_w)
di "Between R-squared: " e(r2_b)
di "Overall R-squared: " e(r2_o)

* --- Model 2: Random Effects with robust SE -----------------------------------
xtreg invest value capital, re vce(robust)
estimates store re_robust

* --- Model 3: Random Effects with clustered SE --------------------------------
xtreg invest value capital, re vce(cluster firm)
estimates store re_clustered

* --- Fixed Effects (for Hausman test) ----------------------------------------
* Must run FE first, then RE, then Hausman
quietly xtreg invest value capital, fe
estimates store fe_model

quietly xtreg invest value capital, re
estimates store re_model2

* --- Hausman Test (FE vs RE) -------------------------------------------------
* H0: RE is consistent and efficient (no correlation between u_i and X)
* H1: RE is inconsistent (use FE instead)
hausman fe_model re_model2

* Display Hausman test results
di "=== Hausman Test ==="
di "chi2 statistic: " r(chi2)
di "p-value: " r(p)
di "df: " r(df)
di "Decision: " cond(r(p) < 0.05, "Use FE", "Use RE")

* --- Compare RE specifications ------------------------------------------------
estimates table re_model re_robust re_clustered, ///
    b(%9.4f) se(%9.4f) stats(N r2_w r2_b r2_o)

* --- Display key results ------------------------------------------------------
di "=== Random Effects Results ==="
estimates restore re_model
di "value coefficient: " _b[value]
di "value SE: " _se[value]
di "capital coefficient: " _b[capital]
di "capital SE: " _se[capital]
di "constant: " _b[_cons]
di "sigma_u: " e(sigma_u)
di "sigma_e: " e(sigma_e)
di "rho: " e(rho)

log close, replace
