* =============================================================================
* Validation Script 02: Fixed Effects (Within Estimator)
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

* --- Model 1: Fixed Effects (one-way, entity) --------------------------------
* Standard within estimator with entity fixed effects
xtreg invest value capital, fe
estimates store fe_oneway

* Display within R-squared
di "Within R-squared: " e(r2_w)
di "Between R-squared: " e(r2_b)
di "Overall R-squared: " e(r2_o)

* --- Model 2: Fixed Effects (two-way, entity + time) -------------------------
* Entity and time fixed effects using dummy variables for time
* Method 1: Using i.year in xtreg
xtreg invest value capital i.year, fe
estimates store fe_twoway

* Method 2: Alternative using reghdfe (if installed)
* reghdfe invest value capital, absorb(firm year) cluster(firm)

* --- F-test for individual effects (FE vs Pooled OLS) ------------------------
* The F-test at the bottom of xtreg output tests H0: all u_i = 0
* Re-run one-way FE to see the F-test
xtreg invest value capital, fe
di "F-test for individual effects:"
di "F(" e(df_a) "," e(df_r) ") = " e(F_f)
di "p-value = " Ftail(e(df_a), e(df_r), e(F_f))

* --- FE with clustered SE (by firm) ------------------------------------------
xtreg invest value capital, fe vce(cluster firm)
estimates store fe_clustered

* --- Entity fixed effects (intercepts) ----------------------------------------
* To recover entity-specific intercepts
quietly xtreg invest value capital, fe
predict alpha_i, u
* Display unique fixed effects
tabstat alpha_i, by(firm) statistics(mean)

* --- LSDV for comparison (dummy variable approach) ----------------------------
* This should give identical coefficients to xtreg, fe
reg invest value capital i.firm
estimates store lsdv

* --- Compare FE specifications ------------------------------------------------
estimates table fe_oneway fe_twoway fe_clustered, ///
    b(%9.4f) se(%9.4f) stats(N r2_w r2_b r2_o)

* --- Display key results ------------------------------------------------------
di "=== Fixed Effects Results ==="
estimates restore fe_oneway
di "value coefficient: " _b[value]
di "capital coefficient: " _b[capital]
di "Within R-squared: " e(r2_w)
di "Number of groups: " e(N_g)
di "N: " e(N)

log close, replace
