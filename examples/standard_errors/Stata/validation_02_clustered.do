*****************************************************************************
* Validation 02 - Clustered Standard Errors
*
* Estimates Fixed Effects on Grunfeld data and computes clustered standard
* errors: by entity (firm), by time (year), and two-way (firm + year).
*
* Dataset: Grunfeld (10 firms, 20 years)
* Model:   invest ~ value + capital (FE by firm)
*****************************************************************************

clear all
set more off

* --- Data Loading -----------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv", clear

describe
summarize invest value capital

* --- Declare panel structure ------------------------------------------------
xtset firm year

* --- Non-robust FE ----------------------------------------------------------
* Standard fixed effects with default (non-robust) standard errors
xtreg invest value capital, fe
estimates store fe_nonrobust

* --- Clustered by Entity (firm) ---------------------------------------------
* Stata's default clustering for xtreg uses the panel variable
xtreg invest value capital, fe vce(cluster firm)
estimates store fe_cl_entity

* --- Clustered by Time (year) -----------------------------------------------
xtreg invest value capital, fe vce(cluster year)
estimates store fe_cl_time

* --- Two-Way Clustering (firm + year) ---------------------------------------
* Stata does not natively support two-way clustering in xtreg.
* Use reghdfe (user-written) or manual Cameron-Gelbach-Miller approach.
*
* Option 1: Using reghdfe (recommended, install with: ssc install reghdfe)
* reghdfe invest value capital, absorb(firm) vce(cluster firm year)
* estimates store fe_twoway
*
* Option 2: Manual CGM (2011) two-way clustering
* Step 1: Cluster by firm
xtreg invest value capital, fe vce(cluster firm)
matrix V_firm = e(V)

* Step 2: Cluster by year
xtreg invest value capital, fe vce(cluster year)
matrix V_year = e(V)

* Step 3: HC1 robust (intersection = individual observation)
xtreg invest value capital, fe vce(robust)
matrix V_robust = e(V)

* Step 4: Two-way = V_firm + V_year - V_robust (CGM 2011)
matrix V_twoway = V_firm + V_year - V_robust

* Display two-way clustered SEs
display "Two-way clustered SE for value:   " sqrt(V_twoway[1,1])
display "Two-way clustered SE for capital: " sqrt(V_twoway[2,2])

* --- Compare all estimates --------------------------------------------------
estimates table fe_nonrobust fe_cl_entity fe_cl_time, ///
    se stats(N r2_w) title("SE Comparison: Non-robust vs Clustered")

* --- Export results to CSV --------------------------------------------------
tempname memhold
tempfile results
postfile `memhold' str20 model_name str20 se_type str20 variable ///
    coefficient std_error t_statistic p_value ///
    using `results', replace

* Non-robust
estimates restore fe_nonrobust
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    post `memhold' ("fe_within") ("nonrobust") ("`var'") (`b') (`se') (`t') (`p')
}

* Clustered by entity
estimates restore fe_cl_entity
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    post `memhold' ("fe_within") ("cluster_entity") ("`var'") (`b') (`se') (`t') (`p')
}

* Clustered by time
estimates restore fe_cl_time
foreach var in value capital {
    local b = _b[`var']
    local se = _se[`var']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    post `memhold' ("fe_within") ("cluster_time") ("`var'") (`b') (`se') (`t') (`p')
}

* Two-way (manual CGM)
* Use coefficients from any FE estimation (they are identical)
estimates restore fe_cl_entity
foreach var in value capital {
    local b = _b[`var']
    * Get index for this variable in V_twoway matrix
    local idx = colnumb(V_twoway, "`var'")
    local se = sqrt(V_twoway[`idx', `idx'])
    local t = `b' / `se'
    * Use min(G1-1, G2-1) as conservative df for two-way
    local df = min(e(N_clust) - 1, 20 - 1)
    local p = 2 * ttail(`df', abs(`t'))
    post `memhold' ("fe_within") ("cluster_twoway") ("`var'") (`b') (`se') (`t') (`p')
}

postclose `memhold'

use `results', clear
list, clean
export delimited using "/home/guhaase/projetos/panelbox/examples/standard_errors/Stata/results_clustered.csv", replace

display "Done."
