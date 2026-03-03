* =============================================================================
* Validation Script 01: Poisson and Negative Binomial Regression
* =============================================================================
* Compares Stata results with PanelBox for count data models.
*
* Models estimated:
*   1. Poisson GLM (healthcare_visits): visits ~ age + income + insurance + chronic
*   2. Poisson GLM (firm_patents): patents ~ log_rd + log_emp + firm_age
*      + tech_sector + public_funding + international (cluster-robust SE)
*   3. Negative Binomial (firm_patents): same formula
*
* Note: This script is for syntax reference only (no Stata license available).
* =============================================================================

clear all
set more off
set seed 12345

* --- Define paths ---
local data_dir "/home/guhaase/projetos/panelbox/examples/count/data"
local output_dir "/home/guhaase/projetos/panelbox/examples/count/Stata"

* =============================================================================
* PART 1: Poisson Regression on healthcare_visits
* =============================================================================

display _newline(2) "============================================="
display "PART 1: Poisson Regression (healthcare_visits)"
display "============================================="

* Load healthcare_visits dataset
import delimited using "`data_dir'/healthcare_visits.csv", clear

* Inspect data
describe
summarize visits age income insurance chronic

* Descriptive statistics for visits
tabstat visits, statistics(mean variance min max n) columns(statistics)

* Check overdispersion: variance/mean ratio
quietly summarize visits
display "Var/Mean ratio = " r(Var) / r(mean)

* --- 1a. Poisson GLM with default (model-based) SE ---
display _newline "--- Poisson GLM (model-based SE) ---"
poisson visits age income insurance chronic
estimates store poisson_model

* Display IRR
display _newline "--- Poisson GLM (Incidence Rate Ratios) ---"
poisson visits age income insurance chronic, irr

* --- 1b. Poisson GLM with robust (sandwich) SE ---
* This matches PanelBox se_type="robust"
display _newline "--- Poisson GLM (robust SE) ---"
poisson visits age income insurance chronic, vce(robust)
estimates store poisson_robust

* Store results
matrix b = e(b)
matrix V = e(V)
scalar poisson_ll = e(ll)
scalar poisson_aic = -2 * e(ll) + 2 * e(k)
scalar poisson_n = e(N)

display "Log-Likelihood: " poisson_ll
display "AIC: " poisson_aic
display "N: " poisson_n

* Cameron-Trivedi overdispersion test
* After Poisson estimation, use estat gof
quietly poisson visits age income insurance chronic
estat gof

* =============================================================================
* PART 2: Poisson and Negative Binomial on firm_patents
* =============================================================================

display _newline(2) "============================================="
display "PART 2: Negative Binomial (firm_patents)"
display "============================================="

* Load firm_patents dataset
import delimited using "`data_dir'/firm_patents.csv", clear

* Inspect data
describe
summarize patents rd_spending employees firm_age tech_sector public_funding international

* Create log variables to match PanelBox notebook
generate log_rd = ln(rd_spending)
generate log_emp = ln(employees)

* Check overdispersion
tabstat patents, statistics(mean variance min max n) columns(statistics)
quietly summarize patents
display "Var/Mean ratio = " r(Var) / r(mean)

* --- 2a. Poisson GLM with cluster-robust SE (by firm_id) ---
* Matches PanelBox se_type="cluster"
display _newline "--- Poisson GLM (cluster-robust SE by firm_id) ---"
poisson patents log_rd log_emp firm_age tech_sector public_funding international, ///
    vce(cluster firm_id)
estimates store poisson_patents

scalar poisson_patents_ll = e(ll)
display "Log-Likelihood: " poisson_patents_ll

* --- 2b. Negative Binomial (NB2) model ---
* nbreg estimates NB2 by default (Var = mu + alpha * mu^2)
display _newline "--- Negative Binomial (NB2) ---"
nbreg patents log_rd log_emp firm_age tech_sector public_funding international
estimates store nb_patents

* Display key statistics
scalar nb_ll = e(ll)
scalar nb_alpha = e(alpha)
scalar nb_aic = -2 * e(ll) + 2 * e(k)
scalar nb_n = e(N)

display _newline "Dispersion parameter (alpha): " nb_alpha
display "Log-Likelihood (NB): " nb_ll
display "AIC (NB): " nb_aic
display "N: " nb_n

* Display IRR
display _newline "--- NB2 (Incidence Rate Ratios) ---"
nbreg patents log_rd log_emp firm_age tech_sector public_funding international, irr

* --- 2c. Likelihood Ratio Test: Poisson vs NB ---
* The LR test is automatically reported at the bottom of nbreg output
* It tests H0: alpha = 0 (Poisson is adequate)
display _newline "--- LR Test: Poisson vs NB ---"
display "LR statistic: " 2 * (nb_ll - poisson_patents_ll)
display "Note: LR test for alpha=0 is reported by nbreg output"
display "      (chibar2(01) at the bottom of nbreg output)"

* --- 2d. Model comparison ---
display _newline "--- Model Comparison ---"
estimates table poisson_patents nb_patents, stats(ll aic bic N)

* =============================================================================
* SAVE RESULTS
* =============================================================================

display _newline(2) "============================================="
display "Saving results..."
display "============================================="

* Export Poisson results to CSV
* (In practice, use esttab or outreg2 for formatted output)
* Example with esttab (requires estout package):
* ssc install estout
* esttab poisson_robust poisson_patents nb_patents using ///
*     "`output_dir'/results_01_poisson_negbin_stata.csv", ///
*     cells(b se t p) stats(ll aic bic N) csv replace

display "Validation 01 complete."
display "============================================="
