* =============================================================================
* Validation Script 03: Zero-Inflated Poisson and Zero-Inflated Negative Binomial
* =============================================================================
* Compares Stata results with PanelBox for zero-inflated count data models.
*
* Models estimated (all on healthcare_zinb):
*   1. Baseline Poisson: doctor_visits ~ age + income + chronic_condition
*   2. Standard Negative Binomial: same formula
*   3. ZIP: count ~ age + income + chronic_condition
*          inflate ~ insurance + distance_clinic + urban
*   4. ZINB: same specification as ZIP, plus alpha
*
* Note: This script is for syntax reference only (no Stata license available).
* =============================================================================

clear all
set more off

* --- Define paths ---
local data_dir "/home/guhaase/projetos/panelbox/examples/count/data"
local output_dir "/home/guhaase/projetos/panelbox/examples/count/Stata"

* =============================================================================
* Load data
* =============================================================================

import delimited using "`data_dir'/healthcare_zinb.csv", clear

* Inspect data
describe
summarize doctor_visits age income chronic_condition insurance distance_clinic urban

* Descriptive statistics for doctor_visits
tabstat doctor_visits, statistics(mean variance min max n) columns(statistics)

* Check fraction of zeros
count if doctor_visits == 0
display "Fraction zeros: " r(N) / _N * 100 "%"

* Distribution of doctor_visits
tabulate doctor_visits if doctor_visits <= 10

* =============================================================================
* PART 1: Baseline Poisson
* =============================================================================

display _newline(2) "============================================="
display "PART 1: Baseline Poisson"
display "============================================="

* Poisson GLM with robust SE
poisson doctor_visits age income chronic_condition, vce(robust)
estimates store poisson_base

* Store log-likelihood
scalar poisson_ll = e(ll)
display "Log-Likelihood: " poisson_ll

* Goodness of fit test
quietly poisson doctor_visits age income chronic_condition
estat gof

* Predicted probability of zero
predict p_poisson, n
generate prob_zero_poisson = exp(-p_poisson)
summarize prob_zero_poisson
display "Mean predicted P(Y=0): " r(mean)

drop p_poisson prob_zero_poisson

* =============================================================================
* PART 2: Standard Negative Binomial
* =============================================================================

display _newline(2) "============================================="
display "PART 2: Standard Negative Binomial"
display "============================================="

* NB2 model
nbreg doctor_visits age income chronic_condition
estimates store nb_base

scalar nb_ll = e(ll)
scalar nb_alpha = e(alpha)
display "Log-Likelihood: " nb_ll
display "Alpha (dispersion): " nb_alpha

* IRR
nbreg doctor_visits age income chronic_condition, irr

* =============================================================================
* PART 3: Zero-Inflated Poisson (ZIP)
* =============================================================================

display _newline(2) "============================================="
display "PART 3: Zero-Inflated Poisson (ZIP)"
display "============================================="

* ZIP model
* Count equation: doctor_visits ~ age + income + chronic_condition
* Inflate equation: insurance + distance_clinic + urban
* In Stata, the inflate() option specifies the zero-inflation equation
zip doctor_visits age income chronic_condition, ///
    inflate(insurance distance_clinic urban)
estimates store zip_model

scalar zip_ll = e(ll)
display "Log-Likelihood: " zip_ll

* Display IRR for count model
zip doctor_visits age income chronic_condition, ///
    inflate(insurance distance_clinic urban) irr

* Vuong test
* Stata reports the Vuong test automatically at the bottom of zip output
* It tests H0: Poisson vs H1: ZIP
display _newline "Note: Vuong test statistic is reported in the zip output"
display "      A large positive value favors ZIP over standard Poisson"

* Predicted probabilities from ZIP
predict p_zip_zero, pr(0)
summarize p_zip_zero
display "Mean predicted P(Y=0) from ZIP: " r(mean)

drop p_zip_zero

* =============================================================================
* PART 4: Zero-Inflated Negative Binomial (ZINB)
* =============================================================================

display _newline(2) "============================================="
display "PART 4: Zero-Inflated Negative Binomial (ZINB)"
display "============================================="

* ZINB model
* Same specification as ZIP but with NB count distribution
zinb doctor_visits age income chronic_condition, ///
    inflate(insurance distance_clinic urban)
estimates store zinb_model

scalar zinb_ll = e(ll)
scalar zinb_alpha = e(alpha)
display "Log-Likelihood: " zinb_ll
display "Alpha (dispersion): " zinb_alpha

* Display IRR for count model
zinb doctor_visits age income chronic_condition, ///
    inflate(insurance distance_clinic urban) irr

* =============================================================================
* MODEL COMPARISON
* =============================================================================

display _newline(2) "============================================="
display "MODEL COMPARISON"
display "============================================="

* AIC/BIC comparison
estimates stats poisson_base nb_base zip_model zinb_model

* Likelihood ratio test: Poisson vs NB (from nbreg output)
display _newline "LR test: Poisson vs NB"
display "LR statistic: " 2 * (nb_ll - poisson_ll)

* Likelihood ratio test: ZIP vs ZINB
display _newline "LR test: ZIP vs ZINB"
display "LR statistic: " 2 * (zinb_ll - zip_ll)

* Vuong test: ZIP vs Poisson (already reported by zip command)

* Full comparison table
estimates table poisson_base nb_base zip_model zinb_model, ///
    stats(ll aic bic N) ///
    keep(age income chronic_condition insurance distance_clinic urban)

* =============================================================================
* SAVE RESULTS
* =============================================================================

display _newline(2) "============================================="
display "Saving results..."
display "============================================="

* Example with esttab (requires estout package):
* ssc install estout
* esttab poisson_base nb_base zip_model zinb_model ///
*     using "`output_dir'/results_03_zero_inflated_stata.csv", ///
*     cells(b se z p) stats(ll aic bic N) csv replace

display "Validation 03 complete."
display "============================================="
