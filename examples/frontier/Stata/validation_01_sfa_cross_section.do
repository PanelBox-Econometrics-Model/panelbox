* ==============================================================================
* Validation Script: Stochastic Frontier Analysis - Cross-Section (Stata)
* ==============================================================================
* Purpose: Reproduce PanelBox SFA cross-section results
* Notebook: examples/frontier/notebooks/01_introduction_sfa.ipynb
* Data: hospital_data.csv (500 hospitals, cross-section)
* Models: Half-normal SFA, Truncated-normal SFA
* NOTE: This script is for reference only. It has NOT been executed.
* ==============================================================================

clear all
set more off

* ------------------------------------------------------------------------------
* 1. Load Data
* ------------------------------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/frontier/data/hospital_data.csv", clear

describe
summarize log_cases log_doctors log_nurses log_beds teaching urban

* ------------------------------------------------------------------------------
* 2. OLS Baseline
* ------------------------------------------------------------------------------
display "============================================================"
display "OLS BASELINE"
display "============================================================"

regress log_cases log_doctors log_nurses log_beds teaching urban

* Store OLS log-likelihood for comparison
scalar ols_ll = e(ll)
display "OLS Log-Likelihood: " ols_ll

* Check residual skewness
predict resid_ols, residuals
summarize resid_ols, detail
display "OLS Residual Skewness: " r(skewness)

* ------------------------------------------------------------------------------
* 3. SFA Model 1: Half-Normal
* ------------------------------------------------------------------------------
display "============================================================"
display "SFA MODEL 1: HALF-NORMAL"
display "============================================================"

* Stata frontier command with half-normal distribution
* In Stata, frontier estimates: y = x'b + v - u (production frontier)
* dist(hnormal) specifies half-normal for u
frontier log_cases log_doctors log_nurses log_beds teaching urban, dist(hnormal)

* Display results
estimates store sfa_hn

* Extract variance parameters
* Stata parameterization: lnsig2v = ln(sigma_v^2), lnsig2u = ln(sigma_u^2)
display "Variance Parameters (Half-Normal):"
display "  ln(sigma_v^2) = " _b[/lnsig2v]
display "  ln(sigma_u^2) = " _b[/lnsig2u]
scalar sigma_v_sq_hn = exp(_b[/lnsig2v])
scalar sigma_u_sq_hn = exp(_b[/lnsig2u])
scalar sigma_v_hn = sqrt(sigma_v_sq_hn)
scalar sigma_u_hn = sqrt(sigma_u_sq_hn)
scalar sigma_sq_hn = sigma_v_sq_hn + sigma_u_sq_hn
scalar gamma_hn = sigma_u_sq_hn / sigma_sq_hn
scalar lambda_hn = sigma_u_hn / sigma_v_hn

display "  sigma_v^2 = " sigma_v_sq_hn
display "  sigma_u^2 = " sigma_u_sq_hn
display "  sigma_v   = " sigma_v_hn
display "  sigma_u   = " sigma_u_hn
display "  gamma     = " gamma_hn
display "  lambda    = " lambda_hn

scalar sfa_hn_ll = e(ll)
display "  Log-Likelihood = " sfa_hn_ll

* Technical Efficiency (Battese-Coelli estimator)
predict te_hn, te
summarize te_hn
display "Mean TE (Half-Normal): " r(mean)

* LR test for inefficiency
* H0: sigma_u^2 = 0 (OLS adequate)
scalar lr_stat_hn = -2 * (ols_ll - sfa_hn_ll)
display "LR Statistic: " lr_stat_hn
display "Critical value (5%, mixed chi-sq): 2.706"

* ------------------------------------------------------------------------------
* 4. SFA Model 2: Truncated Normal
* ------------------------------------------------------------------------------
display "============================================================"
display "SFA MODEL 2: TRUNCATED NORMAL"
display "============================================================"

frontier log_cases log_doctors log_nurses log_beds teaching urban, dist(tnormal)

estimates store sfa_tn

* Extract parameters
display "Variance Parameters (Truncated Normal):"
display "  ln(sigma_v^2) = " _b[/lnsig2v]
display "  ln(sigma_u^2) = " _b[/lnsig2u]
display "  mu             = " _b[/mu]
scalar sigma_v_sq_tn = exp(_b[/lnsig2v])
scalar sigma_u_sq_tn = exp(_b[/lnsig2u])
scalar sigma_v_tn = sqrt(sigma_v_sq_tn)
scalar sigma_u_tn = sqrt(sigma_u_sq_tn)
scalar gamma_tn = sigma_u_sq_tn / (sigma_v_sq_tn + sigma_u_sq_tn)

display "  sigma_v^2 = " sigma_v_sq_tn
display "  sigma_u^2 = " sigma_u_sq_tn
display "  sigma_v   = " sigma_v_tn
display "  sigma_u   = " sigma_u_tn
display "  gamma     = " gamma_tn

scalar sfa_tn_ll = e(ll)
display "  Log-Likelihood = " sfa_tn_ll

* Technical Efficiency
predict te_tn, te
summarize te_tn
display "Mean TE (Truncated Normal): " r(mean)

* ------------------------------------------------------------------------------
* 5. Model Comparison
* ------------------------------------------------------------------------------
display "============================================================"
display "MODEL COMPARISON"
display "============================================================"

estimates table sfa_hn sfa_tn, stats(ll aic bic N)

* Correlation between efficiency scores
correlate te_hn te_tn

* Export efficiency scores
export delimited hospital_id te_hn te_tn using ///
    "/home/guhaase/projetos/panelbox/examples/frontier/Stata/efficiency_scores_cross_section.csv", ///
    replace

display "============================================================"
display "CROSS-SECTION SFA VALIDATION (STATA) COMPLETE"
display "============================================================"

* ==============================================================================
* NOTES ON PARAMETERIZATION DIFFERENCES:
* ==============================================================================
* R frontier package uses:
*   sigmaSq = sigma_v^2 + sigma_u^2
*   gamma = sigma_u^2 / sigmaSq
*
* Stata frontier command uses:
*   lnsig2v = ln(sigma_v^2)
*   lnsig2u = ln(sigma_u^2)
*
* To compare:
*   R sigma_v^2 = Stata exp(lnsig2v)
*   R sigma_u^2 = Stata exp(lnsig2u)
*   R gamma = Stata exp(lnsig2u) / (exp(lnsig2v) + exp(lnsig2u))
*
* Both use the Battese-Coelli efficiency estimator by default.
* ==============================================================================
