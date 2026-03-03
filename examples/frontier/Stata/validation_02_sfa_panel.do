* ==============================================================================
* Validation Script: Stochastic Frontier Analysis - Panel Data (Stata)
* ==============================================================================
* Purpose: Reproduce PanelBox panel SFA results
* Notebook: examples/frontier/notebooks/02_panel_sfa.ipynb
* Data: bank_panel.csv (100 banks x 10 years)
* Models: Pitt-Lee (time-invariant), BC92 (time-varying with eta)
* NOTE: This script is for reference only. It has NOT been executed.
* ==============================================================================

clear all
set more off

* ------------------------------------------------------------------------------
* 1. Load Data
* ------------------------------------------------------------------------------
import delimited "/home/guhaase/projetos/panelbox/examples/frontier/data/bank_panel.csv", clear

describe
summarize log_loans log_labor log_capital log_deposits public_ownership npl_ratio

* Set panel structure
xtset bank_id year
xtdescribe

* ------------------------------------------------------------------------------
* 2. Pooled OLS Baseline
* ------------------------------------------------------------------------------
display "============================================================"
display "POOLED OLS BASELINE"
display "============================================================"

regress log_loans log_labor log_capital log_deposits

scalar ols_ll = e(ll)
display "OLS Log-Likelihood: " ols_ll

* Skewness of residuals
predict resid_ols, residuals
summarize resid_ols, detail
display "OLS Residual Skewness: " r(skewness)

* ------------------------------------------------------------------------------
* 3. Pitt-Lee Model (Time-Invariant Inefficiency)
* ------------------------------------------------------------------------------
display "============================================================"
display "PITT-LEE (1981) - TIME-INVARIANT INEFFICIENCY"
display "============================================================"

* xtfrontier with ti option: time-invariant inefficiency (Pitt-Lee)
xtfrontier log_loans log_labor log_capital log_deposits, ti

estimates store pl_model

* Extract variance parameters
* Stata xtfrontier ti parameterization: /mu, /lnsigma2, /ilgtgamma
display "Model Parameters (Pitt-Lee):"

* For ti model, Stata reports sigma2 and gamma directly
scalar sigma_sq_pl = e(sigma_u)^2 + e(sigma_v)^2
scalar sigma_u_pl = e(sigma_u)
scalar sigma_v_pl = e(sigma_v)
scalar gamma_pl = e(sigma_u)^2 / sigma_sq_pl

display "  sigma_u   = " sigma_u_pl
display "  sigma_v   = " sigma_v_pl
display "  gamma     = " gamma_pl
display "  Log-Lik   = " e(ll)

scalar pl_ll = e(ll)

* Technical Efficiency
predict te_pl, te
summarize te_pl
display "Mean TE (Pitt-Lee): " r(mean)

* Average TE by year
bysort year: summarize te_pl

* LR test: OLS vs Pitt-Lee (inefficiency presence)
scalar lr_pl = -2 * (ols_ll - pl_ll)
display "LR Test (OLS vs Pitt-Lee):"
display "  LR Statistic: " lr_pl
display "  Critical value (5%, mixed chi-sq): 2.706"

* ------------------------------------------------------------------------------
* 4. Battese-Coelli 1992 (Time-Varying Inefficiency)
* ------------------------------------------------------------------------------
display "============================================================"
display "BATTESE-COELLI (1992) - TIME-VARYING INEFFICIENCY"
display "============================================================"

* xtfrontier with tvd option: time-varying decay (BC92)
* u_it = u_i * exp[-eta * (t - T)]
xtfrontier log_loans log_labor log_capital log_deposits, tvd

estimates store bc92_model

* Extract parameters
scalar sigma_u_bc92 = e(sigma_u)
scalar sigma_v_bc92 = e(sigma_v)
scalar sigma_sq_bc92 = sigma_u_bc92^2 + sigma_v_bc92^2
scalar gamma_bc92 = sigma_u_bc92^2 / sigma_sq_bc92
scalar eta_bc92 = _b[/eta]
scalar bc92_ll = e(ll)

display "Model Parameters (BC92):"
display "  sigma_u = " sigma_u_bc92
display "  sigma_v = " sigma_v_bc92
display "  gamma   = " gamma_bc92
display "  eta     = " eta_bc92
display "  Log-Lik = " bc92_ll

* Interpret eta
if (eta_bc92 > 0) {
    display "  Interpretation: eta > 0 => Efficiency IMPROVES over time"
}
else if (eta_bc92 < 0) {
    display "  Interpretation: eta < 0 => Efficiency WORSENS over time"
}
else {
    display "  Interpretation: eta = 0 => Time-invariant"
}

* Technical Efficiency (time-varying)
predict te_bc92, te
summarize te_bc92

* Average TE by year
bysort year: summarize te_bc92

* ------------------------------------------------------------------------------
* 5. LR Test: Pitt-Lee vs BC92
* ------------------------------------------------------------------------------
display "============================================================"
display "LR TEST: Pitt-Lee vs BC92"
display "============================================================"
display "H0: eta = 0 (time-invariant, Pitt-Lee)"
display "H1: eta != 0 (time-varying, BC92)"

scalar lr_stat = -2 * (pl_ll - bc92_ll)
display "  LR Statistic: " lr_stat
display "  df = 1"
display "  P-value: " chi2tail(1, lr_stat)

if (chi2tail(1, lr_stat) < 0.05) {
    display "  Conclusion: Reject H0 - BC92 is preferred"
}
else {
    display "  Conclusion: Fail to reject H0 - Pitt-Lee is adequate"
}

* ------------------------------------------------------------------------------
* 6. BC95 with Inefficiency Determinants
* ------------------------------------------------------------------------------
display "============================================================"
display "BATTESE-COELLI (1995) - INEFFICIENCY DETERMINANTS"
display "============================================================"

* Stata does not have a built-in xtfrontier for BC95.
* Use sfpanel (user-written) or frontier with panel dummies.
* Alternative: Use cross-sectional frontier with z-variables on pooled data.

* Method 1: Using sfpanel (if installed)
* ssc install sfpanel
capture noisily {
    sfpanel log_loans log_labor log_capital log_deposits, ///
        model(bc95) ///
        emean(log_assets public_ownership npl_ratio) ///
        distribution(tnormal)

    estimates store bc95_model
    predict te_bc95, jlms
    summarize te_bc95
}

* Method 2: Fallback - use frontier command on pooled data with zu()
* This is a cross-sectional approximation for BC95
capture noisily {
    frontier log_loans log_labor log_capital log_deposits, ///
        dist(tnormal) ///
        cm(log_assets public_ownership npl_ratio)

    predict te_bc95_pool, te
    summarize te_bc95_pool
}

* ------------------------------------------------------------------------------
* 7. Model Comparison
* ------------------------------------------------------------------------------
display "============================================================"
display "MODEL COMPARISON"
display "============================================================"

estimates table pl_model bc92_model, stats(ll aic bic N)

* Correlation between efficiency scores
correlate te_pl te_bc92

* Export results
export delimited bank_id year te_pl te_bc92 using ///
    "/home/guhaase/projetos/panelbox/examples/frontier/Stata/efficiency_scores_panel.csv", ///
    replace

display "============================================================"
display "PANEL SFA VALIDATION (STATA) COMPLETE"
display "============================================================"

* ==============================================================================
* NOTES:
* ==============================================================================
* 1. Stata xtfrontier uses a different parameterization than R's frontier:
*    - Stata ti: Battese-Coelli (1992) time-invariant version
*    - Stata tvd: Battese-Coelli (1992) time-varying decay
*    - eta in Stata = eta in R frontier (same sign convention)
*
* 2. For BC95 (inefficiency determinants), Stata requires sfpanel (user-written)
*    or sfcross for cross-sectional models. Install with: ssc install sfpanel
*
* 3. Efficiency estimates from Stata predict, te are E[exp(-u)|e] (BC estimator),
*    same as R's efficiencies() function.
*
* 4. The variance decomposition is:
*    gamma = sigma_u^2 / (sigma_u^2 + sigma_v^2)
*    Same definition in both R and Stata.
* ==============================================================================
