/*==============================================================================
Validation Script 02: Difference GMM with Collapsed Instruments
==============================================================================

Replicates Roodman (2009), Section 4
Difference GMM with collapsed instruments to avoid proliferation

Reference:
Roodman, D. (2009). Stata Journal, 9(1), 86-136.

Key difference from 01: Uses collapse option
Expected: Fewer instruments, slightly larger SEs, similar coefficients
==============================================================================*/

clear all
set more off
set seed 12345

* Load dataset
use "../data/abdata.dta", clear

* Set panel structure
xtset id year

/*------------------------------------------------------------------------------
Difference GMM with Collapsed Instruments
------------------------------------------------------------------------------*/

* Command with collapse option
xtabond2 n L.n w k, ///
    gmm(L.n, lag(2 .) collapse) ///
    iv(w k) ///
    robust ///
    small ///
    twostep

* Save results
estimates store diff_gmm_collapsed

* Export results
log using "../results/stata/02_difference_gmm_collapsed.txt", text replace

display "=========================================="
display "Difference GMM Collapsed - Detailed Results"
display "=========================================="
display ""

display "Model Specification:"
display "  Dependent variable: n (employment)"
display "  Lagged dependent: L.n"
display "  Exogenous: w (wages), k (capital)"
display "  GMM instruments: L.n, lags(2 .) COLLAPSED"
display "  IV instruments: w, k"
display "  Estimation: Two-step with robust SE"
display ""

display "Sample Information:"
display "  Number of observations: " e(N)
display "  Number of groups: " e(N_g)
display ""

display "Instrument Information:"
display "  Number of instruments: " e(j)
display "  Number of parameters: " e(rank)
display "  Instrument ratio: " e(j)/e(N_g)
display ""

* Coefficients
matrix list e(b)
matrix list e(V)

* Tests
display ""
display "Specification Tests:"
display "  Hansen J: stat=" e(hansen) " p=" e(hansenp)

quietly estat abond, artests(2)
display "  AR(1): z=" r(ar1) " p=" r(ar1p)
display "  AR(2): z=" r(ar2) " p=" r(ar2p)

* Parseable output
display ""
display "PARSEABLE OUTPUT:"
display "COEF_L_n " _b[L.n]
display "SE_L_n " _se[L.n]
display "COEF_w " _b[w]
display "SE_w " _se[w]
display "COEF_k " _b[k]
display "SE_k " _se[k]
display "HANSEN_STAT " e(hansen)
display "HANSEN_P " e(hansenp)
display "HANSEN_DF " e(j) - e(rank)
display "N_OBS " e(N)
display "N_GROUPS " e(N_g)
display "N_INSTRUMENTS " e(j)

quietly estat abond, artests(2)
display "AR1_Z " r(ar1)
display "AR1_P " r(ar1p)
display "AR2_Z " r(ar2)
display "AR2_P " r(ar2p)

log close

display "Results saved to: ../results/stata/02_difference_gmm_collapsed.txt"

/* End of script */
