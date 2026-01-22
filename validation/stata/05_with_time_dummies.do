/*==============================================================================
Validation Script 05: Difference GMM with Time Dummies
==============================================================================

Tests GMM with time fixed effects
Challenge: Time dummies increase dimensionality

==============================================================================*/

clear all
set more off
set seed 12345

use "../data/abdata.dta", clear
xtset id year

/*------------------------------------------------------------------------------
Difference GMM with Year Dummies
------------------------------------------------------------------------------*/

* Create year dummies
xi i.year, noomit

* Run GMM with time dummies
xtabond2 n L.n w k _Iyear*, ///
    gmm(L.n, lag(2 .) collapse) ///
    iv(w k _Iyear*) ///
    robust ///
    small ///
    twostep

estimates store diff_gmm_timedummies

log using "../results/stata/05_with_time_dummies.txt", text replace

display "=========================================="
display "Difference GMM with Time Dummies"
display "=========================================="
display ""

display "Model Specification:"
display "  Includes year fixed effects"
display "  Collapsed instruments to manage dimensionality"
display ""

display "Sample Information:"
display "  Observations: " e(N)
display "  Groups: " e(N_g)
display "  Instruments: " e(j)
display ""

matrix list e(b)

display ""
display "Key Tests:"
display "  Hansen J: stat=" e(hansen) " p=" e(hansenp)

quietly estat abond, artests(2)
display "  AR(2): z=" r(ar2) " p=" r(ar2p)

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
display "N_OBS " e(N)
display "N_GROUPS " e(N_g)
display "N_INSTRUMENTS " e(j)

quietly estat abond, artests(2)
display "AR2_Z " r(ar2)
display "AR2_P " r(ar2p)

log close

display "Results saved to: ../results/stata/05_with_time_dummies.txt"

/* End of script */
