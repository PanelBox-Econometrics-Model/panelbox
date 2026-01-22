/*==============================================================================
Validation Script 04: System GMM with Collapsed Instruments
==============================================================================

System GMM with collapsed instruments
Expected: Fewer instruments than script 03, similar results

==============================================================================*/

clear all
set more off
set seed 12345

use "../data/abdata.dta", clear
xtset id year

/*------------------------------------------------------------------------------
System GMM with Collapsed Instruments
------------------------------------------------------------------------------*/

xtabond2 n L.n w k, ///
    gmm(L.n, lag(2 .) collapse) ///
    gmm(L.n, lag(1 1) diff eq(level) collapse) ///
    iv(w k) ///
    iv(w k, eq(level)) ///
    robust ///
    small ///
    twostep

estimates store sys_gmm_collapsed

log using "../results/stata/04_system_gmm_collapsed.txt", text replace

display "=========================================="
display "System GMM Collapsed - Detailed Results"
display "=========================================="
display ""

display "Model Specification:"
display "  Dependent variable: n"
display "  System GMM with COLLAPSED instruments"
display ""

display "Sample Information:"
display "  Observations: " e(N)
display "  Groups: " e(N_g)
display "  Instruments: " e(j)
display "  Instrument ratio: " e(j)/e(N_g)
display ""

matrix list e(b)
matrix list e(V)

display ""
display "Tests:"
display "  Hansen J: stat=" e(hansen) " p=" e(hansenp)
display "  Diff-Hansen: stat=" e(hansendf) " p=" e(hansendfp)

quietly estat abond, artests(2)
display "  AR(1): z=" r(ar1) " p=" r(ar1p)
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
display "AR1_Z " r(ar1)
display "AR1_P " r(ar1p)
display "AR2_Z " r(ar2)
display "AR2_P " r(ar2p)

log close

display "Results saved to: ../results/stata/04_system_gmm_collapsed.txt"

/* End of script */
