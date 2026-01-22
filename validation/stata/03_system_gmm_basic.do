/*==============================================================================
Validation Script 03: System GMM Basic
==============================================================================

Replicates Roodman (2009), Section 5, Example 1
Basic System GMM (Blundell-Bond 1998)

Reference:
Roodman, D. (2009). Stata Journal, 9(1), 86-136, page 120.

System GMM combines:
- Difference equations with level instruments
- Level equations with differenced instruments

Expected: More efficient than Difference GMM for persistent series
==============================================================================*/

clear all
set more off
set seed 12345

use "../data/abdata.dta", clear
xtset id year

/*------------------------------------------------------------------------------
System GMM (Blundell-Bond 1998)
------------------------------------------------------------------------------*/

* System GMM command
xtabond2 n L.n w k, ///
    gmm(L.n, lag(2 .)) ///
    gmm(L.n, lag(1 1) diff eq(level)) ///
    iv(w k) ///
    iv(w k, eq(level)) ///
    robust ///
    small ///
    twostep

estimates store sys_gmm_basic

log using "../results/stata/03_system_gmm_basic.txt", text replace

display "=========================================="
display "System GMM Basic - Detailed Results"
display "=========================================="
display ""

display "Model Specification:"
display "  Dependent variable: n (employment)"
display "  Lagged dependent: L.n"
display "  Exogenous: w (wages), k (capital)"
display ""
display "  DIFFERENCE EQUATIONS:"
display "    GMM instruments: levels of L.n, lags(2 .)"
display "    IV instruments: w, k"
display ""
display "  LEVEL EQUATIONS:"
display "    GMM instruments: differences of L.n, lag(1 1)"
display "    IV instruments: w, k"
display ""
display "  Estimation: Two-step System GMM with robust SE"
display ""

display "Sample Information:"
display "  Number of observations: " e(N)
display "  Number of groups: " e(N_g)
display ""

display "Instrument Information:"
display "  Total instruments: " e(j)
display "  Number of parameters: " e(rank)
display "  Instrument ratio: " e(j)/e(N_g)
display ""

* Coefficients
matrix list e(b)
matrix list e(V)

* Tests
display ""
display "Specification Tests:"
display ""
display "  Hansen J-test (full system):"
display "    Statistic: " e(hansen)
display "    P-value: " e(hansenp)
display "    DF: " e(j) - e(rank)
display ""

* Difference-in-Hansen test for level instruments
display "  Difference-in-Hansen test:"
display "    (tests validity of additional level moments)"
display "    Statistic: " e(hansendf)
display "    P-value: " e(hansendfp)
display ""

* AR tests
quietly estat abond, artests(2)
display "  Arellano-Bond AR tests:"
display "    AR(1): z=" r(ar1) " p=" r(ar1p)
display "    AR(2): z=" r(ar2) " p=" r(ar2p)
display ""

* Parseable output
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
display "DIFF_HANSEN_STAT " e(hansendf)
display "DIFF_HANSEN_P " e(hansendfp)
display "N_OBS " e(N)
display "N_GROUPS " e(N_g)
display "N_INSTRUMENTS " e(j)

quietly estat abond, artests(2)
display "AR1_Z " r(ar1)
display "AR1_P " r(ar1p)
display "AR2_Z " r(ar2)
display "AR2_P " r(ar2p)

log close

display "Results saved to: ../results/stata/03_system_gmm_basic.txt"

/* End of script */
