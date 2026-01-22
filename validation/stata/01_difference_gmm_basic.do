/*==============================================================================
Validation Script 01: Difference GMM Basic
==============================================================================

Replicates Roodman (2009), Section 4, Example 1
Basic Difference GMM with Arellano-Bond dataset

Reference:
Roodman, D. (2009). "How to do xtabond2: An introduction to difference and
system GMM in Stata." Stata Journal, 9(1), 86-136.

Dataset:
Arellano and Bond (1991) - Employment equations for UK firms
==============================================================================*/

clear all
set more off
set seed 12345

* Load dataset
use "../data/abdata.dta", clear

* Describe data
describe
summarize n w k

* Set panel structure
xtset id year

/*------------------------------------------------------------------------------
Basic Difference GMM (Arellano-Bond 1991)
------------------------------------------------------------------------------*/

* Command replicating Roodman (2009), page 106
xtabond2 n L.n w k, ///
    gmm(L.n, lag(2 .)) ///
    iv(w k) ///
    robust ///
    small ///
    twostep

* Save results
estimates store diff_gmm_basic

* Display detailed output
estimates table diff_gmm_basic, star stats(N N_g j_p ar1p ar2p)

* Export results to text file
log using "../results/stata/01_difference_gmm_basic.txt", text replace

* Header
display "=========================================="
display "Difference GMM Basic - Detailed Results"
display "=========================================="
display ""

* Model specification
display "Model Specification:"
display "  Dependent variable: n (employment)"
display "  Lagged dependent: L.n"
display "  Exogenous: w (wages), k (capital)"
display "  GMM instruments: L.n, lags(2 .)"
display "  IV instruments: w, k"
display "  Estimation: Two-step with robust SE"
display "  Small sample correction: Yes"
display ""

* Basic statistics
display "Sample Information:"
display "  Number of observations: " e(N)
display "  Number of groups: " e(N_g)
display "  Min observations per group: " e(g_min)
display "  Avg observations per group: " e(g_avg)
display "  Max observations per group: " e(g_max)
display ""

* Instruments
display "Instrument Information:"
display "  Number of instruments: " e(j)
display "  Number of parameters: " e(rank)
display "  Instrument ratio: " e(j)/e(N_g)
display ""

* Coefficients table
display "Coefficients:"
ereturn list
matrix list e(b)
matrix list e(V)

* Specification tests
display ""
display "Specification Tests:"
display "  Hansen J-test:"
display "    Statistic: " e(hansen)
display "    P-value: " e(hansenp)
display "    Degrees of freedom: " e(j) - e(rank)
display ""

* AR tests
display "  Arellano-Bond AR tests:"
quietly estat abond, artests(2)
display "    AR(1) z-statistic: " r(ar1)
display "    AR(1) p-value: " r(ar1p)
display "    AR(2) z-statistic: " r(ar2)
display "    AR(2) p-value: " r(ar2p)
display ""

* Sargan test
display "  Sargan test:"
display "    Statistic: " e(sargan)
display "    P-value: " e(sarganp)
display "    Degrees of freedom: " e(j) - e(rank)
display ""

* Export coefficient matrix in parseable format
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
display "SARGAN_STAT " e(sargan)
display "SARGAN_P " e(sarganp)
display "N_OBS " e(N)
display "N_GROUPS " e(N_g)
display "N_INSTRUMENTS " e(j)

* Get AR test statistics
quietly estat abond, artests(2)
display "AR1_Z " r(ar1)
display "AR1_P " r(ar1p)
display "AR2_Z " r(ar2)
display "AR2_P " r(ar2p)

log close

display ""
display "Results saved to: ../results/stata/01_difference_gmm_basic.txt"
display ""

/* End of script */
