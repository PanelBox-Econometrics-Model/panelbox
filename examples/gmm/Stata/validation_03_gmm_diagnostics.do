***********************************************************************
* Validation 03 - GMM Diagnostics
*
* Comprehensive diagnostic tests for GMM estimators:
* - Instrument count (collapsed vs uncollapsed)
* - Sargan vs Hansen test
* - AR(1) and AR(2) serial correlation tests
* - Difference-in-Hansen test for level instruments
* - Windmeijer correction impact
* - One-step vs Two-step consistency
*
* Dataset: abdata (N=140 firms, T=9 years, unbalanced)
*
* NOTE: This script is for reference only. Not executed (no Stata license).
***********************************************************************

clear all
set more off

* --- Data Loading -----------------------------------------------------------
local datapath "/home/guhaase/projetos/panelbox/examples/gmm/data/abdata.csv"
import delimited using "`datapath'", clear

xtset firm year
summarize n w k ys

***********************************************************************
* PART 1: Instrument Count Comparison
***********************************************************************

display _newline(3)
display "============================================================="
display "PART 1: Instrument Count Comparison"
display "============================================================="

* 1a. Full instruments (uncollapsed)
display _newline
display "--- Difference GMM: Full instruments (uncollapsed) ---"

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

display "Number of instruments: " e(j)
display "Number of groups: " e(N_g)
display "Instrument/group ratio: " e(j)/e(N_g)

estimates store diag_full

* 1b. Collapsed instruments
display _newline
display "--- Difference GMM: Collapsed instruments ---"

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .) collapse) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

display "Number of instruments: " e(j)
display "Number of groups: " e(N_g)
display "Instrument/group ratio: " e(j)/e(N_g)

estimates store diag_collapsed

* 1c. Restricted lag depth (2:4)
display _newline
display "--- Difference GMM: Restricted instruments (lag 2:4) ---"

xtabond2 n L.n w k, ///
    gmm(n, lag(2 4)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

display "Number of instruments: " e(j)
display "Number of groups: " e(N_g)
display "Instrument/group ratio: " e(j)/e(N_g)

estimates store diag_restricted

* 1d. Minimal instruments (lag 2:2)
display _newline
display "--- Difference GMM: Minimal instruments (lag 2:2) ---"

xtabond2 n L.n w k, ///
    gmm(n, lag(2 2)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

display "Number of instruments: " e(j)
display "Number of groups: " e(N_g)
display "Instrument/group ratio: " e(j)/e(N_g)

estimates store diag_minimal

* Comparison of instrument strategies
display _newline
display "=== Instrument Strategy Comparison ==="
estimates table diag_full diag_collapsed diag_restricted diag_minimal, ///
    b(%9.4f) se(%9.4f) star

***********************************************************************
* PART 2: Sargan vs Hansen Test
***********************************************************************

display _newline(3)
display "============================================================="
display "PART 2: Sargan (One-Step) vs Hansen (Two-Step) Test"
display "============================================================="

* One-step: Reports Sargan test
display _newline
display "--- One-Step: Sargan Test ---"

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    robust ///
    small

* The Sargan test statistic is reported in the output
* e(sarganp) contains the p-value

display "Sargan p-value: " e(sarganp)

* Two-step: Reports Hansen J test
display _newline
display "--- Two-Step: Hansen J Test ---"

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

display "Hansen p-value: " e(hansenp)

***********************************************************************
* PART 3: AR(1) and AR(2) Tests
***********************************************************************

display _newline(3)
display "============================================================="
display "PART 3: Arellano-Bond Serial Correlation Tests"
display "============================================================="

display _newline
display "--- Difference GMM Two-Step ---"

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

estat abond, artests(3)

display _newline
display "Interpretation:"
display "  AR(1): Expected to be significant (reject H0 of no autocorrelation)"
display "  AR(2): Should NOT be significant (fail to reject H0)"
display "  AR(2) rejection => instruments invalid, model misspecified"

display _newline
display "--- System GMM Two-Step ---"

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    twostep ///
    robust ///
    small

estat abond, artests(3)

***********************************************************************
* PART 4: Difference-in-Hansen Test (System GMM)
***********************************************************************

display _newline(3)
display "============================================================="
display "PART 4: Difference-in-Hansen Test"
display "============================================================="

display _newline
display "The Difference-in-Hansen test examines whether the additional"
display "instruments used in the level equation are valid."
display ""
display "H0: The subset of instruments is exogenous."
display "If p-value < 0.05 => Level instruments are invalid."

* System GMM automatically reports Difference-in-Hansen
xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    twostep ///
    robust ///
    small

* The output includes:
*   "Difference-in-Hansen tests of exogeneity of instrument subsets:"
*   This tests the validity of the level-equation instruments

***********************************************************************
* PART 5: One-Step vs Two-Step Consistency
***********************************************************************

display _newline(3)
display "============================================================="
display "PART 5: One-Step vs Two-Step Coefficient Consistency"
display "============================================================="

* Difference GMM
display _newline
display "--- Difference GMM: One-Step vs Two-Step ---"

quietly xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    robust ///
    small
estimates store diff_1step

quietly xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small
estimates store diff_2step

estimates table diff_1step diff_2step, ///
    b(%9.6f) se(%9.6f) star ///
    title("Difference GMM: One-Step vs Two-Step")

* System GMM
display _newline
display "--- System GMM: One-Step vs Two-Step ---"

quietly xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    robust ///
    small
estimates store sys_1step

quietly xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    twostep ///
    robust ///
    small
estimates store sys_2step

estimates table sys_1step sys_2step, ///
    b(%9.6f) se(%9.6f) star ///
    title("System GMM: One-Step vs Two-Step")

***********************************************************************
* PART 6: Windmeijer Correction Impact
***********************************************************************

display _newline(3)
display "============================================================="
display "PART 6: Windmeijer Correction Impact"
display "============================================================="

display _newline
display "Comparing two-step SE with and without Windmeijer correction."
display "xtabond2 with 'robust' applies Windmeijer correction automatically."

* Two-step without robust (conventional SE)
quietly xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    small
estimates store twostep_conventional

* Two-step with robust (Windmeijer-corrected SE)
quietly xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small
estimates store twostep_windmeijer

estimates table twostep_conventional twostep_windmeijer, ///
    b(%9.6f) se(%9.6f) star ///
    title("Two-Step: Conventional SE vs Windmeijer-Corrected SE")

***********************************************************************
* PART 7: Extended Model with ys
***********************************************************************

display _newline(3)
display "============================================================="
display "PART 7: Extended Model: n ~ L.n + w + k + ys"
display "============================================================="

display _newline
display "--- Difference GMM Two-Step ---"

xtabond2 n L.n w k ys, ///
    gmm(n, lag(2 .)) ///
    iv(w k ys) ///
    noleveleq ///
    twostep ///
    robust ///
    small

estat abond, artests(2)

display _newline
display "--- System GMM Two-Step ---"

xtabond2 n L.n w k ys, ///
    gmm(n, lag(2 .)) ///
    iv(w k ys) ///
    twostep ///
    robust ///
    small

estat abond, artests(2)

***********************************************************************
* End of script
***********************************************************************
display _newline(3)
display "============================================================="
display "=== GMM Diagnostics Validation Complete ==="
display "============================================================="
