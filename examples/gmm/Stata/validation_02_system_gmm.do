***********************************************************************
* Validation 02 - System GMM (Blundell-Bond)
*
* Replicates PanelBox SystemGMM results using Stata's xtabond2 command.
* System GMM includes both the difference and level equations.
*
* Dataset: abdata (N=140 firms, T=9 years, unbalanced)
* Model:   n = L.n + w + k, GMM instruments for n
*
* NOTE: This script is for reference only. Not executed (no Stata license).
***********************************************************************

clear all
set more off

* --- Data Loading -----------------------------------------------------------
local datapath "/home/guhaase/projetos/panelbox/examples/gmm/data/abdata.csv"
import delimited using "`datapath'", clear

* Declare panel structure
xtset firm year

* Summary
describe
summarize n w k ys

***********************************************************************
* MODEL 1: System GMM One-Step (xtabond2)
*
* By default (without noleveleq), xtabond2 estimates System GMM.
* GMM-style instruments: lags of n (lag 2+ for diff eq, lag 1 for level eq)
* IV-style instruments: w, k (exogenous)
***********************************************************************

display _newline(2)
display "=== System GMM: One-Step ==="

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store sys_onestep

***********************************************************************
* MODEL 2: System GMM Two-Step
***********************************************************************

display _newline(2)
display "=== System GMM: Two-Step (Windmeijer-corrected) ==="

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    twostep ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store sys_twostep

***********************************************************************
* MODEL 3: System GMM Two-Step with Time Dummies
***********************************************************************

display _newline(2)
display "=== System GMM: Two-Step with Time Dummies ==="

* Generate year dummies
tab year, gen(yr)

xtabond2 n L.n w k yr2-yr8, ///
    gmm(n, lag(2 .)) ///
    iv(w k yr2-yr8) ///
    twostep ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store sys_twostep_td

***********************************************************************
* MODEL 4: System GMM Two-Step, Collapsed Instruments
***********************************************************************

display _newline(2)
display "=== System GMM: Two-Step, Collapsed ==="

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .) collapse) ///
    iv(w k) ///
    twostep ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store sys_twostep_col

***********************************************************************
* MODEL 5: System GMM One-Step, no time dummies
***********************************************************************

display _newline(2)
display "=== System GMM: One-Step, Individual Effects Only ==="

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    robust ///
    small ///
    nodiffsargan

estat abond, artests(2)
estat overid

estimates store sys_onestep_ind

***********************************************************************
* MODEL 6: System GMM Two-Step, no time dummies
***********************************************************************

display _newline(2)
display "=== System GMM: Two-Step, Individual Effects Only ==="

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    twostep ///
    robust ///
    small ///
    nodiffsargan

estat abond, artests(2)
estat overid

estimates store sys_twostep_ind

***********************************************************************
* MODEL 7: Extended Model with ys (sales)
***********************************************************************

display _newline(2)
display "=== System GMM: Two-Step with ys (sales) ==="

xtabond2 n L.n w k ys, ///
    gmm(n, lag(2 .)) ///
    iv(w k ys) ///
    twostep ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store sys_twostep_ys

***********************************************************************
* COMPARISON: Difference GMM vs System GMM
***********************************************************************

display _newline(2)
display "=== Comparison: Difference GMM vs System GMM ==="

* Run Difference GMM for comparison
xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

estimates store diff_compare

* Comparison table
estimates table diff_compare sys_twostep sys_twostep_col, ///
    stats(N N_g sarganp hansenp ar1p ar2p j) ///
    b(%9.4f) se(%9.4f) star

***********************************************************************
* DIFFERENCE-IN-HANSEN TEST
*
* xtabond2 automatically reports the Difference-in-Hansen test
* when System GMM is estimated. This tests the validity of the
* additional instruments used in the level equation.
***********************************************************************

display _newline(2)
display "=== Difference-in-Hansen Test (from System GMM) ==="
display "Check the 'Difference-in-Hansen' rows in the xtabond2 output above."
display "H0: Level instruments are valid"
display "Reject => Level instruments invalid, prefer Difference GMM"

***********************************************************************
* COMPARISON TABLE
***********************************************************************

display _newline(2)
display "=== Full Comparison Table ==="

estimates table sys_onestep sys_twostep sys_twostep_td ///
    sys_twostep_col sys_twostep_ys, ///
    stats(N N_g) b(%9.4f) se(%9.4f) star

***********************************************************************
* End of script
***********************************************************************
display _newline(2)
display "=== System GMM Validation Complete ==="
