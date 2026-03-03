***********************************************************************
* Validation 01 - Difference GMM (Arellano-Bond)
*
* Replicates PanelBox DifferenceGMM results using Stata's xtabond
* and xtabond2 commands.
*
* Dataset: abdata (N=140 firms, T=9 years, unbalanced)
* Model:   n = L.n + w + k + year_dummies, GMM instruments for n
*
* NOTE: This script is for reference only. Not executed (no Stata license).
***********************************************************************

clear all
set more off

* --- Data Loading -----------------------------------------------------------
* Adjust the path below to your local environment
local datapath "/home/guhaase/projetos/panelbox/examples/gmm/data/abdata.csv"
import delimited using "`datapath'", clear

* Declare panel structure
xtset firm year

* Summary
describe
summarize n w k ys
xtdescribe

***********************************************************************
* MODEL 1: Difference GMM One-Step using xtabond (built-in Arellano-Bond)
***********************************************************************

* xtabond estimates one-step Arellano-Bond by default.
* The dependent variable lag is included automatically.
* Exogenous regressors: w, k
* Instruments: lags of n from t-2 onwards

display _newline(2)
display "=== xtabond: Difference GMM One-Step ==="

xtabond n w k, lags(1) vce(robust)
estat abond      /* Arellano-Bond test for AR(1) and AR(2) */
estat sargan     /* Sargan test of overidentifying restrictions */

* Store results
estimates store diff_onestep_xtabond

***********************************************************************
* MODEL 2: Difference GMM Two-Step using xtabond
***********************************************************************

display _newline(2)
display "=== xtabond: Difference GMM Two-Step ==="

xtabond n w k, lags(1) twostep vce(robust)
estat abond
estat sargan

estimates store diff_twostep_xtabond

***********************************************************************
* MODEL 3: Difference GMM One-Step using xtabond2 (Roodman)
*
* xtabond2 provides more control over instrument specification.
* noleveleq restricts to difference equation only (Arellano-Bond).
***********************************************************************

display _newline(2)
display "=== xtabond2: Difference GMM One-Step ==="

* GMM-style instruments for n (lags 2+)
* iv-style instruments for w and k (assumed exogenous)
xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    robust ///
    small

* Post-estimation tests
estat abond, artests(2)
estat overid

estimates store diff_onestep_xtabond2

***********************************************************************
* MODEL 4: Difference GMM Two-Step using xtabond2
***********************************************************************

display _newline(2)
display "=== xtabond2: Difference GMM Two-Step ==="

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store diff_twostep_xtabond2

***********************************************************************
* MODEL 5: Difference GMM Two-Step with Time Dummies (xtabond2)
***********************************************************************

display _newline(2)
display "=== xtabond2: Difference GMM Two-Step with Time Dummies ==="

* Generate year dummies
tab year, gen(yr)

xtabond2 n L.n w k yr2-yr8, ///
    gmm(n, lag(2 .)) ///
    iv(w k yr2-yr8) ///
    noleveleq ///
    twostep ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store diff_twostep_timedummies

***********************************************************************
* MODEL 6: Difference GMM with Collapsed Instruments
***********************************************************************

display _newline(2)
display "=== xtabond2: Difference GMM Two-Step, Collapsed ==="

xtabond2 n L.n w k, ///
    gmm(n, lag(2 .) collapse) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store diff_twostep_collapsed

***********************************************************************
* MODEL 7: Difference GMM with Restricted Lag Depth
***********************************************************************

display _newline(2)
display "=== xtabond2: Difference GMM Two-Step, Lag 2:4 ==="

xtabond2 n L.n w k, ///
    gmm(n, lag(2 4)) ///
    iv(w k) ///
    noleveleq ///
    twostep ///
    robust ///
    small

estat abond, artests(2)
estat overid

estimates store diff_twostep_lag24

***********************************************************************
* COMPARISON TABLE
***********************************************************************

display _newline(2)
display "=== Comparison of Difference GMM Estimates ==="

estimates table diff_onestep_xtabond diff_twostep_xtabond ///
    diff_onestep_xtabond2 diff_twostep_xtabond2 ///
    diff_twostep_timedummies diff_twostep_collapsed, ///
    stats(N N_g) b(%9.4f) se(%9.4f) star

***********************************************************************
* End of script
***********************************************************************
display _newline(2)
display "=== Difference GMM Validation Complete ==="
