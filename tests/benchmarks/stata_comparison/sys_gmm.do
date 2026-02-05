*==============================================================================
* Stata Reference Script: System GMM (Blundell-Bond)
* For benchmark comparison with PanelBox
* Requires xtabond2 (ssc install xtabond2)
*==============================================================================

clear all
set more off

* Load Grunfeld data
use "https://www.stata-press.com/data/r18/grunfeld.dta", clear

* Declare panel structure
xtset company year

* System GMM: invest = rho*L.invest + b1*value + b2*capital + u_i + e_it
* Uses both difference and level equations
* Using xtabond2 (Roodman 2009)

display "================================================"
display "System GMM - Basic Specification"
display "================================================"

* Basic System GMM (one-step)
xtabond2 invest L.invest value capital, ///
    gmm(L.invest, lag(2 .)) ///
    iv(value capital) ///
    robust ///
    small

* Store results
matrix coef = e(b)
matrix se = e(V)

display ""
display "Coefficients:"
matrix list coef
display ""
display "Variance-Covariance Matrix:"
matrix list se
display ""
display "N: " e(N)
display "Groups: " e(N_g)
display "Instruments: " e(j)
display "Sargan test statistic: " e(sargan)
display "Sargan test p-value: " e(sarganp)
display "Hansen J test statistic: " e(hansen)
display "Hansen J test p-value: " e(hansenp)

display ""
display "================================================"
display "System GMM - Two-Step with Windmeijer"
display "================================================"

* Two-step System GMM with Windmeijer correction
xtabond2 invest L.invest value capital, ///
    gmm(L.invest, lag(2 .)) ///
    iv(value capital) ///
    robust ///
    small ///
    twostep

display ""
display "Two-Step Results:"
display "N: " e(N)
display "Groups: " e(N_g)
display "Instruments: " e(j)
display "Hansen J: " e(hansen) " (p=" e(hansenp) ")"
display "AR(1) test: p=" e(ar1p)
display "AR(2) test: p=" e(ar2p)

display ""
display "================================================"
display "System GMM - Collapsed Instruments"
display "================================================"

* With collapse option to avoid instrument proliferation
xtabond2 invest L.invest value capital, ///
    gmm(L.invest, lag(2 .) collapse) ///
    iv(value capital) ///
    robust ///
    small ///
    twostep

display ""
display "Collapsed Results:"
display "N: " e(N)
display "Groups: " e(N_g)
display "Instruments: " e(j)
display "Hansen J: " e(hansen) " (p=" e(hansenp) ")"

display ""
display "================================================"
display "System GMM - Alternative Specifications"
display "================================================"

* With orthogonal deviations (useful for unbalanced panels)
xtabond2 invest L.invest value capital, ///
    gmm(L.invest, lag(2 .) collapse orthogonal) ///
    iv(value capital) ///
    robust ///
    small ///
    twostep

display ""
display "Orthogonal Deviations Results:"
display "N: " e(N)
display "Groups: " e(N_g)
display "Instruments: " e(j)

display ""
display "========================"
display "Benchmark completed!"
display "========================"
