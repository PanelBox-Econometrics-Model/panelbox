*==============================================================================
* validation_03_vecm.do
*
* Johansen Cointegration Test and VECM Estimation
* using Stata's vecrank and vec commands.
*
* Dataset: macro_panel.csv (filtered to USA)
* Endogenous variables: gdp_growth, inflation, interest_rate
* Lags in levels (K): 2
* Deterministic specification: constant in cointegrating equation
*==============================================================================

clear all
set more off

* --------------------------------------------------------------------------
* Load data
* --------------------------------------------------------------------------
import delimited using "/home/guhaase/projetos/panelbox/examples/var/data/macro_panel.csv", clear

* Filter to USA only
keep if country == "USA"

* Generate numeric time variable
gen year = real(substr(quarter, 1, 4))
gen qtr = real(substr(quarter, 7, 7))
gen time = yq(year, qtr)
format time %tq
tsset time

* --------------------------------------------------------------------------
* Johansen Cointegration Rank Test
* --------------------------------------------------------------------------
* vecrank performs the Johansen trace and max-eigenvalue tests
* lags(2): 2 lags in levels (equivalent to K=2 in ca.jo)
* The trace option requests the trace test statistic
* The max option requests the max-eigenvalue test statistic

* Trace test (default)
vecrank gdp_growth inflation interest_rate, lags(2) trend(constant)

* Max-eigenvalue test
vecrank gdp_growth inflation interest_rate, lags(2) trend(constant) max

* Note: vecrank reports trace statistics, eigenvalues, and critical values
* at 5% significance level. The selected rank is determined by the first
* non-rejection of H0: r <= r0.

* --------------------------------------------------------------------------
* VECM Estimation
* --------------------------------------------------------------------------
* Based on the cointegration rank test results, estimate VECM
* Assuming rank = 1 (one cointegrating relationship)
* lags(2): 2 lags in levels
* rank(1): one cointegrating equation
* trend(constant): restricted constant (constant in cointegrating equation)

vec gdp_growth inflation interest_rate, lags(2) rank(1) trend(constant)

* Display cointegrating equation
* The beta (cointegrating vector) and alpha (loading/adjustment) are shown
* in the output

* --------------------------------------------------------------------------
* Post-estimation diagnostics
* --------------------------------------------------------------------------
* Test for autocorrelation in VECM residuals
veclmar, mlag(4)

* Test for normality of VECM residuals
vecnorm

* Stability condition
vecstable, graph

* --------------------------------------------------------------------------
* VECM-based IRFs
* --------------------------------------------------------------------------
* Create IRF from VECM
irf create vecm_irf, set(vecm_irf_results) step(10) replace

* Plot orthogonalized IRFs from VECM
irf graph oirf, set(vecm_irf_results) level(95) ///
    title("VECM-based Orthogonalized IRFs")

* --------------------------------------------------------------------------
* Alternative: VECM with rank = 2 (if rank test suggests r=2)
* --------------------------------------------------------------------------
* Uncomment if the rank test indicates r=2
* vec gdp_growth inflation interest_rate, lags(2) rank(2) trend(constant)

* --------------------------------------------------------------------------
* Display summary
* --------------------------------------------------------------------------
display "Johansen cointegration test and VECM estimation complete."
display "Endogenous variables: gdp_growth, inflation, interest_rate"
display "Lags in levels (K): 2"
display "Deterministic: constant in cointegrating equation"
display "Cointegration rank tested: r = 0, 1, 2"
