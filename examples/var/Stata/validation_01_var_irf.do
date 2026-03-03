*==============================================================================
* validation_01_var_irf.do
*
* VAR(2) Estimation and Impulse Response Functions (IRFs)
* using Stata's built-in var and irf commands.
*
* Dataset: macro_panel.csv (filtered to USA)
* Endogenous variables: gdp_growth, inflation, interest_rate
* Lag order: 2
*==============================================================================

clear all
set more off

* --------------------------------------------------------------------------
* Load data
* --------------------------------------------------------------------------
import delimited using "/home/guhaase/projetos/panelbox/examples/var/data/macro_panel.csv", clear

* Filter to USA only (for time-series VAR)
keep if country == "USA"

* Generate a numeric time variable from quarter (e.g., 2010-Q1 -> 2010.00)
gen year = real(substr(quarter, 1, 4))
gen qtr = real(substr(quarter, 7, 7))
gen time = yq(year, qtr)
format time %tq
tsset time

* Verify data structure
describe
summarize gdp_growth inflation interest_rate

* --------------------------------------------------------------------------
* Lag order selection
* --------------------------------------------------------------------------
* varsoc estimates information criteria for different lag orders
varsoc gdp_growth inflation interest_rate, maxlag(8)

* --------------------------------------------------------------------------
* VAR(2) Estimation
* --------------------------------------------------------------------------
* Estimate VAR(2) with constant (default)
var gdp_growth inflation interest_rate, lags(1/2)

* Display results
estimates store var2

* Show coefficient table
var, coeflegend

* --------------------------------------------------------------------------
* Stability check
* --------------------------------------------------------------------------
* Check eigenvalue stability condition (all eigenvalues inside unit circle)
varstable, graph

* --------------------------------------------------------------------------
* Impulse Response Functions (Cholesky, orthogonalized)
* --------------------------------------------------------------------------
* Create IRF results using Cholesky decomposition
* Variable ordering: gdp_growth, inflation, interest_rate
* (ordering matters for Cholesky identification)

irf create var2_irf, set(var2_irf_results) step(10) replace

* Graph selected IRFs
* Response of gdp_growth to interest_rate shock
irf graph oirf, impulse(interest_rate) response(gdp_growth) ///
    set(var2_irf_results) level(95) ///
    title("Response of GDP Growth to Interest Rate Shock")

* Response of inflation to gdp_growth shock
irf graph oirf, impulse(gdp_growth) response(inflation) ///
    set(var2_irf_results) level(95) ///
    title("Response of Inflation to GDP Growth Shock")

* All orthogonalized IRFs
irf graph oirf, set(var2_irf_results) level(95)

* --------------------------------------------------------------------------
* Cumulative IRFs
* --------------------------------------------------------------------------
irf graph coirf, set(var2_irf_results) level(95) ///
    title("Cumulative Orthogonalized IRFs")

* --------------------------------------------------------------------------
* Export IRF table to CSV
* --------------------------------------------------------------------------
irf table oirf, set(var2_irf_results) level(95)

* --------------------------------------------------------------------------
* Display summary
* --------------------------------------------------------------------------
display "VAR(2) estimation and IRF analysis complete."
display "Endogenous variables: gdp_growth, inflation, interest_rate"
display "Lag order: 2"
display "IRF horizon: 10 periods"
display "Identification: Cholesky decomposition"
