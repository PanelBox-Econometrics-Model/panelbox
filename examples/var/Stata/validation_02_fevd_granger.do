*==============================================================================
* validation_02_fevd_granger.do
*
* Forecast Error Variance Decomposition (FEVD) and Granger Causality Tests
* using Stata's built-in var, irf, and vargranger commands.
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

* Filter to USA only
keep if country == "USA"

* Generate numeric time variable
gen year = real(substr(quarter, 1, 4))
gen qtr = real(substr(quarter, 7, 7))
gen time = yq(year, qtr)
format time %tq
tsset time

* --------------------------------------------------------------------------
* VAR(2) Estimation (same as validation_01)
* --------------------------------------------------------------------------
var gdp_growth inflation interest_rate, lags(1/2)

* --------------------------------------------------------------------------
* Forecast Error Variance Decomposition (FEVD)
* --------------------------------------------------------------------------
* Create IRF set (needed for FEVD computation)
irf create var2_fevd, set(var2_fevd_results) step(10) replace

* FEVD tables: proportion of forecast error variance explained by each shock
* FEVD for gdp_growth
irf table fevd, impulse(gdp_growth inflation interest_rate) ///
    response(gdp_growth) set(var2_fevd_results)

* FEVD for inflation
irf table fevd, impulse(gdp_growth inflation interest_rate) ///
    response(inflation) set(var2_fevd_results)

* FEVD for interest_rate
irf table fevd, impulse(gdp_growth inflation interest_rate) ///
    response(interest_rate) set(var2_fevd_results)

* Graph FEVD
irf graph fevd, set(var2_fevd_results) ///
    title("Forecast Error Variance Decomposition")

* --------------------------------------------------------------------------
* Granger Causality Tests
* --------------------------------------------------------------------------
* vargranger performs Granger causality Wald tests for each equation
* Tests whether lags of one variable help predict another
vargranger

* The output shows:
* - For each equation: which variables Granger-cause the dependent variable
* - F-statistic and p-value for each test
* - "ALL" row tests joint significance of all other variables

* --------------------------------------------------------------------------
* Pairwise Granger Causality (alternative approach)
* --------------------------------------------------------------------------
* Test if interest_rate Granger-causes gdp_growth
* (test that coefficients on lagged interest_rate in gdp_growth equation are 0)
test [gdp_growth]L.interest_rate [gdp_growth]L2.interest_rate

* Test if inflation Granger-causes interest_rate
test [interest_rate]L.inflation [interest_rate]L2.inflation

* Test if gdp_growth Granger-causes inflation
test [inflation]L.gdp_growth [inflation]L2.gdp_growth

* --------------------------------------------------------------------------
* Display summary
* --------------------------------------------------------------------------
display "FEVD and Granger causality analysis complete."
display "Endogenous variables: gdp_growth, inflation, interest_rate"
display "Lag order: 2"
display "FEVD horizon: 10 periods"
