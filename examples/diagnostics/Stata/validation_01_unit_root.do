* =============================================================================
* Validation Script 01: Panel Unit Root Tests
* PanelBox vs Stata
*
* Tests: IPS, LLC, Fisher-ADF
* Dataset: Penn World Table (30 countries x 50 years)
* =============================================================================

clear all
set more off

* --- Data Loading ---
* Adjust path as needed for your system
local data_path "/home/guhaase/projetos/panelbox/examples/diagnostics/data/unit_root/penn_world_table.csv"

import delimited using "`data_path'", clear

describe
summarize

* --- Create log-transformed variables ---
gen log_gdp = ln(rgdpna)
gen log_capital = ln(rkna)
gen log_labor = ln(emp)
gen log_productivity = log_gdp - log_labor

* --- Encode country variable for xtset ---
encode countrycode, gen(country_id)
xtset country_id year
xtdescribe

* =============================================================================
* 1. IPS Test (Im-Pesaran-Shin)
* H0: All panels contain unit roots
* H1: Some panels are stationary
* =============================================================================

display "=== 1. IPS Tests ==="

* IPS test with trend for trended variables
foreach var in log_gdp log_capital log_labor log_productivity {
    display ""
    display "IPS test on `var' (with trend):"
    xtunitroot ips `var', trend lags(aic 4)
}

* IPS test without trend for mean-reverting variables
display ""
display "IPS test on labsh (without trend):"
xtunitroot ips labsh, lags(aic 4)

* =============================================================================
* 2. LLC Test (Levin-Lin-Chu)
* H0: All panels contain unit roots (common rho)
* H1: All panels are stationary
* =============================================================================

display ""
display "=== 2. LLC Tests ==="

foreach var in log_gdp log_capital log_labor log_productivity {
    display ""
    display "LLC test on `var' (with trend):"
    xtunitroot llc `var', trend lags(aic 4)
}

display ""
display "LLC test on labsh (without trend):"
xtunitroot llc labsh, lags(aic 4)

* =============================================================================
* 3. Hadri Test (stationarity under H0)
* H0: All panels are stationary
* H1: Some panels contain unit roots
* =============================================================================

display ""
display "=== 3. Hadri Tests ==="

foreach var in log_gdp log_capital log_labor log_productivity {
    display ""
    display "Hadri test on `var' (with trend):"
    xtunitroot hadri `var', trend
}

display ""
display "Hadri test on labsh (without trend):"
xtunitroot hadri labsh

* =============================================================================
* 4. Fisher-type ADF Test (combines individual ADF tests)
* =============================================================================

display ""
display "=== 4. Fisher-type ADF Tests ==="

foreach var in log_gdp log_capital {
    display ""
    display "Fisher ADF on `var' (with trend):"
    xtunitroot fisher `var', dfuller trend lags(4)
}

* =============================================================================
* 5. First-difference tests (should reject H0)
* =============================================================================

display ""
display "=== 5. IPS on First Differences (should be stationary) ==="

foreach var in log_gdp log_capital {
    display ""
    display "IPS test on D.`var':"
    xtunitroot ips D.`var', lags(aic 4)
}

display ""
display "=== Unit Root Tests Complete ==="
