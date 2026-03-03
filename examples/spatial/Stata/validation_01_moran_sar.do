* =============================================================================
* Validation Script 01: Moran's I and Spatial Autoregressive (SAR) Model
* =============================================================================
* Dataset: Columbus - 49 neighborhoods in Columbus, Ohio
* Model: CRIME ~ INC + HOVAL
* Weight matrix: Queen contiguity (row-standardized)
* Estimation: GS2SLS (Generalized Spatial Two-Stage Least Squares)
* =============================================================================

clear all
set more off

* ---- Load Columbus dataset ----
* The Columbus dataset is a classic spatial econometrics example.
* It should be available via Stata's spatial module or loaded from CSV.
* If using CSV exported from R:
* import delimited "/home/guhaase/projetos/panelbox/examples/spatial/data/columbus.csv", clear

* Alternative: use Stata's built-in example data or download from web
* For this script, we assume the Columbus dataset is available.
* The dataset has variables: CRIME, INC, HOVAL and polygon geometries.

* If loading from a shapefile:
* spshape2dta columbus, replace
* use columbus, clear

* ---- Create spatial weights matrix ----
* Stata requires the data to be in sp format with a spatial ID.
* First, set the spatial data:
* spset _ID

* Create queen contiguity weights matrix
* spatwmat using columbus_shp, name(W) standardize

* Alternative using spmatrix:
* spmatrix create contiguity W, replace normalize(row)

* ---- 1. OLS Baseline ----
display "=== OLS Baseline Model ==="
regress CRIME INC HOVAL

* Store OLS results
estimates store ols_baseline

* ---- 2. Moran's I Test ----
* Test for spatial autocorrelation in OLS residuals
display "=== Moran's I Test for Spatial Autocorrelation ==="

* Method 1: Using estat moran after spregress
* First need to run a spatial model or use spatgsa

* Method 2: Using spatgsa (user-written command)
* spatgsa CRIME INC HOVAL, weights(W) moran

* Method 3: After OLS, compute residuals and test
predict double resid_ols, residuals
* Then use Moran's I on residuals:
* spatwmat using W_matrix, name(W) standardize
* spatgsa resid_ols, weights(W) moran

* ---- 3. LM Tests for Spatial Dependence ----
display "=== LM Tests for Spatial Dependence ==="

* After OLS estimation, test for spatial lag and error dependence
* estat moran, errorlag(W)
* Or use the dedicated commands:
* spatdiag, weights(W)

* The LM tests include:
* - LM-lag: tests for spatial lag (SAR)
* - LM-error: tests for spatial error (SEM)
* - Robust LM-lag and Robust LM-error

* ---- 4. SAR Model (Spatial Lag) via GS2SLS ----
display "=== SAR Model (Spatial Lag) - GS2SLS ==="

* Using spregress (Stata 15+)
spregress CRIME INC HOVAL, gs2sls dvarlag(W)

* Display results
estimates store sar_gs2sls

* Extract rho (spatial autoregressive parameter)
display "Rho: " _b[/rho]

* ---- 5. SAR Model via ML (if available) ----
display "=== SAR Model (Spatial Lag) - ML ==="

* Using spregress with ML
spregress CRIME INC HOVAL, ml dvarlag(W)

estimates store sar_ml

* Display log-likelihood and AIC
display "Log-Likelihood: " e(ll)
display "AIC: " -2*e(ll) + 2*e(k)

* ---- 6. Export results ----
display "=== Export Results ==="

* Export SAR ML results to a matrix and then to CSV
matrix results = e(b)
matrix se = vecdiag(e(V))

* For a more structured export:
* esttab ols_baseline sar_gs2sls sar_ml using "results_01_moran_sar.csv", ///
*     cells(b se t p) csv replace

display "Done."
