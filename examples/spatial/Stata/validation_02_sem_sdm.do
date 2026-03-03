* =============================================================================
* Validation Script 02: Spatial Error Model (SEM) and Spatial Durbin Model (SDM)
* =============================================================================
* Dataset: Columbus - 49 neighborhoods in Columbus, Ohio
* Model: CRIME ~ INC + HOVAL
* Weight matrix: Queen contiguity (row-standardized)
* Estimation: GS2SLS and ML
* =============================================================================

clear all
set more off

* ---- Load Columbus dataset ----
* Assume Columbus dataset is loaded and spatial weights W are defined.
* See validation_01_moran_sar.do for data loading and weight matrix setup.

* If loading from shapefile:
* spshape2dta columbus, replace
* use columbus, clear
* spset _ID
* spmatrix create contiguity W, replace normalize(row)

* ---- 1. Spatial Error Model (SEM) via GS2SLS ----
display "=== Spatial Error Model (SEM) - GS2SLS ==="

* SEM: y = Xb + u, where u = lambda*W*u + e
* In Stata, use errorlag() to specify spatial error structure
spregress CRIME INC HOVAL, gs2sls errorlag(W)

estimates store sem_gs2sls

* Extract lambda (spatial error parameter)
display "Lambda: " _b[/lambda]

* ---- 2. Spatial Error Model (SEM) via ML ----
display "=== Spatial Error Model (SEM) - ML ==="

spregress CRIME INC HOVAL, ml errorlag(W)

estimates store sem_ml

display "Lambda (ML): " _b[/lambda]
display "Log-Likelihood: " e(ll)
display "AIC: " -2*e(ll) + 2*e(k)

* ---- 3. Spatial Durbin Model (SDM) via GS2SLS ----
display "=== Spatial Durbin Model (SDM) - GS2SLS ==="

* SDM: y = rho*W*y + Xb + W*X*theta + e
* In Stata, use dvarlag() for W*y and ivarlag() for W*X
spregress CRIME INC HOVAL, gs2sls dvarlag(W) ivarlag(W: INC HOVAL)

estimates store sdm_gs2sls

* Extract rho (spatial lag parameter)
display "Rho: " _b[/rho]

* Display coefficients for W*X terms (spatial lags of X)
display "W*INC coefficient: " _b[CRIME:W_INC]
display "W*HOVAL coefficient: " _b[CRIME:W_HOVAL]

* ---- 4. Spatial Durbin Model (SDM) via ML ----
display "=== Spatial Durbin Model (SDM) - ML ==="

spregress CRIME INC HOVAL, ml dvarlag(W) ivarlag(W: INC HOVAL)

estimates store sdm_ml

display "Rho (ML): " _b[/rho]
display "Log-Likelihood: " e(ll)
display "AIC: " -2*e(ll) + 2*e(k)

* ---- 5. Spatial Durbin Error Model (SDEM) ----
display "=== Spatial Durbin Error Model (SDEM) ==="

* SDEM: y = Xb + W*X*theta + u, where u = lambda*W*u + e
* In Stata, combine errorlag() with ivarlag()
spregress CRIME INC HOVAL, gs2sls errorlag(W) ivarlag(W: INC HOVAL)

estimates store sdem_gs2sls

display "Lambda (SDEM): " _b[/lambda]

* ---- 6. Model Comparison ----
display "=== Model Comparison ==="

* Compare all models side by side
estimates table sem_gs2sls sem_ml sdm_gs2sls sdm_ml sdem_gs2sls, ///
    stats(N ll aic bic) b(%9.4f) se(%9.4f)

* ---- 7. LR Test: SDM vs SAR ----
display "=== LR Test: SDM vs SAR ==="
* Test whether W*X terms are jointly significant
* This tests H0: theta = 0 (SDM reduces to SAR)
* lrtest sar_ml sdm_ml

* ---- 8. Export results ----
display "=== Export Results ==="

* Export all model results to CSV
* esttab sem_gs2sls sem_ml sdm_gs2sls sdm_ml sdem_gs2sls ///
*     using "results_02_sem_sdm.csv", ///
*     cells(b se t p) csv replace

display "Done."
