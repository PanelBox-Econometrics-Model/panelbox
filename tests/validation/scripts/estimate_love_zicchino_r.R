#!/usr/bin/env Rscript
# Estimate Panel VAR on Love & Zicchino (2006) dataset using R panelvar package
# This provides reference outputs for validation

library(panelvar)
library(jsonlite)

# Load data
data <- read.csv("tests/validation/data/love_zicchino_2006.csv")

cat("Loading data...\n")
cat(sprintf("  N = %d unique firms\n", length(unique(data$firm_id))))
cat(sprintf("  T = %d to %d periods\n", min(table(data$firm_id)), max(table(data$firm_id))))
cat(sprintf("  Total obs = %d\n", nrow(data)))

# Prepare data for panelvar (requires pdata.frame)
library(plm)
pdata <- pdata.frame(data, index = c("firm_id", "year"))

# Variable names
vars <- c("sales", "inventory", "ar", "debt")

cat("\n--- Estimating Panel VAR(2) with OLS ---\n")
# OLS estimation (for comparison)
# Note: panelvar's pvargmm is the main function, but we can use it with transformation="demean" for OLS-like
pvar_ols <- pvargmm(
  dependent_vars = vars,
  lags = 2,
  transformation = "fd",  # First differences
  data = pdata,
  panel_identifier = c("firm_id", "year"),
  steps = c("onestep"),
  system_instruments = FALSE,
  max_instr_dependent_vars = 99,
  min_instr_dependent_vars = 2L,
  collapse = FALSE
)

cat("\n--- Estimating Panel VAR(2) with GMM (FOD transformation) ---\n")
# GMM estimation with Forward Orthogonal Deviations
pvar_gmm_fod <- pvargmm(
  dependent_vars = vars,
  lags = 2,
  transformation = "fod",
  data = pdata,
  panel_identifier = c("firm_id", "year"),
  steps = c("twostep"),
  system_instruments = FALSE,
  max_instr_dependent_vars = 3,
  min_instr_dependent_vars = 2L,
  collapse = TRUE
)

# Extract results
extract_results <- function(model, name) {
  cat(sprintf("\nExtracting results for %s...\n", name))

  # Coefficients
  coefs <- coef(model)

  # Standard errors (if available)
  se <- tryCatch({
    sqrt(diag(vcov(model)))
  }, error = function(e) {
    rep(NA, length(coefs))
  })

  # Residuals
  residuals_matrix <- residuals(model)

  # Get coefficient matrices A1 and A2
  # panelvar stores coefficients in specific structure
  # We need to reshape them to K×K matrices for each lag

  K <- length(vars)
  n_coefs <- length(coefs)
  n_lags <- 2

  # Extract A matrices
  A_matrices <- list()
  coef_idx <- 1

  for (lag in 1:n_lags) {
    A_lag <- matrix(0, nrow=K, ncol=K)
    for (i in 1:K) {  # equation
      for (j in 1:K) {  # variable
        if (coef_idx <= length(coefs)) {
          A_lag[i, j] <- coefs[coef_idx]
          coef_idx <- coef_idx + 1
        }
      }
    }
    A_matrices[[lag]] <- A_lag
  }

  results <- list(
    model_name = name,
    coefficients = coefs,
    std_errors = se,
    A_matrices = A_matrices,
    n_obs = nrow(residuals_matrix),
    K = K,
    lags = n_lags
  )

  # Add GMM-specific diagnostics if available
  if (inherits(model, "pvargmm")) {
    results$hansen_j <- tryCatch(model$hansen$p.value, error = function(e) NA)
    results$ar1_pvalue <- tryCatch(model$ar1$p.value, error = function(e) NA)
    results$ar2_pvalue <- tryCatch(model$ar2$p.value, error = function(e) NA)
  }

  return(results)
}

# Extract all results
results_ols <- extract_results(pvar_ols, "OLS_FD")
results_gmm_fod <- extract_results(pvar_gmm_fod, "GMM_FOD")

# Combine results
all_results <- list(
  dataset = "love_zicchino_2006",
  variables = vars,
  ols_fd = results_ols,
  gmm_fod = results_gmm_fod,
  metadata = list(
    r_version = as.character(getRversion()),
    panelvar_version = as.character(packageVersion("panelvar")),
    date = Sys.Date()
  )
)

# Save results as JSON
write_json(
  all_results,
  "tests/validation/reference_outputs/love_zicchino_r_results.json",
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 10
)

cat("\n✓ R reference results saved successfully\n")
cat("  Output: tests/validation/reference_outputs/love_zicchino_r_results.json\n")

# Print summary
cat("\n=== GMM FOD Results Summary ===\n")
print(summary(pvar_gmm_fod))

cat("\n=== A1 Matrix (GMM FOD) ===\n")
print(results_gmm_fod$A_matrices[[1]])

cat("\n=== A2 Matrix (GMM FOD) ===\n")
print(results_gmm_fod$A_matrices[[2]])

if (!is.na(results_gmm_fod$hansen_j)) {
  cat(sprintf("\nHansen J p-value: %.4f\n", results_gmm_fod$hansen_j))
}
if (!is.na(results_gmm_fod$ar1_pvalue)) {
  cat(sprintf("AR(1) test p-value: %.4f\n", results_gmm_fod$ar1_pvalue))
}
if (!is.na(results_gmm_fod$ar2_pvalue)) {
  cat(sprintf("AR(2) test p-value: %.4f\n", results_gmm_fod$ar2_pvalue))
}
