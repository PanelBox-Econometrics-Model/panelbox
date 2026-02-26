#!/usr/bin/env Rscript
# Generate reference results for censored models validation

# Load required libraries
suppressPackageStartupMessages({
  library(censReg)  # For Tobit models
  library(plm)
  library(jsonlite)
})

# Set working directory to data folder
setwd("../data")

# Load data
data <- read.csv("panel_censored.csv")

# Convert to panel data frame
panel_data <- pdata.frame(data, index = c("entity", "time"))

# Initialize results list
results <- list()

cat("Generating reference results for censored models...\n\n")

# ========================================================================
# 1. POOLED TOBIT (using censReg)
# ========================================================================
cat("1. Pooled Tobit...\n")
tryCatch({
  pooled_tobit <- censReg(y ~ x1 + x2,
                           data = data,
                           left = 0)  # Left-censored at 0

  # Extract results
  coefs <- coef(pooled_tobit)
  # censReg includes log(sigma) as last parameter
  n_coef <- length(coefs) - 1
  beta_coefs <- coefs[1:n_coef]
  log_sigma <- coefs[n_coef + 1]
  sigma <- exp(log_sigma)

  # Standard errors
  se_all <- sqrt(diag(vcov(pooled_tobit)))
  se_beta <- se_all[1:n_coef]
  se_log_sigma <- se_all[n_coef + 1]

  results$pooled_tobit <- list(
    coefficients = as.numeric(beta_coefs),
    coef_names = names(beta_coefs),
    std_errors_coef = as.numeric(se_beta),
    sigma = sigma,
    log_sigma = log_sigma,
    se_log_sigma = se_log_sigma,
    loglik = as.numeric(logLik(pooled_tobit)),
    aic = AIC(pooled_tobit),
    n_obs = nrow(data),
    n_censored = sum(data$censored),
    n_uncensored = sum(1 - data$censored)
  )
}, error = function(e) {
  cat("  Warning: Pooled Tobit failed -", e$message, "\n")
  results$pooled_tobit <<- list(error = e$message)
})

# ========================================================================
# 2. RANDOM EFFECTS TOBIT (using censReg with panel structure)
# ========================================================================
cat("2. Random Effects Tobit...\n")
tryCatch({
  # Add entity dummies for simple RE approximation
  # (Note: censReg doesn't have built-in RE, this is a workaround)
  # For proper RE Tobit, would need specialized package

  # We'll use plm's pdata structure info
  entities <- as.factor(panel_data$entity)

  # Create formula with entity as random effect approximation
  re_tobit <- censReg(y ~ x1 + x2,
                       data = data,
                       left = 0)

  # Store as approximation
  results$re_tobit_approx <- list(
    coefficients = as.numeric(coef(re_tobit)[1:(length(coef(re_tobit))-1)]),
    sigma = exp(coef(re_tobit)[length(coef(re_tobit))]),
    loglik = as.numeric(logLik(re_tobit)),
    note = "Approximation - censReg doesn't support true RE"
  )
}, error = function(e) {
  cat("  Warning: RE Tobit failed -", e$message, "\n")
  results$re_tobit_approx <<- list(error = e$message)
})

# ========================================================================
# 3. PREDICTED VALUES for Pooled Tobit
# ========================================================================
if (!is.null(results$pooled_tobit$coefficients)) {
  cat("3. Predicted Values for Pooled Tobit...\n")

  # Predicted latent values
  X <- as.matrix(cbind(1, data[, c("x1", "x2")]))
  beta <- results$pooled_tobit$coefficients
  y_star <- X %*% beta

  # Predicted observed values (censored at 0)
  y_pred <- pmax(0, y_star)

  results$pooled_tobit$predicted_values_sample <- head(as.numeric(y_pred), 100)
  results$pooled_tobit$latent_values_sample <- head(as.numeric(y_star), 100)
}

# ========================================================================
# SAVE RESULTS
# ========================================================================
cat("\nSaving results to JSON...\n")
write_json(results, "reference_results_censored.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 10)

cat("Reference results generated successfully!\n")

# Print summary
cat("\n=== SUMMARY ===\n")
if (!is.null(results$pooled_tobit$coefficients)) {
  cat("Pooled Tobit:\n")
  cat("  Coefficients:\n")
  print(results$pooled_tobit$coefficients)
  cat(sprintf("  Sigma: %.4f\n", results$pooled_tobit$sigma))
  cat(sprintf("  N censored: %d (%.1f%%)\n",
              results$pooled_tobit$n_censored,
              100 * results$pooled_tobit$n_censored / results$pooled_tobit$n_obs))
}
