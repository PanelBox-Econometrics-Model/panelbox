#!/usr/bin/env Rscript
# ==============================================================================
# CUE-GMM Validation Script for R
# ==============================================================================
#
# This script validates PanelBox's CUE-GMM implementation against R's gmm package.
#
# Dependencies:
# - gmm (CRAN)
# - jsonlite (CRAN)
#
# Usage:
#   Rscript gmm_cue.R
#
# Output:
#   Creates JSON files in tests/validation/outputs/ with reference results
#
# ==============================================================================

library(gmm)
library(jsonlite)

# Set seed for reproducibility
set.seed(42)

# Create output directory
dir.create("../outputs", showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Test Case 1: Simple Instrumental Variables
# ==============================================================================

cat("Test Case 1: Simple IV Model\n")
cat(paste(rep("=", 70), collapse=""), "\n")

n <- 500

# Generate instruments
z1 <- rnorm(n, 0, 1)
z2 <- rnorm(n, 0, 1)

# Endogenous regressor
v <- rnorm(n, 0, 1)
x <- 0.5 + 0.8 * z1 + 0.6 * z2 + v

# Error with endogeneity
epsilon <- rnorm(n, 0, 1) + 0.5 * v

# True parameters: beta0 = 1.0, beta1 = 2.0
y <- 1.0 + 2.0 * x + epsilon

# Create data frame
data1 <- data.frame(y = y, x = x, z1 = z1, z2 = z2)

# Define moment conditions
# g(beta) = Z'(y - X'beta) = 0
# where X = [1, x], Z = [1, z1, z2]
moment_fn <- function(theta, data) {
  y <- data$y
  x <- cbind(1, data$x)
  z <- cbind(1, data$z1, data$z2)

  # Residuals
  residuals <- y - x %*% theta

  # Moment conditions: Z'ε
  moments <- z * as.vector(residuals)

  return(moments)
}

# Estimate CUE-GMM
cat("Estimating CUE-GMM...\n")
cue_fit <- gmm(
  g = moment_fn,
  x = data1,
  t0 = c(1, 2),  # Starting values
  type = "cue",  # Continuous updated
  vcov = "HAC",  # HAC-robust variance
  kernel = "Bartlett",
  prewhite = FALSE
)

cat("CUE-GMM Results:\n")
summ1 <- summary(cue_fit)
print(summ1)

# Extract results (include data for Python validation)
test1_results <- list(
  test_case = "simple_iv",
  description = "Simple IV model: y = beta0 + beta1*x + e, instruments: z1, z2",
  n = n,
  true_params = c(1.0, 2.0),
  coefficients = as.numeric(coef(cue_fit)),
  std_errors = sqrt(diag(vcov(cue_fit))),
  vcov_matrix = vcov(cue_fit),
  j_statistic = as.numeric(summ1$stest$test[1, 1]),  # Hansen J-statistic
  j_pvalue = as.numeric(summ1$stest$test[1, 2]),
  j_df = 1,  # 3 instruments - 2 parameters
  convergence = (cue_fit$algoInfo$convergence == 0),
  # Include data for Python validation
  data = list(
    y = y,
    x = x,
    z1 = z1,
    z2 = z2
  )
)

# Save to JSON
write_json(
  test1_results,
  "../outputs/gmm_cue_test1.json",
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 10
)

cat("✓ Test 1 complete. Results saved to gmm_cue_test1.json\n\n")

# ==============================================================================
# Test Case 2: Overidentified Model
# ==============================================================================

cat("Test Case 2: Overidentified Model\n")
cat(paste(rep("=", 70), collapse=""), "\n")

n <- 1000

# Generate 3 instruments for 1 endogenous regressor
z1 <- rnorm(n, 0, 1)
z2 <- rnorm(n, 0, 1)
z3 <- rnorm(n, 0, 1)

# Endogenous regressor
v <- rnorm(n, 0, 1)
x <- 0.5 + 0.7 * z1 + 0.5 * z2 + 0.6 * z3 + v

# Error
epsilon <- rnorm(n, 0, 1) + 0.4 * v

# True params: beta0 = 1.5, beta1 = -0.8
y <- 1.5 - 0.8 * x + epsilon

data2 <- data.frame(y = y, x = x, z1 = z1, z2 = z2, z3 = z3)

# Moment function with 3 instruments
moment_fn_overid <- function(theta, data) {
  y <- data$y
  x <- cbind(1, data$x)
  z <- cbind(1, data$z1, data$z2, data$z3)

  residuals <- y - x %*% theta
  moments <- z * as.vector(residuals)

  return(moments)
}

cat("Estimating CUE-GMM (overidentified)...\n")
cue_fit2 <- gmm(
  g = moment_fn_overid,
  x = data2,
  t0 = c(1.5, -0.8),
  type = "cue",
  vcov = "HAC",
  kernel = "Bartlett"
)

cat("CUE-GMM Results:\n")
summ2 <- summary(cue_fit2)
print(summ2)

test2_results <- list(
  test_case = "overidentified",
  description = "Overidentified: 4 instruments (const, z1, z2, z3), 2 parameters",
  n = n,
  true_params = c(1.5, -0.8),
  coefficients = as.numeric(coef(cue_fit2)),
  std_errors = sqrt(diag(vcov(cue_fit2))),
  vcov_matrix = vcov(cue_fit2),
  j_statistic = as.numeric(summ2$stest$test[1, 1]),
  j_pvalue = as.numeric(summ2$stest$test[1, 2]),
  j_df = 2,  # 4 instruments - 2 parameters
  convergence = (cue_fit2$algoInfo$convergence == 0),
  # Include data for Python validation
  data = list(
    y = y,
    x = x,
    z1 = z1,
    z2 = z2,
    z3 = z3
  )
)

write_json(
  test2_results,
  "../outputs/gmm_cue_test2.json",
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 10
)

cat("✓ Test 2 complete. Results saved to gmm_cue_test2.json\n\n")

# ==============================================================================
# Test Case 3: Dynamic Panel (Arellano-Bond setup)
# ==============================================================================
# # Note: Commented out due to issues with lag construction
# # Will be addressed in future validation
# #
# # cat("Test Case 3: Dynamic Panel Data\n")
# # cat(paste(rep("=", 70), collapse=""), "\n")
#
# # Note: This is a simplified version. Full Arellano-Bond would require
# # panel-specific transformations. Here we demonstrate CUE-GMM with lagged y.
#
# n_entities <- 100
# n_time <- 10
# n <- n_entities * n_time
#
# # Generate panel data
# entity_id <- rep(1:n_entities, each = n_time)
# time_id <- rep(1:n_time, n_entities)
#
# # Simulate AR(1) with fixed effects
# rho <- 0.6
# beta_x <- 0.3
#
# y <- numeric(n)
# x <- rnorm(n, 0, 1)
#
# for (i in 1:n_entities) {
#   idx <- which(entity_id == i)
#   alpha_i <- rnorm(1, 0, 1)  # Fixed effect
#
#   y[idx[1]] <- alpha_i + rnorm(1, 0, 0.5)
#
#   for (t in 2:n_time) {
#     y[idx[t]] <- rho * y[idx[t-1]] + beta_x * x[idx[t]] + alpha_i + rnorm(1, 0, 0.5)
#   }
# }
#
# # Create lagged variables
# y_lag1 <- c(NA, y[-n])
# y_lag2 <- c(NA, NA, y[-(n-1:n)])
#
# # Remove first observations for each entity (no lags available)
# # Fix: ensure proper recycling by computing for each index
# valid_idx <- rep(FALSE, length(y))
# for (i in 1:n_entities) {
#   idx <- which(entity_id == i)
#   if (length(idx) >= 3) {
#     # Skip first two observations (need 2 lags)
#     valid_idx[idx[3:length(idx)]] <- TRUE
#   }
# }
#
# data3 <- data.frame(
#   y = y[valid_idx],
#   y_lag1 = y_lag1[valid_idx],
#   y_lag2 = y_lag2[valid_idx],
#   x = x[valid_idx]
# )
#
# # Moment function for dynamic model
# # Instruments: y_lag2 (valid instrument for differenced equation)
# moment_fn_dynamic <- function(theta, data) {
#   # theta = [rho, beta_x]
#   y <- data$y
#   regressors <- cbind(data$y_lag1, data$x)
#   instruments <- cbind(1, data$y_lag2)
#
#   residuals <- y - regressors %*% theta
#   moments <- instruments * as.vector(residuals)
#
#   return(moments)
# }
#
# cat("Estimating CUE-GMM (dynamic panel)...\n")
# cue_fit3 <- gmm(
#   g = moment_fn_dynamic,
#   x = data3,
#   t0 = c(0.6, 0.3),
#   type = "cue",
#   vcov = "HAC"
# )
#
# cat("CUE-GMM Results:\n")
# print(summary(cue_fit3))
#
# test3_results <- list(
#   test_case = "dynamic_panel",
#   description = "Dynamic panel: y = rho*y_lag1 + beta*x + e, instrument: y_lag2",
#   n = nrow(data3),
#   true_params = c(rho, beta_x),
#   coefficients = as.numeric(coef(cue_fit3)),
#   std_errors = sqrt(diag(vcov(cue_fit3))),
#   vcov_matrix = vcov(cue_fit3),
#   j_statistic = cue_fit3$stest$test[1],
#   j_pvalue = cue_fit3$stest$test[2],
#   j_df = 0,  # Exactly identified
#   convergence = cue_fit3$conv.code == 0
# )
#
# write_json(
#   test3_results,
#   "../outputs/gmm_cue_test3.json",
#   pretty = TRUE,
#   auto_unbox = TRUE,
#   digits = 10
# )
#
# cat("✓ Test 3 complete. Results saved to gmm_cue_test3.json\n\n")
#
# # ==============================================================================
# # Summary
# # ==============================================================================

cat(paste(rep("=", 70), collapse=""), "\n")
cat("Validation Complete\n")
cat(paste(rep("=", 70), collapse=""), "\n")
cat("Generated reference outputs:\n")
cat("  - gmm_cue_test1.json (Simple IV)\n")
cat("  - gmm_cue_test2.json (Overidentified)\n")
cat("  - gmm_cue_test3.json (Dynamic Panel) - SKIPPED\n")
cat("\nUse these files to validate PanelBox CUE-GMM implementation.\n")
cat("Tolerance: coefficients ± 1e-4, J-statistic ± 1e-2\n")
