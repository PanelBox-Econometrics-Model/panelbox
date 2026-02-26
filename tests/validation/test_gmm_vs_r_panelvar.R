#!/usr/bin/env Rscript
# Validation script for Panel VAR GMM using R panelvar package
# This script generates test data, estimates using panelvar, and exports results

library(panelvar)
library(jsonlite)

set.seed(42)

# ============================================================================
# Generate Panel VAR data
# ============================================================================

n_entities <- 50
n_periods <- 15
K <- 2  # Number of variables

# True VAR(1) coefficient matrix (stable system)
A1_true <- matrix(c(
  0.5, 0.1,
  0.2, 0.6
), nrow = K, byrow = TRUE)

# Check stability
eigenvalues <- eigen(A1_true)$values
cat("True eigenvalues:", eigenvalues, "\n")
stopifnot(max(abs(eigenvalues)) < 1.0)

# Generate data
data_list <- list()
row_idx <- 1

for (entity in 1:n_entities) {
  # Initial values
  y_prev <- rnorm(K, sd = 0.5)

  for (t in 1:n_periods) {
    # VAR(1): y_t = A1 %*% y_{t-1} + epsilon_t
    epsilon <- rnorm(K, sd = 0.3)
    y <- A1_true %*% y_prev + epsilon

    data_list[[row_idx]] <- data.frame(
      entity = entity,
      time = t,
      y1 = y[1],
      y2 = y[2]
    )

    row_idx <- row_idx + 1
    y_prev <- y
  }
}

df <- do.call(rbind, data_list)

# Save data for Python
write.csv(df, "/tmp/pvar_gmm_test_data.csv", row.names = FALSE)
cat("Data saved to /tmp/pvar_gmm_test_data.csv\n")
cat("Data dimensions:", nrow(df), "rows,", ncol(df), "columns\n")

# ============================================================================
# Estimate Panel VAR using panelvar package
# ============================================================================

cat("\n" , "Estimating Panel VAR GMM in R...\n")

# Prepare data for panelvar
# panelvar expects panel data with entity and time indices
pdata <- df

# Estimate Panel VAR GMM with FOD transformation
# pvargmm() is the main function for GMM estimation in panelvar
tryCatch({
  pvar_result <- pvargmm(
    dependent_vars = c("y1", "y2"),
    lags = 1,
    transformation = "fod",  # Forward orthogonal deviations
    data = pdata,
    panel_identifier = c("entity", "time"),
    steps = c("twostep"),  # Two-step GMM
    system_instruments = TRUE,
    max_instr_dependent_vars = 3,
    min_instr_dependent_vars = 2L,
    collapse = TRUE  # Collapsed instruments to avoid proliferation
  )

  cat("Estimation successful!\n")

  # Extract results
  # Get coefficient matrices
  coef_names <- names(coef(pvar_result))
  coefs <- coef(pvar_result)

  # Organize coefficients into matrix format
  # For VAR(1) with K=2, we have 4 coefficients per lag
  # Coefficients are typically organized as [y1.L1.y1, y1.L1.y2, y2.L1.y1, y2.L1.y2]

  A1_estimated <- matrix(0, nrow = K, ncol = K)

  # Map coefficient names to matrix positions
  # This depends on panelvar's naming convention
  # Typically: eq1.L1.var1, eq1.L1.var2, eq2.L1.var1, eq2.L1.var2

  # Try to extract intelligently
  coef_vec <- as.numeric(coefs)

  # If we have 4 coefficients, arrange them as 2x2 matrix
  if (length(coef_vec) >= 4) {
    # panelvar typically stores by equation: all lags for eq1, then all lags for eq2
    # For VAR(1), K=2: [eq1: y1.L1, y2.L1] [eq2: y1.L1, y2.L1]
    A1_estimated[1, ] <- coef_vec[1:2]  # First equation
    A1_estimated[2, ] <- coef_vec[3:4]  # Second equation
  } else {
    warning("Unexpected number of coefficients")
  }

  # Get standard errors
  ses <- summary(pvar_result)$coefs[, "Std. Error"]
  se_matrix <- matrix(0, nrow = K, ncol = K)
  if (length(ses) >= 4) {
    se_matrix[1, ] <- ses[1:2]
    se_matrix[2, ] <- ses[3:4]
  }

  # Hansen J test
  hansen_test <- pvar_result$hansen

  # Number of instruments
  n_instruments <- pvar_result$n_instruments

  # Package results
  results <- list(
    A1 = A1_estimated,
    A1_true = A1_true,
    standard_errors = se_matrix,
    hansen_j_stat = hansen_test$statistic,
    hansen_j_pvalue = hansen_test$p.value,
    hansen_j_df = hansen_test$parameter,
    n_instruments = n_instruments,
    n_obs = nrow(pvar_result$model),
    n_entities = n_entities,
    n_periods = n_periods,
    transformation = "fod",
    gmm_step = "two-step",
    coefficient_names = coef_names,
    all_coefficients = as.numeric(coefs)
  )

  # Display results
  cat("\n=================================================================\n")
  cat("R panelvar GMM Estimation Results\n")
  cat("=================================================================\n")
  cat("True A1:\n")
  print(A1_true)
  cat("\nEstimated A1:\n")
  print(A1_estimated)
  cat("\nStandard Errors:\n")
  print(se_matrix)
  cat("\nHansen J test:\n")
  cat("  Statistic:", hansen_test$statistic, "\n")
  cat("  p-value:", hansen_test$p.value, "\n")
  cat("  DF:", hansen_test$parameter, "\n")
  cat("\nNumber of instruments:", n_instruments, "\n")
  cat("Number of observations:", nrow(pvar_result$model), "\n")
  cat("=================================================================\n")

  # Save results as JSON
  write_json(results, "/tmp/pvar_gmm_r_results.json", auto_unbox = TRUE, digits = 10)
  cat("\nResults saved to /tmp/pvar_gmm_r_results.json\n")

}, error = function(e) {
  cat("ERROR during estimation:\n")
  cat(conditionMessage(e), "\n")

  # Save error information
  error_results <- list(
    error = TRUE,
    message = conditionMessage(e)
  )
  write_json(error_results, "/tmp/pvar_gmm_r_results.json")
  quit(status = 1)
})

cat("\nValidation script completed successfully!\n")
