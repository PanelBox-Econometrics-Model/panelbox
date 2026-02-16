#!/usr/bin/env Rscript
# R script to validate LM tests using the splm package
#
# This script runs LM tests for spatial dependence on the test data
# and saves results in JSON format for comparison with Python implementation.
#
# Required packages: splm, spdep, plm, jsonlite

# Check if required packages are installed
required_packages <- c("splm", "spdep", "plm", "jsonlite")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(missing_packages) > 0) {
  cat("Missing required packages:", paste(missing_packages, collapse=", "), "\n")
  cat("Please install them using:\n")
  cat("install.packages(c(", paste(sprintf("'%s'", missing_packages), collapse=", "), "))\n")
  quit(status=1)
}

# Load required libraries
suppressPackageStartupMessages({
  library(splm)
  library(spdep)
  library(plm)
  library(jsonlite)
})

cat("=== R LM Tests Validation ===\n\n")

# Set working directory to script location
setwd(dirname(sys.frame(1)$ofile))

# Load test data
cat("Loading test data...\n")
data <- read.csv("spatial_test_data.csv")
W <- as.matrix(read.csv("spatial_weights.csv", header=FALSE))

cat("Data shape:", nrow(data), "x", ncol(data), "\n")
cat("W shape:", nrow(W), "x", ncol(W), "\n")
cat("W row sums (first 5):", head(rowSums(W), 5), "\n\n")

# Create spatial weights list
W_list <- mat2listw(W, style="W")

# Convert to pdata.frame
cat("Creating panel data frame...\n")
pdata <- pdata.frame(data, index=c("entity", "time"))

# Fit pooled OLS (for LM tests)
cat("Fitting pooled OLS model...\n")
pooled_ols <- plm(y ~ x1 + x2 + x3, data = pdata, model = "pooling")

cat("\nOLS Summary:\n")
print(summary(pooled_ols))

# Run LM tests using splm
cat("\n=== Running LM Tests ===\n")
cat("This may take a moment...\n\n")

# The slmtest function runs all tests at once
lm_results <- slmtest(pooled_ols, listw = W_list, test = "all")

# Print results
cat("LM Test Results:\n")
print(lm_results)

# Extract individual test results
# Note: slmtest returns a multi-test object
# We need to extract each component

results_list <- list(
  lm_lag_stat = as.numeric(lm_results$statistic["LM-lag"]),
  lm_lag_pvalue = as.numeric(lm_results$p.value["LM-lag"]),

  lm_error_stat = as.numeric(lm_results$statistic["LM-error"]),
  lm_error_pvalue = as.numeric(lm_results$p.value["LM-error"]),

  robust_lm_lag_stat = as.numeric(lm_results$statistic["RLM-lag"]),
  robust_lm_lag_pvalue = as.numeric(lm_results$p.value["RLM-lag"]),

  robust_lm_error_stat = as.numeric(lm_results$statistic["RLM-error"]),
  robust_lm_error_pvalue = as.numeric(lm_results$p.value["RLM-error"])
)

# Print extracted results
cat("\n=== Extracted Results for JSON ===\n")
print(results_list)

# Save to JSON for comparison with Python
output_file <- "r_lm_results.json"
write_json(results_list, output_file, pretty = TRUE, auto_unbox = TRUE)

cat("\n✓ Results saved to:", output_file, "\n")

# Also create a summary table
summary_df <- data.frame(
  Test = c("LM-Lag", "LM-Error", "Robust LM-Lag", "Robust LM-Error"),
  Statistic = c(
    results_list$lm_lag_stat,
    results_list$lm_error_stat,
    results_list$robust_lm_lag_stat,
    results_list$robust_lm_error_stat
  ),
  P_value = c(
    results_list$lm_lag_pvalue,
    results_list$lm_error_pvalue,
    results_list$robust_lm_lag_pvalue,
    results_list$robust_lm_error_pvalue
  ),
  Significant = c(
    results_list$lm_lag_pvalue < 0.05,
    results_list$lm_error_pvalue < 0.05,
    results_list$robust_lm_lag_pvalue < 0.05,
    results_list$robust_lm_error_pvalue < 0.05
  )
)

cat("\n=== Summary Table ===\n")
print(summary_df)

# Save summary as CSV
write.csv(summary_df, "r_lm_summary.csv", row.names = FALSE)
cat("\n✓ Summary saved to: r_lm_summary.csv\n")

cat("\n=== Validation Complete ===\n")
