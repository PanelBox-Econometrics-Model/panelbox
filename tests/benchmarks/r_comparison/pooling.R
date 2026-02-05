################################################################################
# Pooled OLS Estimation with R plm
################################################################################
#
# Purpose: Reference implementation using R's plm package for comparison
#          with PanelBox's PooledOLS estimator
#
# Dataset: Grunfeld (built-in to plm package)
#
# Output: Saves coefficients, standard errors, and statistics to text file
#
################################################################################

# Load required packages
if (!require("plm")) install.packages("plm")
library(plm)

# Load Grunfeld data
data("Grunfeld", package = "plm")

# Display dataset info
cat("================================================================================\n")
cat("Pooled OLS Estimation - R plm package\n")
cat("================================================================================\n\n")
cat("Dataset: Grunfeld\n")
cat("Observations:", nrow(Grunfeld), "\n")
cat("Firms:", length(unique(Grunfeld$firm)), "\n")
cat("Years:", length(unique(Grunfeld$year)), "\n\n")

# Estimate Pooled OLS model
# Model: invest ~ value + capital
cat("Estimating model: invest ~ value + capital\n\n")

pooled_model <- plm(inv ~ value + capital,
                    data = Grunfeld,
                    model = "pooling",
                    index = c("firm", "year"))

# Display results
cat("================================================================================\n")
cat("ESTIMATION RESULTS\n")
cat("================================================================================\n\n")

# Get summary
model_summary <- summary(pooled_model)
print(model_summary)

# Extract key statistics
coefs <- coef(pooled_model)
std_errors <- sqrt(diag(vcov(pooled_model)))
tvalues <- coefs / std_errors
pvalues <- 2 * pt(-abs(tvalues), df = pooled_model$df.residual)

# R-squared
r_squared <- r.squared(pooled_model)
adj_r_squared <- r.squared(pooled_model, model = "pooled")

# Degrees of freedom
n_obs <- nobs(pooled_model)
n_params <- length(coefs)
df_resid <- n_obs - n_params

# Residual standard error
rss <- sum(residuals(pooled_model)^2)
sigma <- sqrt(rss / df_resid)

# F-statistic
f_stat <- model_summary$fstatistic

cat("\n================================================================================\n")
cat("EXPORT RESULTS FOR PYTHON COMPARISON\n")
cat("================================================================================\n\n")

# Create output data frame
results <- data.frame(
  variable = names(coefs),
  coefficient = as.numeric(coefs),
  std_error = as.numeric(std_errors),
  t_value = as.numeric(tvalues),
  p_value = as.numeric(pvalues),
  stringsAsFactors = FALSE
)

# Print formatted results
cat("\nCoefficients:\n")
print(results, row.names = FALSE, digits = 8)

cat("\n\nModel Statistics:\n")
cat(sprintf("R-squared:           %.8f\n", r_squared))
cat(sprintf("Adjusted R-squared:  %.8f\n", adj_r_squared))
cat(sprintf("Residual Std Error:  %.8f\n", sigma))
cat(sprintf("N:                   %d\n", n_obs))
cat(sprintf("K:                   %d\n", n_params))
cat(sprintf("DF Residual:         %d\n", df_resid))
if (!is.null(f_stat)) {
  cat(sprintf("F-statistic:         %.8f\n", f_stat["value"]))
  cat(sprintf("F p-value:           %.8e\n", pf(f_stat["value"],
                                                 f_stat["numdf"],
                                                 f_stat["dendf"],
                                                 lower.tail = FALSE)))
}

# Save results to file
output_file <- "pooling_results.txt"
sink(output_file)

cat("# Pooled OLS Results - R plm\n")
cat("# Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("## Coefficients\n")
for (i in 1:nrow(results)) {
  cat(sprintf("%s: coef=%.10f, se=%.10f, t=%.6f, p=%.8e\n",
              results$variable[i],
              results$coefficient[i],
              results$std_error[i],
              results$t_value[i],
              results$p_value[i]))
}

cat("\n## Model Statistics\n")
cat(sprintf("r_squared: %.10f\n", r_squared))
cat(sprintf("adj_r_squared: %.10f\n", adj_r_squared))
cat(sprintf("sigma: %.10f\n", sigma))
cat(sprintf("n_obs: %d\n", n_obs))
cat(sprintf("n_params: %d\n", n_params))
cat(sprintf("df_resid: %d\n", df_resid))
if (!is.null(f_stat)) {
  cat(sprintf("f_statistic: %.10f\n", f_stat["value"]))
  cat(sprintf("f_pvalue: %.10e\n", pf(f_stat["value"],
                                       f_stat["numdf"],
                                       f_stat["dendf"],
                                       lower.tail = FALSE)))
}

sink()

cat("\n\nResults saved to:", output_file, "\n")
cat("\n================================================================================\n")
cat("INSTRUCTIONS FOR PYTHON\n")
cat("================================================================================\n\n")
cat("1. Copy the coefficient and standard error values above\n")
cat("2. Update test_pooled_vs_plm.py with these reference values\n")
cat("3. Run the Python test to compare PanelBox vs R plm\n\n")
cat("Expected tolerance: < 1e-6 for coefficients and standard errors\n\n")
