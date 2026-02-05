################################################################################
# Fixed Effects (Within) Estimation with R plm
################################################################################
#
# Purpose: Reference implementation using R's plm package for comparison
#          with PanelBox's FixedEffects estimator
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
cat("Fixed Effects (Within) Estimation - R plm package\n")
cat("================================================================================\n\n")
cat("Dataset: Grunfeld\n")
cat("Observations:", nrow(Grunfeld), "\n")
cat("Firms:", length(unique(Grunfeld$firm)), "\n")
cat("Years:", length(unique(Grunfeld$year)), "\n\n")

# Estimate Fixed Effects model
# Model: invest ~ value + capital
cat("Estimating model: invest ~ value + capital (within estimator)\n\n")

fe_model <- plm(inv ~ value + capital,
                data = Grunfeld,
                model = "within",
                effect = "individual",
                index = c("firm", "year"))

# Display results
cat("================================================================================\n")
cat("ESTIMATION RESULTS\n")
cat("================================================================================\n\n")

# Get summary
model_summary <- summary(fe_model)
print(model_summary)

# Extract key statistics
coefs <- coef(fe_model)
std_errors <- sqrt(diag(vcov(fe_model)))
tvalues <- coefs / std_errors
pvalues <- 2 * pt(-abs(tvalues), df = fe_model$df.residual)

# R-squared (within, between, overall)
r2_within <- r.squared(fe_model)
# For between and overall, need to calculate manually or use alternative
# plm provides these in summary

# Extract fixed effects
fixed_effects <- fixef(fe_model)

# Degrees of freedom
n_obs <- nobs(fe_model)
n_params <- length(coefs)
n_groups <- length(unique(index(fe_model)[[1]]))
df_resid <- n_obs - n_params - n_groups + 1

# Residual standard error
rss <- sum(residuals(fe_model)^2)
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

cat("\n\nFixed Effects (first 5):\n")
print(head(fixed_effects, 5), digits = 8)

cat("\n\nModel Statistics:\n")
cat(sprintf("R-squared (within):  %.8f\n", r2_within))
cat(sprintf("Residual Std Error:  %.8f\n", sigma))
cat(sprintf("N:                   %d\n", n_obs))
cat(sprintf("K:                   %d\n", n_params))
cat(sprintf("N Groups:            %d\n", n_groups))
cat(sprintf("DF Residual:         %d\n", df_resid))
if (!is.null(f_stat)) {
  cat(sprintf("F-statistic:         %.8f\n", f_stat["value"]))
  cat(sprintf("F p-value:           %.8e\n", pf(f_stat["value"],
                                                 f_stat["numdf"],
                                                 f_stat["dendf"],
                                                 lower.tail = FALSE)))
}

# Variance components (if available from ercomp)
tryCatch({
  ercomp_result <- ercomp(inv ~ value + capital,
                          data = Grunfeld,
                          index = c("firm", "year"))
  cat("\n\nVariance Components:\n")
  cat(sprintf("Sigma_u (between):   %.8f\n", sqrt(ercomp_result$sigma2$id)))
  cat(sprintf("Sigma_e (within):    %.8f\n", sqrt(ercomp_result$sigma2$idios)))
  cat(sprintf("Rho:                 %.8f\n", ercomp_result$sigma2$id /
              (ercomp_result$sigma2$id + ercomp_result$sigma2$idios)))
}, error = function(e) {
  cat("\nVariance components not available\n")
})

# Save results to file
output_file <- "within_results.txt"
sink(output_file)

cat("# Fixed Effects (Within) Results - R plm\n")
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

cat("\n## Fixed Effects\n")
for (i in 1:length(fixed_effects)) {
  cat(sprintf("%s: %.10f\n", names(fixed_effects)[i], fixed_effects[i]))
}

cat("\n## Model Statistics\n")
cat(sprintf("r2_within: %.10f\n", r2_within))
cat(sprintf("sigma: %.10f\n", sigma))
cat(sprintf("n_obs: %d\n", n_obs))
cat(sprintf("n_params: %d\n", n_params))
cat(sprintf("n_groups: %d\n", n_groups))
cat(sprintf("df_resid: %d\n", df_resid))
if (!is.null(f_stat)) {
  cat(sprintf("f_statistic: %.10f\n", f_stat["value"]))
  cat(sprintf("f_pvalue: %.10e\n", pf(f_stat["value"],
                                       f_stat["numdf"],
                                       f_stat["dendf"],
                                       lower.tail = FALSE)))
}

tryCatch({
  ercomp_result <- ercomp(inv ~ value + capital,
                          data = Grunfeld,
                          index = c("firm", "year"))
  cat(sprintf("\nsigma_u: %.10f\n", sqrt(ercomp_result$sigma2$id)))
  cat(sprintf("sigma_e: %.10f\n", sqrt(ercomp_result$sigma2$idios)))
  cat(sprintf("rho: %.10f\n", ercomp_result$sigma2$id /
              (ercomp_result$sigma2$id + ercomp_result$sigma2$idios)))
}, error = function(e) {})

sink()

cat("\n\nResults saved to:", output_file, "\n")
cat("\n================================================================================\n")
cat("INSTRUCTIONS FOR PYTHON\n")
cat("================================================================================\n\n")
cat("1. Copy the coefficient and standard error values above\n")
cat("2. Update test_fe_vs_plm.py with these reference values\n")
cat("3. Run the Python test to compare PanelBox vs R plm\n\n")
cat("Expected tolerance: < 1e-6 for coefficients and standard errors\n\n")
