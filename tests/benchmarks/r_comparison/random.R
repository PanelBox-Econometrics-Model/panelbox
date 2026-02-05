################################################################################
# Random Effects Estimation with R plm
################################################################################
#
# Purpose: Reference implementation using R's plm package for comparison
#          with PanelBox's RandomEffects estimator
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
cat("Random Effects Estimation - R plm package\n")
cat("================================================================================\n\n")
cat("Dataset: Grunfeld\n")
cat("Observations:", nrow(Grunfeld), "\n")
cat("Firms:", length(unique(Grunfeld$firm)), "\n")
cat("Years:", length(unique(Grunfeld$year)), "\n\n")

# Estimate Random Effects model
# Model: invest ~ value + capital
cat("Estimating model: invest ~ value + capital (random effects)\n\n")

re_model <- plm(inv ~ value + capital,
                data = Grunfeld,
                model = "random",
                effect = "individual",
                index = c("firm", "year"))

# Display results
cat("================================================================================\n")
cat("ESTIMATION RESULTS\n")
cat("================================================================================\n\n")

# Get summary
model_summary <- summary(re_model)
print(model_summary)

# Extract key statistics
coefs <- coef(re_model)
std_errors <- sqrt(diag(vcov(re_model)))
tvalues <- coefs / std_errors
pvalues <- 2 * pt(-abs(tvalues), df = re_model$df.residual)

# R-squared
r2_within <- r.squared(re_model)

# Degrees of freedom
n_obs <- nobs(re_model)
n_params <- length(coefs)
n_groups <- length(unique(index(re_model)[[1]]))
df_resid <- n_obs - n_params

# Residual standard error
rss <- sum(residuals(re_model)^2)
sigma <- sqrt(rss / df_resid)

# F-statistic
f_stat <- model_summary$fstatistic

# Extract variance components and theta
ercomp_result <- ercomp(re_model)
sigma_u <- sqrt(ercomp_result$sigma2$id)
sigma_e <- sqrt(ercomp_result$sigma2$idios)
rho <- ercomp_result$sigma2$id / (ercomp_result$sigma2$id + ercomp_result$sigma2$idios)

# Theta (transformation parameter)
# For balanced panels: theta = 1 - sqrt(sigma_e^2 / (sigma_e^2 + T * sigma_u^2))
T_bar <- nobs(re_model) / n_groups
theta <- 1 - sqrt(sigma_e^2 / (sigma_e^2 + T_bar * sigma_u^2))

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

cat("\n\nVariance Components:\n")
cat(sprintf("Sigma_u (between):   %.8f\n", sigma_u))
cat(sprintf("Sigma_e (within):    %.8f\n", sigma_e))
cat(sprintf("Rho:                 %.8f\n", rho))
cat(sprintf("Theta:               %.8f\n", theta))

cat("\n\nModel Statistics:\n")
cat(sprintf("R-squared:           %.8f\n", r2_within))
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

# Hausman test (RE vs FE)
cat("\n\nHausman Test (RE vs FE):\n")
fe_model <- plm(inv ~ value + capital,
                data = Grunfeld,
                model = "within",
                effect = "individual",
                index = c("firm", "year"))

hausman_test <- tryCatch({
  phtest(fe_model, re_model)
}, error = function(e) {
  cat("Hausman test failed:", e$message, "\n")
  NULL
})

if (!is.null(hausman_test)) {
  print(hausman_test)
  cat(sprintf("\nHausman statistic:   %.8f\n", hausman_test$statistic))
  cat(sprintf("Hausman p-value:     %.8e\n", hausman_test$p.value))
  cat(sprintf("Degrees of freedom:  %d\n", hausman_test$parameter))
}

# Save results to file
output_file <- "random_results.txt"
sink(output_file)

cat("# Random Effects Results - R plm\n")
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

cat("\n## Variance Components\n")
cat(sprintf("sigma_u: %.10f\n", sigma_u))
cat(sprintf("sigma_e: %.10f\n", sigma_e))
cat(sprintf("rho: %.10f\n", rho))
cat(sprintf("theta: %.10f\n", theta))

cat("\n## Model Statistics\n")
cat(sprintf("r2: %.10f\n", r2_within))
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

if (!is.null(hausman_test)) {
  cat(sprintf("\n## Hausman Test\n"))
  cat(sprintf("hausman_stat: %.10f\n", hausman_test$statistic))
  cat(sprintf("hausman_pvalue: %.10e\n", hausman_test$p.value))
  cat(sprintf("hausman_df: %d\n", hausman_test$parameter))
}

sink()

cat("\n\nResults saved to:", output_file, "\n")
cat("\n================================================================================\n")
cat("INSTRUCTIONS FOR PYTHON\n")
cat("================================================================================\n\n")
cat("1. Copy the coefficient and standard error values above\n")
cat("2. Update test_re_vs_plm.py with these reference values\n")
cat("3. Run the Python test to compare PanelBox vs R plm\n\n")
cat("Expected tolerance: < 1e-6 for coefficients and standard errors\n\n")
