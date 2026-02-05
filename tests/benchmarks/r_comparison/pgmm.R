################################################################################
# GMM Estimation with R plm (pgmm function)
################################################################################
#
# Purpose: Reference implementation using R's plm package for comparison
#          with PanelBox's DifferenceGMM and SystemGMM estimators
#
# Dataset: Grunfeld (built-in to plm package)
#
# Output: Saves coefficients, standard errors, and statistics to text file
#
# Note: plm's pgmm implements Arellano-Bond (1991) difference GMM
#       and Blundell-Bond (1998) system GMM
#
################################################################################

# Load required packages
if (!require("plm")) install.packages("plm")
library(plm)

# Load Grunfeld data
data("Grunfeld", package = "plm")

# Display dataset info
cat("================================================================================\n")
cat("GMM Estimation - R plm package (pgmm)\n")
cat("================================================================================\n\n")
cat("Dataset: Grunfeld\n")
cat("Observations:", nrow(Grunfeld), "\n")
cat("Firms:", length(unique(Grunfeld$firm)), "\n")
cat("Years:", length(unique(Grunfeld$year)), "\n\n")

################################################################################
# DIFFERENCE GMM (Arellano-Bond 1991)
################################################################################

cat("================================================================================\n")
cat("DIFFERENCE GMM (Arellano-Bond 1991)\n")
cat("================================================================================\n\n")

cat("Estimating model: inv ~ lag(inv, 1) + value + capital\n")
cat("Transformation: First differences\n")
cat("GMM type: Two-step\n\n")

# Estimate Difference GMM
# Model: inv ~ lag(inv) + value + capital
# GMM instruments: lags 2+ of inv
# IV instruments: value, capital

diff_gmm <- pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99),
                 data = Grunfeld,
                 effect = "twoways",
                 model = "twosteps",
                 transformation = "d")

cat("Estimation Results:\n\n")
model_summary <- summary(diff_gmm)
print(model_summary)

# Extract coefficients
coefs <- coef(diff_gmm)
std_errors <- sqrt(diag(vcov(diff_gmm)))
tvalues <- coefs / std_errors
pvalues <- 2 * pnorm(-abs(tvalues))

# Create results dataframe
diff_results <- data.frame(
  variable = names(coefs),
  coefficient = as.numeric(coefs),
  std_error = as.numeric(std_errors),
  z_value = as.numeric(tvalues),
  p_value = as.numeric(pvalues),
  stringsAsFactors = FALSE
)

cat("\n\nCoefficients:\n")
print(diff_results, row.names = FALSE, digits = 8)

# Sargan test
sargan_test <- tryCatch({
  sargan(diff_gmm)
}, error = function(e) {
  cat("\nSargan test not available\n")
  NULL
})

if (!is.null(sargan_test)) {
  cat("\n\nSargan Test of Overidentifying Restrictions:\n")
  print(sargan_test)
}

# AR tests (if mtest is available)
ar_tests <- tryCatch({
  mtest(diff_gmm, order = 2)
}, error = function(e) {
  cat("\nAR tests not available\n")
  NULL
})

if (!is.null(ar_tests)) {
  cat("\n\nArellano-Bond Tests for Autocorrelation:\n")
  print(ar_tests)
}

cat("\n\nModel Info:\n")
cat(sprintf("N observations:      %d\n", nobs(diff_gmm)))
cat(sprintf("N instruments:       %d\n", length(diff_gmm$args$namest)))

################################################################################
# SYSTEM GMM (Blundell-Bond 1998)
################################################################################

cat("\n\n================================================================================\n")
cat("SYSTEM GMM (Blundell-Bond 1998)\n")
cat("================================================================================\n\n")

cat("Estimating model: inv ~ lag(inv, 1) + value + capital\n")
cat("Transformation: Differences + levels\n")
cat("GMM type: Two-step\n\n")

# Estimate System GMM
# Model: inv ~ lag(inv) + value + capital
# GMM instruments: lags 2+ of inv (diff eq) + lags 1 of diff(inv) (level eq)
# IV instruments: value, capital

sys_gmm <- pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99) + lag(inv, 1),
                data = Grunfeld,
                effect = "twoways",
                model = "twosteps",
                transformation = "ld")

cat("Estimation Results:\n\n")
model_summary <- summary(sys_gmm)
print(model_summary)

# Extract coefficients
coefs <- coef(sys_gmm)
std_errors <- sqrt(diag(vcov(sys_gmm)))
tvalues <- coefs / std_errors
pvalues <- 2 * pnorm(-abs(tvalues))

# Create results dataframe
sys_results <- data.frame(
  variable = names(coefs),
  coefficient = as.numeric(coefs),
  std_error = as.numeric(std_errors),
  z_value = as.numeric(tvalues),
  p_value = as.numeric(pvalues),
  stringsAsFactors = FALSE
)

cat("\n\nCoefficients:\n")
print(sys_results, row.names = FALSE, digits = 8)

# Sargan test
sargan_test <- tryCatch({
  sargan(sys_gmm)
}, error = function(e) {
  cat("\nSargan test not available\n")
  NULL
})

if (!is.null(sargan_test)) {
  cat("\n\nSargan Test of Overidentifying Restrictions:\n")
  print(sargan_test)
}

# AR tests
ar_tests <- tryCatch({
  mtest(sys_gmm, order = 2)
}, error = function(e) {
  cat("\nAR tests not available\n")
  NULL
})

if (!is.null(ar_tests)) {
  cat("\n\nArellano-Bond Tests for Autocorrelation:\n")
  print(ar_tests)
}

cat("\n\nModel Info:\n")
cat(sprintf("N observations:      %d\n", nobs(sys_gmm)))
cat(sprintf("N instruments:       %d\n", length(sys_gmm$args$namest)))

################################################################################
# SAVE RESULTS
################################################################################

output_file <- "pgmm_results.txt"
sink(output_file)

cat("# GMM Results - R plm (pgmm)\n")
cat("# Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("## Difference GMM Coefficients\n")
for (i in 1:nrow(diff_results)) {
  cat(sprintf("%s: coef=%.10f, se=%.10f, z=%.6f, p=%.8e\n",
              diff_results$variable[i],
              diff_results$coefficient[i],
              diff_results$std_error[i],
              diff_results$z_value[i],
              diff_results$p_value[i]))
}

cat("\n## System GMM Coefficients\n")
for (i in 1:nrow(sys_results)) {
  cat(sprintf("%s: coef=%.10f, se=%.10f, z=%.6f, p=%.8e\n",
              sys_results$variable[i],
              sys_results$coefficient[i],
              sys_results$std_error[i],
              sys_results$z_value[i],
              sys_results$p_value[i]))
}

cat("\n## Note\n")
cat("R plm's pgmm has different syntax and options compared to Stata's xtabond2\n")
cat("Direct comparison may require matching:\n")
cat("- Lag structure for instruments\n")
cat("- Transformation (differences vs orthogonal)\n")
cat("- One-step vs two-step estimation\n")
cat("- Robust standard errors\n\n")
cat("For detailed comparison, consult:\n")
cat("- plm documentation: ?pgmm\n")
cat("- Croissant & Millo (2008) 'Panel Data Econometrics in R'\n")

sink()

cat("\n\nResults saved to:", output_file, "\n")
cat("\n================================================================================\n")
cat("INSTRUCTIONS FOR PYTHON\n")
cat("================================================================================\n\n")
cat("1. Copy the coefficient and standard error values above\n")
cat("2. Update test_gmm_vs_plm.py with these reference values\n")
cat("3. Run the Python test to compare PanelBox vs R plm\n\n")
cat("Expected tolerance: < 1e-3 for GMM (algorithm differences)\n")
cat("\nNote: GMM results may differ more than static models due to:\n")
cat("  - Different lag specifications\n")
cat("  - Different weighting matrix calculations\n")
cat("  - Different initial values\n")
cat("  - Numerical precision in iterative optimization\n\n")
