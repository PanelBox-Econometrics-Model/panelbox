#!/usr/bin/env Rscript
# Generate reference results for count models validation

# Load required libraries
suppressPackageStartupMessages({
  library(pglm)
  library(plm)
  library(jsonlite)
  library(MASS)  # For negative binomial
})

# Set working directory to data folder
setwd("data")

# Load data
data <- read.csv("panel_count.csv")

# Convert to panel data frame
panel_data <- pdata.frame(data, index = c("entity", "time"))

# Initialize results list
results <- list()

cat("Generating reference results for count models...\n\n")

# ========================================================================
# 1. POOLED POISSON
# ========================================================================
cat("1. Pooled Poisson...\n")
pooled_poisson <- glm(y ~ x1 + x2,
                       data = data,
                       family = poisson(link = "log"))

results$pooled_poisson <- list(
  coefficients = as.numeric(coef(pooled_poisson)),
  coef_names = names(coef(pooled_poisson)),
  std_errors = as.numeric(sqrt(diag(vcov(pooled_poisson)))),
  loglik = as.numeric(logLik(pooled_poisson)),
  aic = as.numeric(AIC(pooled_poisson)),
  bic = as.numeric(BIC(pooled_poisson)),
  nobs = as.numeric(nobs(pooled_poisson)),
  deviance = as.numeric(deviance(pooled_poisson)),
  df_residual = as.numeric(df.residual(pooled_poisson))
)

# Check for overdispersion
dispersion_test <- results$pooled_poisson$deviance / results$pooled_poisson$df_residual
results$pooled_poisson$dispersion = dispersion_test

# ========================================================================
# 2. POISSON FIXED EFFECTS (using pglm)
# ========================================================================
cat("2. Poisson Fixed Effects...\n")
tryCatch({
  fe_poisson <- pglm(y ~ x1 + x2,
                      data = panel_data,
                      family = poisson(link = "log"),
                      model = "within",
                      method = "bfgs")

  results$fe_poisson <- list(
    coefficients = as.numeric(coef(fe_poisson)),
    coef_names = names(coef(fe_poisson)),
    std_errors = as.numeric(sqrt(diag(vcov(fe_poisson)))),
    loglik = as.numeric(logLik(fe_poisson))
  )
}, error = function(e) {
  cat("  Warning: FE Poisson failed -", e$message, "\n")
  results$fe_poisson <<- list(error = e$message)
})

# ========================================================================
# 3. RANDOM EFFECTS POISSON (using pglm)
# ========================================================================
cat("3. Random Effects Poisson...\n")
tryCatch({
  re_poisson <- pglm(y ~ x1 + x2,
                      data = panel_data,
                      family = poisson(link = "log"),
                      model = "random",
                      method = "bfgs")

  results$re_poisson <- list(
    coefficients = as.numeric(coef(re_poisson)),
    coef_names = names(coef(re_poisson)),
    std_errors = as.numeric(sqrt(diag(vcov(re_poisson)))),
    loglik = as.numeric(logLik(re_poisson)),
    sigma = re_poisson$sigma  # Random effect variance
  )
}, error = function(e) {
  cat("  Warning: RE Poisson failed -", e$message, "\n")
  results$re_poisson <<- list(error = e$message)
})

# ========================================================================
# 4. NEGATIVE BINOMIAL (using MASS)
# ========================================================================
cat("4. Negative Binomial...\n")
tryCatch({
  nb_model <- glm.nb(y ~ x1 + x2, data = data)

  results$negative_binomial <- list(
    coefficients = as.numeric(coef(nb_model)),
    coef_names = names(coef(nb_model)),
    std_errors = as.numeric(sqrt(diag(vcov(nb_model)))),
    loglik = as.numeric(logLik(nb_model)),
    aic = as.numeric(AIC(nb_model)),
    bic = as.numeric(BIC(nb_model)),
    theta = nb_model$theta,  # Dispersion parameter
    se_theta = nb_model$SE.theta  # SE of dispersion parameter
  )
}, error = function(e) {
  cat("  Warning: Negative Binomial failed -", e$message, "\n")
  results$negative_binomial <<- list(error = e$message)
})

# ========================================================================
# 5. PREDICTED COUNTS for Pooled Poisson
# ========================================================================
cat("5. Predicted Counts for Pooled Poisson...\n")
pred_counts <- predict(pooled_poisson, type = "response")
results$pooled_poisson$predicted_counts_sample <- head(as.numeric(pred_counts), 100)

# ========================================================================
# SAVE RESULTS
# ========================================================================
cat("\nSaving results to JSON...\n")
write_json(results, "reference_results_count.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 10)

cat("Reference results generated successfully!\n")

# Print summary
cat("\n=== SUMMARY ===\n")
cat("Pooled Poisson coefficients:\n")
print(results$pooled_poisson$coefficients)
cat(sprintf("Dispersion: %.3f (overdispersion if > 1)\n", results$pooled_poisson$dispersion))

if (!is.null(results$fe_poisson$coefficients)) {
  cat("\nFE Poisson coefficients:\n")
  print(results$fe_poisson$coefficients)
}
if (!is.null(results$re_poisson$coefficients)) {
  cat("\nRE Poisson coefficients:\n")
  print(results$re_poisson$coefficients)
}
if (!is.null(results$negative_binomial$coefficients)) {
  cat("\nNegative Binomial coefficients:\n")
  print(results$negative_binomial$coefficients)
  cat(sprintf("Theta (dispersion): %.3f\n", results$negative_binomial$theta))
}
