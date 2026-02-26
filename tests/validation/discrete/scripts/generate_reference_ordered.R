#!/usr/bin/env Rscript
# Generate reference results for ordered choice models validation

# Load required libraries
suppressPackageStartupMessages({
  library(MASS)     # For polr
  library(jsonlite)
})

# Set working directory to data folder
setwd("../data")

# Load data
data <- read.csv("panel_ordered.csv")
data$y <- as.factor(data$y)  # polr requires factor outcome

# Initialize results list
results <- list()

cat("Generating reference results for ordered choice models...\n\n")

# ========================================================================
# 1. ORDERED LOGIT (using MASS::polr)
# ========================================================================
cat("1. Ordered Logit...\n")
tryCatch({
  ordered_logit <- polr(y ~ x1 + x2,
                        data = data,
                        method = "logistic",
                        Hess = TRUE)

  # Extract coefficients and thresholds
  coefs <- coef(ordered_logit)
  thresholds <- ordered_logit$zeta

  # Standard errors
  se_all <- sqrt(diag(vcov(ordered_logit)))
  n_coef <- length(coefs)
  se_coefs <- se_all[1:n_coef]
  se_thresholds <- se_all[(n_coef+1):length(se_all)]

  results$ordered_logit <- list(
    coefficients = as.numeric(coefs),
    coef_names = names(coefs),
    std_errors_coef = as.numeric(se_coefs),
    thresholds = as.numeric(thresholds),
    threshold_names = names(thresholds),
    std_errors_threshold = as.numeric(se_thresholds),
    loglik = as.numeric(logLik(ordered_logit)),
    aic = AIC(ordered_logit),
    deviance = deviance(ordered_logit),
    edf = ordered_logit$edf,  # effective degrees of freedom
    n_obs = nrow(data)
  )
}, error = function(e) {
  cat("  Warning: Ordered Logit failed -", e$message, "\n")
  results$ordered_logit <<- list(error = e$message)
})

# ========================================================================
# 2. ORDERED PROBIT (using MASS::polr)
# ========================================================================
cat("2. Ordered Probit...\n")
tryCatch({
  ordered_probit <- polr(y ~ x1 + x2,
                          data = data,
                          method = "probit",
                          Hess = TRUE)

  # Extract coefficients and thresholds
  coefs <- coef(ordered_probit)
  thresholds <- ordered_probit$zeta

  # Standard errors
  se_all <- sqrt(diag(vcov(ordered_probit)))
  n_coef <- length(coefs)
  se_coefs <- se_all[1:n_coef]
  se_thresholds <- se_all[(n_coef+1):length(se_all)]

  results$ordered_probit <- list(
    coefficients = as.numeric(coefs),
    coef_names = names(coefs),
    std_errors_coef = as.numeric(se_coefs),
    thresholds = as.numeric(thresholds),
    threshold_names = names(thresholds),
    std_errors_threshold = as.numeric(se_thresholds),
    loglik = as.numeric(logLik(ordered_probit)),
    aic = AIC(ordered_probit),
    deviance = deviance(ordered_probit),
    edf = ordered_probit$edf,
    n_obs = nrow(data)
  )
}, error = function(e) {
  cat("  Warning: Ordered Probit failed -", e$message, "\n")
  results$ordered_probit <<- list(error = e$message)
})

# ========================================================================
# 3. PREDICTED PROBABILITIES for Ordered Logit
# ========================================================================
if (!is.null(results$ordered_logit$coefficients)) {
  cat("3. Predicted Probabilities for Ordered Logit...\n")
  pred_probs <- predict(ordered_logit, type = "probs")
  # Save first 50 rows
  results$ordered_logit$predicted_probs_sample <- as.matrix(head(pred_probs, 50))

  # Also save predicted classes
  pred_class <- predict(ordered_logit, type = "class")
  results$ordered_logit$predicted_class_sample <- as.character(head(pred_class, 50))
}

# ========================================================================
# SAVE RESULTS
# ========================================================================
cat("\nSaving results to JSON...\n")
write_json(results, "reference_results_ordered.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 10)

cat("Reference results generated successfully!\n")

# Print summary
cat("\n=== SUMMARY ===\n")
if (!is.null(results$ordered_logit$coefficients)) {
  cat("Ordered Logit:\n")
  cat("  Coefficients:\n")
  print(results$ordered_logit$coefficients)
  cat("  Thresholds:\n")
  print(results$ordered_logit$thresholds)
}

if (!is.null(results$ordered_probit$coefficients)) {
  cat("\nOrdered Probit:\n")
  cat("  Coefficients:\n")
  print(results$ordered_probit$coefficients)
  cat("  Thresholds:\n")
  print(results$ordered_probit$thresholds)
}
