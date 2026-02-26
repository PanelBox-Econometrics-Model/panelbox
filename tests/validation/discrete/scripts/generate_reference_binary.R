#!/usr/bin/env Rscript
# Generate reference results for binary choice models validation

# Load required libraries
suppressPackageStartupMessages({
  library(pglm)
  library(plm)
  library(jsonlite)
  library(margins)
})

# Set working directory to data folder
setwd("../data")

# Load data
data <- read.csv("panel_binary.csv")

# Convert to panel data frame
panel_data <- pdata.frame(data, index = c("entity", "time"))

# Initialize results list
results <- list()

cat("Generating reference results for binary choice models...\n\n")

# ========================================================================
# 1. POOLED LOGIT
# ========================================================================
cat("1. Pooled Logit...\n")
pooled_logit <- glm(y ~ x1 + x2,
                     data = data,
                     family = binomial(link = "logit"))

results$pooled_logit <- list(
  coefficients = as.numeric(coef(pooled_logit)),
  coef_names = names(coef(pooled_logit)),
  std_errors = as.numeric(sqrt(diag(vcov(pooled_logit)))),
  loglik = as.numeric(logLik(pooled_logit)),
  aic = AIC(pooled_logit),
  bic = BIC(pooled_logit),
  nobs = nobs(pooled_logit)
)

# ========================================================================
# 2. POOLED PROBIT
# ========================================================================
cat("2. Pooled Probit...\n")
pooled_probit <- glm(y ~ x1 + x2,
                      data = data,
                      family = binomial(link = "probit"))

results$pooled_probit <- list(
  coefficients = as.numeric(coef(pooled_probit)),
  coef_names = names(coef(pooled_probit)),
  std_errors = as.numeric(sqrt(diag(vcov(pooled_probit)))),
  loglik = as.numeric(logLik(pooled_probit)),
  aic = AIC(pooled_probit),
  bic = BIC(pooled_probit),
  nobs = nobs(pooled_probit)
)

# ========================================================================
# 3. FIXED EFFECTS LOGIT (using pglm)
# ========================================================================
cat("3. Fixed Effects Logit...\n")
tryCatch({
  fe_logit <- pglm(y ~ x1 + x2,
                    data = panel_data,
                    family = binomial(link = "logit"),
                    model = "within",
                    method = "bfgs")

  results$fe_logit <- list(
    coefficients = as.numeric(coef(fe_logit)),
    coef_names = names(coef(fe_logit)),
    std_errors = as.numeric(sqrt(diag(vcov(fe_logit)))),
    loglik = as.numeric(logLik(fe_logit))
  )
}, error = function(e) {
  cat("  Warning: FE Logit failed -", e$message, "\n")
  results$fe_logit <<- list(error = e$message)
})

# ========================================================================
# 4. RANDOM EFFECTS PROBIT (using pglm)
# ========================================================================
cat("4. Random Effects Probit...\n")
tryCatch({
  re_probit <- pglm(y ~ x1 + x2,
                     data = panel_data,
                     family = binomial(link = "probit"),
                     model = "random",
                     method = "bfgs")

  results$re_probit <- list(
    coefficients = as.numeric(coef(re_probit)),
    coef_names = names(coef(re_probit)),
    std_errors = as.numeric(sqrt(diag(vcov(re_probit)))),
    loglik = as.numeric(logLik(re_probit)),
    sigma = re_probit$sigma  # Random effect variance component
  )
}, error = function(e) {
  cat("  Warning: RE Probit failed -", e$message, "\n")
  results$re_probit <<- list(error = e$message)
})

# ========================================================================
# 5. MARGINAL EFFECTS for Pooled Logit
# ========================================================================
cat("5. Average Marginal Effects (AME) for Pooled Logit...\n")
tryCatch({
  # Calculate AME using margins package
  ame_logit <- margins(pooled_logit)
  ame_summary <- summary(ame_logit)

  results$ame_logit <- list(
    marginal_effects = as.numeric(ame_summary$AME),
    variable_names = ame_summary$factor,
    std_errors = as.numeric(ame_summary$SE),
    lower_ci = as.numeric(ame_summary$lower),
    upper_ci = as.numeric(ame_summary$upper)
  )
}, error = function(e) {
  cat("  Warning: AME calculation failed -", e$message, "\n")
  results$ame_logit <<- list(error = e$message)
})

# ========================================================================
# 6. PREDICTED PROBABILITIES for Pooled Logit
# ========================================================================
cat("6. Predicted Probabilities for Pooled Logit...\n")
pred_probs <- predict(pooled_logit, type = "response")
results$pooled_logit$predicted_probs_sample <- head(as.numeric(pred_probs), 100)

# ========================================================================
# 7. MODEL FIT STATISTICS
# ========================================================================
cat("7. Model Fit Statistics...\n")

# McFadden R-squared for pooled logit
null_logit <- glm(y ~ 1, data = data, family = binomial(link = "logit"))
mcfadden_r2 <- 1 - (logLik(pooled_logit)/logLik(null_logit))
results$pooled_logit$mcfadden_r2 <- as.numeric(mcfadden_r2)

# Classification accuracy
threshold <- 0.5
predicted_class <- as.integer(pred_probs > threshold)
actual_class <- data$y
accuracy <- mean(predicted_class == actual_class)
results$pooled_logit$accuracy <- accuracy

# Confusion matrix
conf_matrix <- table(Actual = actual_class, Predicted = predicted_class)
results$pooled_logit$confusion_matrix <- as.matrix(conf_matrix)

# ========================================================================
# SAVE RESULTS
# ========================================================================
cat("\nSaving results to JSON...\n")
write_json(results, "reference_results_binary.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 10)

cat("Reference results generated successfully!\n")

# Print summary
cat("\n=== SUMMARY ===\n")
cat("Pooled Logit coefficients:\n")
print(results$pooled_logit$coefficients)
cat("\nPooled Probit coefficients:\n")
print(results$pooled_probit$coefficients)
if (!is.null(results$fe_logit$coefficients)) {
  cat("\nFE Logit coefficients:\n")
  print(results$fe_logit$coefficients)
}
if (!is.null(results$re_probit$coefficients)) {
  cat("\nRE Probit coefficients:\n")
  print(results$re_probit$coefficients)
}
