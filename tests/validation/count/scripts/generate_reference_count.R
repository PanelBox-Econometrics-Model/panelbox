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
setwd("../data")

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
pooled_poisson <- glm(y ~ x1 + x2 + x3,
                       data = data,
                       family = poisson())

results$pooled_poisson <- list(
  coefficients = as.numeric(coef(pooled_poisson)),
  coef_names = names(coef(pooled_poisson)),
  std_errors = as.numeric(sqrt(diag(vcov(pooled_poisson)))),
  loglik = as.numeric(logLik(pooled_poisson)),
  aic = AIC(pooled_poisson),
  bic = BIC(pooled_poisson),
  nobs = nobs(pooled_poisson),
  deviance = deviance(pooled_poisson),
  df_residual = df.residual(pooled_poisson)
)

# ========================================================================
# 2. NEGATIVE BINOMIAL
# ========================================================================
cat("2. Negative Binomial...\n")
nb_model <- glm.nb(y ~ x1 + x2 + x3,
                    data = data)

results$negative_binomial <- list(
  coefficients = as.numeric(coef(nb_model)),
  coef_names = names(coef(nb_model)),
  std_errors = as.numeric(sqrt(diag(vcov(nb_model)))),
  loglik = as.numeric(logLik(nb_model)),
  aic = AIC(nb_model),
  bic = BIC(nb_model),
  theta = nb_model$theta,  # Dispersion parameter
  se_theta = nb_model$SE.theta
)

# ========================================================================
# 3. POISSON FIXED EFFECTS (using pglm)
# ========================================================================
cat("3. Poisson Fixed Effects...\n")
tryCatch({
  fe_poisson <- pglm(y ~ x1 + x2 + x3,
                      data = panel_data,
                      family = poisson(),
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
# 4. RANDOM EFFECTS POISSON (using pglm)
# ========================================================================
cat("4. Random Effects Poisson...\n")
tryCatch({
  re_poisson <- pglm(y ~ x1 + x2 + x3,
                      data = panel_data,
                      family = poisson(),
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
# 5. OVERDISPERSION TEST
# ========================================================================
cat("5. Overdispersion Test...\n")
# Cameron & Trivedi (1990) test
residuals_pearson <- residuals(pooled_poisson, type = "pearson")
n <- length(residuals_pearson)
p <- length(coef(pooled_poisson))
overdispersion_stat <- sum(residuals_pearson^2) / (n - p)
results$overdispersion <- list(
  statistic = overdispersion_stat,
  p_value = pchisq(sum(residuals_pearson^2), df = n - p, lower.tail = FALSE)
)

# ========================================================================
# 6. PREDICTED COUNTS
# ========================================================================
cat("6. Predicted Counts...\n")
pred_counts <- predict(pooled_poisson, type = "response")
results$pooled_poisson$predicted_counts_sample <- head(as.numeric(pred_counts), 100)

# Mean and variance of predictions
results$pooled_poisson$pred_mean <- mean(pred_counts)
results$pooled_poisson$pred_var <- var(pred_counts)

# ========================================================================
# 7. MARGINAL EFFECTS
# ========================================================================
cat("7. Marginal Effects for Poisson...\n")
# For Poisson, marginal effect = beta * E[y|x]
mean_y <- mean(pred_counts)
me_poisson <- coef(pooled_poisson)[-1] * mean_y  # Exclude intercept
results$marginal_effects_poisson <- list(
  effects = as.numeric(me_poisson),
  variable_names = names(me_poisson),
  mean_y = mean_y
)

# ========================================================================
# 8. GOODNESS OF FIT
# ========================================================================
cat("8. Goodness of Fit...\n")
# Chi-squared goodness of fit test
observed <- table(factor(data$y, levels = 0:max(data$y)))
expected_probs <- dpois(0:max(data$y), lambda = mean(pred_counts))
expected <- expected_probs * nrow(data)
# Combine categories with expected < 5
min_expected <- 5
if (any(expected < min_expected)) {
  last_valid <- max(which(expected >= min_expected))
  if (last_valid < length(expected)) {
    observed_grouped <- c(observed[1:last_valid],
                          sum(observed[(last_valid+1):length(observed)]))
    expected_grouped <- c(expected[1:last_valid],
                          sum(expected[(last_valid+1):length(expected)]))
    observed <- observed_grouped
    expected <- expected_grouped
  }
}
chi_sq_stat <- sum((observed - expected)^2 / expected)
df_chi <- length(observed) - length(coef(pooled_poisson)) - 1
results$goodness_of_fit <- list(
  chi_squared = chi_sq_stat,
  df = df_chi,
  p_value = pchisq(chi_sq_stat, df = df_chi, lower.tail = FALSE)
)

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
cat("\nNegative Binomial coefficients:\n")
print(results$negative_binomial$coefficients)
cat(sprintf("\nTheta (dispersion): %.4f\n", results$negative_binomial$theta))
cat(sprintf("Overdispersion statistic: %.4f\n", results$overdispersion$statistic))
if (results$overdispersion$statistic > 1.5) {
  cat("  -> Evidence of overdispersion (consider Negative Binomial)\n")
}
