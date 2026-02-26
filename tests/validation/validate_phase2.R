# R Validation Script for FASE 2 - Marginal Effects and Random Effects Probit
#
# This script validates panelbox implementations against R packages:
# - margins: for marginal effects (AME, MEM, MER)
# - pglm: for Random Effects Probit
#
# Required packages:
# install.packages(c("margins", "pglm", "plm", "MASS"))

library(margins)
library(pglm)
library(plm)
library(MASS)

# Set seed for reproducibility
set.seed(42)

# ============================================================================
# 1. Generate Test Data
# ============================================================================

generate_panel_data <- function(n_entities = 100, n_periods = 5,
                                sigma_alpha = 0.8) {
  n_obs <- n_entities * n_periods

  # Entity and time indices
  entity <- rep(1:n_entities, each = n_periods)
  time <- rep(1:n_periods, n_entities)

  # Random effects
  alpha_i <- rnorm(n_entities, 0, sigma_alpha)
  alpha_expanded <- rep(alpha_i, each = n_periods)

  # Covariates
  x1 <- rnorm(n_obs)
  x2 <- rnorm(n_obs)
  x3 <- rbinom(n_obs, 1, 0.5)  # Binary variable

  # True parameters
  beta_0 <- 0.5
  beta_1 <- 0.8
  beta_2 <- -0.6
  beta_3 <- 1.0

  # Linear predictor with random effects
  eta <- beta_0 + beta_1 * x1 + beta_2 * x2 + beta_3 * x3 + alpha_expanded

  # Generate binary outcome
  prob <- pnorm(eta)
  y <- rbinom(n_obs, 1, prob)

  # Create data frame
  data <- data.frame(
    entity = factor(entity),
    time = factor(time),
    y = y,
    x1 = x1,
    x2 = x2,
    x3 = x3
  )

  return(data)
}

# Generate data
data <- generate_panel_data(n_entities = 100, n_periods = 5)

# Convert to pdata.frame for panel models
pdata <- pdata.frame(data, index = c("entity", "time"))

# Save data for Python comparison
write.csv(data, "test_panel_data.csv", row.names = FALSE)

cat("Generated panel data with", nrow(data), "observations\n")
cat("Number of entities:", length(unique(data$entity)), "\n")
cat("Number of periods:", length(unique(data$time)), "\n\n")

# ============================================================================
# 2. Test Marginal Effects (AME, MEM, MER)
# ============================================================================

cat("=" , rep("=", 60), "\n", sep="")
cat("MARGINAL EFFECTS VALIDATION\n")
cat("=" , rep("=", 60), "\n\n", sep="")

# Fit Pooled Logit
logit_model <- glm(y ~ x1 + x2 + x3, data = data, family = binomial(link = "logit"))

cat("Pooled Logit Coefficients:\n")
print(summary(logit_model)$coefficients)
cat("\n")

# Compute Average Marginal Effects (AME)
ame_logit <- margins(logit_model)
cat("Average Marginal Effects (AME) - Logit:\n")
print(summary(ame_logit))
cat("\n")

# Marginal Effects at Means (MEM)
mem_logit <- margins(logit_model, at = list(x1 = mean(data$x1),
                                            x2 = mean(data$x2),
                                            x3 = mean(data$x3)))
cat("Marginal Effects at Means (MEM) - Logit:\n")
print(summary(mem_logit))
cat("\n")

# Marginal Effects at Representative values (MER)
mer_logit <- margins(logit_model, at = list(x1 = 0, x2 = 0, x3 = 1))
cat("Marginal Effects at Representative values (MER) - Logit:\n")
cat("Evaluated at: x1=0, x2=0, x3=1\n")
print(summary(mer_logit))
cat("\n")

# Fit Pooled Probit for comparison
probit_model <- glm(y ~ x1 + x2 + x3, data = data, family = binomial(link = "probit"))

cat("Pooled Probit Coefficients:\n")
print(summary(probit_model)$coefficients)
cat("\n")

# AME for Probit
ame_probit <- margins(probit_model)
cat("Average Marginal Effects (AME) - Probit:\n")
print(summary(ame_probit))
cat("\n")

# Save marginal effects results
me_results <- list(
  logit_ame = summary(ame_logit),
  logit_mem = summary(mem_logit),
  logit_mer = summary(mer_logit),
  probit_ame = summary(ame_probit)
)

saveRDS(me_results, "marginal_effects_results.rds")

# ============================================================================
# 3. Test Random Effects Probit
# ============================================================================

cat("=" , rep("=", 60), "\n", sep="")
cat("RANDOM EFFECTS PROBIT VALIDATION\n")
cat("=" , rep("=", 60), "\n\n", sep="")

# Fit Random Effects Probit using pglm
# Note: pglm uses adaptive Gauss-Hermite quadrature
re_probit <- pglm(y ~ x1 + x2 + x3,
                  data = pdata,
                  family = binomial(link = "probit"),
                  model = "random",
                  method = "nr",  # Newton-Raphson
                  print.level = 1)

cat("Random Effects Probit Results:\n")
print(summary(re_probit))
cat("\n")

# Extract key parameters
beta_est <- coef(re_probit)
cat("Coefficient estimates:\n")
print(beta_est)
cat("\n")

# Get sigma (standard deviation of random effects)
# In pglm, this is stored as sigma
sigma_alpha_est <- re_probit$sigma
cat("Sigma_alpha (RE standard deviation):", sigma_alpha_est, "\n")

# Calculate rho (intra-class correlation)
rho_est <- sigma_alpha_est^2 / (1 + sigma_alpha_est^2)
cat("Rho (intra-class correlation):", rho_est, "\n\n")

# Log-likelihood
cat("Log-likelihood:", logLik(re_probit), "\n")
cat("AIC:", AIC(re_probit), "\n")
cat("BIC:", BIC(re_probit), "\n\n")

# Compare with Pooled Probit
cat("Comparison with Pooled Probit:\n")
cat("----------------------------------\n")
pooled_probit <- pglm(y ~ x1 + x2 + x3,
                      data = pdata,
                      family = binomial(link = "probit"),
                      model = "pooling")

cat("Pooled Probit (using pglm):\n")
print(summary(pooled_probit))
cat("\n")

# Likelihood Ratio Test
lr_statistic <- 2 * (logLik(re_probit) - logLik(pooled_probit))
lr_pvalue <- pchisq(lr_statistic, df = 1, lower.tail = FALSE)
cat("LR Test (H0: sigma_alpha = 0):\n")
cat("  LR statistic:", lr_statistic, "\n")
cat("  P-value:", lr_pvalue, "\n")
cat("  Conclusion:", ifelse(lr_pvalue < 0.05,
                           "Reject H0 - Random effects are significant",
                           "Fail to reject H0"), "\n\n")

# ============================================================================
# 4. Test with Different Quadrature Points
# ============================================================================

cat("=" , rep("=", 60), "\n", sep="")
cat("QUADRATURE POINTS COMPARISON\n")
cat("=" , rep("=", 60), "\n\n", sep="")

# Note: pglm doesn't allow easy control of quadrature points
# But we can compare with different optimization methods

cat("Testing different optimization methods in pglm:\n")
cat("(as proxy for quadrature accuracy)\n\n")

# Method 1: Newton-Raphson
re_probit_nr <- pglm(y ~ x1 + x2 + x3,
                     data = pdata,
                     family = binomial(link = "probit"),
                     model = "random",
                     method = "nr")

# Method 2: BFGS
re_probit_bfgs <- pglm(y ~ x1 + x2 + x3,
                       data = pdata,
                       family = binomial(link = "probit"),
                       model = "random",
                       method = "bfgs")

cat("Newton-Raphson:\n")
cat("  Beta[x1]:", coef(re_probit_nr)["x1"], "\n")
cat("  Sigma_alpha:", re_probit_nr$sigma, "\n")
cat("  Log-likelihood:", logLik(re_probit_nr), "\n\n")

cat("BFGS:\n")
cat("  Beta[x1]:", coef(re_probit_bfgs)["x1"], "\n")
cat("  Sigma_alpha:", re_probit_bfgs$sigma, "\n")
cat("  Log-likelihood:", logLik(re_probit_bfgs), "\n\n")

# ============================================================================
# 5. Export Results for Python Comparison
# ============================================================================

cat("=" , rep("=", 60), "\n", sep="")
cat("EXPORTING RESULTS\n")
cat("=" , rep("=", 60), "\n\n", sep="")

# Create results list
validation_results <- list(
  # Data info
  n_obs = nrow(data),
  n_entities = length(unique(data$entity)),
  n_periods = length(unique(data$time)),

  # Pooled Logit
  pooled_logit = list(
    coefficients = coef(logit_model),
    std_errors = summary(logit_model)$coefficients[, "Std. Error"],
    loglik = logLik(logit_model)[1],
    aic = AIC(logit_model),
    ame = summary(ame_logit),
    mem = summary(mem_logit),
    mer = summary(mer_logit)
  ),

  # Pooled Probit
  pooled_probit = list(
    coefficients = coef(probit_model),
    std_errors = summary(probit_model)$coefficients[, "Std. Error"],
    loglik = logLik(probit_model)[1],
    aic = AIC(probit_model),
    ame = summary(ame_probit)
  ),

  # Random Effects Probit
  re_probit = list(
    coefficients = coef(re_probit),
    std_errors = sqrt(diag(vcov(re_probit))),
    sigma_alpha = re_probit$sigma,
    rho = rho_est,
    loglik = logLik(re_probit)[1],
    aic = AIC(re_probit),
    bic = BIC(re_probit),
    converged = re_probit$convergence
  )
)

# Save as JSON for easy Python reading
library(jsonlite)
write_json(validation_results, "phase2_validation_results.json",
           pretty = TRUE, digits = 10)

cat("Results exported to phase2_validation_results.json\n")

# Also save as RDS for R users
saveRDS(validation_results, "phase2_validation_results.rds")
cat("Results also saved as phase2_validation_results.rds\n\n")

# ============================================================================
# 6. Summary Table
# ============================================================================

cat("=" , rep("=", 60), "\n", sep="")
cat("SUMMARY COMPARISON TABLE\n")
cat("=" , rep("=", 60), "\n\n", sep="")

# Create comparison table
comparison <- data.frame(
  Model = c("Pooled Logit", "Pooled Probit", "RE Probit"),
  Beta_x1 = c(
    coef(logit_model)["x1"],
    coef(probit_model)["x1"],
    coef(re_probit)["x1"]
  ),
  Beta_x2 = c(
    coef(logit_model)["x2"],
    coef(probit_model)["x2"],
    coef(re_probit)["x2"]
  ),
  Beta_x3 = c(
    coef(logit_model)["x3"],
    coef(probit_model)["x3"],
    coef(re_probit)["x3"]
  ),
  Sigma_alpha = c(
    NA,
    NA,
    re_probit$sigma
  ),
  LogLik = c(
    logLik(logit_model)[1],
    logLik(probit_model)[1],
    logLik(re_probit)[1]
  ),
  AIC = c(
    AIC(logit_model),
    AIC(probit_model),
    AIC(re_probit)
  )
)

print(comparison, row.names = FALSE)
cat("\n")

cat("Marginal Effects Comparison (AME):\n")
cat("-----------------------------------\n")
me_comparison <- data.frame(
  Variable = c("x1", "x2", "x3"),
  Logit_AME = summary(ame_logit)$AME[1:3],
  Probit_AME = summary(ame_probit)$AME[1:3]
)
print(me_comparison, row.names = FALSE)

cat("\n")
cat("Validation script completed successfully!\n")
cat("=" , rep("=", 60), "\n", sep="")
