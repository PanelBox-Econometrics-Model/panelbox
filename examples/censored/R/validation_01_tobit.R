# =============================================================================
# Validation Script 01: Tobit Pooled Model
# PanelBox vs R (AER::tobit / censReg::censReg)
#
# The labor_supply.csv on disk has no censored observations (min hours = 8.9).
# The PanelBox notebook regenerates data with a specific DGP that produces
# ~7% censoring (35 zeros out of 500). We replicate that DGP here in R
# to produce comparable censored data, then estimate the Tobit models.
#
# Model: hours ~ wage + education + experience + experience_sq +
#                children + married + non_labor_income
# Censoring: left-censored at 0
# =============================================================================

library(AER)
library(censReg)

# --- Replicate the PanelBox notebook DGP ------------------------------------
# The notebook uses numpy RNG seed=42 with default_rng (PCG64).
# We cannot exactly replicate numpy's RNG in R, so we use the CSV data
# as covariates and apply the same DGP formula to generate latent hours.
# This gives us the same X matrix but potentially different error draws.
# Instead, we just read the CSV and artificially censor it using the same
# approach the notebook uses.

data_path <- "/home/guhaase/projetos/panelbox/examples/censored/data/labor_supply.csv"
df <- read.csv(data_path)

# The CSV has hours that are all positive (min=8.9).
# The notebook regenerates the data in-cell using a DGP with lower intercept
# that produces ~7% censoring. Since we cannot reproduce the numpy RNG in R,
# we generate our own censored labor supply data using R's RNG with the same DGP.
set.seed(42)
n <- 500

education <- sample(8:20, n, replace = TRUE)
age <- sample(25:59, n, replace = TRUE)
experience <- pmax(age - education - 6 + rnorm(n, 0, 2), 0)
experience_sq <- experience^2
children <- rpois(n, 0.8)
married <- rbinom(n, 1, 0.6)
non_labor_income <- abs(rnorm(n, 20, 15))
wage <- exp(0.8 + 0.07 * education + 0.03 * experience -
              0.0005 * experience_sq + rnorm(n, 0, 0.4))

# Latent hours (same DGP as notebook)
latent_hours <- (-5.0
                 + 3.0 * log(wage)
                 + 0.8 * education
                 + 1.2 * experience
                 - 0.02 * experience_sq
                 - 3.5 * children
                 + 1.5 * married
                 - 0.25 * non_labor_income
                 + rnorm(n, 0, 12))

# Apply censoring
hours <- pmax(latent_hours, 0)

df <- data.frame(
  hours = round(hours, 1),
  wage = round(wage, 2),
  education = education,
  experience = round(experience, 1),
  experience_sq = round(experience_sq, 1),
  age = age,
  children = children,
  married = married,
  non_labor_income = round(non_labor_income, 2)
)

cat("=== Dataset Summary ===\n")
cat("N observations:", nrow(df), "\n")
cat("N censored (hours == 0):", sum(df$hours == 0), "\n")
cat("N uncensored (hours > 0):", sum(df$hours > 0), "\n")
cat("Censoring rate:", round(mean(df$hours == 0) * 100, 2), "%\n\n")

# --- Model 1: OLS (for comparison) ------------------------------------------
cat("=== OLS Estimation (biased baseline) ===\n")
ols_model <- lm(hours ~ wage + education + experience + experience_sq +
                  children + married + non_labor_income, data = df)
cat(capture.output(summary(ols_model)), sep = "\n")
cat("\n")

# --- Model 2: Tobit with AER::tobit -----------------------------------------
cat("=== Tobit Model (AER::tobit) ===\n")
tobit_aer <- tobit(hours ~ wage + education + experience + experience_sq +
                     children + married + non_labor_income,
                   left = 0, right = Inf, data = df)
cat(capture.output(summary(tobit_aer)), sep = "\n")
cat("\n")

# --- Model 3: Tobit with censReg::censReg -----------------------------------
cat("=== Tobit Model (censReg::censReg) ===\n")
tobit_censreg <- censReg(hours ~ wage + education + experience + experience_sq +
                           children + married + non_labor_income,
                         left = 0, right = Inf, data = df)
cat(capture.output(summary(tobit_censreg)), sep = "\n")
cat("\n")

# --- Extract results ---------------------------------------------------------

# OLS results
ols_coefs <- summary(ols_model)$coefficients
ols_results <- data.frame(
  model_name = "ols",
  variable = rownames(ols_coefs),
  coefficient = ols_coefs[, "Estimate"],
  std_error = ols_coefs[, "Std. Error"],
  statistic = ols_coefs[, "t value"],
  p_value = ols_coefs[, "Pr(>|t|)"],
  stringsAsFactors = FALSE
)

# AER Tobit results
aer_coefs <- summary(tobit_aer)$coefficients
aer_results <- data.frame(
  model_name = "tobit_aer",
  variable = rownames(aer_coefs),
  coefficient = aer_coefs[, "Estimate"],
  std_error = aer_coefs[, "Std. Error"],
  statistic = aer_coefs[, "z value"],
  p_value = aer_coefs[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)

# censReg Tobit results
cr_coefs <- summary(tobit_censreg)$estimate
cr_results <- data.frame(
  model_name = "tobit_censreg",
  variable = rownames(cr_coefs),
  coefficient = cr_coefs[, "Estimate"],
  std_error = cr_coefs[, "Std. error"],
  statistic = cr_coefs[, "t value"],
  p_value = cr_coefs[, "Pr(> t)"],
  stringsAsFactors = FALSE
)

# Combine all results
all_results <- rbind(ols_results, aer_results, cr_results)
row.names(all_results) <- NULL

# Add model-level statistics
sigma_aer <- tobit_aer$scale
sigma_censreg <- exp(coef(tobit_censreg)["logSigma"])
loglik_aer <- as.numeric(logLik(tobit_aer))
loglik_censreg <- as.numeric(logLik(tobit_censreg))

model_stats <- data.frame(
  model_name = c("tobit_aer", "tobit_aer", "tobit_aer", "tobit_aer",
                 "tobit_censreg", "tobit_censreg", "tobit_censreg", "tobit_censreg",
                 "ols", "ols"),
  variable = c("_sigma", "_log_likelihood", "_n_censored", "_n_uncensored",
               "_sigma", "_log_likelihood", "_n_censored", "_n_uncensored",
               "_r_squared", "_r_squared_adj"),
  coefficient = c(sigma_aer, loglik_aer, sum(df$hours == 0), sum(df$hours > 0),
                  sigma_censreg, loglik_censreg, sum(df$hours == 0), sum(df$hours > 0),
                  summary(ols_model)$r.squared, summary(ols_model)$adj.r.squared),
  std_error = NA,
  statistic = NA,
  p_value = NA,
  stringsAsFactors = FALSE
)

all_results <- rbind(all_results, model_stats)

# --- Save results to CSV ----------------------------------------------------
output_path <- "/home/guhaase/projetos/panelbox/examples/censored/R/results_tobit.csv"
write.csv(all_results, output_path, row.names = FALSE)
cat("\nResults saved to:", output_path, "\n")

# --- Print comparison table --------------------------------------------------
cat("\n=== Coefficient Comparison: OLS vs Tobit (AER) vs Tobit (censReg) ===\n")
vars <- rownames(ols_coefs)
# censReg does not include 'age' since we dropped it from the model
# but both models use the same formula, so all vars should match
common_vars <- intersect(vars, rownames(aer_coefs))
common_vars <- intersect(common_vars, rownames(cr_coefs))
comparison <- data.frame(
  Variable = common_vars,
  OLS = round(ols_coefs[common_vars, "Estimate"], 6),
  Tobit_AER = round(aer_coefs[common_vars, "Estimate"], 6),
  Tobit_censReg = round(cr_coefs[common_vars, "Estimate"], 6)
)
print(comparison, row.names = FALSE)

cat("\n=== Sigma Estimates ===\n")
cat("AER sigma:", round(sigma_aer, 6), "\n")
cat("censReg sigma:", round(sigma_censreg, 6), "\n")

cat("\n=== Log-Likelihood ===\n")
cat("AER log-lik:", round(loglik_aer, 4), "\n")
cat("censReg log-lik:", round(loglik_censreg, 4), "\n")

cat("\nDone.\n")
