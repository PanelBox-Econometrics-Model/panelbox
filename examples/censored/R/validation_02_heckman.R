# =============================================================================
# Validation Script 02: Heckman Selection Models (Two-Step and MLE)
# PanelBox vs R (sampleSelection)
#
# Dataset: mroz_1987.csv
# Outcome equation: wage ~ education + experience + experience_sq
# Selection equation: lfp ~ education + experience + age + children_lt6 +
#                           children_6_18 + husband_income
# =============================================================================

library(sampleSelection)

# --- Load data ---------------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/censored/data/mroz_1987.csv"
df <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("N observations:", nrow(df), "\n")
cat("N working (lfp == 1):", sum(df$lfp == 1), "\n")
cat("N not working (lfp == 0):", sum(df$lfp == 0), "\n")
cat("Participation rate:", round(mean(df$lfp) * 100, 2), "%\n\n")

# --- Model 1: OLS on working women only (biased baseline) -------------------
cat("=== OLS on Working Women Only (biased baseline) ===\n")
df_working <- df[df$lfp == 1, ]
ols_model <- lm(wage ~ education + experience + experience_sq, data = df_working)
cat(capture.output(summary(ols_model)), sep = "\n")
cat("\n")

# --- Model 2: Heckman Two-Step (heckit) -------------------------------------
cat("=== Heckman Two-Step (sampleSelection::heckit) ===\n")
heckman_2step <- heckit(
  selection = lfp ~ education + experience + age + children_lt6 +
    children_6_18 + husband_income,
  outcome = wage ~ education + experience + experience_sq,
  data = df,
  method = "2step"
)
cat(capture.output(summary(heckman_2step)), sep = "\n")
cat("\n")

# --- Model 3: Heckman MLE ---------------------------------------------------
cat("=== Heckman MLE (sampleSelection::selection) ===\n")
heckman_mle <- selection(
  selection = lfp ~ education + experience + age + children_lt6 +
    children_6_18 + husband_income,
  outcome = wage ~ education + experience + experience_sq,
  data = df,
  method = "ml"
)
cat(capture.output(summary(heckman_mle)), sep = "\n")
cat("\n")

# --- Extract results ---------------------------------------------------------
# sampleSelection uses flat rownames without S:/O: prefixes.
# Selection equation coefficients come first, then outcome, then error terms.
# We identify them by position based on the number of parameters in each equation.

n_sel_params <- 7   # intercept + education + experience + age + children_lt6 + children_6_18 + husband_income
n_out_params <- 4   # intercept + education + experience + experience_sq

# Helper to extract and label coefficients from sampleSelection summary
extract_heckman_coefs <- function(model_summary, model_name) {
  est <- model_summary$estimate
  n_rows <- nrow(est)

  # Build equation labels by position
  eq_labels <- c(
    rep("selection", n_sel_params),
    rep("outcome", n_out_params),
    rep("error_terms", n_rows - n_sel_params - n_out_params)
  )

  data.frame(
    model_name = model_name,
    equation = eq_labels,
    variable = rownames(est),
    coefficient = est[, 1],
    std_error = est[, 2],
    statistic = est[, 3],
    p_value = est[, 4],
    stringsAsFactors = FALSE
  )
}

# OLS results
ols_coefs <- summary(ols_model)$coefficients
ols_results <- data.frame(
  model_name = "ols_working_only",
  equation = "outcome",
  variable = rownames(ols_coefs),
  coefficient = ols_coefs[, "Estimate"],
  std_error = ols_coefs[, "Std. Error"],
  statistic = ols_coefs[, "t value"],
  p_value = ols_coefs[, "Pr(>|t|)"],
  stringsAsFactors = FALSE
)

# Heckman Two-Step results
h2_results <- extract_heckman_coefs(summary(heckman_2step), "heckman_2step")

# Heckman MLE results
hm_results <- extract_heckman_coefs(summary(heckman_mle), "heckman_mle")

# Combine all results
all_results <- rbind(ols_results, h2_results, hm_results)
row.names(all_results) <- NULL

# --- Model-level statistics --------------------------------------------------

# Two-step: rho, sigma, lambda (IMR coefficient)
rho_2step <- heckman_2step$rho
sigma_2step <- heckman_2step$sigma
lambda_2step <- rho_2step * sigma_2step

# MLE: rho, sigma, log-likelihood
rho_mle <- heckman_mle$estimate["rho"]
sigma_mle <- heckman_mle$estimate["sigma"]
lambda_mle <- rho_mle * sigma_mle
loglik_mle <- as.numeric(logLik(heckman_mle))

model_stats <- data.frame(
  model_name = c("heckman_2step", "heckman_2step", "heckman_2step",
                 "heckman_2step", "heckman_2step",
                 "heckman_mle", "heckman_mle", "heckman_mle",
                 "heckman_mle", "heckman_mle", "heckman_mle",
                 "ols_working_only", "ols_working_only"),
  equation = "model_stat",
  variable = c("_rho", "_sigma", "_lambda", "_n_selected", "_n_total",
               "_rho", "_sigma", "_lambda", "_log_likelihood", "_n_selected", "_n_total",
               "_r_squared", "_r_squared_adj"),
  coefficient = c(rho_2step, sigma_2step, lambda_2step,
                  sum(df$lfp == 1), nrow(df),
                  rho_mle, sigma_mle, lambda_mle, loglik_mle,
                  sum(df$lfp == 1), nrow(df),
                  summary(ols_model)$r.squared, summary(ols_model)$adj.r.squared),
  std_error = NA,
  statistic = NA,
  p_value = NA,
  stringsAsFactors = FALSE
)

all_results <- rbind(all_results, model_stats)

# --- Save results to CSV ----------------------------------------------------
output_path <- "/home/guhaase/projetos/panelbox/examples/censored/R/results_heckman.csv"
write.csv(all_results, output_path, row.names = FALSE)
cat("\nResults saved to:", output_path, "\n")

# --- Print comparison --------------------------------------------------------
# Extract outcome coefficients by position
h2_est <- summary(heckman_2step)$estimate
hm_est <- summary(heckman_mle)$estimate

# Outcome equation rows: positions (n_sel_params+1):(n_sel_params+n_out_params)
out_start <- n_sel_params + 1
out_end <- n_sel_params + n_out_params
out_vars <- c("(Intercept)", "education", "experience", "experience_sq")

cat("\n=== Outcome Equation Comparison ===\n")
cat(sprintf("%-18s %-14s %-14s %-14s\n", "Variable", "OLS", "Heckman_2step", "Heckman_MLE"))
for (i in seq_along(out_vars)) {
  idx <- out_start + i - 1
  cat(sprintf("%-18s %-14.6f %-14.6f %-14.6f\n", out_vars[i],
              coef(ols_model)[out_vars[i]],
              h2_est[idx, "Estimate"],
              hm_est[idx, "Estimate"]))
}

cat("\n=== Selection Equation (Heckman Two-Step) ===\n")
sel_vars <- c("(Intercept)", "education", "experience", "age",
              "children_lt6", "children_6_18", "husband_income")
cat(sprintf("%-18s %-14s %-14s\n", "Variable", "Two-Step", "MLE"))
for (i in seq_along(sel_vars)) {
  cat(sprintf("%-18s %-14.6f %-14.6f\n", sel_vars[i],
              h2_est[i, "Estimate"],
              hm_est[i, "Estimate"]))
}

cat("\n=== Selection Parameters ===\n")
cat(sprintf("%-15s %-14s %-14s\n", "Parameter", "Two-Step", "MLE"))
cat(sprintf("%-15s %-14.6f %-14.6f\n", "rho", rho_2step, rho_mle))
cat(sprintf("%-15s %-14.6f %-14.6f\n", "sigma", sigma_2step, sigma_mle))
cat(sprintf("%-15s %-14.6f %-14.6f\n", "lambda", lambda_2step, lambda_mle))

cat("\nDone.\n")
