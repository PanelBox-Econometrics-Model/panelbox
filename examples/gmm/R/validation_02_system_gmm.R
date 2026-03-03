###############################################################################
# Validation 02 - System GMM (Blundell-Bond)
#
# Replicates PanelBox SystemGMM results using plm::pgmm() in R.
# Estimates one-step and two-step System GMM on the abdata dataset.
# Reports AR(1)/AR(2), Sargan/Hansen J tests, Windmeijer-corrected SE.
#
# Dataset: abdata (N=140 firms, T=9 years, unbalanced)
# Model:   n ~ lag(n) + w + k | gmm(n, 2:99)
# System GMM: transformation = "ld" (levels + differences)
###############################################################################

library(plm)

# --- Data Loading -----------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/gmm/data/abdata.csv"
abdata <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("Rows:", nrow(abdata), "\n")
cat("Firms:", length(unique(abdata$firm)), "\n")
cat("Years:", paste(sort(unique(abdata$year)), collapse=", "), "\n\n")

# Convert to pdata.frame
pab <- pdata.frame(abdata, index = c("firm", "year"))

# --- Helper: extract results from pgmm --------------------------------------
extract_pgmm_results <- function(model, model_name, robust_summary) {
  coefs <- robust_summary$coefficients
  var_names <- rownames(coefs)

  # pgmm residuals is a list (one vector per group)
  n_groups <- length(model$residuals)
  n_obs <- sum(sapply(model$residuals, length))
  n_instruments <- ncol(model$W[[1]])

  # AR tests from summary
  ar1 <- robust_summary$m1
  ar2 <- robust_summary$m2

  ar1_stat <- if (!is.null(ar1)) ar1$statistic else NA
  ar1_pval <- if (!is.null(ar1)) ar1$p.value else NA
  ar2_stat <- if (!is.null(ar2)) ar2$statistic else NA
  ar2_pval <- if (!is.null(ar2)) ar2$p.value else NA

  # Sargan/Hansen test from summary
  sargan_test <- robust_summary$sargan
  sargan_stat <- if (!is.null(sargan_test)) sargan_test$statistic else NA
  sargan_pval <- if (!is.null(sargan_test)) sargan_test$p.value else NA
  sargan_df   <- if (!is.null(sargan_test)) sargan_test$parameter else NA

  results <- data.frame(
    model_name    = model_name,
    variable      = var_names,
    coefficient   = coefs[, "Estimate"],
    std_error     = coefs[, "Std. Error"],
    z_statistic   = coefs[, "z-value"],
    p_value       = coefs[, "Pr(>|z|)"],
    n_obs         = n_obs,
    n_groups      = n_groups,
    n_instruments = n_instruments,
    ar1_statistic = ar1_stat,
    ar1_pvalue    = ar1_pval,
    ar2_statistic = ar2_stat,
    ar2_pvalue    = ar2_pval,
    sargan_statistic = NA,
    sargan_pvalue    = NA,
    sargan_df        = NA,
    hansen_statistic  = NA,
    hansen_pvalue     = NA,
    hansen_df         = NA,
    stringsAsFactors = FALSE,
    row.names = NULL
  )

  # One-step: Sargan test. Two-step: Hansen J test
  if (grepl("twostep", model_name)) {
    results$hansen_statistic <- sargan_stat
    results$hansen_pvalue    <- sargan_pval
    results$hansen_df        <- sargan_df
  } else {
    results$sargan_statistic <- sargan_stat
    results$sargan_pvalue    <- sargan_pval
    results$sargan_df        <- sargan_df
  }

  return(results)
}

# --- Model 1: System GMM One-Step (twoways) ---------------------------------
cat("=== System GMM: One-Step (twoways) ===\n")
sys_gmm_1step <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "onestep",
  transformation = "ld"
)
summ_sys_1step <- summary(sys_gmm_1step, robust = TRUE)
print(summ_sys_1step)
cat("\n")

# --- Model 2: System GMM Two-Step (twoways) ---------------------------------
cat("=== System GMM: Two-Step (twoways, Windmeijer-corrected SE) ===\n")
sys_gmm_2step <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "ld"
)
summ_sys_2step <- summary(sys_gmm_2step, robust = TRUE)
print(summ_sys_2step)
cat("\n")

# --- Model 3: System GMM One-Step (individual, no time dummies) --------------
cat("=== System GMM: One-Step (individual, no time dummies) ===\n")
sys_gmm_1step_ind <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "onestep",
  transformation = "ld"
)
summ_sys_1step_ind <- summary(sys_gmm_1step_ind, robust = TRUE)
print(summ_sys_1step_ind)
cat("\n")

# --- Model 4: System GMM Two-Step (individual, no time dummies) --------------
cat("=== System GMM: Two-Step (individual, no time dummies) ===\n")
sys_gmm_2step_ind <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "twosteps",
  transformation = "ld"
)
summ_sys_2step_ind <- summary(sys_gmm_2step_ind, robust = TRUE)
print(summ_sys_2step_ind)
cat("\n")

# --- Collect Results ---------------------------------------------------------
results <- rbind(
  extract_pgmm_results(sys_gmm_1step, "sys_gmm_onestep_twoways", summ_sys_1step),
  extract_pgmm_results(sys_gmm_2step, "sys_gmm_twostep_twoways", summ_sys_2step),
  extract_pgmm_results(sys_gmm_1step_ind, "sys_gmm_onestep_individual", summ_sys_1step_ind),
  extract_pgmm_results(sys_gmm_2step_ind, "sys_gmm_twostep_individual", summ_sys_2step_ind)
)

# --- Save to CSV -------------------------------------------------------------
output_path <- "/home/guhaase/projetos/panelbox/examples/gmm/R/results_system_gmm.csv"
write.csv(results, output_path, row.names = FALSE)
cat("Results saved to:", output_path, "\n")

# --- Print summary table -----------------------------------------------------
cat("\n=== Results Summary ===\n")
print(results[, c("model_name", "variable", "coefficient", "std_error", "z_statistic", "p_value")])

cat("\n=== Diagnostic Tests Summary ===\n")
diag_cols <- c("model_name", "n_obs", "n_groups", "n_instruments",
               "ar1_statistic", "ar1_pvalue", "ar2_statistic", "ar2_pvalue",
               "sargan_statistic", "sargan_pvalue", "hansen_statistic", "hansen_pvalue")
diag_results <- unique(results[, diag_cols])
print(diag_results)

# --- Comparison: Diff GMM vs System GMM (two-step, no time dummies) ----------
cat("\n=== Comparison: Difference GMM vs System GMM ===\n")
cat("(Both two-step, individual effects, no time dummies)\n\n")

# Run Difference GMM for comparison
diff_gmm_2step <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "twosteps",
  transformation = "d"
)
summ_diff <- summary(diff_gmm_2step, robust = TRUE)

cat("Difference GMM coefficients:\n")
print(summ_diff$coefficients[, c("Estimate", "Std. Error")])
cat("\nSystem GMM coefficients:\n")
print(summ_sys_2step_ind$coefficients[, c("Estimate", "Std. Error")])

cat("\nDone.\n")
