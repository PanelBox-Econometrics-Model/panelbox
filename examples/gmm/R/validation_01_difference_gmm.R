###############################################################################
# Validation 01 - Difference GMM (Arellano-Bond)
#
# Replicates PanelBox DifferenceGMM results using plm::pgmm() in R.
# Estimates one-step and two-step Difference GMM on the abdata dataset.
# Reports AR(1)/AR(2) tests, Sargan/Hansen J tests, and saves results to CSV.
#
# Dataset: abdata (N=140 firms, T=9 years, unbalanced)
# Model:   n ~ lag(n) + w + k | gmm(n, 2:99)
###############################################################################

library(plm)

# --- Data Loading -----------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/gmm/data/abdata.csv"
abdata <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("Rows:", nrow(abdata), "\n")
cat("Firms:", length(unique(abdata$firm)), "\n")
cat("Years:", paste(sort(unique(abdata$year)), collapse=", "), "\n")
cat("Columns:", paste(names(abdata), collapse=", "), "\n\n")

# Convert to pdata.frame
pab <- pdata.frame(abdata, index = c("firm", "year"))

# --- Helper: extract results from pgmm --------------------------------------
extract_pgmm_results <- function(model, model_name, robust_summary) {
  coefs <- robust_summary$coefficients
  var_names <- rownames(coefs)

  # pgmm residuals is a list (one vector per group)
  n_groups <- length(model$residuals)
  n_obs <- sum(sapply(model$residuals, length))

  # Instrument count
  n_instruments <- ncol(model$W[[1]])

  # AR tests from summary (uses robust vcov when summary is robust)
  ar1 <- robust_summary$m1
  ar2 <- robust_summary$m2

  ar1_stat <- if (!is.null(ar1)) ar1$statistic else NA
  ar1_pval <- if (!is.null(ar1)) ar1$p.value else NA
  ar2_stat <- if (!is.null(ar2)) ar2$statistic else NA
  ar2_pval <- if (!is.null(ar2)) ar2$p.value else NA

  # Sargan/Hansen test from summary
  sargan <- robust_summary$sargan
  sargan_stat <- if (!is.null(sargan)) sargan$statistic else NA
  sargan_pval <- if (!is.null(sargan)) sargan$p.value else NA
  sargan_df   <- if (!is.null(sargan)) sargan$parameter else NA

  # Build data frame
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
    sargan_statistic = sargan_stat,
    sargan_pvalue    = sargan_pval,
    sargan_df        = sargan_df,
    hansen_statistic = NA,
    hansen_pvalue    = NA,
    hansen_df        = NA,
    stringsAsFactors = FALSE,
    row.names = NULL
  )

  # For two-step, the Sargan test is actually Hansen J
  if (grepl("twostep", model_name)) {
    results$hansen_statistic <- results$sargan_statistic
    results$hansen_pvalue    <- results$sargan_pvalue
    results$hansen_df        <- results$sargan_df
    results$sargan_statistic <- NA
    results$sargan_pvalue    <- NA
    results$sargan_df        <- NA
  }

  return(results)
}

# --- Model 1: Difference GMM One-Step (with time dummies) --------------------
cat("=== Difference GMM: One-Step (twoways) ===\n")
diff_gmm_1step <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "onestep",
  transformation = "d"
)
summ_1step <- summary(diff_gmm_1step, robust = TRUE)
print(summ_1step)
cat("\n")

# --- Model 2: Difference GMM Two-Step (with time dummies) --------------------
cat("=== Difference GMM: Two-Step (twoways) ===\n")
diff_gmm_2step <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "d"
)
summ_2step <- summary(diff_gmm_2step, robust = TRUE)
print(summ_2step)
cat("\n")

# --- Model 3: Difference GMM One-Step (no time dummies) ----------------------
cat("=== Difference GMM: One-Step (individual only) ===\n")
diff_gmm_1step_ind <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "onestep",
  transformation = "d"
)
summ_1step_ind <- summary(diff_gmm_1step_ind, robust = TRUE)
print(summ_1step_ind)
cat("\n")

# --- Model 4: Difference GMM Two-Step (no time dummies) ----------------------
cat("=== Difference GMM: Two-Step (individual only) ===\n")
diff_gmm_2step_ind <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "onestep",     # one-step for no-time-dummies variant
  transformation = "d"
)
# Actually run two-step for no-time-dummies
diff_gmm_2step_ind <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "twosteps",
  transformation = "d"
)
summ_2step_ind <- summary(diff_gmm_2step_ind, robust = TRUE)
print(summ_2step_ind)
cat("\n")

# --- Collect Results ---------------------------------------------------------
results <- rbind(
  extract_pgmm_results(diff_gmm_1step, "diff_gmm_onestep_twoways", summ_1step),
  extract_pgmm_results(diff_gmm_2step, "diff_gmm_twostep_twoways", summ_2step),
  extract_pgmm_results(diff_gmm_1step_ind, "diff_gmm_onestep_individual", summ_1step_ind),
  extract_pgmm_results(diff_gmm_2step_ind, "diff_gmm_twostep_individual", summ_2step_ind)
)

# --- Save to CSV -------------------------------------------------------------
output_path <- "/home/guhaase/projetos/panelbox/examples/gmm/R/results_difference_gmm.csv"
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

cat("\nDone.\n")
