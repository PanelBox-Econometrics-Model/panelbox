###############################################################################
# Validation 03 - GMM Diagnostics
#
# Comprehensive diagnostic comparison:
# - Instrument counts (collapsed vs uncollapsed)
# - Specification tests (AR1, AR2, Sargan, Hansen)
# - Difference GMM vs System GMM
# - One-step vs Two-step consistency
# - Windmeijer-corrected vs uncorrected SE
#
# Dataset: abdata (N=140 firms, T=9 years, unbalanced)
###############################################################################

library(plm)

# --- Data Loading -----------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/gmm/data/abdata.csv"
abdata <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("Rows:", nrow(abdata), "\n")
cat("Firms:", length(unique(abdata$firm)), "\n")
cat("Years:", paste(sort(unique(abdata$year)), collapse=", "), "\n\n")

pab <- pdata.frame(abdata, index = c("firm", "year"))

# --- Helper Functions --------------------------------------------------------
get_diagnostics <- function(model, model_name, robust_summ, non_robust_summ = NULL) {
  coefs_robust <- robust_summ$coefficients
  var_names <- rownames(coefs_robust)

  # pgmm residuals is a list (one vector per group)
  n_groups <- length(model$residuals)
  n_obs <- sum(sapply(model$residuals, length))
  n_instruments <- ncol(model$W[[1]])

  # AR tests and Sargan/Hansen from summary
  ar1 <- robust_summ$m1
  ar2 <- robust_summ$m2
  sar <- robust_summ$sargan

  # Get non-robust SE for comparison if available
  se_non_robust <- if (!is.null(non_robust_summ)) {
    non_robust_summ$coefficients[, "Std. Error"]
  } else {
    rep(NA, length(var_names))
  }

  results <- data.frame(
    model_name       = model_name,
    variable         = var_names,
    coefficient      = coefs_robust[, "Estimate"],
    std_error_robust = coefs_robust[, "Std. Error"],
    std_error_conventional = se_non_robust,
    z_statistic      = coefs_robust[, "z-value"],
    p_value          = coefs_robust[, "Pr(>|z|)"],
    n_obs            = n_obs,
    n_groups         = n_groups,
    n_instruments    = n_instruments,
    instrument_ratio = round(n_instruments / n_groups, 3),
    ar1_statistic    = if (!is.null(ar1)) ar1$statistic else NA,
    ar1_pvalue       = if (!is.null(ar1)) ar1$p.value else NA,
    ar2_statistic    = if (!is.null(ar2)) ar2$statistic else NA,
    ar2_pvalue       = if (!is.null(ar2)) ar2$p.value else NA,
    sargan_statistic = NA,
    sargan_pvalue    = NA,
    sargan_df        = NA,
    hansen_statistic = NA,
    hansen_pvalue    = NA,
    hansen_df        = NA,
    stringsAsFactors = FALSE,
    row.names = NULL
  )

  if (grepl("twostep", model_name)) {
    results$hansen_statistic <- if (!is.null(sar)) sar$statistic else NA
    results$hansen_pvalue    <- if (!is.null(sar)) sar$p.value else NA
    results$hansen_df        <- if (!is.null(sar)) sar$parameter else NA
  } else {
    results$sargan_statistic <- if (!is.null(sar)) sar$statistic else NA
    results$sargan_pvalue    <- if (!is.null(sar)) sar$p.value else NA
    results$sargan_df        <- if (!is.null(sar)) sar$parameter else NA
  }

  return(results)
}

all_results <- data.frame()

# =============================================================================
# PART 1: Instrument Count Comparison (collapsed vs uncollapsed not available
#         in plm::pgmm, so we vary lag depth instead)
# =============================================================================

cat("=================================================================\n")
cat("PART 1: Instrument Depth Comparison\n")
cat("=================================================================\n\n")

# Full instrument set (lag 2:99)
cat("--- Difference GMM Two-Step: Full instruments (lag 2:99) ---\n")
m_full <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "d"
)
s_full_r <- summary(m_full, robust = TRUE)
s_full   <- summary(m_full, robust = FALSE)
print(s_full_r)
cat("Instruments:", ncol(m_full$W[[1]]), "\n\n")

res_full <- get_diagnostics(m_full, "diff_twostep_full_instruments", s_full_r, s_full)
all_results <- rbind(all_results, res_full)

# Restricted instrument set (lag 2:4)
cat("--- Difference GMM Two-Step: Restricted instruments (lag 2:4) ---\n")
m_rest <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:4),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "d"
)
s_rest_r <- summary(m_rest, robust = TRUE)
s_rest   <- summary(m_rest, robust = FALSE)
print(s_rest_r)
cat("Instruments:", ncol(m_rest$W[[1]]), "\n\n")

res_rest <- get_diagnostics(m_rest, "diff_twostep_restricted_instruments", s_rest_r, s_rest)
all_results <- rbind(all_results, res_rest)

# Minimal instrument set (lag 2:2)
cat("--- Difference GMM Two-Step: Minimal instruments (lag 2:2) ---\n")
m_min <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:2),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "d"
)
s_min_r <- summary(m_min, robust = TRUE)
s_min   <- summary(m_min, robust = FALSE)
print(s_min_r)
cat("Instruments:", ncol(m_min$W[[1]]), "\n\n")

res_min <- get_diagnostics(m_min, "diff_twostep_minimal_instruments", s_min_r, s_min)
all_results <- rbind(all_results, res_min)

# =============================================================================
# PART 2: Specification Tests Comparison
# =============================================================================

cat("=================================================================\n")
cat("PART 2: Difference GMM vs System GMM (twoways, two-step)\n")
cat("=================================================================\n\n")

# Difference GMM
cat("--- Difference GMM Two-Step (twoways) ---\n")
m_diff <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "d"
)
s_diff_r <- summary(m_diff, robust = TRUE)
s_diff   <- summary(m_diff, robust = FALSE)
print(s_diff_r)

res_diff <- get_diagnostics(m_diff, "diff_gmm_twostep_twoways", s_diff_r, s_diff)
all_results <- rbind(all_results, res_diff)

# System GMM
cat("\n--- System GMM Two-Step (twoways) ---\n")
m_sys <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "ld"
)
s_sys_r <- summary(m_sys, robust = TRUE)
s_sys   <- summary(m_sys, robust = FALSE)
print(s_sys_r)

res_sys <- get_diagnostics(m_sys, "sys_gmm_twostep_twoways", s_sys_r, s_sys)
all_results <- rbind(all_results, res_sys)

# =============================================================================
# PART 3: One-step vs Two-step Consistency Check
# =============================================================================

cat("=================================================================\n")
cat("PART 3: One-Step vs Two-Step Consistency\n")
cat("=================================================================\n\n")

# One-step Difference
cat("--- Difference GMM One-Step (individual) ---\n")
m_diff_1 <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "onestep",
  transformation = "d"
)
s_diff_1r <- summary(m_diff_1, robust = TRUE)
s_diff_1  <- summary(m_diff_1, robust = FALSE)
print(s_diff_1r)

res_diff_1 <- get_diagnostics(m_diff_1, "diff_gmm_onestep_individual", s_diff_1r, s_diff_1)
all_results <- rbind(all_results, res_diff_1)

# Two-step Difference
cat("\n--- Difference GMM Two-Step (individual) ---\n")
m_diff_2 <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "twosteps",
  transformation = "d"
)
s_diff_2r <- summary(m_diff_2, robust = TRUE)
s_diff_2  <- summary(m_diff_2, robust = FALSE)
print(s_diff_2r)

res_diff_2 <- get_diagnostics(m_diff_2, "diff_gmm_twostep_individual", s_diff_2r, s_diff_2)
all_results <- rbind(all_results, res_diff_2)

# One-step System
cat("\n--- System GMM One-Step (individual) ---\n")
m_sys_1 <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "onestep",
  transformation = "ld"
)
s_sys_1r <- summary(m_sys_1, robust = TRUE)
s_sys_1  <- summary(m_sys_1, robust = FALSE)
print(s_sys_1r)

res_sys_1 <- get_diagnostics(m_sys_1, "sys_gmm_onestep_individual", s_sys_1r, s_sys_1)
all_results <- rbind(all_results, res_sys_1)

# Two-step System
cat("\n--- System GMM Two-Step (individual) ---\n")
m_sys_2 <- pgmm(
  n ~ lag(n, 1) + w + k | lag(n, 2:99),
  data = pab,
  effect = "individual",
  model = "twosteps",
  transformation = "ld"
)
s_sys_2r <- summary(m_sys_2, robust = TRUE)
s_sys_2  <- summary(m_sys_2, robust = FALSE)
print(s_sys_2r)

res_sys_2 <- get_diagnostics(m_sys_2, "sys_gmm_twostep_individual", s_sys_2r, s_sys_2)
all_results <- rbind(all_results, res_sys_2)

# =============================================================================
# PART 4: Windmeijer Correction Impact
# =============================================================================

cat("=================================================================\n")
cat("PART 4: Windmeijer Correction Impact (Two-Step Models)\n")
cat("=================================================================\n\n")

# Compare conventional vs robust (Windmeijer) SE for two-step models
windmeijer_comparison <- data.frame()
for (i in seq_len(nrow(all_results))) {
  if (grepl("twostep", all_results$model_name[i]) &&
      !is.na(all_results$std_error_conventional[i])) {
    windmeijer_comparison <- rbind(windmeijer_comparison, data.frame(
      model_name = all_results$model_name[i],
      variable = all_results$variable[i],
      se_conventional = all_results$std_error_conventional[i],
      se_windmeijer = all_results$std_error_robust[i],
      ratio = all_results$std_error_robust[i] / all_results$std_error_conventional[i],
      stringsAsFactors = FALSE
    ))
  }
}

if (nrow(windmeijer_comparison) > 0) {
  cat("Windmeijer correction ratios (robust_SE / conventional_SE):\n")
  cat("Values > 1 indicate Windmeijer correction inflates SE (as expected).\n\n")
  print(windmeijer_comparison)
}

# =============================================================================
# PART 5: Including ys (sales) as additional regressor
# =============================================================================

cat("\n=================================================================\n")
cat("PART 5: Extended Model with ys (sales)\n")
cat("=================================================================\n\n")

cat("--- Difference GMM Two-Step: n ~ lag(n) + w + k + ys ---\n")
m_ext <- pgmm(
  n ~ lag(n, 1) + w + k + ys | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "d"
)
s_ext_r <- summary(m_ext, robust = TRUE)
s_ext   <- summary(m_ext, robust = FALSE)
print(s_ext_r)

res_ext <- get_diagnostics(m_ext, "diff_twostep_extended_with_ys", s_ext_r, s_ext)
all_results <- rbind(all_results, res_ext)

cat("\n--- System GMM Two-Step: n ~ lag(n) + w + k + ys ---\n")
m_ext_sys <- pgmm(
  n ~ lag(n, 1) + w + k + ys | lag(n, 2:99),
  data = pab,
  effect = "twoways",
  model = "twosteps",
  transformation = "ld"
)
s_ext_sys_r <- summary(m_ext_sys, robust = TRUE)
s_ext_sys   <- summary(m_ext_sys, robust = FALSE)
print(s_ext_sys_r)

res_ext_sys <- get_diagnostics(m_ext_sys, "sys_twostep_extended_with_ys", s_ext_sys_r, s_ext_sys)
all_results <- rbind(all_results, res_ext_sys)

# =============================================================================
# Save All Results
# =============================================================================

output_path <- "/home/guhaase/projetos/panelbox/examples/gmm/R/results_gmm_diagnostics.csv"
write.csv(all_results, output_path, row.names = FALSE)
cat("\nAll results saved to:", output_path, "\n")

# --- Print instrument count summary ------------------------------------------
cat("\n=== Instrument Count Summary ===\n")
inst_summary <- unique(all_results[, c("model_name", "n_obs", "n_groups",
                                        "n_instruments", "instrument_ratio")])
print(inst_summary)

# --- Print diagnostic tests summary ------------------------------------------
cat("\n=== Diagnostic Tests Summary ===\n")
diag_summary <- unique(all_results[, c("model_name",
                                         "ar1_statistic", "ar1_pvalue",
                                         "ar2_statistic", "ar2_pvalue",
                                         "sargan_statistic", "sargan_pvalue",
                                         "hansen_statistic", "hansen_pvalue")])
print(diag_summary)

cat("\nDone.\n")
