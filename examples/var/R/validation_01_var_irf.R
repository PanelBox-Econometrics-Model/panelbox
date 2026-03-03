###############################################################################
# validation_01_var_irf.R
#
# VAR(p) estimation with lag selection, coefficient extraction, and
# Impulse Response Functions (IRFs) using the vars package.
#
# Uses the Canada dataset from the vars package for full reproducibility.
# Endogenous variables: e (employment), prod (productivity),
#                       rw (real wage), U (unemployment rate)
#
# Also estimates a VAR(2) on macro_panel.csv (single country: USA)
# with variables: gdp_growth, inflation, interest_rate
###############################################################################

library(vars)
library(lmtest)

cat("============================================================\n")
cat("  VAR Estimation and IRF Analysis - R Validation Script\n")
cat("============================================================\n\n")

# Output directory for CSV results
output_dir <- "/home/guhaase/projetos/panelbox/examples/var/R"

###############################################################################
# PART 1: VAR with Canada dataset (vars built-in)
###############################################################################

cat("--- Part 1: VAR with Canada dataset ---\n\n")

data(Canada)
cat("Canada dataset dimensions:", nrow(Canada), "x", ncol(Canada), "\n")
cat("Variables:", colnames(Canada), "\n")
cat("Time range:", start(Canada), "to", end(Canada), "\n\n")

# 1a. Lag order selection
cat("--- Lag Order Selection ---\n")
lag_select <- VARselect(Canada, lag.max = 8, type = "const")
cat("Information criteria by lag:\n")
print(lag_select$criteria)
cat("\nSelected lags:\n")
print(lag_select$selection)
cat("\n")

# Store lag selection results
lag_df <- data.frame(
  lag = 1:8,
  AIC = lag_select$criteria["AIC(n)", ],
  HQ = lag_select$criteria["HQ(n)", ],
  SC = lag_select$criteria["SC(n)", ],
  FPE = lag_select$criteria["FPE(n)", ]
)

# 1b. Estimate VAR(2) - matching PanelBox tutorial (p=2)
cat("--- VAR(2) Estimation ---\n")
var2 <- VAR(Canada, p = 2, type = "const")
cat("\nVAR(2) Summary:\n")
print(summary(var2))

# Extract coefficients per equation
coef_list <- list()
equations <- names(var2$varresult)

for (eq in equations) {
  eq_summary <- summary(var2$varresult[[eq]])
  coef_table <- as.data.frame(eq_summary$coefficients)
  coef_table$variable <- rownames(coef_table)
  coef_table$equation <- eq
  rownames(coef_table) <- NULL
  colnames(coef_table) <- c("coefficient", "std_error", "statistic", "p_value",
                             "variable", "equation")
  coef_list[[eq]] <- coef_table
}

coefs_canada <- do.call(rbind, coef_list)
coefs_canada$model_name <- "VAR2_Canada"
coefs_canada$dataset <- "Canada"
rownames(coefs_canada) <- NULL

cat("\n--- Coefficients (all equations) ---\n")
print(coefs_canada[, c("equation", "variable", "coefficient", "std_error",
                        "statistic", "p_value")])

# Information criteria for the estimated VAR(2)
cat("\n--- Model Information Criteria ---\n")
aic_val <- AIC(var2$varresult$e) + AIC(var2$varresult$prod) +
           AIC(var2$varresult$rw) + AIC(var2$varresult$U)
cat("Log-Likelihood:", logLik(var2), "\n")

# 1c. Stability check
cat("\n--- Stability (Eigenvalue Moduli) ---\n")
roots_var2 <- roots(var2)
cat("Eigenvalue moduli:", roots_var2, "\n")
cat("Max modulus:", max(roots_var2), "\n")
cat("Stable:", all(roots_var2 < 1), "\n\n")

# 1d. Impulse Response Functions (Cholesky, n.ahead=10)
cat("--- IRF: Cholesky Decomposition (n.ahead=10) ---\n")
irf_cholesky <- irf(var2, impulse = NULL, response = NULL, n.ahead = 10,
                     ortho = TRUE, boot = TRUE, runs = 200, ci = 0.95,
                     seed = 42)

# Extract IRF values into a data frame
irf_results <- data.frame()
for (imp in names(irf_cholesky$irf)) {
  irf_mat <- irf_cholesky$irf[[imp]]
  lower_mat <- irf_cholesky$Lower[[imp]]
  upper_mat <- irf_cholesky$Upper[[imp]]

  for (resp in colnames(irf_mat)) {
    temp <- data.frame(
      impulse_var = imp,
      response_var = resp,
      horizon = 0:10,
      irf_value = irf_mat[, resp],
      lower_ci = lower_mat[, resp],
      upper_ci = upper_mat[, resp],
      dataset = "Canada",
      model_name = "VAR2_Canada"
    )
    irf_results <- rbind(irf_results, temp)
  }
}

cat("IRF results extracted:", nrow(irf_results), "rows\n")
cat("\nSample IRF values (e -> e):\n")
print(irf_results[irf_results$impulse_var == "e" &
                   irf_results$response_var == "e", ])

# 1e. Cumulative IRF
cat("\n--- Cumulative IRF (n.ahead=10) ---\n")
irf_cumul <- irf(var2, impulse = NULL, response = NULL, n.ahead = 10,
                  ortho = TRUE, cumulative = TRUE, boot = TRUE,
                  runs = 200, ci = 0.95, seed = 42)

cirf_results <- data.frame()
for (imp in names(irf_cumul$irf)) {
  irf_mat <- irf_cumul$irf[[imp]]
  lower_mat <- irf_cumul$Lower[[imp]]
  upper_mat <- irf_cumul$Upper[[imp]]

  for (resp in colnames(irf_mat)) {
    temp <- data.frame(
      impulse_var = imp,
      response_var = resp,
      horizon = 0:10,
      irf_value = irf_mat[, resp],
      lower_ci = lower_mat[, resp],
      upper_ci = upper_mat[, resp],
      dataset = "Canada",
      model_name = "VAR2_Canada_cumulative"
    )
    cirf_results <- rbind(cirf_results, temp)
  }
}

cat("Cumulative IRF results:", nrow(cirf_results), "rows\n")

###############################################################################
# PART 2: VAR with macro_panel (USA only)
###############################################################################

cat("\n--- Part 2: VAR with macro_panel (USA) ---\n\n")

macro_file <- "/home/guhaase/projetos/panelbox/examples/var/data/macro_panel.csv"
if (file.exists(macro_file)) {
  macro <- read.csv(macro_file, stringsAsFactors = FALSE)
  cat("macro_panel dimensions:", nrow(macro), "x", ncol(macro), "\n")
  cat("Countries:", unique(macro$country)[1:5], "...\n")

  # Filter USA
  usa <- macro[macro$country == "USA", ]
  usa <- usa[order(usa$quarter), ]
  cat("USA observations:", nrow(usa), "\n")

  # Create time series
  endog_vars <- c("gdp_growth", "inflation", "interest_rate")
  usa_ts <- ts(usa[, endog_vars], start = c(2010, 1), frequency = 4)

  # Lag selection
  cat("\n--- Lag Selection (macro_panel USA) ---\n")
  lag_sel_usa <- VARselect(usa_ts, lag.max = 8, type = "const")
  print(lag_sel_usa$selection)

  # VAR(2) estimation
  cat("\n--- VAR(2) Estimation (macro_panel USA) ---\n")
  var2_usa <- VAR(usa_ts, p = 2, type = "const")
  print(summary(var2_usa))

  # Extract coefficients
  coef_list_usa <- list()
  for (eq in names(var2_usa$varresult)) {
    eq_summary <- summary(var2_usa$varresult[[eq]])
    coef_table <- as.data.frame(eq_summary$coefficients)
    coef_table$variable <- rownames(coef_table)
    coef_table$equation <- eq
    rownames(coef_table) <- NULL
    colnames(coef_table) <- c("coefficient", "std_error", "statistic", "p_value",
                               "variable", "equation")
    coef_list_usa[[eq]] <- coef_table
  }
  coefs_usa <- do.call(rbind, coef_list_usa)
  coefs_usa$model_name <- "VAR2_macro_USA"
  coefs_usa$dataset <- "macro_panel_USA"
  rownames(coefs_usa) <- NULL

  # Stability
  roots_usa <- roots(var2_usa)
  cat("Max eigenvalue modulus (USA):", max(roots_usa), "\n")
  cat("Stable:", all(roots_usa < 1), "\n")

  # IRF (Cholesky) - same ordering as PanelBox tutorials
  cat("\n--- IRF: Cholesky (macro USA, n.ahead=10) ---\n")
  irf_usa <- irf(var2_usa,
                  impulse = NULL, response = NULL,
                  n.ahead = 10, ortho = TRUE,
                  boot = TRUE, runs = 200, ci = 0.95, seed = 42)

  irf_usa_df <- data.frame()
  for (imp in names(irf_usa$irf)) {
    irf_mat <- irf_usa$irf[[imp]]
    lower_mat <- irf_usa$Lower[[imp]]
    upper_mat <- irf_usa$Upper[[imp]]

    for (resp in colnames(irf_mat)) {
      temp <- data.frame(
        impulse_var = imp,
        response_var = resp,
        horizon = 0:10,
        irf_value = irf_mat[, resp],
        lower_ci = lower_mat[, resp],
        upper_ci = upper_mat[, resp],
        dataset = "macro_panel_USA",
        model_name = "VAR2_macro_USA"
      )
      irf_usa_df <- rbind(irf_usa_df, temp)
    }
  }

  cat("IRF results (macro USA):", nrow(irf_usa_df), "rows\n")

} else {
  cat("WARNING: macro_panel.csv not found. Skipping Part 2.\n")
  coefs_usa <- data.frame()
  irf_usa_df <- data.frame()
}

###############################################################################
# PART 3: Save all results to CSV
###############################################################################

cat("\n============================================================\n")
cat("  Saving Results\n")
cat("============================================================\n\n")

# 3a. Coefficients CSV
all_coefs <- rbind(coefs_canada,
                   if (nrow(coefs_usa) > 0) coefs_usa else NULL)
all_coefs <- all_coefs[, c("model_name", "dataset", "equation", "variable",
                            "coefficient", "std_error", "statistic", "p_value")]
coef_file <- file.path(output_dir, "results_var_coefficients.csv")
write.csv(all_coefs, coef_file, row.names = FALSE)
cat("Coefficients saved to:", coef_file, "\n")

# 3b. Lag selection CSV
lag_file <- file.path(output_dir, "results_var_lagselection.csv")
write.csv(lag_df, lag_file, row.names = FALSE)
cat("Lag selection saved to:", lag_file, "\n")

# 3c. IRF CSV (all)
all_irf <- rbind(irf_results, cirf_results,
                 if (nrow(irf_usa_df) > 0) irf_usa_df else NULL)
irf_file <- file.path(output_dir, "results_var_irf.csv")
write.csv(all_irf, irf_file, row.names = FALSE)
cat("IRF results saved to:", irf_file, "\n")

# 3d. Combined results CSV (for main validation)
results_var <- data.frame()

# Add VAR coefficients in standard format
for (i in seq_len(nrow(all_coefs))) {
  results_var <- rbind(results_var, data.frame(
    model_name = all_coefs$model_name[i],
    variable = paste0(all_coefs$equation[i], ":", all_coefs$variable[i]),
    coefficient = all_coefs$coefficient[i],
    std_error = all_coefs$std_error[i],
    statistic = all_coefs$statistic[i],
    p_value = all_coefs$p_value[i],
    metric_type = "coefficient"
  ))
}

# Add stability info
results_var <- rbind(results_var, data.frame(
  model_name = "VAR2_Canada",
  variable = "max_eigenvalue_modulus",
  coefficient = max(roots_var2),
  std_error = NA,
  statistic = NA,
  p_value = NA,
  metric_type = "stability"
))

results_file <- file.path(output_dir, "results_var.csv")
write.csv(results_var, results_file, row.names = FALSE)
cat("Combined results saved to:", results_file, "\n")

cat("\n============================================================\n")
cat("  Script completed successfully!\n")
cat("============================================================\n")
