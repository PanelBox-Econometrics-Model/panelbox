###############################################################################
# validation_03_vecm.R
#
# Johansen Cointegration Test and VECM estimation using urca and vars packages.
#
# Uses the Canada dataset from the vars package for full reproducibility.
# Also tests on macro_panel.csv (single country: USA).
#
# Johansen test: urca::ca.jo() with trace and maxeigen tests
# VECM: vars::vec2var() to convert VECM to VAR-level representation
###############################################################################

library(vars)
library(urca)
library(lmtest)

cat("============================================================\n")
cat("  VECM and Cointegration - R Validation Script\n")
cat("============================================================\n\n")

# Output directory
output_dir <- "/home/guhaase/projetos/panelbox/examples/var/R"

###############################################################################
# PART 1: Johansen Cointegration Test - Canada dataset
###############################################################################

cat("--- Part 1: Canada dataset ---\n\n")

data(Canada)

# 1a. Johansen Trace Test (K=2, type="trace")
cat("--- Johansen Trace Test (K=2) ---\n")
jo_trace <- ca.jo(Canada, type = "trace", ecdet = "const", K = 2)
cat("\nJohansen Trace Test Summary:\n")
print(summary(jo_trace))

# Extract trace test results
trace_stats <- jo_trace@teststat
trace_cvals <- jo_trace@cval
trace_eigenvalues <- jo_trace@lambda

cat("\nTrace Statistics:\n")
print(trace_stats)
cat("\nCritical Values (10%, 5%, 1%):\n")
print(trace_cvals)
cat("\nEigenvalues:\n")
print(trace_eigenvalues)

# Build trace results data frame
# ca.jo returns K+1 eigenvalues when ecdet="const" but only K test statistics
# The eigenvalues that correspond to the test rows are the first n_ranks
n_ranks <- length(trace_stats)
trace_df <- data.frame(
  rank = (n_ranks - 1):0,
  trace_statistic = trace_stats,
  critical_value_10pct = trace_cvals[, 1],
  critical_value_5pct = trace_cvals[, 2],
  critical_value_1pct = trace_cvals[, 3],
  eigenvalue = trace_eigenvalues[1:n_ranks],
  reject_5pct = trace_stats > trace_cvals[, 2],
  test_type = "trace",
  dataset = "Canada",
  model_name = "VECM_Canada"
)

cat("\nTrace Test Results (r=0,...,K-1):\n")
print(trace_df[, c("rank", "trace_statistic", "critical_value_5pct",
                    "eigenvalue", "reject_5pct")])

# 1b. Johansen Max-Eigenvalue Test
cat("\n--- Johansen Max-Eigenvalue Test (K=2) ---\n")
jo_eigen <- ca.jo(Canada, type = "eigen", ecdet = "const", K = 2)
cat("\nMax-Eigenvalue Test Summary:\n")
print(summary(jo_eigen))

eigen_stats <- jo_eigen@teststat
eigen_cvals <- jo_eigen@cval
eigen_eigenvalues <- jo_eigen@lambda

n_eigen <- length(eigen_stats)
eigen_df <- data.frame(
  rank = (n_eigen - 1):0,
  maxeigen_statistic = eigen_stats,
  critical_value_10pct = eigen_cvals[, 1],
  critical_value_5pct = eigen_cvals[, 2],
  critical_value_1pct = eigen_cvals[, 3],
  eigenvalue = eigen_eigenvalues[1:n_eigen],
  reject_5pct = eigen_stats > eigen_cvals[, 2],
  test_type = "maxeigen",
  dataset = "Canada",
  model_name = "VECM_Canada"
)

cat("\nMax-Eigenvalue Test Results:\n")
print(eigen_df[, c("rank", "maxeigen_statistic", "critical_value_5pct",
                    "eigenvalue", "reject_5pct")])

# Determine selected rank from trace test
selected_rank <- sum(trace_df$reject_5pct)
cat(sprintf("\nSelected cointegration rank (trace, 5%%): r = %d\n", selected_rank))

# 1c. Cointegrating vectors (beta) and loading matrix (alpha)
cat("\n--- Cointegrating Vectors ---\n")
cat("Beta (normalized cointegrating vectors):\n")
print(jo_trace@V)
cat("\nAlpha (loading/adjustment coefficients):\n")
print(jo_trace@W)

# 1d. Convert VECM to VAR level representation
cat("\n--- VECM to VAR conversion ---\n")
if (selected_rank > 0 && selected_rank < ncol(Canada)) {
  vecm_var <- vec2var(jo_trace, r = selected_rank)
  cat("VECM converted to VAR. Coefficient matrices:\n")
  for (i in seq_along(vecm_var$A)) {
    cat(sprintf("\nA%d:\n", i))
    print(round(vecm_var$A[[i]], 6))
  }

  # IRF from VECM
  cat("\n--- IRF from VECM (n.ahead=10) ---\n")
  irf_vecm <- irf(vecm_var, n.ahead = 10, ortho = TRUE,
                   boot = TRUE, runs = 200, ci = 0.95, seed = 42)

  irf_vecm_df <- data.frame()
  for (imp in names(irf_vecm$irf)) {
    irf_mat <- irf_vecm$irf[[imp]]
    lower_mat <- irf_vecm$Lower[[imp]]
    upper_mat <- irf_vecm$Upper[[imp]]

    for (resp in colnames(irf_mat)) {
      temp <- data.frame(
        impulse_var = imp,
        response_var = resp,
        horizon = 0:10,
        irf_value = irf_mat[, resp],
        lower_ci = lower_mat[, resp],
        upper_ci = upper_mat[, resp],
        dataset = "Canada",
        model_name = "VECM_Canada"
      )
      irf_vecm_df <- rbind(irf_vecm_df, temp)
    }
  }
  cat("VECM IRF results:", nrow(irf_vecm_df), "rows\n")

  # FEVD from VECM
  cat("\n--- FEVD from VECM (n.ahead=10) ---\n")
  fevd_vecm <- fevd(vecm_var, n.ahead = 10)
  for (resp in names(fevd_vecm)) {
    cat(sprintf("FEVD for %s:\n", resp))
    print(round(fevd_vecm[[resp]], 4))
    cat("\n")
  }
} else {
  cat("No cointegration found or full rank. Skipping VECM conversion.\n")
  irf_vecm_df <- data.frame()
}

###############################################################################
# PART 2: Johansen Test on macro_panel (USA)
###############################################################################

cat("\n--- Part 2: macro_panel (USA) ---\n\n")

macro_file <- "/home/guhaase/projetos/panelbox/examples/var/data/macro_panel.csv"
trace_usa_df <- data.frame()
eigen_usa_df <- data.frame()

if (file.exists(macro_file)) {
  macro <- read.csv(macro_file, stringsAsFactors = FALSE)
  usa <- macro[macro$country == "USA", ]
  usa <- usa[order(usa$quarter), ]

  endog_vars <- c("gdp_growth", "inflation", "interest_rate")
  usa_ts <- ts(usa[, endog_vars], start = c(2010, 1), frequency = 4)

  # Johansen Trace Test
  cat("--- Johansen Trace Test (macro USA, K=2) ---\n")
  jo_trace_usa <- ca.jo(usa_ts, type = "trace", ecdet = "const", K = 2)
  print(summary(jo_trace_usa))

  ts_usa <- jo_trace_usa@teststat
  cv_usa <- jo_trace_usa@cval
  ev_usa <- jo_trace_usa@lambda

  n_ts <- length(ts_usa)
  trace_usa_df <- data.frame(
    rank = (n_ts - 1):0,
    trace_statistic = ts_usa,
    critical_value_10pct = cv_usa[, 1],
    critical_value_5pct = cv_usa[, 2],
    critical_value_1pct = cv_usa[, 3],
    eigenvalue = ev_usa[1:n_ts],
    reject_5pct = ts_usa > cv_usa[, 2],
    test_type = "trace",
    dataset = "macro_panel_USA",
    model_name = "VECM_macro_USA"
  )

  cat("\nTrace Test Results (macro USA):\n")
  print(trace_usa_df[, c("rank", "trace_statistic", "critical_value_5pct",
                          "eigenvalue", "reject_5pct")])

  # Johansen Max-Eigenvalue Test
  cat("\n--- Johansen Max-Eigenvalue Test (macro USA, K=2) ---\n")
  jo_eigen_usa <- ca.jo(usa_ts, type = "eigen", ecdet = "const", K = 2)
  print(summary(jo_eigen_usa))

  es_usa <- jo_eigen_usa@teststat
  ecv_usa <- jo_eigen_usa@cval

  n_es <- length(es_usa)
  eigen_usa_df <- data.frame(
    rank = (n_es - 1):0,
    maxeigen_statistic = es_usa,
    critical_value_10pct = ecv_usa[, 1],
    critical_value_5pct = ecv_usa[, 2],
    critical_value_1pct = ecv_usa[, 3],
    eigenvalue = jo_eigen_usa@lambda[1:n_es],
    reject_5pct = es_usa > ecv_usa[, 2],
    test_type = "maxeigen",
    dataset = "macro_panel_USA",
    model_name = "VECM_macro_USA"
  )

  selected_rank_usa <- sum(trace_usa_df$reject_5pct)
  cat(sprintf("\nSelected rank (macro USA, trace, 5%%): r = %d\n",
              selected_rank_usa))

  # VECM if cointegration found
  if (selected_rank_usa > 0 && selected_rank_usa < length(endog_vars)) {
    cat("\n--- VECM from macro USA ---\n")
    vecm_usa <- vec2var(jo_trace_usa, r = selected_rank_usa)
    cat("VECM converted to VAR.\n")
    for (i in seq_along(vecm_usa$A)) {
      cat(sprintf("A%d:\n", i))
      print(round(vecm_usa$A[[i]], 6))
      cat("\n")
    }
  } else {
    cat("No cointegration found in macro USA data.\n")
  }
} else {
  cat("WARNING: macro_panel.csv not found. Skipping Part 2.\n")
}

###############################################################################
# PART 3: Save Results
###############################################################################

cat("\n============================================================\n")
cat("  Saving Results\n")
cat("============================================================\n\n")

# 3a. Johansen test results CSV
johansen_all <- rbind(
  trace_df[, c("rank", "test_type", "eigenvalue", "dataset", "model_name")],
  eigen_df[, c("rank", "test_type", "eigenvalue", "dataset", "model_name")]
)

# Create a comprehensive VECM results file
vecm_results <- data.frame()

# Add trace test
for (i in seq_len(nrow(trace_df))) {
  vecm_results <- rbind(vecm_results, data.frame(
    model_name = trace_df$model_name[i],
    variable = paste0("trace_r", trace_df$rank[i]),
    coefficient = trace_df$trace_statistic[i],
    std_error = NA,
    statistic = trace_df$trace_statistic[i],
    p_value = NA,
    metric_type = "johansen_trace",
    critical_value_5pct = trace_df$critical_value_5pct[i],
    eigenvalue = trace_df$eigenvalue[i]
  ))
}

# Add eigenvalue test
for (i in seq_len(nrow(eigen_df))) {
  vecm_results <- rbind(vecm_results, data.frame(
    model_name = eigen_df$model_name[i],
    variable = paste0("maxeigen_r", eigen_df$rank[i]),
    coefficient = eigen_df$maxeigen_statistic[i],
    std_error = NA,
    statistic = eigen_df$maxeigen_statistic[i],
    p_value = NA,
    metric_type = "johansen_maxeigen",
    critical_value_5pct = eigen_df$critical_value_5pct[i],
    eigenvalue = eigen_df$eigenvalue[i]
  ))
}

# Add USA trace results if available
if (nrow(trace_usa_df) > 0) {
  for (i in seq_len(nrow(trace_usa_df))) {
    vecm_results <- rbind(vecm_results, data.frame(
      model_name = trace_usa_df$model_name[i],
      variable = paste0("trace_r", trace_usa_df$rank[i]),
      coefficient = trace_usa_df$trace_statistic[i],
      std_error = NA,
      statistic = trace_usa_df$trace_statistic[i],
      p_value = NA,
      metric_type = "johansen_trace",
      critical_value_5pct = trace_usa_df$critical_value_5pct[i],
      eigenvalue = trace_usa_df$eigenvalue[i]
    ))
  }
}

if (nrow(eigen_usa_df) > 0) {
  for (i in seq_len(nrow(eigen_usa_df))) {
    vecm_results <- rbind(vecm_results, data.frame(
      model_name = eigen_usa_df$model_name[i],
      variable = paste0("maxeigen_r", eigen_usa_df$rank[i]),
      coefficient = eigen_usa_df$maxeigen_statistic[i],
      std_error = NA,
      statistic = eigen_usa_df$maxeigen_statistic[i],
      p_value = NA,
      metric_type = "johansen_maxeigen",
      critical_value_5pct = eigen_usa_df$critical_value_5pct[i],
      eigenvalue = eigen_usa_df$eigenvalue[i]
    ))
  }
}

# Save VECM-specific results
vecm_file <- file.path(output_dir, "results_vecm.csv")
write.csv(vecm_results, vecm_file, row.names = FALSE)
cat("VECM results saved to:", vecm_file, "\n")

# Save VECM IRFs if available
if (nrow(irf_vecm_df) > 0) {
  irf_vecm_file <- file.path(output_dir, "results_vecm_irf.csv")
  write.csv(irf_vecm_df, irf_vecm_file, row.names = FALSE)
  cat("VECM IRF results saved to:", irf_vecm_file, "\n")
}

# Append to combined results_var.csv
existing_file <- file.path(output_dir, "results_var.csv")
if (file.exists(existing_file)) {
  existing <- read.csv(existing_file, stringsAsFactors = FALSE)
  # Add VECM results (without extra columns)
  vecm_for_combined <- vecm_results[, c("model_name", "variable", "coefficient",
                                         "std_error", "statistic", "p_value",
                                         "metric_type")]
  combined <- rbind(existing, vecm_for_combined)
  write.csv(combined, existing_file, row.names = FALSE)
  cat("Combined results updated in:", existing_file, "\n")
} else {
  vecm_for_combined <- vecm_results[, c("model_name", "variable", "coefficient",
                                         "std_error", "statistic", "p_value",
                                         "metric_type")]
  write.csv(vecm_for_combined, existing_file, row.names = FALSE)
  cat("Combined results saved to:", existing_file, "\n")
}

cat("\n============================================================\n")
cat("  Script completed successfully!\n")
cat("============================================================\n")
