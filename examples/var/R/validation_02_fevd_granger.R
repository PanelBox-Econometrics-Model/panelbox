###############################################################################
# validation_02_fevd_granger.R
#
# Forecast Error Variance Decomposition (FEVD) and Granger Causality tests
# using the vars package.
#
# Uses the Canada dataset from the vars package for full reproducibility.
# Also estimates on macro_panel.csv (single country: USA).
###############################################################################

library(vars)
library(lmtest)

cat("============================================================\n")
cat("  FEVD and Granger Causality - R Validation Script\n")
cat("============================================================\n\n")

# Output directory
output_dir <- "/home/guhaase/projetos/panelbox/examples/var/R"

###############################################################################
# PART 1: FEVD and Granger with Canada dataset
###############################################################################

cat("--- Part 1: Canada dataset ---\n\n")

data(Canada)

# Estimate VAR(2) (same as validation_01)
var2 <- VAR(Canada, p = 2, type = "const")

# 1a. Forecast Error Variance Decomposition (n.ahead=10)
cat("--- FEVD (n.ahead=10) ---\n")
fevd_result <- fevd(var2, n.ahead = 10)

# Extract FEVD into a data frame
fevd_df <- data.frame()
for (resp in names(fevd_result)) {
  mat <- fevd_result[[resp]]
  for (h in seq_len(nrow(mat))) {
    for (imp in colnames(mat)) {
      fevd_df <- rbind(fevd_df, data.frame(
        response_var = resp,
        impulse_var = imp,
        horizon = h,
        percentage = mat[h, imp],
        dataset = "Canada",
        model_name = "VAR2_Canada"
      ))
    }
  }
}

cat("FEVD results extracted:", nrow(fevd_df), "rows\n\n")

# Print FEVD tables
for (resp in names(fevd_result)) {
  cat(sprintf("FEVD for %s:\n", resp))
  print(round(fevd_result[[resp]], 4))
  cat("\n")
}

# 1b. Granger Causality Tests
cat("--- Granger Causality Tests ---\n\n")

# Test each variable pair using vars::causality()
vars_names <- colnames(Canada)
granger_results <- data.frame()

for (cause_var in vars_names) {
  # vars::causality() tests if cause_var Granger-causes the rest
  gc_test <- causality(var2, cause = cause_var)

  # Granger test (F-test)
  granger_stat <- gc_test$Granger$statistic
  granger_p <- gc_test$Granger$p.value
  granger_df1 <- gc_test$Granger$parameter[1]
  granger_df2 <- gc_test$Granger$parameter[2]

  cat(sprintf("Granger: %s -> {others}: F=%.4f, df=(%d,%d), p=%.6f\n",
              cause_var, granger_stat, granger_df1, granger_df2, granger_p))

  granger_results <- rbind(granger_results, data.frame(
    cause_var = cause_var,
    effect_var = paste(setdiff(vars_names, cause_var), collapse = ","),
    test_type = "Granger",
    f_statistic = as.numeric(granger_stat),
    df1 = as.numeric(granger_df1),
    df2 = as.numeric(granger_df2),
    p_value = as.numeric(granger_p),
    dataset = "Canada",
    model_name = "VAR2_Canada"
  ))

  # Instantaneous causality test
  inst_stat <- gc_test$Instant$statistic
  inst_p <- gc_test$Instant$p.value
  inst_df <- gc_test$Instant$parameter

  cat(sprintf("Instantaneous: %s <-> {others}: chi2=%.4f, df=%d, p=%.6f\n\n",
              cause_var, inst_stat, inst_df, inst_p))

  granger_results <- rbind(granger_results, data.frame(
    cause_var = cause_var,
    effect_var = paste(setdiff(vars_names, cause_var), collapse = ","),
    test_type = "Instantaneous",
    f_statistic = as.numeric(inst_stat),
    df1 = as.numeric(inst_df),
    df2 = NA,
    p_value = as.numeric(inst_p),
    dataset = "Canada",
    model_name = "VAR2_Canada"
  ))
}

# 1c. Pairwise Granger causality using grangertest from lmtest
cat("--- Pairwise Granger Tests (lmtest::grangertest) ---\n\n")

pairwise_granger <- data.frame()
for (cause_var in vars_names) {
  for (effect_var in vars_names) {
    if (cause_var != effect_var) {
      gt <- grangertest(Canada[, effect_var] ~ Canada[, cause_var], order = 2)
      f_val <- gt$F[2]
      p_val <- gt$`Pr(>F)`[2]

      cat(sprintf("%s -> %s: F=%.4f, p=%.6f %s\n",
                  cause_var, effect_var, f_val, p_val,
                  ifelse(p_val < 0.05, "*", "")))

      pairwise_granger <- rbind(pairwise_granger, data.frame(
        cause_var = cause_var,
        effect_var = effect_var,
        test_type = "Pairwise_Granger",
        f_statistic = f_val,
        df1 = 2,
        df2 = NA,
        p_value = p_val,
        dataset = "Canada",
        model_name = "VAR2_Canada"
      ))
    }
  }
}
cat("\n")

###############################################################################
# PART 2: FEVD and Granger with macro_panel (USA)
###############################################################################

cat("--- Part 2: macro_panel (USA) ---\n\n")

macro_file <- "/home/guhaase/projetos/panelbox/examples/var/data/macro_panel.csv"
fevd_usa_df <- data.frame()
granger_usa <- data.frame()

if (file.exists(macro_file)) {
  macro <- read.csv(macro_file, stringsAsFactors = FALSE)
  usa <- macro[macro$country == "USA", ]
  usa <- usa[order(usa$quarter), ]

  endog_vars <- c("gdp_growth", "inflation", "interest_rate")
  usa_ts <- ts(usa[, endog_vars], start = c(2010, 1), frequency = 4)

  # VAR(2) estimation
  var2_usa <- VAR(usa_ts, p = 2, type = "const")

  # FEVD
  cat("--- FEVD (macro USA, n.ahead=10) ---\n")
  fevd_usa <- fevd(var2_usa, n.ahead = 10)

  for (resp in names(fevd_usa)) {
    mat <- fevd_usa[[resp]]
    cat(sprintf("FEVD for %s:\n", resp))
    print(round(mat, 4))
    cat("\n")

    for (h in seq_len(nrow(mat))) {
      for (imp in colnames(mat)) {
        fevd_usa_df <- rbind(fevd_usa_df, data.frame(
          response_var = resp,
          impulse_var = imp,
          horizon = h,
          percentage = mat[h, imp],
          dataset = "macro_panel_USA",
          model_name = "VAR2_macro_USA"
        ))
      }
    }
  }

  # Granger causality (vars::causality)
  cat("--- Granger Causality (macro USA) ---\n")
  for (cause_var in endog_vars) {
    gc_test <- causality(var2_usa, cause = cause_var)

    granger_stat <- gc_test$Granger$statistic
    granger_p <- gc_test$Granger$p.value

    cat(sprintf("Granger: %s -> {others}: F=%.4f, p=%.6f\n",
                cause_var, granger_stat, granger_p))

    granger_usa <- rbind(granger_usa, data.frame(
      cause_var = cause_var,
      effect_var = paste(setdiff(endog_vars, cause_var), collapse = ","),
      test_type = "Granger",
      f_statistic = as.numeric(granger_stat),
      df1 = as.numeric(gc_test$Granger$parameter[1]),
      df2 = as.numeric(gc_test$Granger$parameter[2]),
      p_value = as.numeric(granger_p),
      dataset = "macro_panel_USA",
      model_name = "VAR2_macro_USA"
    ))
  }

  # Pairwise Granger (lmtest)
  cat("\n--- Pairwise Granger (macro USA) ---\n")
  for (cause_var in endog_vars) {
    for (effect_var in endog_vars) {
      if (cause_var != effect_var) {
        gt <- grangertest(usa_ts[, effect_var] ~ usa_ts[, cause_var], order = 2)
        f_val <- gt$F[2]
        p_val <- gt$`Pr(>F)`[2]

        cat(sprintf("%s -> %s: F=%.4f, p=%.6f %s\n",
                    cause_var, effect_var, f_val, p_val,
                    ifelse(p_val < 0.05, "*", "")))

        granger_usa <- rbind(granger_usa, data.frame(
          cause_var = cause_var,
          effect_var = effect_var,
          test_type = "Pairwise_Granger",
          f_statistic = f_val,
          df1 = 2,
          df2 = NA,
          p_value = p_val,
          dataset = "macro_panel_USA",
          model_name = "VAR2_macro_USA"
        ))
      }
    }
  }
  cat("\n")

} else {
  cat("WARNING: macro_panel.csv not found. Skipping Part 2.\n\n")
}

###############################################################################
# PART 3: Save Results
###############################################################################

cat("============================================================\n")
cat("  Saving Results\n")
cat("============================================================\n\n")

# 3a. FEVD CSV
all_fevd <- rbind(fevd_df,
                  if (nrow(fevd_usa_df) > 0) fevd_usa_df else NULL)
fevd_file <- file.path(output_dir, "results_fevd.csv")
write.csv(all_fevd, fevd_file, row.names = FALSE)
cat("FEVD results saved to:", fevd_file, "\n")

# 3b. Granger CSV
all_granger <- rbind(granger_results, pairwise_granger,
                     if (nrow(granger_usa) > 0) granger_usa else NULL)
granger_file <- file.path(output_dir, "results_granger.csv")
write.csv(all_granger, granger_file, row.names = FALSE)
cat("Granger results saved to:", granger_file, "\n")

# 3c. Append to combined results_var.csv
results_var <- data.frame()

# Add FEVD summary (horizon 1 and 10 for Canada)
for (h in c(1, 5, 10)) {
  sub <- fevd_df[fevd_df$horizon == h, ]
  for (i in seq_len(nrow(sub))) {
    results_var <- rbind(results_var, data.frame(
      model_name = "VAR2_Canada",
      variable = paste0("FEVD_h", h, "_", sub$response_var[i], "_<-_",
                        sub$impulse_var[i]),
      coefficient = sub$percentage[i],
      std_error = NA,
      statistic = NA,
      p_value = NA,
      metric_type = "fevd"
    ))
  }
}

# Add Granger results
for (i in seq_len(nrow(granger_results))) {
  results_var <- rbind(results_var, data.frame(
    model_name = "VAR2_Canada",
    variable = paste0(granger_results$test_type[i], "_",
                      granger_results$cause_var[i]),
    coefficient = granger_results$f_statistic[i],
    std_error = NA,
    statistic = granger_results$f_statistic[i],
    p_value = granger_results$p_value[i],
    metric_type = "granger"
  ))
}

# Append to existing results_var.csv if it exists
existing_file <- file.path(output_dir, "results_var.csv")
if (file.exists(existing_file)) {
  existing <- read.csv(existing_file, stringsAsFactors = FALSE)
  results_var <- rbind(existing, results_var)
}
write.csv(results_var, existing_file, row.names = FALSE)
cat("Combined results updated in:", existing_file, "\n")

cat("\n============================================================\n")
cat("  Script completed successfully!\n")
cat("============================================================\n")
