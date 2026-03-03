# =============================================================================
# Validation Script 03: Specification Tests
# PanelBox vs R (plm)
#
# Tests: Hausman, Breusch-Pagan LM, F-test for FE, Poolability
# Datasets: nlswork.csv, firm_productivity.csv, grunfeld.csv
# =============================================================================

library(plm)
library(lmtest)

# --- Data Loading ---
nls_path <- "/home/guhaase/projetos/panelbox/examples/diagnostics/data/specification/nlswork.csv"
firm_path <- "/home/guhaase/projetos/panelbox/examples/diagnostics/data/specification/firm_productivity.csv"
grunfeld_path <- "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv"
output_path <- "/home/guhaase/projetos/panelbox/examples/diagnostics/R/results_specification.csv"

cat("=== Specification Tests Validation ===\n\n")

# --- Load datasets ---
nls <- read.csv(nls_path, stringsAsFactors = FALSE)
firm <- read.csv(firm_path, stringsAsFactors = FALSE)
grunfeld <- read.csv(grunfeld_path, stringsAsFactors = FALSE)

cat("NLS Work:", nrow(nls), "rows,", length(unique(nls$idcode)), "workers\n")
cat("Firm Productivity:", nrow(firm), "rows,", length(unique(firm$firm_id)), "firms\n")
cat("Grunfeld:", nrow(grunfeld), "rows,", length(unique(grunfeld$firm)), "firms\n\n")

# Convert to panel data frames
nls_pdf <- pdata.frame(nls, index = c("idcode", "year"), drop.index = FALSE)
firm_pdf <- pdata.frame(firm, index = c("firm_id", "year"), drop.index = FALSE)
grunfeld_pdf <- pdata.frame(grunfeld, index = c("firm", "year"), drop.index = FALSE)

# --- Initialize results storage ---
results <- data.frame(
  test_name = character(),
  dataset = character(),
  model_spec = character(),
  statistic = numeric(),
  p_value = numeric(),
  df = character(),
  df1 = numeric(),
  df2 = numeric(),
  stringsAsFactors = FALSE
)

# =============================================================================
# 1. Hausman Test: FE vs RE (NLS Wage Equation)
# =============================================================================
cat("--- 1. Hausman Test: FE vs RE ---\n")
cat("H0: RE is consistent and efficient (no correlation between c_i and X)\n")
cat("H1: RE is inconsistent, FE preferred\n\n")

# Wage equation: ln_wage ~ experience + tenure + union + married
cat("  Model: ln_wage ~ experience + tenure + union + married\n\n")

tryCatch({
  fe_nls <- plm(ln_wage ~ experience + tenure + union + married,
                data = nls_pdf, model = "within")
  re_nls <- plm(ln_wage ~ experience + tenure + union + married,
                data = nls_pdf, model = "random")

  # Show FE estimates
  cat("  Fixed Effects estimates:\n")
  fe_summary <- summary(fe_nls)
  print(coef(fe_summary))
  cat("\n")

  # Show RE estimates
  cat("  Random Effects estimates:\n")
  re_summary <- summary(re_nls)
  print(coef(re_summary))
  cat("\n")

  # Hausman test
  haus <- phtest(fe_nls, re_nls)
  cat(sprintf("  Hausman statistic: %.4f\n", haus$statistic))
  cat(sprintf("  p-value:           %.6f\n", haus$p.value))
  cat(sprintf("  df:                %d\n", haus$parameter))
  cat(sprintf("  Decision:          %s\n\n",
              ifelse(haus$p.value < 0.05, "REJECT H0 -> use FE",
                                          "FAIL TO REJECT -> RE is acceptable")))

  results <- rbind(results, data.frame(
    test_name = "Hausman",
    dataset = "NLS_Work",
    model_spec = "ln_wage ~ experience + tenure + union + married",
    statistic = as.numeric(haus$statistic),
    p_value = as.numeric(haus$p.value),
    df = as.character(haus$parameter),
    df1 = as.numeric(haus$parameter),
    df2 = NA,
    stringsAsFactors = FALSE
  ))

  # Save FE and RE coefficients for comparison
  fe_coefs <- coef(fe_summary)
  re_coefs <- coef(re_summary)
  for (v in rownames(fe_coefs)) {
    results <- rbind(results, data.frame(
      test_name = "FE_coefficient",
      dataset = "NLS_Work",
      model_spec = "ln_wage ~ experience + tenure + union + married",
      statistic = fe_coefs[v, "Estimate"],
      p_value = fe_coefs[v, "Pr(>|t|)"],
      df = v,
      df1 = fe_coefs[v, "Std. Error"],
      df2 = NA,
      stringsAsFactors = FALSE
    ))
  }
  for (v in rownames(re_coefs)) {
    results <- rbind(results, data.frame(
      test_name = "RE_coefficient",
      dataset = "NLS_Work",
      model_spec = "ln_wage ~ experience + tenure + union + married",
      statistic = re_coefs[v, "Estimate"],
      p_value = re_coefs[v, "Pr(>|z|)"],
      df = v,
      df1 = re_coefs[v, "Std. Error"],
      df2 = NA,
      stringsAsFactors = FALSE
    ))
  }
}, error = function(e) {
  cat(sprintf("  Hausman test (NLS): ERROR - %s\n\n", e$message))
})

# =============================================================================
# 2. Hausman Test on Grunfeld (classic dataset)
# =============================================================================
cat("--- 2. Hausman Test: Grunfeld ---\n")
cat("  Model: invest ~ value + capital\n\n")

tryCatch({
  fe_grun <- plm(invest ~ value + capital, data = grunfeld_pdf, model = "within")
  re_grun <- plm(invest ~ value + capital, data = grunfeld_pdf, model = "random")

  cat("  FE coefficients:\n")
  print(coef(summary(fe_grun)))
  cat("\n  RE coefficients:\n")
  print(coef(summary(re_grun)))

  haus_grun <- phtest(fe_grun, re_grun)
  cat(sprintf("\n  Hausman statistic: %.4f\n", haus_grun$statistic))
  cat(sprintf("  p-value:           %.6f\n", haus_grun$p.value))
  cat(sprintf("  df:                %d\n", haus_grun$parameter))
  cat(sprintf("  Decision:          %s\n\n",
              ifelse(haus_grun$p.value < 0.05, "REJECT H0 -> use FE",
                                                "FAIL TO REJECT -> RE is acceptable")))

  results <- rbind(results, data.frame(
    test_name = "Hausman",
    dataset = "Grunfeld",
    model_spec = "invest ~ value + capital",
    statistic = as.numeric(haus_grun$statistic),
    p_value = as.numeric(haus_grun$p.value),
    df = as.character(haus_grun$parameter),
    df1 = as.numeric(haus_grun$parameter),
    df2 = NA,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  Hausman test (Grunfeld): ERROR - %s\n\n", e$message))
})

# =============================================================================
# 3. Breusch-Pagan LM Test: RE vs Pooled OLS
# =============================================================================
cat("--- 3. Breusch-Pagan LM Test ---\n")
cat("H0: Var(alpha_i) = 0 (Pooled OLS adequate)\n")
cat("H1: Var(alpha_i) > 0 (RE/FE needed)\n\n")

# On NLS data
tryCatch({
  pooled_nls <- plm(ln_wage ~ experience + tenure + union + married,
                    data = nls_pdf, model = "pooling")

  bp_nls <- plmtest(pooled_nls, type = "bp")
  cat(sprintf("  NLS Work (BP LM):  stat = %8.4f, p-value = %.6e, df = %d\n",
              bp_nls$statistic, bp_nls$p.value, bp_nls$parameter))
  cat(sprintf("    Decision: %s\n\n",
              ifelse(bp_nls$p.value < 0.05, "REJECT H0 -> individual effects present",
                                            "FAIL TO REJECT -> Pooled OLS adequate")))

  results <- rbind(results, data.frame(
    test_name = "Breusch_Pagan_LM",
    dataset = "NLS_Work",
    model_spec = "ln_wage ~ experience + tenure + union + married",
    statistic = as.numeric(bp_nls$statistic),
    p_value = as.numeric(bp_nls$p.value),
    df = as.character(bp_nls$parameter),
    df1 = as.numeric(bp_nls$parameter),
    df2 = NA,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  BP LM (NLS): ERROR - %s\n\n", e$message))
})

# On Grunfeld data
tryCatch({
  pooled_grun <- plm(invest ~ value + capital,
                     data = grunfeld_pdf, model = "pooling")

  bp_grun <- plmtest(pooled_grun, type = "bp")
  cat(sprintf("  Grunfeld (BP LM):  stat = %8.4f, p-value = %.6e, df = %d\n",
              bp_grun$statistic, bp_grun$p.value, bp_grun$parameter))
  cat(sprintf("    Decision: %s\n\n",
              ifelse(bp_grun$p.value < 0.05, "REJECT H0 -> individual effects present",
                                              "FAIL TO REJECT -> Pooled OLS adequate")))

  results <- rbind(results, data.frame(
    test_name = "Breusch_Pagan_LM",
    dataset = "Grunfeld",
    model_spec = "invest ~ value + capital",
    statistic = as.numeric(bp_grun$statistic),
    p_value = as.numeric(bp_grun$p.value),
    df = as.character(bp_grun$parameter),
    df1 = as.numeric(bp_grun$parameter),
    df2 = NA,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  BP LM (Grunfeld): ERROR - %s\n\n", e$message))
})

# Honda test variant
tryCatch({
  honda_nls <- plmtest(pooled_nls, type = "honda")
  cat(sprintf("  NLS Work (Honda):  stat = %8.4f, p-value = %.6e\n",
              honda_nls$statistic, honda_nls$p.value))

  results <- rbind(results, data.frame(
    test_name = "Honda_LM",
    dataset = "NLS_Work",
    model_spec = "ln_wage ~ experience + tenure + union + married",
    statistic = as.numeric(honda_nls$statistic),
    p_value = as.numeric(honda_nls$p.value),
    df = "1",
    df1 = 1,
    df2 = NA,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  Honda LM (NLS): ERROR - %s\n\n", e$message))
})

cat("\n")

# =============================================================================
# 4. F-test for Fixed Effects
# =============================================================================
cat("--- 4. F-test for Fixed Effects ---\n")
cat("H0: All individual effects alpha_i = 0 (Pooled OLS)\n")
cat("H1: At least one alpha_i != 0 (FE needed)\n\n")

# On NLS data
tryCatch({
  fe_nls2 <- plm(ln_wage ~ experience + tenure + union + married,
                 data = nls_pdf, model = "within")
  pooled_nls2 <- plm(ln_wage ~ experience + tenure + union + married,
                     data = nls_pdf, model = "pooling")

  ftest_nls <- pFtest(fe_nls2, pooled_nls2)
  cat(sprintf("  NLS Work:  F = %8.4f, p-value = %.6e, df1 = %d, df2 = %d\n",
              ftest_nls$statistic, ftest_nls$p.value,
              ftest_nls$parameter[1], ftest_nls$parameter[2]))
  cat(sprintf("    Decision: %s\n\n",
              ifelse(ftest_nls$p.value < 0.05, "REJECT H0 -> FE justified",
                                                "FAIL TO REJECT -> Pooled OLS adequate")))

  results <- rbind(results, data.frame(
    test_name = "F_test_FE",
    dataset = "NLS_Work",
    model_spec = "ln_wage ~ experience + tenure + union + married",
    statistic = as.numeric(ftest_nls$statistic),
    p_value = as.numeric(ftest_nls$p.value),
    df = paste(ftest_nls$parameter, collapse = ","),
    df1 = as.numeric(ftest_nls$parameter[1]),
    df2 = as.numeric(ftest_nls$parameter[2]),
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  F-test (NLS): ERROR - %s\n\n", e$message))
})

# On Grunfeld data
tryCatch({
  fe_grun2 <- plm(invest ~ value + capital, data = grunfeld_pdf, model = "within")
  pooled_grun2 <- plm(invest ~ value + capital, data = grunfeld_pdf, model = "pooling")

  ftest_grun <- pFtest(fe_grun2, pooled_grun2)
  cat(sprintf("  Grunfeld:  F = %8.4f, p-value = %.6e, df1 = %d, df2 = %d\n",
              ftest_grun$statistic, ftest_grun$p.value,
              ftest_grun$parameter[1], ftest_grun$parameter[2]))
  cat(sprintf("    Decision: %s\n\n",
              ifelse(ftest_grun$p.value < 0.05, "REJECT H0 -> FE justified",
                                                  "FAIL TO REJECT -> Pooled OLS adequate")))

  results <- rbind(results, data.frame(
    test_name = "F_test_FE",
    dataset = "Grunfeld",
    model_spec = "invest ~ value + capital",
    statistic = as.numeric(ftest_grun$statistic),
    p_value = as.numeric(ftest_grun$p.value),
    df = paste(ftest_grun$parameter, collapse = ","),
    df1 = as.numeric(ftest_grun$parameter[1]),
    df2 = as.numeric(ftest_grun$parameter[2]),
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  F-test (Grunfeld): ERROR - %s\n\n", e$message))
})

# =============================================================================
# 5. Poolability Test (Chow-type)
# =============================================================================
cat("--- 5. Poolability Test ---\n")
cat("H0: Slopes are homogeneous across entities\n")
cat("H1: Slopes differ across entities\n\n")

# On Grunfeld data (smaller, more tractable)
tryCatch({
  pool_test <- pooltest(invest ~ value + capital, data = grunfeld_pdf, model = "within")
  cat(sprintf("  Grunfeld:  F = %8.4f, p-value = %.6e, df1 = %d, df2 = %d\n",
              pool_test$statistic, pool_test$p.value,
              pool_test$parameter[1], pool_test$parameter[2]))
  cat(sprintf("    Decision: %s\n\n",
              ifelse(pool_test$p.value < 0.05, "REJECT H0 -> heterogeneous slopes",
                                                "FAIL TO REJECT -> pooling is acceptable")))

  results <- rbind(results, data.frame(
    test_name = "Poolability",
    dataset = "Grunfeld",
    model_spec = "invest ~ value + capital",
    statistic = as.numeric(pool_test$statistic),
    p_value = as.numeric(pool_test$p.value),
    df = paste(pool_test$parameter, collapse = ","),
    df1 = as.numeric(pool_test$parameter[1]),
    df2 = as.numeric(pool_test$parameter[2]),
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  Poolability (Grunfeld): ERROR - %s\n\n", e$message))
})

# =============================================================================
# 6. Hausman Test on Firm Productivity
# =============================================================================
cat("--- 6. Hausman Test: Firm Productivity ---\n")
cat("  Model: log_output ~ log_capital + log_labor + log_materials\n\n")

tryCatch({
  fe_firm <- plm(log_output ~ log_capital + log_labor + log_materials,
                 data = firm_pdf, model = "within")
  re_firm <- plm(log_output ~ log_capital + log_labor + log_materials,
                 data = firm_pdf, model = "random")

  cat("  FE coefficients:\n")
  print(coef(summary(fe_firm)))
  cat("\n  RE coefficients:\n")
  print(coef(summary(re_firm)))

  haus_firm <- phtest(fe_firm, re_firm)
  cat(sprintf("\n  Hausman statistic: %.4f\n", haus_firm$statistic))
  cat(sprintf("  p-value:           %.6f\n", haus_firm$p.value))
  cat(sprintf("  df:                %d\n", haus_firm$parameter))
  cat(sprintf("  Decision:          %s\n\n",
              ifelse(haus_firm$p.value < 0.05, "REJECT H0 -> use FE",
                                                "FAIL TO REJECT -> RE is acceptable")))

  results <- rbind(results, data.frame(
    test_name = "Hausman",
    dataset = "Firm_Productivity",
    model_spec = "log_output ~ log_capital + log_labor + log_materials",
    statistic = as.numeric(haus_firm$statistic),
    p_value = as.numeric(haus_firm$p.value),
    df = as.character(haus_firm$parameter),
    df1 = as.numeric(haus_firm$parameter),
    df2 = NA,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  Hausman (Firm): ERROR - %s\n\n", e$message))
})

# =============================================================================
# 7. Cross-sectional Dependence Test (Pesaran CD)
# =============================================================================
cat("--- 7. Pesaran CD Test for Cross-sectional Dependence ---\n\n")

tryCatch({
  fe_grun3 <- plm(invest ~ value + capital, data = grunfeld_pdf, model = "within")
  cd_test <- pcdtest(fe_grun3, test = "cd")
  cat(sprintf("  Grunfeld (Pesaran CD): stat = %8.4f, p-value = %.6f\n",
              cd_test$statistic, cd_test$p.value))
  cat(sprintf("    Decision: %s\n\n",
              ifelse(cd_test$p.value < 0.05, "REJECT H0 -> cross-sectional dependence",
                                              "FAIL TO REJECT -> no cross-sectional dependence")))

  results <- rbind(results, data.frame(
    test_name = "Pesaran_CD",
    dataset = "Grunfeld",
    model_spec = "invest ~ value + capital",
    statistic = as.numeric(cd_test$statistic),
    p_value = as.numeric(cd_test$p.value),
    df = "NA",
    df1 = NA,
    df2 = NA,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  Pesaran CD: ERROR - %s\n\n", e$message))
})

# =============================================================================
# 8. Serial Correlation Test (Wooldridge)
# =============================================================================
cat("--- 8. Wooldridge Test for Serial Correlation ---\n\n")

tryCatch({
  # pbgtest - Breusch-Godfrey test for serial correlation in panel models
  bg_grun <- pbgtest(fe_grun3)
  cat(sprintf("  Grunfeld (BG):     stat = %8.4f, p-value = %.6f, df = %d\n",
              bg_grun$statistic, bg_grun$p.value, bg_grun$parameter))
  cat(sprintf("    Decision: %s\n\n",
              ifelse(bg_grun$p.value < 0.05, "REJECT H0 -> serial correlation present",
                                              "FAIL TO REJECT -> no serial correlation")))

  results <- rbind(results, data.frame(
    test_name = "Breusch_Godfrey_SC",
    dataset = "Grunfeld",
    model_spec = "invest ~ value + capital",
    statistic = as.numeric(bg_grun$statistic),
    p_value = as.numeric(bg_grun$p.value),
    df = as.character(bg_grun$parameter),
    df1 = as.numeric(bg_grun$parameter),
    df2 = NA,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  BG test: ERROR - %s\n\n", e$message))
})

# =============================================================================
# Save results
# =============================================================================
write.csv(results, output_path, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", output_path))
cat(sprintf("Total tests recorded: %d\n", nrow(results)))

cat("\n=== Summary Table ===\n")
print(results[, c("test_name", "dataset", "statistic", "p_value", "df1", "df2")])
cat("\nDone.\n")
