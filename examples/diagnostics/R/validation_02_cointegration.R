# =============================================================================
# Validation Script 02: Panel Cointegration Tests
# PanelBox vs R (plm, urca)
#
# Tests: Pedroni (group-mean ADF), Kao (residual-based), per-entity ADF
# Dataset: OECD Macro (20 countries x 40 years) - Consumption-Income
#          PPP Data (25 countries x 35 years) - Exchange rate / price ratio
# =============================================================================

library(plm)
library(urca)

# --- Data Loading ---
data_path_oecd <- "/home/guhaase/projetos/panelbox/examples/diagnostics/data/cointegration/oecd_macro.csv"
data_path_ppp  <- "/home/guhaase/projetos/panelbox/examples/diagnostics/data/cointegration/ppp_data.csv"
output_path <- "/home/guhaase/projetos/panelbox/examples/diagnostics/R/results_cointegration.csv"

cat("=== Panel Cointegration Tests Validation ===\n\n")

# --- Load OECD Macro Data ---
oecd <- read.csv(data_path_oecd, stringsAsFactors = FALSE)
cat("OECD Macro dataset:", nrow(oecd), "rows x", ncol(oecd), "columns\n")
cat("Countries:", length(unique(oecd$country)), "\n")
cat("Years:", min(oecd$year), "-", max(oecd$year), "\n\n")

# --- Load PPP Data ---
ppp <- read.csv(data_path_ppp, stringsAsFactors = FALSE)
cat("PPP dataset:", nrow(ppp), "rows x", ncol(ppp), "columns\n")
cat("Countries:", length(unique(ppp$country)), "\n")
cat("Years:", min(ppp$year), "-", max(ppp$year), "\n\n")

# Convert to panel data frames
oecd_pdf <- pdata.frame(oecd, index = c("country", "year"), drop.index = FALSE)
ppp_pdf  <- pdata.frame(ppp, index = c("country", "year"), drop.index = FALSE)

# --- Initialize results storage ---
results <- data.frame(
  test_name = character(),
  dataset = character(),
  y_var = character(),
  x_vars = character(),
  statistic = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

# =============================================================================
# 1. First verify I(1) behavior: Unit root tests on levels
# =============================================================================
cat("--- 1. Preliminary: Verify I(1) behavior ---\n\n")

for (v in c("log_C", "log_Y")) {
  tryCatch({
    ips <- purtest(oecd_pdf[[v]], test = "ips", exo = "trend", lags = "AIC", pmax = 4)
    cat(sprintf("  IPS on %-8s: stat = %8.4f, p-value = %6.4f\n",
                v, ips$statistic$statistic, ips$statistic$p.value))
  }, error = function(e) {
    cat(sprintf("  IPS on %-8s: ERROR - %s\n", v, e$message))
  })
}

for (v in c("log_C", "log_Y")) {
  tryCatch({
    dv <- diff(oecd_pdf[[v]])
    ips <- purtest(dv, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)
    cat(sprintf("  IPS on D(%-5s): stat = %8.4f, p-value = %6.4f\n",
                v, ips$statistic$statistic, ips$statistic$p.value))
  }, error = function(e) {
    cat(sprintf("  IPS on D(%-5s): ERROR - %s\n", v, e$message))
  })
}
cat("\n")

# =============================================================================
# 2. Kao-style Cointegration Test (residual-based panel ADF)
# =============================================================================
cat("--- 2. Kao-style Cointegration Test ---\n")
cat("H0: No cointegration\n")
cat("H1: Cointegration exists\n\n")

# OECD: log_C ~ log_Y
tryCatch({
  # Step 1: FE cointegrating regression
  fe_coint <- plm(log_C ~ log_Y, data = oecd_pdf, model = "within")
  cat("  Cointegrating regression (FE): log_C ~ log_Y\n")
  cat(sprintf("    beta(log_Y) = %.6f (SE = %.6f)\n",
              coef(fe_coint)["log_Y"],
              sqrt(vcov(fe_coint)["log_Y", "log_Y"])))

  # Step 2: Test residuals for stationarity with LLC (allows exo="intercept")
  resids <- residuals(fe_coint)
  llc_resid <- purtest(resids, test = "levinlin", exo = "intercept", lags = "AIC", pmax = 4)
  stat_llc <- llc_resid$statistic$statistic
  pval_llc <- llc_resid$statistic$p.value

  cat(sprintf("  LLC on residuals (Kao-style): stat = %8.4f, p-value = %6.4f\n", stat_llc, pval_llc))
  cat(sprintf("    Interpretation: %s\n",
              ifelse(pval_llc < 0.05, "REJECT H0 -> cointegration exists",
                                      "FAIL TO REJECT -> no cointegration")))

  results <- rbind(results, data.frame(
    test_name = "Kao_LLC_residual",
    dataset = "OECD",
    y_var = "log_C",
    x_vars = "log_Y",
    statistic = stat_llc,
    p_value = pval_llc,
    stringsAsFactors = FALSE
  ))

  # Also try IPS with intercept on residuals
  ips_resid <- purtest(resids, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)
  stat_ips <- ips_resid$statistic$statistic
  pval_ips <- ips_resid$statistic$p.value

  cat(sprintf("  IPS on residuals (Kao-style): stat = %8.4f, p-value = %6.4f\n\n", stat_ips, pval_ips))

  results <- rbind(results, data.frame(
    test_name = "Kao_IPS_residual",
    dataset = "OECD",
    y_var = "log_C",
    x_vars = "log_Y",
    statistic = stat_ips,
    p_value = pval_ips,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  Kao-style test (OECD): ERROR - %s\n\n", e$message))
})

# =============================================================================
# 3. Pedroni-style Tests (Engle-Granger based, per-entity)
# =============================================================================
cat("--- 3. Pedroni-style Cointegration Tests ---\n")
cat("Per-entity cointegrating regressions + residual ADF tests\n\n")

countries <- unique(oecd$country)
entity_results <- data.frame(
  country = character(),
  beta = numeric(),
  adf_stat = numeric(),
  lags = integer(),
  stringsAsFactors = FALSE
)

for (cc in countries) {
  tryCatch({
    sub <- oecd[oecd$country == cc, ]
    sub <- sub[order(sub$year), ]

    # OLS regression: log_C ~ log_Y
    ols <- lm(log_C ~ log_Y, data = sub)
    beta <- coef(ols)["log_Y"]

    # ADF on residuals (type="none" for EG residuals - no intercept needed)
    resid_ts <- residuals(ols)
    if (length(resid_ts) >= 10) {
      adf <- ur.df(resid_ts, type = "none", selectlags = "AIC")
      adf_stat <- adf@teststat[1, "tau1"]
      lags_used <- adf@lags

      entity_results <- rbind(entity_results, data.frame(
        country = cc,
        beta = beta,
        adf_stat = adf_stat,
        lags = lags_used,
        stringsAsFactors = FALSE
      ))
    }
  }, error = function(e) {
    cat(sprintf("    Entity %s: ERROR - %s\n", cc, e$message))
  })
}

if (nrow(entity_results) > 0) {
  mean_adf <- mean(entity_results$adf_stat)
  sd_adf <- sd(entity_results$adf_stat)
  n_ent <- nrow(entity_results)
  z_stat <- sqrt(n_ent) * mean_adf / sd_adf

  cat(sprintf("  Number of entities: %d\n", n_ent))
  cat(sprintf("  Mean ADF statistic: %8.4f\n", mean_adf))
  cat(sprintf("  SD of ADF stats:    %8.4f\n", sd_adf))
  cat(sprintf("  Group-mean Z-stat:  %8.4f\n", z_stat))
  cat(sprintf("  Approx p-value:     %6.4f\n", pnorm(z_stat)))
  cat(sprintf("  Mean beta (MPC):    %8.4f\n\n", mean(entity_results$beta)))

  cat("  Per-entity results:\n")
  for (i in 1:nrow(entity_results)) {
    cat(sprintf("    %-20s: beta = %6.4f, ADF = %7.3f, lags = %d\n",
                entity_results$country[i],
                entity_results$beta[i],
                entity_results$adf_stat[i],
                entity_results$lags[i]))
  }
  cat("\n")

  results <- rbind(results, data.frame(
    test_name = "Pedroni_group_ADF",
    dataset = "OECD",
    y_var = "log_C",
    x_vars = "log_Y",
    statistic = z_stat,
    p_value = pnorm(z_stat),
    stringsAsFactors = FALSE
  ))
} else {
  cat("  WARNING: No entity results could be computed.\n\n")
}

# =============================================================================
# 4. PPP Cointegration: log_S ~ log_P_ratio
# =============================================================================
cat("--- 4. PPP Cointegration: log_S ~ log_P_ratio ---\n\n")

tryCatch({
  fe_ppp <- plm(log_S ~ log_P_ratio, data = ppp_pdf, model = "within")
  cat(sprintf("  Cointegrating regression (FE): log_S ~ log_P_ratio\n"))
  cat(sprintf("    beta(log_P_ratio) = %.6f (SE = %.6f)\n",
              coef(fe_ppp)["log_P_ratio"],
              sqrt(vcov(fe_ppp)["log_P_ratio", "log_P_ratio"])))

  # LLC on residuals
  resids_ppp <- residuals(fe_ppp)
  llc_ppp <- purtest(resids_ppp, test = "levinlin", exo = "intercept", lags = "AIC", pmax = 4)
  stat <- llc_ppp$statistic$statistic
  pval <- llc_ppp$statistic$p.value

  cat(sprintf("  LLC on residuals: stat = %8.4f, p-value = %6.4f\n", stat, pval))

  # IPS on residuals
  ips_ppp <- purtest(resids_ppp, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)
  stat_ips <- ips_ppp$statistic$statistic
  pval_ips <- ips_ppp$statistic$p.value

  cat(sprintf("  IPS on residuals: stat = %8.4f, p-value = %6.4f\n", stat_ips, pval_ips))
  cat(sprintf("    Interpretation: %s\n\n",
              ifelse(pval_ips < 0.05, "REJECT H0 -> PPP holds in long run",
                                      "FAIL TO REJECT -> weak PPP evidence")))

  results <- rbind(results, data.frame(
    test_name = "Kao_LLC_residual",
    dataset = "PPP",
    y_var = "log_S",
    x_vars = "log_P_ratio",
    statistic = stat,
    p_value = pval,
    stringsAsFactors = FALSE
  ))

  results <- rbind(results, data.frame(
    test_name = "Kao_IPS_residual",
    dataset = "PPP",
    y_var = "log_S",
    x_vars = "log_P_ratio",
    statistic = stat_ips,
    p_value = pval_ips,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  PPP cointegration test: ERROR - %s\n\n", e$message))
})

# =============================================================================
# 5. PPP Per-entity cointegration (Pedroni-style)
# =============================================================================
cat("--- 5. PPP Per-Entity ADF Tests ---\n\n")

ppp_countries <- unique(ppp$country)
ppp_entity <- data.frame(
  country = character(),
  beta = numeric(),
  adf_stat = numeric(),
  lags = integer(),
  stringsAsFactors = FALSE
)

for (cc in ppp_countries) {
  tryCatch({
    sub <- ppp[ppp$country == cc, ]
    sub <- sub[order(sub$year), ]
    ols <- lm(log_S ~ log_P_ratio, data = sub)
    beta <- coef(ols)["log_P_ratio"]
    resid_ts <- residuals(ols)
    if (length(resid_ts) >= 10) {
      adf <- ur.df(resid_ts, type = "none", selectlags = "AIC")
      adf_stat <- adf@teststat[1, "tau1"]
      ppp_entity <- rbind(ppp_entity, data.frame(
        country = cc, beta = beta,
        adf_stat = adf_stat, lags = adf@lags,
        stringsAsFactors = FALSE
      ))
    }
  }, error = function(e) {
    cat(sprintf("    Entity %s: ERROR - %s\n", cc, e$message))
  })
}

if (nrow(ppp_entity) > 0) {
  mean_adf <- mean(ppp_entity$adf_stat)
  n_ent <- nrow(ppp_entity)
  z_stat <- sqrt(n_ent) * mean_adf / sd(ppp_entity$adf_stat)

  cat(sprintf("  Number of entities: %d\n", n_ent))
  cat(sprintf("  Mean ADF statistic: %8.4f\n", mean_adf))
  cat(sprintf("  Group-mean Z-stat:  %8.4f\n", z_stat))
  cat(sprintf("  Approx p-value:     %6.4f\n", pnorm(z_stat)))
  cat(sprintf("  Mean beta (PPP):    %8.4f (theory predicts ~1)\n\n", mean(ppp_entity$beta)))

  cat("  Per-entity results:\n")
  for (i in 1:nrow(ppp_entity)) {
    cat(sprintf("    %-5s: beta = %7.4f, ADF = %7.3f, lags = %d\n",
                ppp_entity$country[i],
                ppp_entity$beta[i],
                ppp_entity$adf_stat[i],
                ppp_entity$lags[i]))
  }
  cat("\n")

  results <- rbind(results, data.frame(
    test_name = "Pedroni_group_ADF",
    dataset = "PPP",
    y_var = "log_S",
    x_vars = "log_P_ratio",
    statistic = z_stat,
    p_value = pnorm(z_stat),
    stringsAsFactors = FALSE
  ))
} else {
  cat("  WARNING: No PPP entity results could be computed.\n\n")
}

# =============================================================================
# 6. Interest Rate Parity (bonus)
# =============================================================================
cat("--- 6. Interest Rate Parity Cointegration ---\n\n")

ir_path <- "/home/guhaase/projetos/panelbox/examples/diagnostics/data/cointegration/interest_rates.csv"
ir <- read.csv(ir_path, stringsAsFactors = FALSE)
cat(sprintf("  Interest rates: %d obs, %d countries, %d-%d\n",
            nrow(ir), length(unique(ir$country)), min(ir$year), max(ir$year)))

ir_pdf <- pdata.frame(ir, index = c("country", "year"), drop.index = FALSE)

tryCatch({
  fe_ir <- plm(domestic_rate ~ us_rate, data = ir_pdf, model = "within")
  cat(sprintf("  Cointegrating regression (FE): domestic_rate ~ us_rate\n"))
  cat(sprintf("    beta(us_rate) = %.6f\n", coef(fe_ir)["us_rate"]))

  resids_ir <- residuals(fe_ir)
  llc_ir <- purtest(resids_ir, test = "levinlin", exo = "intercept", lags = "AIC", pmax = 4)
  cat(sprintf("  LLC on residuals: stat = %8.4f, p-value = %6.4f\n",
              llc_ir$statistic$statistic, llc_ir$statistic$p.value))

  ips_ir <- purtest(resids_ir, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)
  cat(sprintf("  IPS on residuals: stat = %8.4f, p-value = %6.4f\n\n",
              ips_ir$statistic$statistic, ips_ir$statistic$p.value))

  results <- rbind(results, data.frame(
    test_name = "Kao_LLC_residual",
    dataset = "Interest_Rates",
    y_var = "domestic_rate",
    x_vars = "us_rate",
    statistic = llc_ir$statistic$statistic,
    p_value = llc_ir$statistic$p.value,
    stringsAsFactors = FALSE
  ))

  results <- rbind(results, data.frame(
    test_name = "Kao_IPS_residual",
    dataset = "Interest_Rates",
    y_var = "domestic_rate",
    x_vars = "us_rate",
    statistic = ips_ir$statistic$statistic,
    p_value = ips_ir$statistic$p.value,
    stringsAsFactors = FALSE
  ))
}, error = function(e) {
  cat(sprintf("  Interest rate test: ERROR - %s\n\n", e$message))
})

# =============================================================================
# Save results
# =============================================================================
write.csv(results, output_path, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", output_path))
cat(sprintf("Total tests recorded: %d\n", nrow(results)))

cat("\n=== Summary Table ===\n")
print(results)
cat("\nDone.\n")
