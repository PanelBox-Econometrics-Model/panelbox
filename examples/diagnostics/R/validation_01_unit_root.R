# =============================================================================
# Validation Script 01: Panel Unit Root Tests
# PanelBox vs R (plm, urca)
#
# Tests: IPS (Im-Pesaran-Shin), LLC (Levin-Lin-Chu), ADF (individual entities)
# Dataset: Penn World Table (30 countries x 50 years)
# =============================================================================

library(plm)
library(urca)

# --- Data Loading ---
data_path <- "/home/guhaase/projetos/panelbox/examples/diagnostics/data/unit_root/penn_world_table.csv"
output_path <- "/home/guhaase/projetos/panelbox/examples/diagnostics/R/results_unit_root.csv"

cat("=== Panel Unit Root Tests Validation ===\n\n")

df <- read.csv(data_path, stringsAsFactors = FALSE)
cat("Dataset dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
cat("Countries:", length(unique(df$countrycode)), "\n")
cat("Years:", min(df$year), "-", max(df$year), "\n\n")

# Create log-transformed variables
df$log_gdp <- log(df$rgdpna)
df$log_capital <- log(df$rkna)
df$log_labor <- log(df$emp)
df$log_productivity <- df$log_gdp - df$log_labor

# Convert to panel data frame
pdf <- pdata.frame(df, index = c("countrycode", "year"), drop.index = FALSE)

# Check balance
cat("Panel balanced:", ifelse(is.pbalanced(pdf), "YES", "NO"), "\n\n")

# --- Initialize results storage ---
results <- data.frame(
  test_name = character(),
  variable = character(),
  statistic = numeric(),
  p_value = numeric(),
  lags_used = character(),
  n_obs = integer(),
  n_groups = integer(),
  stringsAsFactors = FALSE
)

# --- Variables to test ---
# Use constant + trend for trended variables, constant only for mean-reverting
variables_ct <- c("log_gdp", "log_capital", "log_labor", "log_productivity")
variables_c  <- c("labsh")

n_groups <- length(unique(df$countrycode))

# =============================================================================
# 1. IPS Test (Im-Pesaran-Shin)
# =============================================================================
cat("--- 1. IPS Tests (Im-Pesaran-Shin) ---\n")
cat("H0: All panels have unit roots\n")
cat("H1: Some panels are stationary\n\n")

# IPS with trend for trended variables
for (v in variables_ct) {
  tryCatch({
    ips_result <- purtest(pdf[[v]], test = "ips", exo = "trend", lags = "AIC", pmax = 4)
    stat <- ips_result$statistic$statistic
    pval <- ips_result$statistic$p.value

    cat(sprintf("  IPS %-20s: stat = %8.4f, p-value = %6.4f\n", v, stat, pval))

    results <- rbind(results, data.frame(
      test_name = "IPS",
      variable = v,
      statistic = stat,
      p_value = pval,
      lags_used = "AIC(max=4)",
      n_obs = nrow(df),
      n_groups = n_groups,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    cat(sprintf("  IPS %-20s: ERROR - %s\n", v, e$message))
  })
}

# IPS without trend for mean-reverting variables
for (v in variables_c) {
  tryCatch({
    ips_result <- purtest(pdf[[v]], test = "ips", exo = "intercept", lags = "AIC", pmax = 4)
    stat <- ips_result$statistic$statistic
    pval <- ips_result$statistic$p.value

    cat(sprintf("  IPS %-20s: stat = %8.4f, p-value = %6.4f (intercept only)\n", v, stat, pval))

    results <- rbind(results, data.frame(
      test_name = "IPS",
      variable = v,
      statistic = stat,
      p_value = pval,
      lags_used = "AIC(max=4)",
      n_obs = nrow(df),
      n_groups = n_groups,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    cat(sprintf("  IPS %-20s: ERROR - %s\n", v, e$message))
  })
}

cat("\n")

# =============================================================================
# 2. LLC Test (Levin-Lin-Chu)
# =============================================================================
cat("--- 2. LLC Tests (Levin-Lin-Chu) ---\n")
cat("H0: All panels have unit roots (common rho)\n")
cat("H1: All panels are stationary\n\n")

for (v in variables_ct) {
  tryCatch({
    llc_result <- purtest(pdf[[v]], test = "levinlin", exo = "trend", lags = "AIC", pmax = 4)
    stat <- llc_result$statistic$statistic
    pval <- llc_result$statistic$p.value

    cat(sprintf("  LLC %-20s: stat = %8.4f, p-value = %6.4f\n", v, stat, pval))

    results <- rbind(results, data.frame(
      test_name = "LLC",
      variable = v,
      statistic = stat,
      p_value = pval,
      lags_used = "AIC(max=4)",
      n_obs = nrow(df),
      n_groups = n_groups,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    cat(sprintf("  LLC %-20s: ERROR - %s\n", v, e$message))
  })
}

for (v in variables_c) {
  tryCatch({
    llc_result <- purtest(pdf[[v]], test = "levinlin", exo = "intercept", lags = "AIC", pmax = 4)
    stat <- llc_result$statistic$statistic
    pval <- llc_result$statistic$p.value

    cat(sprintf("  LLC %-20s: stat = %8.4f, p-value = %6.4f (intercept only)\n", v, stat, pval))

    results <- rbind(results, data.frame(
      test_name = "LLC",
      variable = v,
      statistic = stat,
      p_value = pval,
      lags_used = "AIC(max=4)",
      n_obs = nrow(df),
      n_groups = n_groups,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    cat(sprintf("  LLC %-20s: ERROR - %s\n", v, e$message))
  })
}

cat("\n")

# =============================================================================
# 3. Hadri Test (stationarity under H0)
# =============================================================================
cat("--- 3. Hadri Tests (Stationarity under H0) ---\n")
cat("H0: All panels are STATIONARY\n")
cat("H1: Some panels contain unit roots\n\n")

for (v in variables_ct) {
  tryCatch({
    hadri_result <- purtest(pdf[[v]], test = "hadri", exo = "trend")
    stat <- hadri_result$statistic$statistic
    pval <- hadri_result$statistic$p.value

    cat(sprintf("  Hadri %-18s: stat = %8.4f, p-value = %6.4f\n", v, stat, pval))

    results <- rbind(results, data.frame(
      test_name = "Hadri",
      variable = v,
      statistic = stat,
      p_value = pval,
      lags_used = "NA",
      n_obs = nrow(df),
      n_groups = n_groups,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    cat(sprintf("  Hadri %-18s: ERROR - %s\n", v, e$message))
  })
}

for (v in variables_c) {
  tryCatch({
    hadri_result <- purtest(pdf[[v]], test = "hadri", exo = "intercept")
    stat <- hadri_result$statistic$statistic
    pval <- hadri_result$statistic$p.value

    cat(sprintf("  Hadri %-18s: stat = %8.4f, p-value = %6.4f (intercept only)\n", v, stat, pval))

    results <- rbind(results, data.frame(
      test_name = "Hadri",
      variable = v,
      statistic = stat,
      p_value = pval,
      lags_used = "NA",
      n_obs = nrow(df),
      n_groups = n_groups,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    cat(sprintf("  Hadri %-18s: ERROR - %s\n", v, e$message))
  })
}

cat("\n")

# =============================================================================
# 4. Individual ADF Tests (per entity, using urca)
# =============================================================================
cat("--- 4. Individual ADF Tests (urca::ur.df) ---\n")
cat("Running ADF for G7 countries on log_gdp...\n\n")

g7 <- c("USA", "GBR", "DEU", "FRA", "JPN", "CAN", "ITA")
# Filter to countries that exist in the data
g7_available <- g7[g7 %in% unique(df$countrycode)]

for (cc in g7_available) {
  tryCatch({
    series <- df[df$countrycode == cc, "log_gdp"]
    series <- na.omit(series)

    if (length(series) < 10) {
      cat(sprintf("  ADF %-5s: insufficient data (n=%d)\n", cc, length(series)))
      next
    }

    adf <- ur.df(series, type = "trend", selectlags = "AIC")
    adf_stat <- adf@teststat[1, "tau3"]
    # Critical values from Dickey-Fuller tables
    cv_1pct <- adf@cval[1, "1pct"]
    cv_5pct <- adf@cval[1, "5pct"]
    cv_10pct <- adf@cval[1, "10pct"]
    lags <- adf@lags

    reject_5 <- ifelse(adf_stat < cv_5pct, "REJECT H0 (stationary)", "FAIL TO REJECT (unit root)")

    cat(sprintf("  ADF %-5s: stat = %7.3f, 5%% CV = %7.3f, lags = %d -> %s\n",
                cc, adf_stat, cv_5pct, lags, reject_5))

    results <- rbind(results, data.frame(
      test_name = paste0("ADF_", cc),
      variable = "log_gdp",
      statistic = adf_stat,
      p_value = NA,  # urca ADF does not provide p-values directly
      lags_used = as.character(lags),
      n_obs = length(series),
      n_groups = 1,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    cat(sprintf("  ADF %-5s: ERROR - %s\n", cc, e$message))
  })
}

cat("\n")

# =============================================================================
# 5. First-difference tests (should be stationary after differencing)
# =============================================================================
cat("--- 5. IPS Tests on First Differences (should reject H0) ---\n\n")

for (v in c("log_gdp", "log_capital")) {
  tryCatch({
    # First difference
    diff_var <- diff(pdf[[v]])
    ips_diff <- purtest(diff_var, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)
    stat <- ips_diff$statistic$statistic
    pval <- ips_diff$statistic$p.value

    cat(sprintf("  IPS D(%-16s): stat = %8.4f, p-value = %6.4f\n", v, stat, pval))

    results <- rbind(results, data.frame(
      test_name = "IPS_diff",
      variable = paste0("D(", v, ")"),
      statistic = stat,
      p_value = pval,
      lags_used = "AIC(max=4)",
      n_obs = nrow(df) - n_groups,
      n_groups = n_groups,
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    cat(sprintf("  IPS D(%-16s): ERROR - %s\n", v, e$message))
  })
}

cat("\n")

# =============================================================================
# Save results
# =============================================================================
write.csv(results, output_path, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", output_path))
cat(sprintf("Total tests recorded: %d\n", nrow(results)))

cat("\n=== Summary Table ===\n")
print(results)
cat("\nDone.\n")
