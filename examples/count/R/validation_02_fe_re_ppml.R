# =============================================================================
# Validation Script 02: FE/RE Poisson and PPML Gravity Models
# =============================================================================
# Compares R results with PanelBox for panel count models.
#
# Models estimated:
#   1. Pooled Poisson (city_crime): crime_count ~ unemployment_rate +
#      police_per_capita + median_income + temperature
#   2. FE Poisson (conditional MLE via fixest::fepois)
#   3. RE Poisson (pglm)
#   4. PPML Gravity (bilateral_trade): trade_value ~ log_gdp_exporter +
#      log_gdp_importer + log_distance + contiguous + common_language +
#      trade_agreement
#
# Packages: fixest, pglm, sandwich, lmtest
# =============================================================================

library(fixest)
library(pglm)
library(sandwich)
library(lmtest)

cat("=============================================================\n")
cat("Validation 02: FE/RE Poisson and PPML Gravity\n")
cat("=============================================================\n\n")

# --- Paths ---
data_dir <- "/home/guhaase/projetos/panelbox/examples/count/data"
output_dir <- "/home/guhaase/projetos/panelbox/examples/count/R"

# Initialize results data frame
results <- data.frame(
    model_name = character(),
    variable = character(),
    coefficient = numeric(),
    std_error = numeric(),
    z_statistic = numeric(),
    p_value = numeric(),
    log_likelihood = numeric(),
    n_obs = numeric(),
    n_groups = numeric(),
    stringsAsFactors = FALSE
)

# =============================================================================
# PART 1: Panel Poisson Models on city_crime
# =============================================================================

cat("PART 1: Panel Poisson Models (city_crime)\n")
cat("-------------------------------------------------------------\n")

cc <- read.csv(file.path(data_dir, "city_crime.csv"))
cat(sprintf("Dataset loaded: %d observations, %d variables\n", nrow(cc), ncol(cc)))
cat(sprintf("Columns: %s\n", paste(names(cc), collapse=", ")))
cat(sprintf("Cities: %d, Years: %d\n\n",
            length(unique(cc$city_id)), length(unique(cc$year))))

# --- 1a. Pooled Poisson (no fixed effects) ---
cat("--- 1a. Pooled Poisson ---\n")
pooled_fit <- glm(crime_count ~ unemployment_rate + police_per_capita +
                  median_income + temperature,
                  family = poisson(link = "log"),
                  data = cc)

# Cluster-robust SE (by city_id)
pooled_vcov <- vcovCL(pooled_fit, cluster = cc$city_id)
pooled_se <- sqrt(diag(pooled_vcov))
pooled_coefs <- coef(pooled_fit)
pooled_z <- pooled_coefs / pooled_se
pooled_p <- 2 * pnorm(abs(pooled_z), lower.tail = FALSE)

pooled_table <- data.frame(
    Variable = names(pooled_coefs),
    Coefficient = pooled_coefs,
    Std_Error = pooled_se,
    z_statistic = pooled_z,
    p_value = pooled_p,
    IRR = exp(pooled_coefs)
)
rownames(pooled_table) <- NULL
cat("Pooled Poisson with Cluster-Robust SE:\n")
print(pooled_table, digits = 6)
cat(sprintf("Log-Likelihood: %.4f\n", logLik(pooled_fit)[1]))

# Save pooled results
for (i in seq_along(pooled_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "pooled_poisson_crime",
        variable = names(pooled_coefs)[i],
        coefficient = pooled_coefs[i],
        std_error = pooled_se[i],
        z_statistic = pooled_z[i],
        p_value = pooled_p[i],
        log_likelihood = logLik(pooled_fit)[1],
        n_obs = nrow(cc),
        n_groups = length(unique(cc$city_id)),
        stringsAsFactors = FALSE
    ))
}

# --- 1b. Fixed Effects Poisson (using fixest::fepois) ---
cat("\n--- 1b. FE Poisson (fixest::fepois) ---\n")

# FE Poisson with entity (city) fixed effects
fe_fit <- fepois(crime_count ~ unemployment_rate + police_per_capita +
                 median_income + temperature | city_id,
                 data = cc,
                 cluster = ~city_id)

cat("FE Poisson Results (city fixed effects, cluster-robust SE):\n")
print(summary(fe_fit))

fe_coefs <- coef(fe_fit)
fe_se <- se(fe_fit)
fe_z <- fe_coefs / fe_se
fe_p <- 2 * pnorm(abs(fe_z), lower.tail = FALSE)

fe_table <- data.frame(
    Variable = names(fe_coefs),
    Coefficient = fe_coefs,
    Std_Error = fe_se,
    z_statistic = fe_z,
    p_value = fe_p,
    IRR = exp(fe_coefs)
)
rownames(fe_table) <- NULL
cat("\nFE Poisson Summary:\n")
print(fe_table, digits = 6)

fe_llf <- tryCatch(logLik(fe_fit)[1], error = function(e) NA)

for (i in seq_along(fe_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "fe_poisson_crime",
        variable = names(fe_coefs)[i],
        coefficient = fe_coefs[i],
        std_error = fe_se[i],
        z_statistic = fe_z[i],
        p_value = fe_p[i],
        log_likelihood = fe_llf,
        n_obs = nrow(cc),
        n_groups = length(unique(cc$city_id)),
        stringsAsFactors = FALSE
    ))
}

# --- 1c. FE Poisson + Year FE ---
cat("\n--- 1c. FE Poisson + Year FE ---\n")

fe_year_fit <- fepois(crime_count ~ unemployment_rate + police_per_capita +
                      median_income + temperature | city_id + year,
                      data = cc,
                      cluster = ~city_id)

cat("FE Poisson + Year FE Results:\n")
print(summary(fe_year_fit))

fe_year_coefs <- coef(fe_year_fit)
fe_year_se <- se(fe_year_fit)
fe_year_z <- fe_year_coefs / fe_year_se
fe_year_p <- 2 * pnorm(abs(fe_year_z), lower.tail = FALSE)

fe_year_table <- data.frame(
    Variable = names(fe_year_coefs),
    Coefficient = fe_year_coefs,
    Std_Error = fe_year_se,
    z_statistic = fe_year_z,
    p_value = fe_year_p,
    IRR = exp(fe_year_coefs)
)
rownames(fe_year_table) <- NULL
print(fe_year_table, digits = 6)

fe_year_llf <- tryCatch(logLik(fe_year_fit)[1], error = function(e) NA)

for (i in seq_along(fe_year_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "fe_year_poisson_crime",
        variable = names(fe_year_coefs)[i],
        coefficient = fe_year_coefs[i],
        std_error = fe_year_se[i],
        z_statistic = fe_year_z[i],
        p_value = fe_year_p[i],
        log_likelihood = fe_year_llf,
        n_obs = nrow(cc),
        n_groups = length(unique(cc$city_id)),
        stringsAsFactors = FALSE
    ))
}

# --- 1d. RE Poisson (pglm) ---
cat("\n--- 1d. RE Poisson (pglm) ---\n")

# Convert to pdata.frame for pglm
cc_panel <- pdata.frame(cc, index = c("city_id", "year"))

re_fit <- tryCatch({
    pglm(crime_count ~ unemployment_rate + police_per_capita +
         median_income + temperature,
         data = cc_panel,
         family = poisson,
         model = "random",
         effect = "individual")
}, error = function(e) {
    cat(sprintf("pglm RE failed: %s\n", e$message))
    NULL
})

if (!is.null(re_fit)) {
    re_est <- summary(re_fit)$estimate
    re_coefs <- re_est[, "Estimate"]
    re_se <- re_est[, "Std. error"]
    re_z <- re_est[, "t value"]
    re_p <- re_est[, "Pr(> t)"]
    re_varnames <- rownames(re_est)

    re_table <- data.frame(
        Variable = re_varnames,
        Coefficient = re_coefs,
        Std_Error = re_se,
        z_statistic = re_z,
        p_value = re_p,
        IRR = exp(re_coefs)
    )
    rownames(re_table) <- NULL
    cat("RE Poisson Results:\n")
    print(re_table, digits = 6)

    re_llf <- as.numeric(logLik(re_fit))
    cat(sprintf("Log-Likelihood: %.4f\n", re_llf))

    for (i in seq_along(re_coefs)) {
        results <- rbind(results, data.frame(
            model_name = "re_poisson_crime",
            variable = re_varnames[i],
            coefficient = re_coefs[i],
            std_error = re_se[i],
            z_statistic = re_z[i],
            p_value = re_p[i],
            log_likelihood = re_llf,
            n_obs = nrow(cc),
            n_groups = length(unique(cc$city_id)),
            stringsAsFactors = FALSE
        ))
    }

    # --- Hausman Test (FE vs RE) ---
    cat("\n--- Hausman Test: FE vs RE ---\n")
    # Compare common coefficients between FE and RE (exclude intercept and sigma)
    common_vars <- intersect(names(fe_coefs), re_varnames)
    common_vars <- common_vars[!common_vars %in% c("(Intercept)", "sigma")]

    if (length(common_vars) > 0) {
        beta_fe <- fe_coefs[common_vars]
        # Index RE estimates by name
        re_idx <- match(common_vars, re_varnames)
        beta_re <- re_coefs[re_idx]
        diff <- beta_fe - beta_re

        # Variance of the difference (simplified diagonal approximation)
        var_fe <- fe_se[common_vars]^2
        var_re <- re_se[re_idx]^2
        var_diff <- abs(var_fe - var_re)

        hausman_stat <- sum(diff^2 / var_diff)
        hausman_df <- length(common_vars)
        hausman_pval <- pchisq(hausman_stat, df = hausman_df, lower.tail = FALSE)

        cat(sprintf("Hausman statistic: %.4f\n", hausman_stat))
        cat(sprintf("Degrees of freedom: %d\n", hausman_df))
        cat(sprintf("p-value: %.4f\n", hausman_pval))
        if (hausman_pval < 0.05) {
            cat("Conclusion: Reject H0 -> Use Fixed Effects\n")
        } else {
            cat("Conclusion: Fail to reject H0 -> RE is consistent\n")
        }
    }
} else {
    cat("RE Poisson could not be estimated.\n")
}


# =============================================================================
# PART 2: PPML Gravity Model on bilateral_trade
# =============================================================================

cat("\n\nPART 2: PPML Gravity Model (bilateral_trade)\n")
cat("-------------------------------------------------------------\n")

bt <- read.csv(file.path(data_dir, "bilateral_trade.csv"))
cat(sprintf("Dataset loaded: %d observations, %d variables\n", nrow(bt), ncol(bt)))
cat(sprintf("Columns: %s\n", paste(names(bt), collapse=", ")))
cat(sprintf("Zero trade observations: %d (%.1f%%)\n\n",
            sum(bt$trade_value == 0), 100 * mean(bt$trade_value == 0)))

# Create log variables
bt$log_gdp_exporter <- log(bt$gdp_exporter)
bt$log_gdp_importer <- log(bt$gdp_importer)
bt$log_distance <- log(bt$distance)

# Create pair identifier
bt$pair_id <- paste(bt$exporter, bt$importer, sep = "_")

# --- 2a. Pooled PPML (no fixed effects) ---
cat("--- 2a. Pooled PPML (fixest::fepois) ---\n")

# PPML = Poisson on levels with robust SE (Santos Silva & Tenreyro 2006)
# Using integer-rounded trade_value since fepois expects counts
# For PPML, we use trade_value directly (can be continuous)
ppml_pooled <- fepois(trade_value ~ log_gdp_exporter + log_gdp_importer +
                      log_distance + contiguous + common_language +
                      trade_agreement,
                      data = bt,
                      cluster = ~pair_id)

cat("Pooled PPML Results:\n")
print(summary(ppml_pooled))

ppml_coefs <- coef(ppml_pooled)
ppml_se <- se(ppml_pooled)
ppml_z <- ppml_coefs / ppml_se
ppml_p <- 2 * pnorm(abs(ppml_z), lower.tail = FALSE)

ppml_table <- data.frame(
    Variable = names(ppml_coefs),
    Coefficient = ppml_coefs,
    Std_Error = ppml_se,
    z_statistic = ppml_z,
    p_value = ppml_p,
    IRR = exp(ppml_coefs),
    Pct_Effect = (exp(ppml_coefs) - 1) * 100
)
rownames(ppml_table) <- NULL
cat("\nPPML Coefficient Table:\n")
print(ppml_table, digits = 6)

ppml_pooled_llf <- tryCatch(logLik(ppml_pooled)[1], error = function(e) NA)

for (i in seq_along(ppml_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "ppml_pooled_gravity",
        variable = names(ppml_coefs)[i],
        coefficient = ppml_coefs[i],
        std_error = ppml_se[i],
        z_statistic = ppml_z[i],
        p_value = ppml_p[i],
        log_likelihood = ppml_pooled_llf,
        n_obs = nrow(bt),
        n_groups = length(unique(bt$pair_id)),
        stringsAsFactors = FALSE
    ))
}

# --- 2b. PPML with Pair Fixed Effects ---
cat("\n--- 2b. PPML with Pair Fixed Effects ---\n")

ppml_pair_fe <- fepois(trade_value ~ trade_agreement | pair_id,
                       data = bt,
                       cluster = ~pair_id)

cat("PPML with Pair FE Results:\n")
print(summary(ppml_pair_fe))

ppml_pfe_coefs <- coef(ppml_pair_fe)
ppml_pfe_se <- se(ppml_pair_fe)
ppml_pfe_z <- ppml_pfe_coefs / ppml_pfe_se
ppml_pfe_p <- 2 * pnorm(abs(ppml_pfe_z), lower.tail = FALSE)

cat(sprintf("FTA coefficient: %.4f (SE: %.4f)\n", ppml_pfe_coefs[1], ppml_pfe_se[1]))
cat(sprintf("IRR = exp(%.4f) = %.4f\n", ppml_pfe_coefs[1], exp(ppml_pfe_coefs[1])))
cat(sprintf("FTA increases trade by %.1f%%\n", (exp(ppml_pfe_coefs[1]) - 1) * 100))

ppml_pfe_llf <- tryCatch(logLik(ppml_pair_fe)[1], error = function(e) NA)

for (i in seq_along(ppml_pfe_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "ppml_pair_fe_gravity",
        variable = names(ppml_pfe_coefs)[i],
        coefficient = ppml_pfe_coefs[i],
        std_error = ppml_pfe_se[i],
        z_statistic = ppml_pfe_z[i],
        p_value = ppml_pfe_p[i],
        log_likelihood = ppml_pfe_llf,
        n_obs = nrow(bt),
        n_groups = length(unique(bt$pair_id)),
        stringsAsFactors = FALSE
    ))
}

# --- 2c. PPML with Year Dummies ---
cat("\n--- 2c. PPML with Year Dummies ---\n")

ppml_year <- fepois(trade_value ~ log_gdp_exporter + log_gdp_importer +
                    log_distance + contiguous + common_language +
                    trade_agreement | year,
                    data = bt,
                    cluster = ~pair_id)

cat("PPML + Year FE Results:\n")
print(summary(ppml_year))

ppml_year_coefs <- coef(ppml_year)
ppml_year_se <- se(ppml_year)
ppml_year_z <- ppml_year_coefs / ppml_year_se
ppml_year_p <- 2 * pnorm(abs(ppml_year_z), lower.tail = FALSE)

ppml_year_table <- data.frame(
    Variable = names(ppml_year_coefs),
    Coefficient = ppml_year_coefs,
    Std_Error = ppml_year_se,
    z_statistic = ppml_year_z,
    p_value = ppml_year_p
)
rownames(ppml_year_table) <- NULL
print(ppml_year_table, digits = 6)

ppml_year_llf <- tryCatch(logLik(ppml_year)[1], error = function(e) NA)

for (i in seq_along(ppml_year_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "ppml_year_fe_gravity",
        variable = names(ppml_year_coefs)[i],
        coefficient = ppml_year_coefs[i],
        std_error = ppml_year_se[i],
        z_statistic = ppml_year_z[i],
        p_value = ppml_year_p[i],
        log_likelihood = ppml_year_llf,
        n_obs = nrow(bt),
        n_groups = length(unique(bt$pair_id)),
        stringsAsFactors = FALSE
    ))
}

# --- 2d. PPML with Pair + Year FE ---
cat("\n--- 2d. PPML with Pair + Year FE ---\n")

ppml_full <- fepois(trade_value ~ trade_agreement | pair_id + year,
                    data = bt,
                    cluster = ~pair_id)

cat("PPML with Pair + Year FE Results:\n")
print(summary(ppml_full))

ppml_full_coefs <- coef(ppml_full)
ppml_full_se <- se(ppml_full)
ppml_full_z <- ppml_full_coefs / ppml_full_se
ppml_full_p <- 2 * pnorm(abs(ppml_full_z), lower.tail = FALSE)

cat(sprintf("FTA coefficient: %.4f (SE: %.4f)\n", ppml_full_coefs[1], ppml_full_se[1]))
cat(sprintf("FTA increases trade by %.1f%% (Pair + Year FE)\n",
            (exp(ppml_full_coefs[1]) - 1) * 100))

ppml_full_llf <- tryCatch(logLik(ppml_full)[1], error = function(e) NA)

for (i in seq_along(ppml_full_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "ppml_pair_year_fe_gravity",
        variable = names(ppml_full_coefs)[i],
        coefficient = ppml_full_coefs[i],
        std_error = ppml_full_se[i],
        z_statistic = ppml_full_z[i],
        p_value = ppml_full_p[i],
        log_likelihood = ppml_full_llf,
        n_obs = nrow(bt),
        n_groups = length(unique(bt$pair_id)),
        stringsAsFactors = FALSE
    ))
}

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

cat("\n\nSaving results to CSV...\n")

rownames(results) <- NULL
output_file <- file.path(output_dir, "results_02_fe_re_ppml.csv")
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", output_file))

cat("\n=============================================================\n")
cat("Validation 02 complete.\n")
cat("=============================================================\n")
