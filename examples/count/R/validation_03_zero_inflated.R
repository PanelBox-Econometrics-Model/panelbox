# =============================================================================
# Validation Script 03: Zero-Inflated Poisson and Zero-Inflated Negative Binomial
# =============================================================================
# Compares R results with PanelBox for zero-inflated count data models.
#
# Models estimated (all on healthcare_zinb):
#   1. Baseline Poisson: doctor_visits ~ age + income + chronic_condition
#   2. Standard Negative Binomial: same formula
#   3. ZIP: count ~ age + income + chronic_condition | inflate ~ insurance + distance_clinic + urban
#   4. ZINB: same specification as ZIP, plus alpha
#
# Packages: pscl (zeroinfl), MASS (glm.nb)
# =============================================================================

library(pscl)
library(MASS)
library(sandwich)
library(lmtest)

cat("=============================================================\n")
cat("Validation 03: Zero-Inflated Models (ZIP / ZINB)\n")
cat("=============================================================\n\n")

# --- Paths ---
data_dir <- "/home/guhaase/projetos/panelbox/examples/count/data"
output_dir <- "/home/guhaase/projetos/panelbox/examples/count/R"

# Initialize results data frame
results <- data.frame(
    model_name = character(),
    equation = character(),
    variable = character(),
    coefficient = numeric(),
    std_error = numeric(),
    z_statistic = numeric(),
    p_value = numeric(),
    log_likelihood = numeric(),
    aic = numeric(),
    bic = numeric(),
    n_obs = numeric(),
    alpha = numeric(),
    vuong_statistic = numeric(),
    vuong_pvalue = numeric(),
    stringsAsFactors = FALSE
)

# =============================================================================
# Load data
# =============================================================================

df <- read.csv(file.path(data_dir, "healthcare_zinb.csv"))
cat(sprintf("Dataset loaded: %d observations, %d variables\n", nrow(df), ncol(df)))
cat(sprintf("Columns: %s\n\n", paste(names(df), collapse=", ")))

# Descriptive statistics
cat("Descriptive statistics for 'doctor_visits':\n")
cat(sprintf("  Mean:        %.4f\n", mean(df$doctor_visits)))
cat(sprintf("  Variance:    %.4f\n", var(df$doctor_visits)))
cat(sprintf("  Var/Mean:    %.4f\n", var(df$doctor_visits) / mean(df$doctor_visits)))
cat(sprintf("  Zeros:       %d (%.1f%%)\n", sum(df$doctor_visits == 0),
            100 * mean(df$doctor_visits == 0)))
cat(sprintf("  Max:         %d\n\n", max(df$doctor_visits)))

# =============================================================================
# PART 1: Baseline Poisson
# =============================================================================

cat("PART 1: Baseline Poisson\n")
cat("-------------------------------------------------------------\n")

poisson_fit <- glm(doctor_visits ~ age + income + chronic_condition,
                   family = poisson(link = "log"),
                   data = df)

cat("Poisson GLM Results:\n")
print(summary(poisson_fit))

# Robust SE
poisson_robust_vcov <- vcovHC(poisson_fit, type = "HC0")
poisson_robust_se <- sqrt(diag(poisson_robust_vcov))
poisson_coefs <- coef(poisson_fit)
poisson_z <- poisson_coefs / poisson_robust_se
poisson_p <- 2 * pnorm(abs(poisson_z), lower.tail = FALSE)

poisson_llf <- logLik(poisson_fit)[1]
poisson_aic <- AIC(poisson_fit)
poisson_bic <- BIC(poisson_fit)

cat(sprintf("Log-Likelihood: %.4f\n", poisson_llf))
cat(sprintf("AIC: %.4f\n", poisson_aic))
cat(sprintf("BIC: %.4f\n\n", poisson_bic))

# Predicted zeros from Poisson
poisson_pred_lambda <- fitted(poisson_fit)
poisson_pred_zeros <- sum(dpois(0, poisson_pred_lambda))
cat(sprintf("Observed zeros:           %d (%.1f%%)\n",
            sum(df$doctor_visits == 0), 100 * mean(df$doctor_visits == 0)))
cat(sprintf("Poisson-predicted zeros:  %.1f (%.1f%%)\n\n",
            poisson_pred_zeros, 100 * poisson_pred_zeros / nrow(df)))

# Save Poisson results
for (i in seq_along(poisson_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "poisson_baseline",
        equation = "count",
        variable = names(poisson_coefs)[i],
        coefficient = poisson_coefs[i],
        std_error = poisson_robust_se[i],
        z_statistic = poisson_z[i],
        p_value = poisson_p[i],
        log_likelihood = poisson_llf,
        aic = poisson_aic,
        bic = poisson_bic,
        n_obs = nrow(df),
        alpha = NA,
        vuong_statistic = NA,
        vuong_pvalue = NA,
        stringsAsFactors = FALSE
    ))
}

# =============================================================================
# PART 2: Standard Negative Binomial
# =============================================================================

cat("PART 2: Standard Negative Binomial\n")
cat("-------------------------------------------------------------\n")

nb_fit <- glm.nb(doctor_visits ~ age + income + chronic_condition,
                 data = df)

cat("Negative Binomial Results:\n")
print(summary(nb_fit))

nb_coefs <- coef(nb_fit)
nb_se <- summary(nb_fit)$coefficients[, "Std. Error"]
nb_z <- summary(nb_fit)$coefficients[, "z value"]
nb_p <- summary(nb_fit)$coefficients[, "Pr(>|z|)"]
nb_alpha <- 1 / nb_fit$theta
nb_llf <- logLik(nb_fit)[1]
nb_aic <- AIC(nb_fit)
nb_bic <- BIC(nb_fit)

cat(sprintf("Alpha (1/theta): %.6f\n", nb_alpha))
cat(sprintf("Log-Likelihood: %.4f\n", nb_llf))
cat(sprintf("AIC: %.4f\n", nb_aic))
cat(sprintf("BIC: %.4f\n\n", nb_bic))

for (i in seq_along(nb_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "negbin_baseline",
        equation = "count",
        variable = names(nb_coefs)[i],
        coefficient = nb_coefs[i],
        std_error = nb_se[i],
        z_statistic = nb_z[i],
        p_value = nb_p[i],
        log_likelihood = nb_llf,
        aic = nb_aic,
        bic = nb_bic,
        n_obs = nrow(df),
        alpha = nb_alpha,
        vuong_statistic = NA,
        vuong_pvalue = NA,
        stringsAsFactors = FALSE
    ))
}

# =============================================================================
# PART 3: Zero-Inflated Poisson (ZIP)
# =============================================================================

cat("PART 3: Zero-Inflated Poisson (ZIP)\n")
cat("-------------------------------------------------------------\n")

# Count equation: doctor_visits ~ age + income + chronic_condition
# Inflate equation: ~ insurance + distance_clinic + urban
zip_fit <- zeroinfl(doctor_visits ~ age + income + chronic_condition |
                    insurance + distance_clinic + urban,
                    data = df,
                    dist = "poisson")

cat("ZIP Results:\n")
print(summary(zip_fit))

zip_coefs_count <- coef(zip_fit, model = "count")
zip_coefs_zero <- coef(zip_fit, model = "zero")
zip_se_count <- summary(zip_fit)$coefficients$count[, "Std. Error"]
zip_se_zero <- summary(zip_fit)$coefficients$zero[, "Std. Error"]
zip_z_count <- summary(zip_fit)$coefficients$count[, "z value"]
zip_z_zero <- summary(zip_fit)$coefficients$zero[, "z value"]
zip_p_count <- summary(zip_fit)$coefficients$count[, "Pr(>|z|)"]
zip_p_zero <- summary(zip_fit)$coefficients$zero[, "Pr(>|z|)"]

zip_llf <- logLik(zip_fit)[1]
zip_aic <- AIC(zip_fit)
zip_bic <- BIC(zip_fit)
zip_n_params <- length(c(zip_coefs_count, zip_coefs_zero))

cat(sprintf("\nZIP Log-Likelihood: %.4f\n", zip_llf))
cat(sprintf("ZIP AIC: %.4f\n", zip_aic))
cat(sprintf("ZIP BIC: %.4f\n", zip_bic))
cat(sprintf("ZIP parameters: %d\n\n", zip_n_params))

# Vuong test: ZIP vs Poisson (manual computation)
cat("Vuong Test: ZIP vs standard Poisson\n")
cat("-------------------------------------------------------------\n")

# Compute Vuong statistic manually
# m_i = log(f_ZIP(y_i) / f_Poisson(y_i))
zip_llf_i <- predprob(zip_fit)
poisson_llf_i <- dpois(df$doctor_visits, lambda = fitted(poisson_fit))
# For ZIP, extract the probability of each observed outcome
zip_prob_i <- sapply(seq_len(nrow(df)), function(i) zip_llf_i[i, as.character(df$doctor_visits[i])])
# Handle missing columns for large counts
zip_prob_i[is.na(zip_prob_i)] <- 1e-300

m_i <- log(zip_prob_i) - log(poisson_llf_i)
vuong_stat <- sqrt(nrow(df)) * mean(m_i) / sd(m_i)
vuong_pval <- pnorm(abs(vuong_stat), lower.tail = FALSE)
cat(sprintf("Vuong statistic: %.4f\n", vuong_stat))
cat(sprintf("Vuong p-value: %.2e\n", vuong_pval))

# Also print pscl::vuong for reference
cat("\npscl::vuong output:\n")
invisible(vuong(zip_fit, poisson_fit))

# Save ZIP count equation results
for (i in seq_along(zip_coefs_count)) {
    results <- rbind(results, data.frame(
        model_name = "zip",
        equation = "count",
        variable = names(zip_coefs_count)[i],
        coefficient = zip_coefs_count[i],
        std_error = zip_se_count[i],
        z_statistic = zip_z_count[i],
        p_value = zip_p_count[i],
        log_likelihood = zip_llf,
        aic = zip_aic,
        bic = zip_bic,
        n_obs = nrow(df),
        alpha = NA,
        vuong_statistic = vuong_stat,
        vuong_pvalue = vuong_pval,
        stringsAsFactors = FALSE
    ))
}

# Save ZIP zero (inflation) equation results
for (i in seq_along(zip_coefs_zero)) {
    results <- rbind(results, data.frame(
        model_name = "zip",
        equation = "zero_inflate",
        variable = names(zip_coefs_zero)[i],
        coefficient = zip_coefs_zero[i],
        std_error = zip_se_zero[i],
        z_statistic = zip_z_zero[i],
        p_value = zip_p_zero[i],
        log_likelihood = zip_llf,
        aic = zip_aic,
        bic = zip_bic,
        n_obs = nrow(df),
        alpha = NA,
        vuong_statistic = vuong_stat,
        vuong_pvalue = vuong_pval,
        stringsAsFactors = FALSE
    ))
}

# =============================================================================
# PART 4: Zero-Inflated Negative Binomial (ZINB)
# =============================================================================

cat("\n\nPART 4: Zero-Inflated Negative Binomial (ZINB)\n")
cat("-------------------------------------------------------------\n")

zinb_fit <- zeroinfl(doctor_visits ~ age + income + chronic_condition |
                     insurance + distance_clinic + urban,
                     data = df,
                     dist = "negbin")

cat("ZINB Results:\n")
print(summary(zinb_fit))

zinb_coefs_count <- coef(zinb_fit, model = "count")
zinb_coefs_zero <- coef(zinb_fit, model = "zero")
zinb_se_count <- summary(zinb_fit)$coefficients$count[, "Std. Error"]
zinb_se_zero <- summary(zinb_fit)$coefficients$zero[, "Std. Error"]
zinb_z_count <- summary(zinb_fit)$coefficients$count[, "z value"]
zinb_z_zero <- summary(zinb_fit)$coefficients$zero[, "z value"]
zinb_p_count <- summary(zinb_fit)$coefficients$count[, "Pr(>|z|)"]
zinb_p_zero <- summary(zinb_fit)$coefficients$zero[, "Pr(>|z|)"]

zinb_alpha <- 1 / zinb_fit$theta
zinb_llf <- logLik(zinb_fit)[1]
zinb_aic <- AIC(zinb_fit)
zinb_bic <- BIC(zinb_fit)
zinb_n_params <- length(c(zinb_coefs_count, zinb_coefs_zero)) + 1  # +1 for theta/alpha

cat(sprintf("\nZINB Alpha (1/theta): %.6f\n", zinb_alpha))
cat(sprintf("ZINB Theta: %.6f\n", zinb_fit$theta))
cat(sprintf("ZINB Log-Likelihood: %.4f\n", zinb_llf))
cat(sprintf("ZINB AIC: %.4f\n", zinb_aic))
cat(sprintf("ZINB BIC: %.4f\n", zinb_bic))
cat(sprintf("ZINB parameters: %d\n\n", zinb_n_params))

# Save ZINB count equation results
for (i in seq_along(zinb_coefs_count)) {
    results <- rbind(results, data.frame(
        model_name = "zinb",
        equation = "count",
        variable = names(zinb_coefs_count)[i],
        coefficient = zinb_coefs_count[i],
        std_error = zinb_se_count[i],
        z_statistic = zinb_z_count[i],
        p_value = zinb_p_count[i],
        log_likelihood = zinb_llf,
        aic = zinb_aic,
        bic = zinb_bic,
        n_obs = nrow(df),
        alpha = zinb_alpha,
        vuong_statistic = NA,
        vuong_pvalue = NA,
        stringsAsFactors = FALSE
    ))
}

# Save ZINB zero (inflation) equation results
for (i in seq_along(zinb_coefs_zero)) {
    results <- rbind(results, data.frame(
        model_name = "zinb",
        equation = "zero_inflate",
        variable = names(zinb_coefs_zero)[i],
        coefficient = zinb_coefs_zero[i],
        std_error = zinb_se_zero[i],
        z_statistic = zinb_z_zero[i],
        p_value = zinb_p_zero[i],
        log_likelihood = zinb_llf,
        aic = zinb_aic,
        bic = zinb_bic,
        n_obs = nrow(df),
        alpha = zinb_alpha,
        vuong_statistic = NA,
        vuong_pvalue = NA,
        stringsAsFactors = FALSE
    ))
}

# =============================================================================
# MODEL COMPARISON SUMMARY
# =============================================================================

cat("\n\n=============================================================\n")
cat("MODEL COMPARISON SUMMARY\n")
cat("=============================================================\n")

comparison <- data.frame(
    Model = c("Poisson", "Neg. Binomial", "ZIP", "ZINB"),
    LogLik = c(poisson_llf, nb_llf, zip_llf, zinb_llf),
    AIC = c(poisson_aic, nb_aic, zip_aic, zinb_aic),
    BIC = c(poisson_bic, nb_bic, zip_bic, zinb_bic),
    N_Params = c(length(poisson_coefs), length(nb_coefs) + 1,
                 zip_n_params, zinb_n_params)
)
print(comparison, digits = 6)

best_aic <- comparison$Model[which.min(comparison$AIC)]
best_bic <- comparison$Model[which.min(comparison$BIC)]
cat(sprintf("\nBest by AIC: %s\n", best_aic))
cat(sprintf("Best by BIC: %s\n", best_bic))

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

cat("\n\nSaving results to CSV...\n")

rownames(results) <- NULL
output_file <- file.path(output_dir, "results_03_zero_inflated.csv")
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", output_file))

cat("\n=============================================================\n")
cat("Validation 03 complete.\n")
cat("=============================================================\n")
