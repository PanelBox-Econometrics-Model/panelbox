# =============================================================================
# Validation Script 01: Poisson and Negative Binomial Regression
# =============================================================================
# Compares R results with PanelBox for count data models.
#
# Models estimated:
#   1. Poisson GLM (healthcare_visits): visits ~ age + income + insurance + chronic
#   2. Negative Binomial GLM (firm_patents): patents ~ log_rd + log_emp + firm_age
#                                            + tech_sector + public_funding + international
#
# Packages: MASS (glm.nb), stats (glm)
# =============================================================================

library(MASS)
library(sandwich)
library(lmtest)

cat("=============================================================\n")
cat("Validation 01: Poisson and Negative Binomial Models\n")
cat("=============================================================\n\n")

# --- Paths ---
data_dir <- "/home/guhaase/projetos/panelbox/examples/count/data"
output_dir <- "/home/guhaase/projetos/panelbox/examples/count/R"

# =============================================================================
# PART 1: Poisson Regression on healthcare_visits
# =============================================================================

cat("PART 1: Poisson Regression (healthcare_visits)\n")
cat("-------------------------------------------------------------\n")

hc <- read.csv(file.path(data_dir, "healthcare_visits.csv"))
cat(sprintf("Dataset loaded: %d observations, %d variables\n", nrow(hc), ncol(hc)))
cat(sprintf("Columns: %s\n\n", paste(names(hc), collapse=", ")))

# Descriptive statistics for visits
cat("Descriptive statistics for 'visits':\n")
cat(sprintf("  Mean:     %.4f\n", mean(hc$visits)))
cat(sprintf("  Variance: %.4f\n", var(hc$visits)))
cat(sprintf("  Var/Mean: %.4f (overdispersion index)\n\n", var(hc$visits) / mean(hc$visits)))

# Fit Poisson model: visits ~ age + income + insurance + chronic
poisson_fit <- glm(visits ~ age + income + insurance + chronic,
                   family = poisson(link = "log"),
                   data = hc)

cat("Poisson GLM Results:\n")
print(summary(poisson_fit))

# Robust (sandwich) standard errors to match PanelBox se_type="robust"
poisson_robust_vcov <- vcovHC(poisson_fit, type = "HC0")
poisson_robust_se <- sqrt(diag(poisson_robust_vcov))
poisson_coefs <- coef(poisson_fit)
poisson_z <- poisson_coefs / poisson_robust_se
poisson_p <- 2 * pnorm(abs(poisson_z), lower.tail = FALSE)

cat("\nPoisson with Robust (HC0) Standard Errors:\n")
cat("-------------------------------------------------------------\n")
poisson_robust_table <- data.frame(
    Variable = names(poisson_coefs),
    Coefficient = poisson_coefs,
    Std_Error_Robust = poisson_robust_se,
    z_statistic = poisson_z,
    p_value = poisson_p,
    IRR = exp(poisson_coefs),
    Pct_Change = (exp(poisson_coefs) - 1) * 100
)
rownames(poisson_robust_table) <- NULL
print(poisson_robust_table, digits = 6)

# Model fit statistics
poisson_llf <- logLik(poisson_fit)[1]
poisson_aic <- AIC(poisson_fit)
poisson_bic <- BIC(poisson_fit)
poisson_deviance <- deviance(poisson_fit)
poisson_n <- nrow(hc)

cat(sprintf("\nLog-Likelihood: %.4f\n", poisson_llf))
cat(sprintf("AIC:            %.4f\n", poisson_aic))
cat(sprintf("BIC:            %.4f\n", poisson_bic))
cat(sprintf("Deviance:       %.4f\n", poisson_deviance))
cat(sprintf("N:              %d\n", poisson_n))

# Overdispersion test (Cameron-Trivedi style)
y_pred <- fitted(poisson_fit)
y_obs <- hc$visits
aux_y <- ((y_obs - y_pred)^2 / y_pred) - 1
aux_reg <- lm(aux_y ~ y_pred)
ct_tstat <- summary(aux_reg)$coefficients["y_pred", "t value"]
ct_pval <- summary(aux_reg)$coefficients["y_pred", "Pr(>|t|)"]
cat(sprintf("\nCameron-Trivedi Overdispersion Test:\n"))
cat(sprintf("  t-statistic: %.4f\n", ct_tstat))
cat(sprintf("  p-value:     %.4f\n", ct_pval))

# =============================================================================
# PART 2: Negative Binomial Regression on firm_patents
# =============================================================================

cat("\n\nPART 2: Negative Binomial Regression (firm_patents)\n")
cat("-------------------------------------------------------------\n")

fp <- read.csv(file.path(data_dir, "firm_patents.csv"))
cat(sprintf("Dataset loaded: %d observations, %d variables\n", nrow(fp), ncol(fp)))
cat(sprintf("Columns: %s\n\n", paste(names(fp), collapse=", ")))

# Create log variables matching PanelBox notebook
fp$log_rd <- log(fp$rd_spending)
fp$log_emp <- log(fp$employees)

cat("Descriptive statistics for 'patents':\n")
cat(sprintf("  Mean:     %.4f\n", mean(fp$patents)))
cat(sprintf("  Variance: %.4f\n", var(fp$patents)))
cat(sprintf("  Var/Mean: %.4f (overdispersion index)\n\n", var(fp$patents) / mean(fp$patents)))

# Model formula
fmla <- patents ~ log_rd + log_emp + firm_age + tech_sector + public_funding + international

# First fit Poisson for comparison
poisson_fp <- glm(fmla, family = poisson(link = "log"), data = fp)

cat("Poisson GLM on firm_patents:\n")
print(summary(poisson_fp))

# Poisson with cluster-robust SE (cluster on firm_id)
# Using vcovCL for cluster-robust variance
poisson_fp_cluster_vcov <- vcovCL(poisson_fp, cluster = fp$firm_id)
poisson_fp_cluster_se <- sqrt(diag(poisson_fp_cluster_vcov))
poisson_fp_coefs <- coef(poisson_fp)
poisson_fp_z <- poisson_fp_coefs / poisson_fp_cluster_se
poisson_fp_p <- 2 * pnorm(abs(poisson_fp_z), lower.tail = FALSE)

cat("\nPoisson with Cluster-Robust SE (by firm_id):\n")
cat("-------------------------------------------------------------\n")
poisson_fp_table <- data.frame(
    Variable = names(poisson_fp_coefs),
    Coefficient = poisson_fp_coefs,
    Std_Error_Cluster = poisson_fp_cluster_se,
    z_statistic = poisson_fp_z,
    p_value = poisson_fp_p
)
rownames(poisson_fp_table) <- NULL
print(poisson_fp_table, digits = 6)

poisson_fp_llf <- logLik(poisson_fp)[1]

# Negative Binomial (NB2) model
cat("\n\nFitting Negative Binomial (NB2) model...\n")
nb_fp <- glm.nb(fmla, data = fp)

cat("Negative Binomial GLM Results:\n")
print(summary(nb_fp))

nb_coefs <- coef(nb_fp)
nb_se <- summary(nb_fp)$coefficients[, "Std. Error"]
nb_z <- summary(nb_fp)$coefficients[, "z value"]
nb_p <- summary(nb_fp)$coefficients[, "Pr(>|z|)"]
nb_alpha <- 1 / nb_fp$theta  # NB2 alpha = 1/theta in R
nb_llf <- logLik(nb_fp)[1]
nb_aic <- AIC(nb_fp)
nb_bic <- BIC(nb_fp)

cat(sprintf("\nDispersion parameter (alpha = 1/theta): %.6f\n", nb_alpha))
cat(sprintf("Theta:                                  %.6f\n", nb_fp$theta))
cat(sprintf("SE(Theta):                              %.6f\n", nb_fp$SE.theta))
cat(sprintf("Log-Likelihood (Poisson): %.4f\n", poisson_fp_llf))
cat(sprintf("Log-Likelihood (NB):      %.4f\n", nb_llf))
cat(sprintf("AIC (NB):                 %.4f\n", nb_aic))
cat(sprintf("BIC (NB):                 %.4f\n", nb_bic))

# Likelihood Ratio Test: Poisson vs NB
lr_stat <- 2 * (nb_llf - poisson_fp_llf)
lr_pval <- 0.5 * pchisq(lr_stat, df = 1, lower.tail = FALSE)  # One-sided test on boundary
cat(sprintf("\nLikelihood Ratio Test (Poisson vs NB):\n"))
cat(sprintf("  LR statistic: %.4f\n", lr_stat))
cat(sprintf("  p-value:      %.2e (chi-sq mixture, 1 df)\n", lr_pval))
if (lr_pval < 0.05) {
    cat("  Conclusion:   Reject H0 -> NB preferred over Poisson\n")
} else {
    cat("  Conclusion:   Fail to reject H0 -> Poisson adequate\n")
}

# IRR for NB
nb_irr_table <- data.frame(
    Variable = names(nb_coefs),
    Coefficient = nb_coefs,
    Std_Error = nb_se,
    z_statistic = nb_z,
    p_value = nb_p,
    IRR = exp(nb_coefs),
    Pct_Change = (exp(nb_coefs) - 1) * 100
)
rownames(nb_irr_table) <- NULL
cat("\nNegative Binomial IRR Table:\n")
print(nb_irr_table, digits = 6)

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

cat("\n\nSaving results to CSV...\n")

# Combine all results
results <- data.frame(
    model_name = character(),
    variable = character(),
    coefficient = numeric(),
    std_error = numeric(),
    z_statistic = numeric(),
    p_value = numeric(),
    log_likelihood = numeric(),
    aic = numeric(),
    bic = numeric(),
    deviance = numeric(),
    n_obs = numeric(),
    alpha = numeric(),
    stringsAsFactors = FALSE
)

# Poisson (healthcare_visits) with robust SE
for (i in seq_along(poisson_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "poisson_healthcare",
        variable = names(poisson_coefs)[i],
        coefficient = poisson_coefs[i],
        std_error = poisson_robust_se[i],
        z_statistic = poisson_z[i],
        p_value = poisson_p[i],
        log_likelihood = poisson_llf,
        aic = poisson_aic,
        bic = poisson_bic,
        deviance = poisson_deviance,
        n_obs = poisson_n,
        alpha = NA,
        stringsAsFactors = FALSE
    ))
}

# Poisson (firm_patents) with cluster SE
for (i in seq_along(poisson_fp_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "poisson_patents_cluster",
        variable = names(poisson_fp_coefs)[i],
        coefficient = poisson_fp_coefs[i],
        std_error = poisson_fp_cluster_se[i],
        z_statistic = poisson_fp_z[i],
        p_value = poisson_fp_p[i],
        log_likelihood = poisson_fp_llf,
        aic = AIC(poisson_fp),
        bic = BIC(poisson_fp),
        deviance = deviance(poisson_fp),
        n_obs = nrow(fp),
        alpha = NA,
        stringsAsFactors = FALSE
    ))
}

# NB (firm_patents) with ML SE
for (i in seq_along(nb_coefs)) {
    results <- rbind(results, data.frame(
        model_name = "negbin_patents",
        variable = names(nb_coefs)[i],
        coefficient = nb_coefs[i],
        std_error = nb_se[i],
        z_statistic = nb_z[i],
        p_value = nb_p[i],
        log_likelihood = nb_llf,
        aic = nb_aic,
        bic = nb_bic,
        deviance = deviance(nb_fp),
        n_obs = nrow(fp),
        alpha = nb_alpha,
        stringsAsFactors = FALSE
    ))
}

rownames(results) <- NULL

output_file <- file.path(output_dir, "results_01_poisson_negbin.csv")
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", output_file))

cat("\n=============================================================\n")
cat("Validation 01 complete.\n")
cat("=============================================================\n")
