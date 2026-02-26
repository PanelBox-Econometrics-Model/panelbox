#!/usr/bin/env Rscript
# Script robusto para gerar benchmarks do pacote frontier
# Autor: PanelBox Development Team
# Data: 2026-02-15

library(frontier)

cat("========================================\n")
cat("Gerando benchmarks do R frontier\n")
cat("========================================\n\n")

# Criar diretório de saída
output_dir <- "../r_results"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# ============================================================================
# Dataset: Rice Production Philippines
# ============================================================================
cat("1. Carregando dataset Rice Production (Philippines)...\n")
data(riceProdPhil)

# Salvar dataset
write.csv(riceProdPhil, file.path(output_dir, "riceProdPhil.csv"), row.names = FALSE)
cat("   Salvo:", file.path(output_dir, "riceProdPhil.csv"), "\n")
cat("   Dimensões:", nrow(riceProdPhil), "obs\n\n")

# ============================================================================
# Cross-Section Model - Half-Normal
# ============================================================================
cat("2. Estimando Cross-Section SFA (Half-Normal)...\n")

# Agregar para cross-section
rice_cs <- aggregate(
    cbind(PROD, AREA, LABOR, NPK, OTHER) ~ FMERCODE,
    data = riceProdPhil,
    FUN = mean
)

# Log transform
rice_cs$log_output <- log(rice_cs$PROD)
rice_cs$log_area <- log(rice_cs$AREA)
rice_cs$log_labor <- log(rice_cs$LABOR)
rice_cs$log_npk <- log(rice_cs$NPK)
rice_cs$log_other <- log(rice_cs$OTHER)

# Estimate
sfa_hn <- sfa(
    log_output ~ log_area + log_labor + log_npk + log_other,
    data = rice_cs,
    ineffDecrease = TRUE
)

# Extract parameters
coefs_hn <- coef(sfa_hn)
params_hn <- data.frame(
    parameter = names(coefs_hn),
    estimate = as.numeric(coefs_hn),
    stringsAsFactors = FALSE
)

# Add derived variance components
sigmaSq <- coefs_hn["sigmaSq"]
gamma <- coefs_hn["gamma"]

derived_params <- data.frame(
    parameter = c('sigma_v_sq', 'sigma_u_sq', 'lambda', 'sigma_v', 'sigma_u'),
    estimate = c(
        sigmaSq * (1 - gamma),  # sigma_v^2
        sigmaSq * gamma,        # sigma_u^2
        sqrt(gamma / (1 - gamma)),  # lambda
        sqrt(sigmaSq * (1 - gamma)),  # sigma_v
        sqrt(sigmaSq * gamma)   # sigma_u
    ),
    stringsAsFactors = FALSE
)

params_hn <- rbind(params_hn, derived_params)

# Efficiencies
eff_hn <- efficiencies(sfa_hn)
eff_df_hn <- data.frame(
    firm_id = rice_cs$FMERCODE,
    efficiency = as.numeric(eff_hn),
    stringsAsFactors = FALSE
)

# Log-likelihood
loglik_hn <- data.frame(
    model = "cross_section_half_normal",
    loglik = as.numeric(logLik(sfa_hn)),
    stringsAsFactors = FALSE
)

# Save
write.csv(params_hn, file.path(output_dir, "r_frontier_cs_halfnormal_params.csv"), row.names = FALSE)
write.csv(eff_df_hn, file.path(output_dir, "r_frontier_cs_halfnormal_efficiency.csv"), row.names = FALSE)
write.csv(loglik_hn, file.path(output_dir, "r_frontier_cs_halfnormal_loglik.csv"), row.names = FALSE)

cat("   Log-likelihood:", loglik_hn$loglik, "\n")
cat("   gamma:", gamma, "\n")
cat("   Mean efficiency:", mean(eff_hn), "\n\n")

# ============================================================================
# Panel Model - Pitt & Lee (Time-Invariant)
# ============================================================================
cat("3. Estimando Panel SFA - Pitt & Lee...\n")

# Prepare panel data
rice_panel <- riceProdPhil
rice_panel$log_output <- log(rice_panel$PROD)
rice_panel$log_area <- log(rice_panel$AREA)
rice_panel$log_labor <- log(rice_panel$LABOR)
rice_panel$log_npk <- log(rice_panel$NPK)
rice_panel$log_other <- log(rice_panel$OTHER)

# Estimate
sfa_pl <- sfa(
    log_output ~ log_area + log_labor + log_npk + log_other,
    data = rice_panel,
    ineffDecrease = TRUE,
    timeEffect = FALSE  # time-invariant inefficiency
)

# Extract parameters
coefs_pl <- coef(sfa_pl)
params_pl <- data.frame(
    parameter = names(coefs_pl),
    estimate = as.numeric(coefs_pl),
    stringsAsFactors = FALSE
)

# Add derived variance components
sigmaSq_pl <- coefs_pl["sigmaSq"]
gamma_pl <- coefs_pl["gamma"]

derived_params_pl <- data.frame(
    parameter = c('sigma_v_sq', 'sigma_u_sq', 'lambda', 'sigma_v', 'sigma_u'),
    estimate = c(
        sigmaSq_pl * (1 - gamma_pl),
        sigmaSq_pl * gamma_pl,
        sqrt(gamma_pl / (1 - gamma_pl)),
        sqrt(sigmaSq_pl * (1 - gamma_pl)),
        sqrt(sigmaSq_pl * gamma_pl)
    ),
    stringsAsFactors = FALSE
)

params_pl <- rbind(params_pl, derived_params_pl)

# Efficiencies
eff_pl <- efficiencies(sfa_pl)
eff_df_pl <- data.frame(
    firm_id = rice_panel$FMERCODE,
    year = rice_panel$YEARDUM,
    efficiency = as.numeric(eff_pl),
    stringsAsFactors = FALSE
)

# Log-likelihood
loglik_pl <- data.frame(
    model = "panel_pitt_lee",
    loglik = as.numeric(logLik(sfa_pl)),
    stringsAsFactors = FALSE
)

# Save
write.csv(params_pl, file.path(output_dir, "r_frontier_panel_pittlee_params.csv"), row.names = FALSE)
write.csv(eff_df_pl, file.path(output_dir, "r_frontier_panel_pittlee_efficiency.csv"), row.names = FALSE)
write.csv(loglik_pl, file.path(output_dir, "r_frontier_panel_pittlee_loglik.csv"), row.names = FALSE)

cat("   Log-likelihood:", loglik_pl$loglik, "\n")
cat("   gamma:", gamma_pl, "\n")
cat("   Mean efficiency:", mean(eff_pl), "\n\n")

# ============================================================================
# Panel Model - Battese & Coelli 1992 (Time-Varying)
# ============================================================================
cat("4. Estimando Panel SFA - Battese & Coelli (1992)...\n")

# Estimate
sfa_bc92 <- sfa(
    log_output ~ log_area + log_labor + log_npk + log_other,
    data = rice_panel,
    ineffDecrease = TRUE,
    timeEffect = TRUE  # time-varying inefficiency (BC92 decay)
)

# Extract parameters
coefs_bc92 <- coef(sfa_bc92)
params_bc92 <- data.frame(
    parameter = names(coefs_bc92),
    estimate = as.numeric(coefs_bc92),
    stringsAsFactors = FALSE
)

# Add derived variance components
sigmaSq_bc92 <- coefs_bc92["sigmaSq"]
gamma_bc92 <- coefs_bc92["gamma"]

derived_params_bc92 <- data.frame(
    parameter = c('sigma_v_sq', 'sigma_u_sq', 'lambda', 'sigma_v', 'sigma_u'),
    estimate = c(
        sigmaSq_bc92 * (1 - gamma_bc92),
        sigmaSq_bc92 * gamma_bc92,
        sqrt(gamma_bc92 / (1 - gamma_bc92)),
        sqrt(sigmaSq_bc92 * (1 - gamma_bc92)),
        sqrt(sigmaSq_bc92 * gamma_bc92)
    ),
    stringsAsFactors = FALSE
)

params_bc92 <- rbind(params_bc92, derived_params_bc92)

# Efficiencies
eff_bc92 <- efficiencies(sfa_bc92)
eff_df_bc92 <- data.frame(
    firm_id = rice_panel$FMERCODE,
    year = rice_panel$YEARDUM,
    efficiency = as.numeric(eff_bc92),
    stringsAsFactors = FALSE
)

# Log-likelihood
loglik_bc92 <- data.frame(
    model = "panel_bc92",
    loglik = as.numeric(logLik(sfa_bc92)),
    stringsAsFactors = FALSE
)

# Save
write.csv(params_bc92, file.path(output_dir, "r_frontier_panel_bc92_params.csv"), row.names = FALSE)
write.csv(eff_df_bc92, file.path(output_dir, "r_frontier_panel_bc92_efficiency.csv"), row.names = FALSE)
write.csv(loglik_bc92, file.path(output_dir, "r_frontier_panel_bc92_loglik.csv"), row.names = FALSE)

cat("   Log-likelihood:", loglik_bc92$loglik, "\n")
cat("   gamma:", gamma_bc92, "\n")
cat("   Mean efficiency:", mean(eff_bc92), "\n\n")

# ============================================================================
# Save session info
# ============================================================================
sink(file.path(output_dir, "r_session_info.txt"))
cat("R Session Info - frontier Benchmarks\n")
cat("=====================================\n\n")
sessionInfo()
sink()

cat("========================================\n")
cat("CONCLUÍDO!\n")
cat("Resultados salvos em:", normalizePath(output_dir), "\n")
cat("========================================\n")
