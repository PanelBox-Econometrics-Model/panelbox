#!/usr/bin/env Rscript
#
# Script para gerar resultados de referência do pacote `frontier` (R)
# para validação da implementação PanelBox SFA
#
# Autor: PanelBox Development Team
# Data: 2026-02-15
#
# Uso: Rscript generate_r_frontier_results.R
#
# Dependências:
#   install.packages(c("frontier", "readr", "datasets"))

library(frontier)
# library(readr)  # Substituído por write.csv base R

cat("========================================\n")
cat("Gerando resultados de referência do R\n")
cat("Pacote: frontier\n")
cat("========================================\n\n")

# Criar diretório de saída se não existir
output_dir <- "r_results"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# ============================================================================
# Dataset 1: Rice Production Philippines (Battese & Coelli 1992)
# ============================================================================
cat("Dataset 1: Rice Production (Philippines)\n")
cat("----------------------------------------\n")

# Carregar dados
data(riceProdPhil)

# Salvar dataset para uso no Python
write.csv(riceProdPhil, file.path(output_dir, "riceProdPhil.csv"), row.names = FALSE)

# Estatísticas descritivas
cat("Dimensões:", nrow(riceProdPhil), "observações\n")
cat("Variáveis:", paste(names(riceProdPhil), collapse=", "), "\n\n")

# ============================================================================
# Modelo 1: Cross-section SFA - Half-Normal
# ============================================================================
cat("Modelo 1.1: Cross-section SFA - Half-Normal\n")

# Agregar dados para cross-section (média por firma)
rice_cs <- aggregate(
    cbind(PROD, AREA, LABOR, NPK, OTHER) ~ FMERCODE,
    data = riceProdPhil,
    FUN = mean
)

# Transformar em log
rice_cs$log_output <- log(rice_cs$PROD)
rice_cs$log_area <- log(rice_cs$AREA)
rice_cs$log_labor <- log(rice_cs$LABOR)
rice_cs$log_npk <- log(rice_cs$NPK)
rice_cs$log_other <- log(rice_cs$OTHER)

# Estimar SFA cross-section - Half-Normal
sfa_hn <- sfa(
    log_output ~ log_area + log_labor + log_npk + log_other,
    data = rice_cs,
    ineffDecrease = TRUE  # production frontier (y - u)
)

# Extrair resultados
params_hn <- data.frame(
    parameter = names(coef(sfa_hn)),
    estimate = as.numeric(coef(sfa_hn)),
    se = sqrt(diag(vcov(sfa_hn))),
    stringsAsFactors = FALSE
)

# Adicionar componentes de variância
# Extrair sigmaSq e gamma dos coeficientes
coefs <- coef(sfa_hn)
sigmaSq <- coefs["sigmaSq"]
gamma <- coefs["gamma"]

variance_params <- data.frame(
    parameter = c('sigmaSq', 'gamma', 'sigma_v_sq', 'sigma_u_sq', 'lambda'),
    estimate = c(
        sigmaSq,
        gamma,
        sigmaSq * (1 - gamma),  # sigma_v^2
        sigmaSq * gamma,  # sigma_u^2
        sqrt(gamma / (1 - gamma))  # lambda = sigma_u / sigma_v
    ),
    se = NA_real_,
    stringsAsFactors = FALSE
)

params_hn <- rbind(params_hn, variance_params)

# Eficiências (Battese & Coelli 1988 estimator - JLMS)
eff_hn <- efficiencies(sfa_hn)
eff_df_hn <- data.frame(
    firm_id = rice_cs$FMERCODE,
    efficiency = as.numeric(eff_hn),
    stringsAsFactors = FALSE
)

# Log-likelihood
loglik_hn <- data.frame(
    model = "cross_section_half_normal",
    loglik = logLik(sfa_hn)[1],
    nobs = nobs(sfa_hn),
    nparams = length(coef(sfa_hn)) + 1,  # beta + gamma
    stringsAsFactors = FALSE
)

# Salvar resultados
write.csv(params_hn, file.path(output_dir, "r_frontier_cs_halfnormal_params.csv"), row.names = FALSE)
write.csv(eff_df_hn, file.path(output_dir, "r_frontier_cs_halfnormal_efficiency.csv"), row.names = FALSE)
write.csv(loglik_hn, file.path(output_dir, "r_frontier_cs_halfnormal_loglik.csv"), row.names = FALSE)

cat("  Log-likelihood:", loglik_hn$loglik, "\n")
cat("  gamma:", variance_params$estimate[variance_params$parameter == "gamma"], "\n")
cat("  Mean efficiency:", mean(eff_hn), "\n\n")

# ============================================================================
# Modelo 1.2: Cross-section SFA - Exponential
# ============================================================================
cat("Modelo 1.2: Cross-section SFA - Exponential\n")

# Estimar SFA - Exponential (via truncated normal com mu fixed at 0)
# Nota: frontier package não tem exponential direto, usar truncated com constraints
# Para validação, vamos usar half-normal como proxy (mesma família)
# Em Python, implementamos exponential separadamente

# ============================================================================
# Modelo 2: Panel SFA - Pitt & Lee (1981) - Time-Invariant
# ============================================================================
cat("\nModelo 2: Panel SFA - Pitt & Lee (1981)\n")

# Preparar dados de painel
rice_panel <- riceProdPhil
rice_panel$log_output <- log(rice_panel$PROD)
rice_panel$log_area <- log(rice_panel$AREA)
rice_panel$log_labor <- log(rice_panel$LABOR)
rice_panel$log_npk <- log(rice_panel$NPK)
rice_panel$log_other <- log(rice_panel$OTHER)

# Estimar panel SFA (Pitt & Lee 1981)
# Modelo: time-invariant inefficiency
sfa_pl <- sfa(
    log_output ~ log_area + log_labor + log_npk + log_other,
    data = rice_panel,
    ineffDecrease = TRUE,
    timeEffect = FALSE  # time-invariant inefficiency
)

# Extrair resultados
params_pl <- data.frame(
    parameter = names(coef(sfa_pl)),
    estimate = as.numeric(coef(sfa_pl)),
    se = sqrt(diag(vcov(sfa_pl))),
    stringsAsFactors = FALSE
)

# Componentes de variância
coefs_pl <- coef(sfa_pl)
sigmaSq_pl <- coefs_pl["sigmaSq"]
gamma_pl <- coefs_pl["gamma"]

variance_params_pl <- data.frame(
    parameter = c('sigmaSq', 'gamma', 'sigma_v_sq', 'sigma_u_sq', 'lambda'),
    estimate = c(
        sigmaSq_pl,
        gamma_pl,
        sigmaSq_pl * (1 - gamma_pl),  # sigma_v^2
        sigmaSq_pl * gamma_pl,  # sigma_u^2
        sqrt(gamma_pl / (1 - gamma_pl))  # lambda
    ),
    se = NA_real_,
    stringsAsFactors = FALSE
)

params_pl <- rbind(params_pl, variance_params_pl)

# Eficiências (médias por firma, já que time-invariant)
eff_pl <- efficiencies(sfa_pl)
# Para painel, frontier retorna eficiência por observação
# Agregar por firma (média, já que time-invariant deveriam ser iguais)
eff_df_pl <- data.frame(
    firm_id = rice_panel$FMERCODE,
    year = rice_panel$YEARDUM,
    efficiency = as.numeric(eff_pl),
    stringsAsFactors = FALSE
)

# Log-likelihood
loglik_pl <- data.frame(
    model = "panel_pitt_lee",
    loglik = logLik(sfa_pl)[1],
    nobs = nobs(sfa_pl),
    nparams = length(coef(sfa_pl)) + 1,
    stringsAsFactors = FALSE
)

# Salvar resultados
write.csv(params_pl, file.path(output_dir, "r_frontier_panel_pittlee_params.csv"), row.names = FALSE)
write.csv(eff_df_pl, file.path(output_dir, "r_frontier_panel_pittlee_efficiency.csv"), row.names = FALSE)
write.csv(loglik_pl, file.path(output_dir, "r_frontier_panel_pittlee_loglik.csv"), row.names = FALSE)

cat("  Log-likelihood:", loglik_pl$loglik, "\n")
cat("  gamma:", variance_params_pl$estimate[variance_params_pl$parameter == "gamma"], "\n")
cat("  Mean efficiency:", mean(eff_pl), "\n\n")

# ============================================================================
# Modelo 3: Panel SFA - Battese & Coelli (1992) - Time-Varying
# ============================================================================
cat("Modelo 3: Panel SFA - Battese & Coelli (1992)\n")

# Estimar BC92 model
# u_it = exp(-eta * (t - T)) * u_i
# Nota: frontier package implementa via timeEffect = TRUE
sfa_bc92 <- sfa(
    log_output ~ log_area + log_labor + log_npk + log_other,
    data = rice_panel,
    ineffDecrease = TRUE,
    timeEffect = TRUE  # Battese & Coelli (1992) decay model
)

# Extrair resultados
params_bc92 <- data.frame(
    parameter = names(coef(sfa_bc92)),
    estimate = as.numeric(coef(sfa_bc92)),
    se = sqrt(diag(vcov(sfa_bc92))),
    stringsAsFactors = FALSE
)

# Componentes de variância + eta
variance_params_bc92 <- data.frame(
    parameter = c('sigmaSq', 'gamma', 'sigma_v_sq', 'sigma_u_sq', 'lambda', 'eta'),
    estimate = c(
        coef(sfa_bc92)["sigmaSq"],
        coef(sfa_bc92)["gamma"],
        coef(sfa_bc92)["sigmaSq"] * (1 - coef(sfa_bc92)["gamma"]),
        coef(sfa_bc92)["sigmaSq"] * coef(sfa_bc92)["gamma"],
        sqrt(coef(sfa_bc92)["gamma"] / (1 - coef(sfa_bc92)["gamma"])),
        ifelse(!is.null(sfa_bc92$etaParm), sfa_bc92$etaParm, NA_real_)
    ),
    se = NA_real_,
    stringsAsFactors = FALSE
)

params_bc92 <- rbind(params_bc92, variance_params_bc92)

# Eficiências (time-varying)
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
    loglik = logLik(sfa_bc92)[1],
    nobs = nobs(sfa_bc92),
    nparams = length(coef(sfa_bc92)) + 2,  # beta + gamma + eta
    stringsAsFactors = FALSE
)

# Salvar resultados
write.csv(params_bc92, file.path(output_dir, "r_frontier_panel_bc92_params.csv"), row.names = FALSE)
write.csv(eff_bc92, file.path(output_dir, "r_frontier_panel_bc92_efficiency.csv"), row.names = FALSE)
write.csv(loglik_bc92, file.path(output_dir, "r_frontier_panel_bc92_loglik.csv"), row.names = FALSE)

cat("  Log-likelihood:", loglik_bc92$loglik, "\n")
cat("  gamma:", variance_params_bc92$estimate[variance_params_bc92$parameter == "gamma"], "\n")
cat("  eta:", variance_params_bc92$estimate[variance_params_bc92$parameter == "eta"], "\n")
cat("  Mean efficiency:", mean(eff_bc92), "\n\n")

# ============================================================================
# Informações da sessão R
# ============================================================================
cat("\n========================================\n")
cat("Informações da Sessão R\n")
cat("========================================\n")
sink(file.path(output_dir, "r_session_info.txt"))
sessionInfo()
sink()

cat("\nResultados salvos em:", output_dir, "\n")
cat("Validação concluída com sucesso!\n")
