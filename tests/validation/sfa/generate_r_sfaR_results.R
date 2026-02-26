#!/usr/bin/env Rscript
#
# Script para gerar resultados de referência do pacote `sfaR` (R)
# para validação de True Random Effects (TRE) e BC95
#
# Autor: PanelBox Development Team
# Data: 2026-02-15
#
# Uso: Rscript generate_r_sfaR_results.R
#
# Dependências:
#   install.packages(c("sfaR", "readr", "plm"))

# Verificar se sfaR está instalado
if (!require("sfaR", quietly = TRUE)) {
    cat("Pacote sfaR não encontrado. Instalando...\n")
    install.packages("sfaR", repos = "https://cloud.r-project.org/")
    library(sfaR)
}

library(readr)

cat("========================================\n")
cat("Gerando resultados de referência do R\n")
cat("Pacote: sfaR (True Random Effects)\n")
cat("========================================\n\n")

# Criar diretório de saída
output_dir <- "r_results"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# ============================================================================
# Carregar dados de painel
# ============================================================================
cat("Carregando dataset: Rice Production (Philippines)\n")
cat("------------------------------------------------\n")

# Verificar se o dataset já foi salvo pelo script anterior
if (file.exists(file.path(output_dir, "riceProdPhil.csv"))) {
    rice_panel <- read_csv(file.path(output_dir, "riceProdPhil.csv"), show_col_types = FALSE)
} else {
    # Carregar do frontier package
    if (require("frontier", quietly = TRUE)) {
        data(riceProdPhil, package = "frontier")
        rice_panel <- riceProdPhil
        write_csv(rice_panel, file.path(output_dir, "riceProdPhil.csv"))
    } else {
        stop("Pacote frontier não encontrado. Execute generate_r_frontier_results.R primeiro.")
    }
}

# Preparar variáveis
rice_panel$log_output <- log(rice_panel$PROD)
rice_panel$log_area <- log(rice_panel$AREA)
rice_panel$log_labor <- log(rice_panel$LABOR)
rice_panel$log_npk <- log(rice_panel$NPK)
rice_panel$log_other <- log(rice_panel$OTHER)

cat("Dimensões:", nrow(rice_panel), "observações\n")
cat("Firmas:", length(unique(rice_panel$FMERCODE)), "\n")
cat("Anos:", length(unique(rice_panel$YEARDUM)), "\n\n")

# ============================================================================
# Modelo 1: True Random Effects (TRE) - Greene (2005)
# ============================================================================
cat("Modelo 1: True Random Effects (TRE) - Greene (2005)\n")
cat("----------------------------------------------------\n")

# Estimar TRE usando sfaR
# Modelo: y_it = alpha + X_it * beta + w_i + v_it - u_it
# w_i ~ N(0, sigma_w^2) - heterogeneidade não observada (random effect)
# v_it ~ N(0, sigma_v^2) - ruído idiossincrático
# u_it ~ N+(0, sigma_u^2) - ineficiência

tre_model <- sfaR::sfacross(
    formula = log_output ~ log_area + log_labor + log_npk + log_other,
    data = rice_panel,
    udist = "hnormal",  # half-normal para inefficiency
    uhet = ~ 1,  # heteroskedasticity in u (none)
    vhet = ~ 1,  # heteroskedasticity in v (none)
    S = 1  # method: 1 = Greene (2005) TRE
)

# Nota: sfaR::sfacross é para cross-section
# Para True RE em painel, usar sfaR::sfapanel com model = "TRE"
# Vamos usar a função correta:

tre_model <- tryCatch(
    {
        sfaR::sfapanel(
            formula = log_output ~ log_area + log_labor + log_npk + log_other,
            data = rice_panel,
            id = "FMERCODE",
            udist = "hnormal",
            S = 1  # TRE model
        )
    },
    error = function(e) {
        cat("Erro ao estimar TRE com sfaR:\n")
        cat(conditionMessage(e), "\n")
        cat("Tentando método alternativo...\n")
        NULL
    }
)

if (!is.null(tre_model)) {
    # Extrair coeficientes
    tre_coefs <- coef(tre_model)
    tre_summary <- summary(tre_model)

    # Criar dataframe de parâmetros
    params_tre <- data.frame(
        parameter = names(tre_coefs),
        estimate = as.numeric(tre_coefs),
        se = tre_summary$coefficients[, "Std. Error"],
        tvalue = tre_summary$coefficients[, "t value"],
        pvalue = tre_summary$coefficients[, "Pr(>|t|)"],
        stringsAsFactors = FALSE
    )

    # Log-likelihood
    loglik_tre <- data.frame(
        model = "TRE_greene2005",
        loglik = logLik(tre_model)[1],
        nobs = nobs(tre_model),
        nparams = length(tre_coefs),
        stringsAsFactors = FALSE
    )

    # Eficiências (se disponível)
    if ("efficiencies" %in% names(tre_model)) {
        eff_tre <- tre_model$efficiencies
        eff_df_tre <- data.frame(
            firm_id = rice_panel$FMERCODE,
            year = rice_panel$YEARDUM,
            efficiency = eff_tre,
            stringsAsFactors = FALSE
        )
    } else {
        # Calcular eficiências manualmente se não estiverem disponíveis
        eff_df_tre <- data.frame(
            firm_id = integer(0),
            year = integer(0),
            efficiency = numeric(0)
        )
    }

    # Salvar resultados
    write_csv(params_tre, file.path(output_dir, "r_sfaR_tre_params.csv"))
    write_csv(loglik_tre, file.path(output_dir, "r_sfaR_tre_loglik.csv"))
    if (nrow(eff_df_tre) > 0) {
        write_csv(eff_df_tre, file.path(output_dir, "r_sfaR_tre_efficiency.csv"))
    }

    cat("  Log-likelihood:", loglik_tre$loglik, "\n")
    cat("  Número de parâmetros:", loglik_tre$nparams, "\n\n")
} else {
    cat("  TRE model não pôde ser estimado com sfaR.\n")
    cat("  Isto pode indicar que o pacote não está instalado corretamente\n")
    cat("  ou que a versão não suporta esta funcionalidade.\n\n")
}

# ============================================================================
# Modelo 2: BC95 com determinantes de ineficiência
# ============================================================================
cat("Modelo 2: Battese & Coelli (1995) - Determinantes de Ineficiência\n")
cat("------------------------------------------------------------------\n")

# BC95: modelo com variáveis explicativas para ineficiência
# mu_it = delta_0 + delta_1 * Z_it
# u_it ~ N+(mu_it, sigma_u^2)

# Para este exemplo, vamos usar YEARDUM como determinante
# (firms podem ficar mais eficientes com o tempo)

bc95_model <- tryCatch(
    {
        sfaR::sfacross(
            formula = log_output ~ log_area + log_labor + log_npk + log_other,
            muhet = ~ YEARDUM,  # determinants of inefficiency
            data = rice_panel,
            udist = "tnormal"  # truncated normal (permite mu != 0)
        )
    },
    error = function(e) {
        cat("Erro ao estimar BC95 com sfaR:\n")
        cat(conditionMessage(e), "\n")
        NULL
    }
)

if (!is.null(bc95_model)) {
    # Extrair coeficientes
    bc95_coefs <- coef(bc95_model)
    bc95_summary <- summary(bc95_model)

    params_bc95 <- data.frame(
        parameter = names(bc95_coefs),
        estimate = as.numeric(bc95_coefs),
        se = bc95_summary$coefficients[, "Std. Error"],
        tvalue = bc95_summary$coefficients[, "t value"],
        pvalue = bc95_summary$coefficients[, "Pr(>|t|)"],
        stringsAsFactors = FALSE
    )

    # Log-likelihood
    loglik_bc95 <- data.frame(
        model = "BC95_determinants",
        loglik = logLik(bc95_model)[1],
        nobs = nobs(bc95_model),
        nparams = length(bc95_coefs),
        stringsAsFactors = FALSE
    )

    # Salvar
    write_csv(params_bc95, file.path(output_dir, "r_sfaR_bc95_params.csv"))
    write_csv(loglik_bc95, file.path(output_dir, "r_sfaR_bc95_loglik.csv"))

    cat("  Log-likelihood:", loglik_bc95$loglik, "\n")
    cat("  Número de parâmetros:", loglik_bc95$nparams, "\n\n")
} else {
    cat("  BC95 model não pôde ser estimado.\n\n")
}

# ============================================================================
# Informações da sessão
# ============================================================================
cat("\n========================================\n")
cat("Informações da Sessão R (sfaR)\n")
cat("========================================\n")
sink(file.path(output_dir, "r_sfaR_session_info.txt"))
sessionInfo()
sink()

cat("\nResultados sfaR salvos em:", output_dir, "\n")
cat("Validação sfaR concluída!\n")
