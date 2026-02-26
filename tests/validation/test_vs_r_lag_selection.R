#!/usr/bin/env Rscript

# Validação da seleção de lags do Panel VAR contra pacote R 'plm'
# Este script gera dados, estima com diferentes lags no R e salva os resultados
# para comparação com a implementação Python

# Verificar e instalar pacotes necessários
packages <- c("plm", "jsonlite")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Set seed para reprodutibilidade
set.seed(42)

# Gerar dados de painel balanceado
N <- 10  # entidades
T_i <- 30  # períodos (mais longo para testar múltiplos lags)
K <- 3   # variáveis

# Criar estrutura de dados
entity_ids <- rep(1:N, each = T_i)
time_ids <- rep(1:T_i, N)

# Gerar dados VAR(2) com coeficientes conhecidos
A1 <- matrix(c(
  0.5, 0.1, 0.05,
  0.2, 0.4, 0.1,
  0.1, 0.15, 0.6
), nrow = 3, byrow = TRUE)

A2 <- matrix(c(
  -0.1, 0, 0,
  0, 0.05, 0,
  0, 0, -0.2
), nrow = 3, byrow = TRUE)

# Gerar séries
y <- matrix(0, nrow = N * T_i, ncol = K)

# Valores iniciais
for (i in 1:N) {
  start_idx <- (i - 1) * T_i + 1
  y[start_idx, ] <- rnorm(K, mean = 0, sd = 0.5)
  y[start_idx + 1, ] <- rnorm(K, mean = 0, sd = 0.5)
}

# Gerar dados
for (i in 1:N) {
  for (t in 3:T_i) {
    idx <- (i - 1) * T_i + t
    idx_lag1 <- idx - 1
    idx_lag2 <- idx - 2

    y[idx, ] <- A1 %*% y[idx_lag1, ] + A2 %*% y[idx_lag2, ] + rnorm(K, sd = 0.1)
  }
}

# Criar dataframe
df <- data.frame(
  entity = entity_ids,
  time = time_ids,
  y1 = y[, 1],
  y2 = y[, 2],
  y3 = y[, 3]
)

# Salvar dados para Python
write.csv(df, "/tmp/pvar_lag_selection_data.csv", row.names = FALSE)

# Criar painel
pdata <- pdata.frame(df, index = c("entity", "time"))

# Função para calcular critérios de informação para um dado lag order
calculate_ic <- function(pdata, p, endog_vars = c("y1", "y2", "y3")) {
  K <- length(endog_vars)

  # Criar lags
  for (var in endog_vars) {
    for (lag in 1:p) {
      lag_name <- paste0(var, "_lag", lag)
      pdata[[lag_name]] <- lag(pdata[[var]], lag)
    }
  }

  # Remover NAs
  pdata <- na.omit(pdata)

  # Criar fórmulas para cada equação
  lag_vars <- paste0(rep(endog_vars, each = p), "_lag", 1:p)
  formula_rhs <- paste(lag_vars, collapse = " + ")

  # Estimar cada equação
  residuals_list <- list()
  n_obs <- 0

  for (i in 1:K) {
    var <- endog_vars[i]
    formula_str <- paste(var, "~", formula_rhs, "- 1")

    eq <- plm(as.formula(formula_str),
              data = pdata, effect = "individual", model = "within")

    residuals_list[[i]] <- residuals(eq)
    if (i == 1) {
      n_obs <- length(residuals(eq))
    }
  }

  # Calcular matriz de covariância dos resíduos
  resid_matrix <- do.call(cbind, residuals_list)
  Sigma_hat <- cov(resid_matrix)

  # Calcular critérios de informação
  n_params <- K * K * p  # K^2 * p parâmetros no sistema

  log_det_sigma <- log(det(Sigma_hat))
  aic <- log_det_sigma + (2 * n_params) / n_obs
  bic <- log_det_sigma + (n_params * log(n_obs)) / n_obs
  hqic <- log_det_sigma + (2 * n_params * log(log(n_obs))) / n_obs
  mbic <- log_det_sigma + (n_params * log(n_obs) * log(log(n_obs))) / n_obs

  return(list(
    lag = p,
    n_obs = n_obs,
    aic = aic,
    bic = bic,
    hqic = hqic,
    mbic = mbic,
    log_det_sigma = log_det_sigma
  ))
}

# Testar múltiplas ordens de lags
max_lags <- 6
ic_results <- list()

for (p in 1:max_lags) {
  cat("Calculating IC for lag", p, "...\n")
  ic_results[[p]] <- calculate_ic(pdata, p)
}

# Determinar lags selecionados por cada critério
aic_values <- sapply(ic_results, function(x) x$aic)
bic_values <- sapply(ic_results, function(x) x$bic)
hqic_values <- sapply(ic_results, function(x) x$hqic)
mbic_values <- sapply(ic_results, function(x) x$mbic)

selected_lags <- list(
  AIC = which.min(aic_values),
  BIC = which.min(bic_values),
  HQIC = which.min(hqic_values),
  MBIC = which.min(mbic_values)
)

# Preparar resultados
results <- list(
  ic_table = data.frame(
    lag = 1:max_lags,
    aic = aic_values,
    bic = bic_values,
    hqic = hqic_values,
    mbic = mbic_values
  ),
  selected = selected_lags,
  true_lag = 2  # Sabemos que geramos VAR(2)
)

# Salvar resultados
write_json(results, "/tmp/pvar_lag_selection_results.json", auto_unbox = TRUE, digits = 12)

cat("\nLag Selection Results saved to /tmp/pvar_lag_selection_results.json\n")
cat("\nSelected lags by criterion:\n")
cat("  AIC:", selected_lags$AIC, "\n")
cat("  BIC:", selected_lags$BIC, "\n")
cat("  HQIC:", selected_lags$HQIC, "\n")
cat("  MBIC:", selected_lags$MBIC, "\n")
cat("\nTrue lag order: 2\n")

cat("\nInformation Criteria Table:\n")
print(results$ic_table)
