#!/usr/bin/env Rscript

# Validação do Panel VAR contra pacote R 'plm'
# Este script gera dados, estima no R e salva os resultados
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
T_i <- 20  # períodos
K <- 3   # variáveis

# Criar estrutura de dados
entity_ids <- rep(1:N, each = T_i)
time_ids <- rep(1:T_i, N)

# Gerar dados VAR(2) com coeficientes conhecidos
# y1_t = 0.5*y1_{t-1} + 0.2*y2_{t-1} + 0.1*y3_{t-1} - 0.1*y1_{t-2} + e1_t
# y2_t = 0.1*y1_{t-1} + 0.4*y2_{t-1} + 0.15*y3_{t-1} + 0.05*y2_{t-2} + e2_t
# y3_t = 0.05*y1_{t-1} + 0.1*y2_{t-1} + 0.6*y3_{t-1} - 0.2*y3_{t-2} + e3_t

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
write.csv(df, "/tmp/pvar_test_data.csv", row.names = FALSE)

# Criar painel
pdata <- pdata.frame(df, index = c("entity", "time"))

# Criar lags manualmente para cada variável
pdata$y1_lag1 <- lag(pdata$y1, 1)
pdata$y1_lag2 <- lag(pdata$y1, 2)
pdata$y2_lag1 <- lag(pdata$y2, 1)
pdata$y2_lag2 <- lag(pdata$y2, 2)
pdata$y3_lag1 <- lag(pdata$y3, 1)
pdata$y3_lag2 <- lag(pdata$y3, 2)

# Remover NAs
pdata <- na.omit(pdata)

# Estimar equação por equação com efeitos fixos (within)
eq1 <- plm(y1 ~ y1_lag1 + y1_lag2 + y2_lag1 + y2_lag2 + y3_lag1 + y3_lag2 - 1,
           data = pdata, effect = "individual", model = "within")

eq2 <- plm(y2 ~ y1_lag1 + y1_lag2 + y2_lag1 + y2_lag2 + y3_lag1 + y3_lag2 - 1,
           data = pdata, effect = "individual", model = "within")

eq3 <- plm(y3 ~ y1_lag1 + y1_lag2 + y2_lag1 + y2_lag2 + y3_lag1 + y3_lag2 - 1,
           data = pdata, effect = "individual", model = "within")

# Extrair coeficientes
coef_eq1 <- coef(eq1)
coef_eq2 <- coef(eq2)
coef_eq3 <- coef(eq3)

# Organizar em matrizes A1 e A2
# Python formato: A[k,:] são coeficientes na equação k
# Cada linha k da matriz são os coeficientes de y1_lag, y2_lag, y3_lag na equação k
A1_est <- matrix(c(
  coef_eq1["y1_lag1"], coef_eq1["y2_lag1"], coef_eq1["y3_lag1"],  # Eq 1
  coef_eq2["y1_lag1"], coef_eq2["y2_lag1"], coef_eq2["y3_lag1"],  # Eq 2
  coef_eq3["y1_lag1"], coef_eq3["y2_lag1"], coef_eq3["y3_lag1"]   # Eq 3
), nrow = 3, byrow = TRUE)

A2_est <- matrix(c(
  coef_eq1["y1_lag2"], coef_eq1["y2_lag2"], coef_eq1["y3_lag2"],  # Eq 1
  coef_eq2["y1_lag2"], coef_eq2["y2_lag2"], coef_eq2["y3_lag2"],  # Eq 2
  coef_eq3["y1_lag2"], coef_eq3["y2_lag2"], coef_eq3["y3_lag2"]   # Eq 3
), nrow = 3, byrow = TRUE)

# Extrair erros padrão
se_eq1 <- sqrt(diag(vcov(eq1)))
se_eq2 <- sqrt(diag(vcov(eq2)))
se_eq3 <- sqrt(diag(vcov(eq3)))

# Calcular matriz de covariância dos resíduos
resid_matrix <- cbind(residuals(eq1), residuals(eq2), residuals(eq3))
Sigma_hat <- cov(resid_matrix)

# Calcular critérios de informação
n_obs <- nrow(pdata)
n_params <- 6 * 3  # 6 coeficientes por equação, 3 equações

log_det_sigma <- log(det(Sigma_hat))
aic <- log_det_sigma + (2 * n_params) / n_obs
bic <- log_det_sigma + (n_params * log(n_obs)) / n_obs
hqic <- log_det_sigma + (2 * n_params * log(log(n_obs))) / n_obs

# Salvar resultados
results <- list(
  A1 = A1_est,
  A2 = A2_est,
  Sigma = Sigma_hat,
  aic = aic,
  bic = bic,
  hqic = hqic,
  n_obs = n_obs,
  coefficients = list(
    eq1 = as.list(coef_eq1),
    eq2 = as.list(coef_eq2),
    eq3 = as.list(coef_eq3)
  ),
  std_errors = list(
    eq1 = as.list(se_eq1),
    eq2 = as.list(se_eq2),
    eq3 = as.list(se_eq3)
  )
)

write_json(results, "/tmp/pvar_r_results.json", auto_unbox = TRUE, digits = 12)

cat("R validation results saved to /tmp/pvar_r_results.json\n")
cat("\nEstimated A1 matrix:\n")
print(A1_est)
cat("\nEstimated A2 matrix:\n")
print(A2_est)
cat("\nAIC:", aic, "BIC:", bic, "HQIC:", hqic, "\n")
