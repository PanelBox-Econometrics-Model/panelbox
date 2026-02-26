#!/usr/bin/env Rscript

# Validação dos eigenvalues/estabilidade do Panel VAR contra R
# Este script calcula os eigenvalues da companion matrix
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
write.csv(df, "/tmp/pvar_stability_data.csv", row.names = FALSE)

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
A1_est <- matrix(c(
  coef_eq1["y1_lag1"], coef_eq1["y2_lag1"], coef_eq1["y3_lag1"],
  coef_eq2["y1_lag1"], coef_eq2["y2_lag1"], coef_eq2["y3_lag1"],
  coef_eq3["y1_lag1"], coef_eq3["y2_lag1"], coef_eq3["y3_lag1"]
), nrow = 3, byrow = TRUE)

A2_est <- matrix(c(
  coef_eq1["y1_lag2"], coef_eq1["y2_lag2"], coef_eq1["y3_lag2"],
  coef_eq2["y1_lag2"], coef_eq2["y2_lag2"], coef_eq2["y3_lag2"],
  coef_eq3["y1_lag2"], coef_eq3["y2_lag2"], coef_eq3["y3_lag2"]
), nrow = 3, byrow = TRUE)

# Construir companion matrix
# Para VAR(2) com K=3 variáveis:
# Companion matrix é (K*p) × (K*p) = 6×6
#
# [ A1  A2 ]
# [ I   0  ]
#
# onde I é identidade K×K e 0 é matriz zero K×K

p <- 2
K <- 3

# Inicializar companion matrix (K*p × K*p)
companion <- matrix(0, nrow = K * p, ncol = K * p)

# Primeira linha de blocos: [A1, A2]
companion[1:K, 1:K] <- A1_est
companion[1:K, (K+1):(2*K)] <- A2_est

# Segunda linha de blocos: [I, 0]
companion[(K+1):(2*K), 1:K] <- diag(K)

cat("\nCompanion Matrix:\n")
print(companion)

# Calcular eigenvalues
eigenvalues <- eigen(companion)$values

cat("\nEigenvalues:\n")
print(eigenvalues)

# Calcular módulos dos eigenvalues
moduli <- abs(eigenvalues)
max_modulus <- max(moduli)

cat("\nEigenvalue moduli:\n")
print(moduli)
cat("\nMax modulus:", max_modulus, "\n")

# Sistema é estável se max_modulus < 1
is_stable <- max_modulus < 1.0
cat("System is stable:", is_stable, "\n")

# Salvar resultados
results <- list(
  A1 = A1_est,
  A2 = A2_est,
  companion_matrix = companion,
  eigenvalues = list(
    real = Re(eigenvalues),
    imag = Im(eigenvalues)
  ),
  eigenvalue_moduli = moduli,
  max_modulus = max_modulus,
  is_stable = is_stable,
  stability_margin = 1.0 - max_modulus
)

write_json(results, "/tmp/pvar_stability_results.json", auto_unbox = TRUE, digits = 12)

cat("\nStability results saved to /tmp/pvar_stability_results.json\n")
