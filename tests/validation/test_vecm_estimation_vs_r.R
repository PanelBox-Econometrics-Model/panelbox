#!/usr/bin/env Rscript

# Validação do Panel VECM Estimation contra pacote R 'urca'
# Este script gera dados cointegrados, estima VECM no R e salva os resultados
# para comparação com a implementação Python
#
# NOTA: urca::ca.jo é para séries temporais (não painéis)
# Validação: comparar metodologia para uma entidade individual

# Verificar e instalar pacotes necessários
packages <- c("urca", "vars", "jsonlite")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Set seed para reprodutibilidade
set.seed(123)

# Gerar dados de painel com cointegração
N <- 20  # entidades
T_i <- 80  # períodos (precisa ser maior para VECM estimation)
K <- 3   # variáveis

# Criar estrutura de dados
entity_ids <- rep(1:N, each = T_i)
time_ids <- rep(1:T_i, N)

# Gerar dados com r=1 relação de cointegração conhecida
# y1 - 1.5*y2 + 0.8*y3 ~ I(0)
all_data <- list()

for (i in 1:N) {
  # Gerar common stochastic trend
  trend <- cumsum(rnorm(T_i, sd = 0.3))

  # Cointegrating relation: y1 - 1.5*y2 + 0.8*y3 = I(0)
  # Construir y2, y3 seguindo o trend
  y2 <- trend + cumsum(rnorm(T_i, sd = 0.1))
  y3 <- 0.5 * trend + cumsum(rnorm(T_i, sd = 0.1))

  # y1 cointegrado com y2 e y3
  stationary_error <- arima.sim(n = T_i, list(ar = c(0.3)), sd = 0.2)
  y1 <- 1.5 * y2 - 0.8 * y3 + as.numeric(stationary_error)

  entity_data <- data.frame(
    entity = i,
    time = 1:T_i,
    y1 = y1,
    y2 = y2,
    y3 = y3
  )

  all_data[[i]] <- entity_data
}

# Combinar todos os dados
df <- do.call(rbind, all_data)

# Salvar dados para Python
write.csv(df, "/tmp/vecm_estimation_test_data.csv", row.names = FALSE)

# ============================================================================
# Estimar VECM usando ca.jo para primeira entidade
# ============================================================================

entity1_data <- all_data[[1]]
y_matrix <- as.matrix(entity1_data[, c("y1", "y2", "y3")])

# Johansen cointegration test
jo_test <- ca.jo(
  y_matrix,
  type = "trace",
  ecdet = "const",
  K = 2,  # lags in VAR representation (VECM(1) = VAR(2))
  spec = "transitory"
)

# Converter VECM para VAR usando vec2var
# Isso nos dá os coeficientes em forma de VAR
vecm_to_var <- vec2var(jo_test, r = 1)  # assume rank = 1

# Coeficientes do VAR em níveis (conversão do VECM)
A_matrices <- vecm_to_var$A  # List of coefficient matrices

# Obter beta (cointegrating vector) do Johansen test
# jo_test@V contém os eigenvectors (cointegrating vectors)
beta_raw <- jo_test@V[, 1, drop = FALSE]  # First cointegrating vector

# Normalizar beta (primeira variável = 1)
beta_normalized <- beta_raw / beta_raw[1]

# Obter Pi matrix (long-run matrix)
# Pi = alpha %*% t(beta)
# Do Johansen test podemos calcular alpha
# alpha = (dY %*% beta) / (Y_{-1} %*% beta)^2

# Eigenvalues
eigenvalues <- jo_test@lambda

# Residual covariance matrix
Sigma <- cov(residuals(vecm_to_var))

# Extract coefficient matrices
A1 <- A_matrices$A1
if (length(A_matrices) >= 2 && !is.null(A_matrices$A2)) {
  A2 <- A_matrices$A2
} else {
  A2 <- matrix(0, nrow = K, ncol = K)
}

# Deterministic component (constant)
deterministic <- if (!is.null(vecm_to_var$deterministic)) {
  vecm_to_var$deterministic
} else {
  NULL
}

# ============================================================================
# Preparar resultados para JSON
# ============================================================================

results <- list(
  # Data dimensions
  N = N,
  T = T_i,
  K = K,

  # VECM parameters
  rank = 1,  # rank usado
  lags = 1,  # lags in VECM (p-1 where p is VAR order)

  # Johansen test results
  eigenvalues = as.numeric(eigenvalues),

  # VECM parameters (entity 1 only)
  entity1 = list(
    # Cointegrating vector (normalized, first var = 1)
    beta = as.numeric(beta_normalized),

    # VAR representation (converted from VECM using vec2var)
    # VECM(p-1) = VAR(p)
    # A matrices are from VAR in levels
    A1 = A1,
    A2 = A2,

    # Residual covariance
    Sigma = Sigma,

    # Deterministic component
    deterministic = deterministic
  ),

  # Metadata
  r_packages = list(
    urca = as.character(packageVersion("urca")),
    vars = as.character(packageVersion("vars"))
  )
)

# Salvar resultados em JSON
write(
  jsonlite::toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = 10),
  "/tmp/vecm_estimation_r_results.json"
)

# Imprimir sumário
cat("\n===== VECM ESTIMATION (Entity 1) =====\n")
cat("\nRank:", results$rank, "\n")
cat("VECM lags:", results$lags, "\n")
cat("VAR order:", results$lags + 1, "\n")
cat("\nCointegrating vector (beta, normalized):\n")
print(beta_normalized)
cat("\nA1 (VAR coefficient matrix, lag 1):\n")
print(A1)
cat("\nA2 (VAR coefficient matrix, lag 2):\n")
print(A2)
cat("\nResidual covariance matrix:\n")
print(Sigma)

cat("\nResults saved to /tmp/vecm_estimation_r_results.json\n")
