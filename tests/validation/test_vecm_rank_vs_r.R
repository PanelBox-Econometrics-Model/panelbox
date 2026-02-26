#!/usr/bin/env Rscript

# Validação do Panel VECM Rank Selection contra pacote R 'urca'
# Este script gera dados cointegrados, estima rank no R e salva os resultados
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
set.seed(42)

# Gerar dados de painel com cointegração
N <- 20  # entidades
T_i <- 50  # períodos (precisa ser maior para Johansen test)
K <- 3   # variáveis

# Criar estrutura de dados
entity_ids <- rep(1:N, each = T_i)
time_ids <- rep(1:T_i, N)

# Gerar dados com r=1 relação de cointegração conhecida
# y1, y2, y3 são I(1), mas y1 - 2*y2 + 0.5*y3 ~ I(0)
all_data <- list()

for (i in 1:N) {
  # Gerar random walk comum (common stochastic trend)
  trend <- cumsum(rnorm(T_i, sd = 0.5))

  # Cointegrating relation: y1 - 2*y2 + 0.5*y3 = I(0)
  # Construir y1, y2, y3 que compartilham trend

  # y2 e y3 seguem o trend com noise
  y2 <- trend + cumsum(rnorm(T_i, sd = 0.1))
  y3 <- trend + cumsum(rnorm(T_i, sd = 0.1))

  # y1 = 2*y2 - 0.5*y3 + I(0) stationary error
  stationary_error <- arima.sim(n = T_i, list(ar = c(0.5)), sd = 0.3)
  y1 <- 2 * y2 - 0.5 * y3 + as.numeric(stationary_error)

  # Adicionar dinâmica de curto prazo (VECM com p=2)
  # Δy_t = α*β'*y_{t-1} + Γ*Δy_{t-1} + ε_t
  # onde β' = [1, -2, 0.5] (já construído acima)
  # α = velocidade de ajuste

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
write.csv(df, "/tmp/vecm_rank_test_data.csv", row.names = FALSE)

# ============================================================================
# Testar rank de cointegração usando ca.jo (Johansen) para primeira entidade
# ============================================================================

entity1_data <- all_data[[1]]
y_matrix <- as.matrix(entity1_data[, c("y1", "y2", "y3")])

# Johansen cointegration test
# type = "trace" (trace test)
# ecdet = "none" (sem constante na relação de cointegração)
# K = 2 (lags, VECM(1) = VAR(2) em níveis)

jo_test <- ca.jo(
  y_matrix,
  type = "trace",
  ecdet = "const",  # constante na relação de cointegração
  K = 2,            # lags in VAR representation
  spec = "transitory"  # constante apenas em EC term, não em VAR
)

# Extrair resultados
summary_jo <- summary(jo_test)

# Eigenvalues
eigenvalues <- jo_test@lambda

# Test statistics e critical values
teststat_trace <- jo_test@teststat  # trace statistics for r=0, r<=1, etc
cval_trace <- jo_test@cval  # critical values (90%, 95%, 99%)

# Max eigenvalue test
jo_maxeig <- ca.jo(
  y_matrix,
  type = "eigen",  # max eigenvalue test
  ecdet = "const",
  K = 2,
  spec = "transitory"
)

teststat_maxeig <- jo_maxeig@teststat
cval_maxeig <- jo_maxeig@cval

# ============================================================================
# Preparar resultados para JSON
# ============================================================================

results <- list(
  # Data dimensions
  N = N,
  T = T_i,
  K = K,

  # Johansen test results (entity 1 only, for methodology comparison)
  entity1 = list(
    eigenvalues = as.numeric(eigenvalues),

    trace_test = list(
      statistics = as.numeric(teststat_trace),
      critical_values = list(
        pct_90 = as.numeric(cval_trace[, 1]),
        pct_95 = as.numeric(cval_trace[, 2]),
        pct_99 = as.numeric(cval_trace[, 3])
      ),
      # Determine rank: highest r where we DON'T reject H0: rank <= r
      # (test stat < critical value at 95%)
      selected_rank = sum(teststat_trace > cval_trace[, 2])
    ),

    maxeig_test = list(
      statistics = as.numeric(teststat_maxeig),
      critical_values = list(
        pct_90 = as.numeric(cval_maxeig[, 1]),
        pct_95 = as.numeric(cval_maxeig[, 2]),
        pct_99 = as.numeric(cval_maxeig[, 3])
      ),
      selected_rank = sum(teststat_maxeig > cval_maxeig[, 2])
    )
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
  "/tmp/vecm_rank_r_results.json"
)

# Imprimir sumário
cat("\n===== JOHANSEN COINTEGRATION TEST (Entity 1) =====\n")
cat("\nTrace Test:\n")
cat("Test statistics:", teststat_trace, "\n")
cat("Critical values (95%):", cval_trace[, 2], "\n")
cat("Selected rank (trace):", results$entity1$trace_test$selected_rank, "\n")

cat("\nMax Eigenvalue Test:\n")
cat("Test statistics:", teststat_maxeig, "\n")
cat("Critical values (95%):", cval_maxeig[, 2], "\n")
cat("Selected rank (max-eig):", results$entity1$maxeig_test$selected_rank, "\n")

cat("\nEigenvalues:", eigenvalues, "\n")

cat("\nResults saved to /tmp/vecm_rank_r_results.json\n")
