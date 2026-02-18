# ============================================
# Benchmark: Panel Count Models
# ============================================

library(plm)
library(MASS)
library(mfx)
library(jsonlite)

# Load test data
data <- read.csv("../data/count_panel_test.csv")
data$entity <- as.factor(data$entity)

# ---- Pooled Poisson ----
cat("\n=== Pooled Poisson ===\n")
pooled_poisson <- glm(y ~ x1 + x2, data = data, family = poisson(link = "log"))
summary(pooled_poisson)

pooled_poisson_results <- list(
  coef = coef(pooled_poisson),
  se = sqrt(diag(vcov(pooled_poisson))),
  loglik = logLik(pooled_poisson)[1]
)

# Cluster-robust SE
library(sandwich)
vcov_cluster <- vcovHC(pooled_poisson, type = "HC1", cluster = data$entity)
pooled_poisson_results$cluster_se <- sqrt(diag(vcov_cluster))

# Marginal effects (AME)
# For Poisson: ME = beta * exp(X'beta)
X <- model.matrix(pooled_poisson)
xb <- X %*% coef(pooled_poisson)
lambda_hat <- exp(xb)

ame_x1 <- mean(coef(pooled_poisson)["x1"] * lambda_hat)
ame_x2 <- mean(coef(pooled_poisson)["x2"] * lambda_hat)

pooled_poisson_results$ame <- list(
  x1 = ame_x1,
  x2 = ame_x2
)

write_json(pooled_poisson_results, "../r/results/pooled_poisson_results.json", pretty = TRUE)

# ---- Fixed Effects Poisson ----
cat("\n=== FE Poisson ===\n")
pdata <- pdata.frame(data, index = c("entity", "time"))
fe_poisson <- pglm(y ~ x1 + x2, data = pdata, family = poisson, model = "within")
summary(fe_poisson)

fe_poisson_results <- list(
  coef = coef(fe_poisson),
  se = sqrt(diag(vcov(fe_poisson))),
  loglik = logLik(fe_poisson)[1]
)

write_json(fe_poisson_results, "../r/results/fe_poisson_results.json", pretty = TRUE)

# ---- Negative Binomial ----
cat("\n=== Negative Binomial ===\n")
negbin <- glm.nb(y ~ x1 + x2, data = data)
summary(negbin)

negbin_results <- list(
  coef = coef(negbin),
  se = sqrt(diag(vcov(negbin))),
  theta = negbin$theta,
  loglik = logLik(negbin)[1]
)

write_json(negbin_results, "../r/results/negbin_results.json", pretty = TRUE)

cat("\nResults saved to r/results/\n")
