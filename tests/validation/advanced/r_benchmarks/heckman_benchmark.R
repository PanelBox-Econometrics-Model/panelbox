# Heckman Selection Model Benchmark usando sampleSelection
# Instalar: install.packages("sampleSelection")

library(sampleSelection)

# Configurar seed
set.seed(42)

# Gerar dados
n <- 1000

# Variaveis
z1 <- rnorm(n)
z2 <- rnorm(n)
x1 <- rnorm(n)

# Selecao latente
s_star <- 0.5 + 0.8*z1 + 0.4*z2 + rnorm(n)
s <- as.numeric(s_star > 0)

# Outcome (com correlacao entre erros)
rho <- 0.5
u <- rho * (s_star - (0.5 + 0.8*z1 + 0.4*z2)) + sqrt(1 - rho^2) * rnorm(n)
y_star <- 1 + 2*x1 + u
y <- ifelse(s == 1, y_star, NA)

# Criar dataframe
df <- data.frame(s=s, y=y, z1=z1, z2=z2, x1=x1)

# Estimar modelo Heckman (two-step)
cat("Estimating Heckman Two-Step...\n")
result_2step <- selection(s ~ z1 + z2, y ~ x1, data=df, method="2step")

# Estimar modelo Heckman (MLE)
cat("Estimating Heckman MLE...\n")
result_mle <- selection(s ~ z1 + z2, y ~ x1, data=df, method="ml")

# Print results
cat("\n=== Heckman Two-Step Results ===\n")
print(summary(result_2step))

cat("\n=== Heckman MLE Results ===\n")
print(summary(result_mle))

# Save results
# Two-step
sel_coef <- coef(result_2step, "selection")
out_coef <- coef(result_2step, "outcome")

results_2step <- data.frame(
    equation = c(rep("selection", length(sel_coef)), rep("outcome", length(out_coef))),
    param = c(names(sel_coef), names(out_coef)),
    coef = c(sel_coef, out_coef)
)
write.csv(results_2step, "heckman_2step_results.csv", row.names=FALSE)

# MLE
sel_coef_mle <- coef(result_mle, "selection")
out_coef_mle <- coef(result_mle, "outcome")

results_mle <- data.frame(
    equation = c(rep("selection", length(sel_coef_mle)), rep("outcome", length(out_coef_mle))),
    param = c(names(sel_coef_mle), names(out_coef_mle)),
    coef = c(sel_coef_mle, out_coef_mle)
)
write.csv(results_mle, "heckman_mle_results.csv", row.names=FALSE)

# Save data
write.csv(df, "heckman_data.csv", row.names=FALSE)

cat("\nResults saved to heckman_*_results.csv\n")
cat("Data saved to heckman_data.csv\n")
