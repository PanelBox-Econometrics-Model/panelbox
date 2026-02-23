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
# Two-step - extract from summary
sum_2step <- summary(result_2step)

# Get coefficients from the summary tables
sel_coef_2step <- sum_2step$estimate[1:3, 1]  # First 3 rows for selection
out_coef_2step <- sum_2step$estimate[4:5, 1]  # Next 2 rows for outcome

results_2step <- data.frame(
    equation = c("selection", "selection", "selection", "outcome", "outcome"),
    param = c("(Intercept)", "z1", "z2", "(Intercept)", "x1"),
    coef = c(sel_coef_2step, out_coef_2step)
)
write.csv(results_2step, "heckman_2step_results.csv", row.names=FALSE)

# MLE
sum_mle <- summary(result_mle)

sel_coef_mle <- sum_mle$estimate[1:3, 1]
out_coef_mle <- sum_mle$estimate[4:5, 1]

results_mle <- data.frame(
    equation = c("selection", "selection", "selection", "outcome", "outcome"),
    param = c("(Intercept)", "z1", "z2", "(Intercept)", "x1"),
    coef = c(sel_coef_mle, out_coef_mle)
)
write.csv(results_mle, "heckman_mle_results.csv", row.names=FALSE)

# Save data
write.csv(df, "heckman_data.csv", row.names=FALSE)

cat("\nResults saved to heckman_*_results.csv\n")
cat("Data saved to heckman_data.csv\n")
