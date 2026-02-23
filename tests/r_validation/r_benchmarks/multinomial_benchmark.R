# Multinomial Logit Benchmark usando mlogit
# Instalar: install.packages("mlogit")

library(mlogit)

# Configurar seed
set.seed(42)

# Gerar dados de escolha
n <- 500
x1 <- rnorm(n)
x2 <- rnorm(n)

# Utilidades (3 alternativas)
# Alternative 1 is base (utility = 0)
u2 <- 0.5 + 1.0*x1 + 0.3*x2 + rlogis(n)
u3 <- 0.8 + 0.5*x1 + 0.8*x2 + rlogis(n)
u1 <- rlogis(n)

# Escolhas
utilities <- cbind(u1, u2, u3)
choice <- apply(utilities, 1, which.max)

# Criar dados em formato wide
df <- data.frame(
    id = 1:n,
    choice = choice,
    x1 = x1,
    x2 = x2
)

# Converter para formato mlogit
df_long <- dfidx(df, choice="choice", shape="wide", varying=NULL, idx=list("id"))

# Estimar modelo
cat("Estimating Multinomial Logit...\n")
result <- mlogit(choice ~ 1 | x1 + x2, data=df_long)

# Print results
cat("\n=== Multinomial Logit Results ===\n")
print(summary(result))

# Save results
coefs <- coef(result)
results_df <- data.frame(
    param = names(coefs),
    coef = coefs,
    se = sqrt(diag(vcov(result)))
)
write.csv(results_df, "multinomial_results.csv", row.names=FALSE)

# Save data
write.csv(df, "multinomial_data.csv", row.names=FALSE)

cat("\nResults saved to multinomial_results.csv\n")
cat("Data saved to multinomial_data.csv\n")
