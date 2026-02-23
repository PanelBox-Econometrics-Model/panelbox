# Conditional Logit Benchmark usando mlogit (McFadden's choice model)
# Instalar: install.packages("mlogit")

library(mlogit)

# Configurar seed
set.seed(42)

# Gerar dados de escolha de modo de transporte
n_trips <- 200
modes <- c("car", "bus", "train")
n_modes <- length(modes)

# Criar dados em formato long
data_list <- list()

for (i in 1:n_trips) {
    # Atributos que variam por alternativa
    cost <- c(
        10 + rnorm(1, sd=2),   # car
        5 + rnorm(1, sd=1),    # bus
        8 + rnorm(1, sd=1.5)   # train
    )
    time <- c(
        30 + rnorm(1, sd=5),   # car
        45 + rnorm(1, sd=10),  # bus
        25 + rnorm(1, sd=5)    # train
    )

    # Utilidade verdadeira
    beta_cost <- -0.3
    beta_time <- -0.05
    utilities <- beta_cost * cost + beta_time * time + rlogis(n_modes)

    # Escolha
    chosen_idx <- which.max(utilities)

    for (j in 1:n_modes) {
        data_list[[length(data_list) + 1]] <- data.frame(
            trip_id = i,
            mode = modes[j],
            chosen = ifelse(j == chosen_idx, TRUE, FALSE),
            cost = cost[j],
            time = time[j]
        )
    }
}

df <- do.call(rbind, data_list)

# Converter para formato mlogit
df_idx <- dfidx(df, choice="chosen", idx=list("trip_id", "mode"))

# Estimar conditional logit
cat("Estimating Conditional Logit...\n")
result <- mlogit(chosen ~ cost + time | 0, data=df_idx)

# Print results
cat("\n=== Conditional Logit Results ===\n")
print(summary(result))

# Save results
coefs <- coef(result)
results_df <- data.frame(
    param = names(coefs),
    coef = coefs,
    se = sqrt(diag(vcov(result)))
)
write.csv(results_df, "conditional_logit_results.csv", row.names=FALSE)

# Save data
write.csv(df, "conditional_logit_data.csv", row.names=FALSE)

cat("\nResults saved to conditional_logit_results.csv\n")
cat("Data saved to conditional_logit_data.csv\n")
