#!/usr/bin/env Rscript
#
# R Validation Script for PanelBox
#
# Runs equivalent panel data tests in R using plm, lmtest packages
# and compares results with PanelBox implementation.
#

library(plm)
library(lmtest)
library(sandwich)

# Suppress warnings for cleaner output
options(warn = -1)

cat("================================================================================\n")
cat("R VALIDATION - RUNNING EQUIVALENT PANEL DATA TESTS\n")
cat("================================================================================\n\n")

output_dir <- "output"

# ============================================================================
# Helper Functions
# ============================================================================

run_all_tests <- function(data, model_type = "within", dataset_name = "unknown") {
  # Run all available panel data tests in R.
  #
  # Parameters:
  #   data: pdata.frame object
  #   model_type: "within" (FE) or "random" (RE)
  #   dataset_name: name for reporting
  #
  # Returns:
  #   List of test results

  cat(sprintf("\nRunning R tests on: %s (model: %s)\n", dataset_name, model_type))
  cat("----------------------------------------\n")

  # Fit model
  if (model_type == "within") {
    formula_obj <- y ~ x1 + x2
    model <- plm(formula_obj, data = data, model = "within")
  } else if (model_type == "random") {
    formula_obj <- y ~ x1 + x2
    model <- plm(formula_obj, data = data, model = "random")
  } else {
    stop(paste("Unknown model type:", model_type))
  }

  cat(sprintf("  Model estimated: %d obs, %d coefficients\n",
              nobs(model), length(coef(model))))

  results <- list(
    model_type = model_type,
    dataset = dataset_name,
    coefficients = list(
      x1 = as.numeric(coef(model)["x1"]),
      x2 = as.numeric(coef(model)["x2"])
    ),
    tests = list()
  )

  # ========================================================================
  # 1. Wooldridge Test for Serial Correlation (FE only)
  # ========================================================================
  if (model_type == "within") {
    cat("  [1/7] Wooldridge AR test...")
    tryCatch({
      # pwfdtest implements Wooldridge first-difference test for AR(1)
      # This is the correct test for serial correlation (not pwtest!)
      # pwtest = test for unobserved effects (different test)
      # pwfdtest = first-difference test for AR(1) (matches PanelBox implementation)
      # NOTE: Must pass formula+data (not model) for pwfdtest to work
      wooldridge <- pwfdtest(formula_obj, data = data, h0 = "fe")
      results$tests$wooldridge <- list(
        statistic = as.numeric(wooldridge$statistic),
        pvalue = as.numeric(wooldridge$p.value),
        df = if(length(wooldridge$parameter) > 0) as.numeric(wooldridge$parameter) else c(),
        method = wooldridge$method
      )
      cat(sprintf(" F=%.3f, p=%.4f\n",
                  results$tests$wooldridge$statistic,
                  results$tests$wooldridge$pvalue))
    }, error = function(e) {
      results$tests$wooldridge <<- list(error = as.character(e$message))
      cat(" ERROR\n")
    })
  } else {
    cat("  [1/7] Wooldridge AR test... SKIPPED (RE model)\n")
  }

  # ========================================================================
  # 2. Breusch-Godfrey Test for Serial Correlation
  # ========================================================================
  cat("  [2/7] Breusch-Godfrey test...")
  tryCatch({
    # Use pbgtest from plm package (panel Breusch-Godfrey)
    bg_test <- pbgtest(model, order = 1)
    results$tests$breusch_godfrey <- list(
      statistic = as.numeric(bg_test$statistic),
      pvalue = as.numeric(bg_test$p.value),
      df = as.numeric(bg_test$parameter),
      method = bg_test$method
    )
    cat(sprintf(" chisq=%.3f, p=%.4f\n",
                results$tests$breusch_godfrey$statistic,
                results$tests$breusch_godfrey$pvalue))
  }, error = function(e) {
    results$tests$breusch_godfrey <<- list(error = as.character(e$message))
    cat(" ERROR\n")
  })

  # ========================================================================
  # 3. Modified Wald Test for Groupwise Heteroskedasticity (FE only)
  # ========================================================================
  if (model_type == "within") {
    cat("  [3/7] Modified Wald test...")
    tryCatch({
      # Compute residuals by group and test for equal variances
      resid_df <- data.frame(
        entity = index(data)$entity,
        resid = residuals(model)
      )

      # Group residuals
      resid_by_entity <- split(resid_df$resid, resid_df$entity)

      # Compute variance for each entity
      variances <- sapply(resid_by_entity, var, na.rm = TRUE)
      n_entities <- length(variances)

      # Modified Wald statistic: sum of (N_i * (log(s_i^2) - log(s^2))^2)
      # Simpler version: chi-square test
      pooled_var <- var(residuals(model), na.rm = TRUE)

      # Compute test statistic
      wald_stat <- sum((variances / pooled_var - 1)^2)
      # This is approximate - exact formula varies

      # For now, use a simpler approach with Levene or Bartlett test
      # Bartlett test for equal variances
      bartlett_result <- bartlett.test(resid ~ entity, data = resid_df)

      results$tests$modified_wald <- list(
        statistic = as.numeric(bartlett_result$statistic),
        pvalue = as.numeric(bartlett_result$p.value),
        df = as.numeric(bartlett_result$parameter),
        method = "Bartlett test (approximate Modified Wald)",
        note = "R approximation - exact Modified Wald not in standard packages"
      )
      cat(sprintf(" chisq=%.3f, p=%.4f (approx)\n",
                  results$tests$modified_wald$statistic,
                  results$tests$modified_wald$pvalue))
    }, error = function(e) {
      results$tests$modified_wald <<- list(error = as.character(e$message))
      cat(" ERROR\n")
    })
  } else {
    cat("  [3/7] Modified Wald test... SKIPPED (RE model)\n")
  }

  # ========================================================================
  # 4. Breusch-Pagan Test for Heteroskedasticity
  # ========================================================================
  cat("  [4/7] Breusch-Pagan test...")
  tryCatch({
    # Use bptest from lmtest
    bp_test <- bptest(model)
    results$tests$breusch_pagan <- list(
      statistic = as.numeric(bp_test$statistic),
      pvalue = as.numeric(bp_test$p.value),
      df = as.numeric(bp_test$parameter),
      method = bp_test$method
    )
    cat(sprintf(" BP=%.3f, p=%.4f\n",
                results$tests$breusch_pagan$statistic,
                results$tests$breusch_pagan$pvalue))
  }, error = function(e) {
    results$tests$breusch_pagan <<- list(error = as.character(e$message))
    cat(" ERROR\n")
  })

  # ========================================================================
  # 5. White Test for Heteroskedasticity
  # ========================================================================
  cat("  [5/7] White test...")
  tryCatch({
    # White test with squared terms (no cross terms to match PanelBox default)
    # Get fitted values
    fitted_vals <- fitted(model)

    # Create squared regressors
    # Note: This is approximate - exact White test implementation varies
    white_test <- bptest(model, ~ fitted_vals + I(fitted_vals^2))

    results$tests$white <- list(
      statistic = as.numeric(white_test$statistic),
      pvalue = as.numeric(white_test$p.value),
      df = as.numeric(white_test$parameter),
      method = "White test (approximate)",
      note = "Using fitted values squared as auxiliary regressors"
    )
    cat(sprintf(" BP=%.3f, p=%.4f\n",
                results$tests$white$statistic,
                results$tests$white$pvalue))
  }, error = function(e) {
    results$tests$white <<- list(error = as.character(e$message))
    cat(" ERROR\n")
  })

  # ========================================================================
  # 6. Pesaran CD Test for Cross-Sectional Dependence
  # ========================================================================
  cat("  [6/7] Pesaran CD test...")
  tryCatch({
    # Use pcdtest from plm
    cd_test <- pcdtest(model, test = "cd")
    results$tests$pesaran_cd <- list(
      statistic = as.numeric(cd_test$statistic),
      pvalue = as.numeric(cd_test$p.value),
      method = cd_test$method
    )
    cat(sprintf(" z=%.3f, p=%.4f\n",
                results$tests$pesaran_cd$statistic,
                results$tests$pesaran_cd$pvalue))
  }, error = function(e) {
    results$tests$pesaran_cd <<- list(error = as.character(e$message))
    cat(" ERROR\n")
  })

  # ========================================================================
  # 7. Mundlak Test (RE only)
  # ========================================================================
  if (model_type == "random") {
    cat("  [7/7] Mundlak test...")
    tryCatch({
      # Mundlak test: RE model augmented with group means
      # Extract original data
      orig_data <- data

      # Compute group means for x1 and x2
      orig_data$x1_mean <- ave(orig_data$x1, index(orig_data)$entity, FUN = mean)
      orig_data$x2_mean <- ave(orig_data$x2, index(orig_data)$entity, FUN = mean)

      # Fit augmented RE model
      mundlak_model <- plm(y ~ x1 + x2 + x1_mean + x2_mean,
                          data = orig_data, model = "random")

      # Wald test on group mean coefficients
      # H0: coefficients on x1_mean and x2_mean are zero
      library(car)
      if (requireNamespace("car", quietly = TRUE)) {
        wald_result <- linearHypothesis(mundlak_model,
                                       c("x1_mean = 0", "x2_mean = 0"))
        results$tests$mundlak <- list(
          statistic = as.numeric(wald_result$Chisq[2]),
          pvalue = as.numeric(wald_result$`Pr(>Chisq)`[2]),
          df = as.numeric(wald_result$Df[2]),
          method = "Mundlak test (Wald test on group means)"
        )
        cat(sprintf(" chisq=%.3f, p=%.4f\n",
                    results$tests$mundlak$statistic,
                    results$tests$mundlak$pvalue))
      } else {
        # Simple F-test alternative
        results$tests$mundlak <- list(
          note = "car package not available for Wald test",
          error = "Manual implementation needed"
        )
        cat(" SKIPPED (car package needed)\n")
      }
    }, error = function(e) {
      results$tests$mundlak <<- list(error = as.character(e$message))
      cat(" ERROR\n")
    })
  } else {
    cat("  [7/7] Mundlak test... SKIPPED (FE model)\n")
  }

  return(results)
}


# ============================================================================
# Main Execution
# ============================================================================

# Test Case 1: AR(1) data with Fixed Effects
cat("\n1. Loading AR(1) data...\n")
data_ar1_raw <- read.csv(file.path(output_dir, "data_ar1.csv"))
data_ar1 <- pdata.frame(data_ar1_raw, index = c("entity", "time"))
cat(sprintf("   Loaded: %d obs, %d entities, %d periods\n",
            nrow(data_ar1_raw),
            length(unique(data_ar1_raw$entity)),
            length(unique(data_ar1_raw$time))))

results_ar1_fe <- run_all_tests(data_ar1, "within", "AR(1) data")

# Test Case 2: Heteroskedastic data with Fixed Effects
cat("\n2. Loading heteroskedastic data...\n")
data_het_raw <- read.csv(file.path(output_dir, "data_het.csv"))
data_het <- pdata.frame(data_het_raw, index = c("entity", "time"))
cat(sprintf("   Loaded: %d obs, %d entities, %d periods\n",
            nrow(data_het_raw),
            length(unique(data_het_raw$entity)),
            length(unique(data_het_raw$time))))

results_het_fe <- run_all_tests(data_het, "within", "Heteroskedastic data")

# Test Case 3: Clean data with Fixed Effects
cat("\n3. Loading clean data...\n")
data_clean_raw <- read.csv(file.path(output_dir, "data_clean.csv"))
data_clean <- pdata.frame(data_clean_raw, index = c("entity", "time"))
cat(sprintf("   Loaded: %d obs, %d entities, %d periods\n",
            nrow(data_clean_raw),
            length(unique(data_clean_raw$entity)),
            length(unique(data_clean_raw$time))))

results_clean_fe <- run_all_tests(data_clean, "within", "Clean data (FE)")

# Test Case 4: Clean data with Random Effects
cat("\n4. Running Random Effects on clean data...\n")
results_clean_re <- run_all_tests(data_clean, "random", "Clean data (RE)")

# ============================================================================
# Save Results
# ============================================================================

cat("\n5. Saving R test results...\n")

library(jsonlite)

write_json(results_ar1_fe,
           file.path(output_dir, "r_results_ar1_fe.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("   Saved: r_results_ar1_fe.json\n")

write_json(results_het_fe,
           file.path(output_dir, "r_results_het_fe.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("   Saved: r_results_het_fe.json\n")

write_json(results_clean_fe,
           file.path(output_dir, "r_results_clean_fe.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("   Saved: r_results_clean_fe.json\n")

write_json(results_clean_re,
           file.path(output_dir, "r_results_clean_re.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("   Saved: r_results_clean_re.json\n")

cat("\n================================================================================\n")
cat("R TESTS COMPLETED\n")
cat("================================================================================\n")
cat("\nNext: Run comparison script\n")
cat("  python compare_results.py\n\n")
