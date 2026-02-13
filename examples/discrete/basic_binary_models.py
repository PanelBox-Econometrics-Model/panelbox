"""
Example: Binary Choice Models for Panel Data

This example demonstrates the use of binary choice models (Logit and Probit)
for panel data analysis, including:
- Pooled Logit and Probit models
- Fixed Effects Logit
- Model comparison and diagnostics
- Prediction and classification

Dataset: Women's Labor Force Participation
We simulate data inspired by the classic Mroz (1987) dataset on married women's
labor force participation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configure display
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.precision", 4)
sns.set_style("whitegrid")

# Import PanelBox models
from panelbox.models.discrete import FixedEffectsLogit, PooledLogit, PooledProbit


def generate_labor_force_data(n_women=500, n_years=7, seed=42):
    """
    Generate simulated panel data for women's labor force participation.

    Parameters
    ----------
    n_women : int
        Number of women in the panel
    n_years : int
        Number of years per woman
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Panel data with labor force participation and covariates
    """
    np.random.seed(seed)

    data_list = []

    for i in range(n_women):
        # Time-invariant characteristics
        education = np.random.normal(12, 3)  # Years of education
        education = np.clip(education, 0, 20)

        # Individual unobserved effect (ability, preferences)
        alpha_i = np.random.normal(0, 0.5)

        for t in range(n_years):
            # Time-varying characteristics
            age = 25 + t + np.random.uniform(-2, 2)
            experience = max(0, t + np.random.normal(0, 1))
            n_kids = np.random.poisson(0.3 * t) if t > 0 else 0
            kids_young = 1 if (n_kids > 0 and t < 3) else 0
            husband_income = np.random.lognormal(10, 0.5)  # Log income

            # Linear index for labor force participation
            linear_index = (
                alpha_i
                - 5.0  # Intercept
                + 0.15 * education
                + 0.08 * experience
                - 0.02 * age
                - 0.5 * kids_young
                - 0.2 * n_kids
                + 0.1 * np.log(husband_income)
            )

            # Probability and outcome
            prob = 1 / (1 + np.exp(-linear_index))
            employed = np.random.binomial(1, prob)

            # Wages (observed only if employed)
            if employed:
                wage = np.exp(1.5 + 0.07 * education + 0.03 * experience + np.random.normal(0, 0.3))
            else:
                wage = np.nan

            data_list.append(
                {
                    "woman_id": i,
                    "year": t,
                    "employed": employed,
                    "education": education,
                    "experience": experience,
                    "age": age,
                    "n_kids": n_kids,
                    "kids_young": kids_young,
                    "husband_income": husband_income,
                    "wage": wage,
                }
            )

    df = pd.DataFrame(data_list)

    # Create additional variables
    df["log_husband_income"] = np.log(df["husband_income"])
    df["age_squared"] = df["age"] ** 2
    df["experience_squared"] = df["experience"] ** 2

    return df


def main():
    """Run the complete example."""

    print("=" * 80)
    print("BINARY CHOICE MODELS FOR PANEL DATA")
    print("Example: Women's Labor Force Participation")
    print("=" * 80)

    # =========================================================================
    # 1. Generate and Explore Data
    # =========================================================================
    print("\n1. GENERATING DATA")
    print("-" * 40)

    df = generate_labor_force_data(n_women=500, n_years=7)
    print(f"Panel dimensions: {df['woman_id'].nunique()} women × {df['year'].nunique()} years")
    print(f"Total observations: {len(df)}")
    print(f"\nLabor force participation rate: {df['employed'].mean():.3f}")

    # Summary statistics
    print("\nSummary Statistics:")
    summary_vars = ["employed", "education", "experience", "age", "n_kids", "kids_young"]
    print(df[summary_vars].describe())

    # Panel balance
    panel_balance = df.groupby("woman_id").size().value_counts()
    print(
        f"\nPanel balance: {panel_balance.iloc[0]} women with {panel_balance.index[0]} observations each"
    )

    # =========================================================================
    # 2. Pooled Logit Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. POOLED LOGIT MODEL")
    print("-" * 40)

    # Define model formula
    formula = """employed ~ education + experience + experience_squared +
                 age + age_squared + n_kids + kids_young + log_husband_income"""

    # Fit Pooled Logit
    pooled_logit = PooledLogit(formula, df, "woman_id", "year")
    logit_results = pooled_logit.fit(se_type="cluster", verbose=False)

    print("\nPooled Logit Results:")
    print(logit_results.summary())

    # Model diagnostics
    print("\nModel Diagnostics:")
    print(f"Pseudo-R² (McFadden): {logit_results.pseudo_r2('mcfadden'):.4f}")
    print(f"AIC: {logit_results.aic:.2f}")
    print(f"BIC: {logit_results.bic:.2f}")

    # Classification performance
    metrics = logit_results.classification_metrics()
    print(f"\nClassification Performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.3f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(logit_results.classification_table())

    # =========================================================================
    # 3. Pooled Probit Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. POOLED PROBIT MODEL")
    print("-" * 40)

    pooled_probit = PooledProbit(formula, df, "woman_id", "year")
    probit_results = pooled_probit.fit(se_type="cluster", verbose=False)

    print("\nPooled Probit Results:")
    print(probit_results.summary())

    # Compare Logit vs Probit coefficients
    print("\nLogit vs Probit Comparison:")
    comparison_df = pd.DataFrame(
        {
            "Logit": logit_results.params,
            "Probit": probit_results.params,
            "Ratio": logit_results.params / probit_results.params,
        }
    )
    print(comparison_df)
    print("\nNote: Logit coefficients ≈ 1.6 × Probit coefficients (rough approximation)")

    # =========================================================================
    # 4. Fixed Effects Logit
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. FIXED EFFECTS LOGIT")
    print("-" * 40)

    # For FE Logit, we use only time-varying covariates
    fe_formula = "employed ~ experience + experience_squared + n_kids + kids_young"

    fe_logit = FixedEffectsLogit(fe_formula, df, "woman_id", "year")
    fe_results = fe_logit.fit(verbose=False)

    print("\nFixed Effects Logit Results:")
    print(fe_results.summary())

    print(f"\nEntities used: {fe_logit.n_used_entities} out of {fe_logit.n_entities}")
    print(f"Entities dropped (no variation): {len(fe_logit.dropped_entities)}")

    # =========================================================================
    # 5. Model Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. MODEL COMPARISON")
    print("-" * 40)

    comparison = pd.DataFrame(
        {
            "Model": ["Pooled Logit", "Pooled Probit", "Fixed Effects Logit"],
            "Log-Likelihood": [logit_results.llf, probit_results.llf, fe_results.llf],
            "AIC": [logit_results.aic, probit_results.aic, fe_results.aic],
            "BIC": [logit_results.bic, probit_results.bic, fe_results.bic],
            "Pseudo-R²": [
                logit_results.pseudo_r2("mcfadden"),
                probit_results.pseudo_r2("mcfadden"),
                fe_results.pseudo_r2("mcfadden"),
            ],
        }
    )
    print(comparison)

    # =========================================================================
    # 6. Predictions and Marginal Effects
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. PREDICTIONS")
    print("-" * 40)

    # Create prediction scenarios
    scenarios = pd.DataFrame(
        {
            "education": [8, 12, 16, 20],
            "experience": [2, 5, 5, 10],
            "experience_squared": [4, 25, 25, 100],
            "age": [25, 30, 35, 40],
            "age_squared": [625, 900, 1225, 1600],
            "n_kids": [0, 1, 2, 0],
            "kids_young": [0, 1, 0, 0],
            "log_husband_income": [9.5, 10.0, 10.5, 11.0],
        }
    )

    # Get predictions from Pooled Logit
    scenarios_with_intercept = scenarios.copy()
    scenarios_with_intercept.insert(0, "intercept", 1.0)
    X_scenarios = scenarios_with_intercept[pooled_logit.exog_names].values

    probabilities = logit_results.predict(X_scenarios, type="prob")

    scenarios["Predicted Probability"] = probabilities
    print("\nPredicted Probabilities for Different Scenarios:")
    print(scenarios[["education", "experience", "n_kids", "kids_young", "Predicted Probability"]])

    # =========================================================================
    # 7. Goodness of Fit Tests
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. GOODNESS OF FIT")
    print("-" * 40)

    # Hosmer-Lemeshow test
    hl_test = logit_results.hosmer_lemeshow_test(n_groups=10)
    print(f"\nHosmer-Lemeshow Test:")
    print(f"Chi-square statistic: {hl_test['statistic']:.4f}")
    print(f"Degrees of freedom: {hl_test['df']}")
    print(f"P-value: {hl_test['p_value']:.4f}")

    if hl_test["p_value"] > 0.05:
        print("Conclusion: No evidence of poor fit (p > 0.05)")
    else:
        print("Conclusion: Evidence of poor fit (p < 0.05)")

    # Link test
    link_test = logit_results.link_test()
    print(f"\nLink Test for Specification:")
    print(f"Coefficient on squared term: {link_test['coefficient']:.4f}")
    print(f"P-value: {link_test['p_value']:.4f}")

    if link_test["p_value"] > 0.05:
        print("Conclusion: No evidence of misspecification (p > 0.05)")
    else:
        print("Conclusion: Evidence of misspecification (p < 0.05)")

    # =========================================================================
    # 8. Visualization
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. VISUALIZATION")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Coefficient comparison
    ax = axes[0, 0]
    coef_names = logit_results.model.exog_names[1:]  # Exclude intercept
    x_pos = np.arange(len(coef_names))

    ax.bar(x_pos - 0.2, logit_results.params[1:], 0.4, label="Logit", alpha=0.8)
    ax.bar(x_pos + 0.2, probit_results.params[1:], 0.4, label="Probit", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(coef_names, rotation=45, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title("Logit vs Probit Coefficients")
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # 2. Predicted probability distribution
    ax = axes[0, 1]
    pred_prob = logit_results.predict(type="prob")
    ax.hist(pred_prob, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Predicted Probabilities")
    ax.axvline(x=0.5, color="red", linestyle="--", label="Classification threshold")
    ax.legend()

    # 3. ROC Curve
    ax = axes[1, 0]
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = df["employed"].values
    y_prob = logit_results.predict(type="prob")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    ax.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Effect of kids on employment probability
    ax = axes[1, 1]
    # Create data for prediction
    n_kids_range = range(0, 5)
    base_data = {
        "education": 12,
        "experience": 5,
        "experience_squared": 25,
        "age": 30,
        "age_squared": 900,
        "kids_young": 0,
        "log_husband_income": 10.0,
    }

    probs_no_young = []
    probs_with_young = []

    for n in n_kids_range:
        # Without young kids
        X_temp = pd.DataFrame([base_data])
        X_temp["n_kids"] = n
        X_temp["kids_young"] = 0
        X_temp.insert(0, "intercept", 1.0)
        X_array = X_temp[pooled_logit.exog_names].values
        prob = logit_results.predict(X_array, type="prob")[0]
        probs_no_young.append(prob)

        # With young kids
        X_temp["kids_young"] = 1 if n > 0 else 0
        X_array = X_temp[pooled_logit.exog_names].values
        prob = logit_results.predict(X_array, type="prob")[0]
        probs_with_young.append(prob)

    ax.plot(n_kids_range, probs_no_young, "o-", label="No young kids", linewidth=2)
    ax.plot(n_kids_range, probs_with_young, "s-", label="With young kids", linewidth=2)
    ax.set_xlabel("Number of Kids")
    ax.set_ylabel("Probability of Employment")
    ax.set_title("Effect of Children on Employment Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_kids_range)

    plt.tight_layout()
    plt.savefig("binary_models_example.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved as 'binary_models_example.png'")
    plt.show()

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
