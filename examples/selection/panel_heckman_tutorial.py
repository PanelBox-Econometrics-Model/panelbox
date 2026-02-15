"""
Panel Heckman Selection Model Tutorial
======================================

This tutorial demonstrates the Panel Heckman two-step and MLE estimators
for correcting sample selection bias in panel data.

**Application: Wage Determination with Labor Force Participation**

We analyze how wages are determined, accounting for the fact that wages
are only observed for individuals who participate in the labor force.
Those who don't participate (selected=0) have unobserved wages.

If participation and wage determination are correlated (ρ ≠ 0), then
simply running OLS on the selected sample will yield biased estimates.

References
----------
.. [1] Heckman, J.J. (1979). "Sample Selection Bias as a Specification Error."
       Econometrica, 47(1), 153-161.
.. [2] Wooldridge, J.M. (1995). "Selection Corrections for Panel Data Models Under
       Conditional Mean Independence Assumptions." Journal of Econometrics, 68(1), 115-132.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# PART 1: Generate Realistic Panel Data with Selection
# ============================================================================


def generate_wage_participation_data(n_individuals=500, n_years=5, rho=0.4, seed=42):
    """
    Generate realistic wage and labor force participation data.

    Model:
    ------
    Selection (Participation):
        participate_it = 1[age_it * γ₁ + educ_i * γ₂ + kids_it * γ₃ +
                           other_income_it * γ₄ + η_i + v_it > 0]

    Outcome (Wage):
        log(wage_it) = β₀ + β₁ * experience_it + β₂ * educ_i +
                       β₃ * tenure_it + α_i + ε_it

    where:
        (η_i, α_i) ~ N(0, Σ_u)  [individual random effects]
        (v_it, ε_it) ~ N(0, Σ_v) with Corr(v_it, ε_it) = ρ

    Parameters
    ----------
    n_individuals : int
        Number of individuals
    n_years : int
        Number of time periods
    rho : float
        Correlation between participation and wage shocks
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with variables:
        - id: individual identifier
        - year: time period
        - participate: binary (1 if working, 0 otherwise)
        - log_wage: observed only if participate=1
        - age, educ, kids, other_income, experience, tenure
    """
    np.random.seed(seed)
    n = n_individuals * n_years

    # Individual and time indices
    individual_id = np.repeat(np.arange(n_individuals), n_years)
    year = np.tile(np.arange(n_years), n_individuals)

    # Time-invariant characteristics
    educ = np.repeat(
        np.random.choice([10, 12, 14, 16, 18], n_individuals, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
        n_years,
    )

    # Time-varying characteristics
    age_base = np.repeat(np.random.randint(22, 55, n_individuals), n_years)
    age = age_base + year  # Age increases with time

    kids = np.random.poisson(1.2, n)  # Number of children
    other_income = np.random.gamma(3, 2, n)  # Non-labor income (thousand $)

    experience = np.maximum(age - educ - 6, 0) + np.random.randint(0, 3, n)
    tenure = np.tile(np.arange(n_years), n_individuals) + np.random.poisson(1, n)

    # True parameters
    # Selection equation: participate ~ age + educ + kids + other_income
    gamma_true = np.array(
        [-2.5, 0.05, 0.15, -0.3, -0.1]
    )  # [intercept, age, educ, kids, other_income]

    # Outcome equation: log(wage) ~ experience + educ + tenure
    beta_true = np.array([1.5, 0.03, 0.08, 0.02])  # [intercept, experience, educ, tenure]

    # Generate correlated errors
    # (v, ε) with correlation rho
    mean = [0, 0]
    cov = [[1, rho], [rho, 0.25]]  # Variance of wage shock is 0.25
    errors = np.random.multivariate_normal(mean, cov, n)
    v = errors[:, 0]  # Participation shock
    epsilon = errors[:, 1]  # Wage shock

    # Participation (selection)
    Z = np.column_stack([np.ones(n), age, educ, kids, other_income])
    participate_latent = Z @ gamma_true + v
    participate = (participate_latent > 0).astype(int)

    # Wage (outcome)
    X = np.column_stack([np.ones(n), experience, educ, tenure])
    log_wage_latent = X @ beta_true + epsilon

    # Observed wage (only if participate)
    log_wage = np.where(participate == 1, log_wage_latent, np.nan)
    wage = np.where(participate == 1, np.exp(log_wage_latent), np.nan)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "id": individual_id,
            "year": year,
            "participate": participate,
            "log_wage": log_wage,
            "wage": wage,
            "age": age,
            "educ": educ,
            "kids": kids,
            "other_income": other_income,
            "experience": experience,
            "tenure": tenure,
            "log_wage_latent": log_wage_latent,  # For validation
        }
    )

    print(f"\n{'='*70}")
    print("Data Generation Summary")
    print(f"{'='*70}")
    print(f"Number of individuals: {n_individuals}")
    print(f"Number of years: {n_years}")
    print(f"Total observations: {n}")
    print(f"Participation rate: {participate.mean():.1%}")
    print(f"True ρ (selection correlation): {rho:.2f}")
    print(f"\nTrue Parameters:")
    print(f"  γ (selection): {gamma_true}")
    print(f"  β (wage): {beta_true}")

    return data, {"gamma": gamma_true, "beta": beta_true, "rho": rho}


# ============================================================================
# PART 2: Estimate Panel Heckman Models
# ============================================================================


def estimate_heckman_models(data):
    """Estimate Heckman two-step and MLE."""
    from panelbox.models.selection import PanelHeckman

    print(f"\n{'='*70}")
    print("Estimating Panel Heckman Models")
    print(f"{'='*70}")

    # Prepare data
    y = data["log_wage"].values
    X_outcome = data[["experience", "educ", "tenure"]].values
    X_outcome = np.column_stack([np.ones(len(data)), X_outcome])

    selection = data["participate"].values
    Z_selection = data[["age", "educ", "kids", "other_income"]].values
    Z_selection = np.column_stack([np.ones(len(data)), Z_selection])

    entity = data["id"].values
    time = data["year"].values

    # Two-Step Estimation
    print("\n" + "-" * 70)
    print("Two-Step Estimation")
    print("-" * 70)

    model_2step = PanelHeckman(
        endog=y,
        exog=X_outcome,
        selection=selection,
        exog_selection=Z_selection,
        entity=entity,
        time=time,
        method="two_step",
    )

    result_2step = model_2step.fit()
    print(result_2step.summary())

    # MLE Estimation
    print("\n" + "-" * 70)
    print("MLE Estimation")
    print("-" * 70)

    model_mle = PanelHeckman(
        endog=y,
        exog=X_outcome,
        selection=selection,
        exog_selection=Z_selection,
        entity=entity,
        time=time,
        method="mle",
    )

    result_mle = model_mle.fit()
    print(result_mle.summary())

    return result_2step, result_mle


# ============================================================================
# PART 3: Diagnostic Analysis
# ============================================================================


def run_diagnostics(result, true_params=None):
    """Run diagnostic tests and comparisons."""
    print(f"\n{'='*70}")
    print("Diagnostic Analysis")
    print(f"{'='*70}")

    # Test 1: Selection Effect Test
    print("\n" + "-" * 70)
    print("1. Selection Effect Test (H0: ρ = 0)")
    print("-" * 70)
    test = result.selection_effect()
    print(test["interpretation"])
    print(f"  Test statistic: {test['statistic']:.3f}")
    print(f"  P-value: {test['pvalue']:.4f}")

    # Test 2: IMR Diagnostics
    print("\n" + "-" * 70)
    print("2. Inverse Mills Ratio Diagnostics")
    print("-" * 70)
    diag = result.imr_diagnostics()
    print(f"  Mean IMR (selected): {diag['imr_mean']:.3f}")
    print(f"  Std Dev IMR: {diag['imr_std']:.3f}")
    print(f"  Range: [{diag['imr_min']:.3f}, {diag['imr_max']:.3f}]")
    print(f"  High IMR observations (>2): {diag['high_imr_count']}")
    print(f"  Selection rate: {diag['selection_rate']:.1%}")

    # Test 3: OLS vs Heckman Comparison
    print("\n" + "-" * 70)
    print("3. OLS vs Heckman Comparison")
    print("-" * 70)
    comparison = result.compare_ols_heckman()
    print("\nCoefficient Estimates:")
    print(f"  {'Variable':<15} {'OLS':>10} {'Heckman':>10} {'Difference':>10} {'% Diff':>10}")
    print("  " + "-" * 60)
    var_names = ["Intercept", "Experience", "Education", "Tenure"]
    for i, var in enumerate(var_names):
        ols_coef = comparison["beta_ols"][i]
        heck_coef = comparison["beta_heckman"][i]
        diff = comparison["difference"][i]
        pct = comparison["pct_difference"][i]
        print(f"  {var:<15} {ols_coef:>10.4f} {heck_coef:>10.4f} {diff:>10.4f} {pct:>9.1f}%")

    print(f"\n{comparison['interpretation']}")

    # Test 4: Parameter Recovery (if true params provided)
    if true_params is not None:
        print("\n" + "-" * 70)
        print("4. Parameter Recovery Check")
        print("-" * 70)
        print(f"\nOutcome Equation (β):")
        print(f"  True:      {true_params['beta']}")
        print(f"  Estimated: {result.outcome_params}")
        print(f"  Bias:      {result.outcome_params - true_params['beta']}")

        print(f"\nSelection Equation (γ):")
        print(f"  True:      {true_params['gamma']}")
        print(f"  Estimated: {result.probit_params}")
        print(f"  Bias:      {result.probit_params - true_params['gamma']}")

        print(f"\nSelection Correlation (ρ):")
        print(f"  True:      {true_params['rho']:.3f}")
        print(f"  Estimated: {result.rho:.3f}")
        print(f"  Bias:      {result.rho - true_params['rho']:.3f}")


def create_diagnostic_plots(result):
    """Create diagnostic visualizations."""
    print(f"\n{'='*70}")
    print("Creating Diagnostic Plots...")
    print(f"{'='*70}")

    # IMR plot
    fig_imr = result.plot_imr(figsize=(12, 5))
    output_path = "/home/guhaase/projetos/panelbox/examples/selection/heckman_imr_diagnostics.png"
    fig_imr.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved IMR diagnostic plot to: {output_path}")
    plt.close(fig_imr)


# ============================================================================
# MAIN TUTORIAL
# ============================================================================


def main():
    """Run complete tutorial."""
    print("\n" + "=" * 70)
    print("PANEL HECKMAN SELECTION MODEL TUTORIAL")
    print("Application: Wage Determination with Labor Force Participation")
    print("=" * 70)

    # Step 1: Generate data
    data, true_params = generate_wage_participation_data(
        n_individuals=500, n_years=5, rho=0.4, seed=42
    )

    # Step 2: Estimate models
    result_2step, result_mle = estimate_heckman_models(data)

    # Step 3: Run diagnostics on two-step result
    run_diagnostics(result_2step, true_params)

    # Step 4: Create plots
    create_diagnostic_plots(result_2step)

    # Step 5: Compare Two-Step vs MLE
    print(f"\n{'='*70}")
    print("Two-Step vs MLE Comparison")
    print(f"{'='*70}")
    print(f"\nEstimated ρ:")
    print(f"  Two-Step: {result_2step.rho:.3f}")
    print(f"  MLE:      {result_mle.rho:.3f}")
    print(f"  True:     {true_params['rho']:.3f}")

    print(f"\nEstimated σ:")
    print(f"  Two-Step: {result_2step.sigma:.3f}")
    print(f"  MLE:      {result_mle.sigma:.3f}")

    if result_mle.llf is not None:
        print(f"\nLog-likelihood (MLE): {result_mle.llf:.2f}")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(
        """
This tutorial demonstrated:
1. ✓ Data generation with selection bias (ρ = 0.4)
2. ✓ Panel Heckman two-step estimation
3. ✓ Panel Heckman MLE estimation
4. ✓ Diagnostic tests (selection effect, IMR, OLS comparison)
5. ✓ Visualization of selection bias
6. ✓ Parameter recovery validation

KEY TAKEAWAYS:
- OLS on selected sample is biased when ρ ≠ 0
- Heckman correction removes selection bias
- IMR measures strength of selection effect
- Two-step is simpler, MLE is more efficient
- Diagnostics help assess model validity
    """
    )

    return data, result_2step, result_mle


if __name__ == "__main__":
    data, result_2step, result_mle = main()
