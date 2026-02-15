"""
Quantile Treatment Effects (QTE) Analysis

This example demonstrates:
1. Estimation of heterogeneous treatment effects across distribution
2. Conditional and unconditional QTE
3. Difference-in-differences with QR
4. Policy evaluation with heterogeneous effects
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from panelbox import PanelData
from panelbox.models.quantile import CanayTwoStep, PooledQuantile, QuantileTreatmentEffects
from panelbox.visualization.quantile import QuantileVisualizer


class QuantileTreatmentEffectsAnalysis:
    """
    Comprehensive QTE analysis for policy evaluation.

    Demonstrates:
    1. Unconditional QTE
    2. Conditional QTE (controlling for covariates)
    3. QTE with panel data (DiD-QR)
    4. Distributional policy impacts
    """

    def __init__(self, data_path=None):
        """Load or simulate treatment data."""
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self.data = self._simulate_treatment_data()

        # Create panel structure if applicable
        if "person_id" in self.data.columns and "period" in self.data.columns:
            self.panel = PanelData(self.data, entity="person_id", time="period")
        else:
            self.panel = None

    def _simulate_treatment_data(self, n_persons=2000, n_periods=2):
        """
        Simulate realistic treatment effect data.

        Setting: Job training program
        - Outcome: Earnings
        - Treatment: Training participation
        - Heterogeneous effects across skill distribution
        - Panel structure for DiD
        """
        np.random.seed(42)

        # Person characteristics
        persons = pd.DataFrame(
            {
                "person_id": range(n_persons),
                "education": np.random.normal(12, 3, n_persons).clip(0, 20),
                "experience": np.random.exponential(5, n_persons).clip(0, 40),
                "female": np.random.binomial(1, 0.5, n_persons),
                "ability": np.random.normal(0, 1, n_persons),  # Unobserved
                # Treatment propensity (not random assignment)
                "treatment_propensity": None,
            }
        )

        # Treatment propensity depends on observables and unobservables
        # Lower skilled workers more likely to participate
        persons["treatment_propensity"] = (
            0.3
            - 0.02 * persons["education"]
            + 0.01 * persons["female"]
            + 0.1 * (persons["ability"] < 0)  # Lower ability more likely
        ).clip(0, 1)

        # Panel structure
        panel_data = []

        for period in range(n_periods):
            period_data = persons.copy()
            period_data["period"] = period

            if period == 0:
                # Pre-treatment period
                period_data["treated"] = 0
            else:
                # Treatment period - stochastic treatment assignment
                period_data["treated"] = (
                    np.random.uniform(0, 1, n_persons) < period_data["treatment_propensity"]
                ).astype(int)

            # Generate earnings
            # Base earnings (Mincer equation)
            log_earnings = (
                1.5
                + 0.08 * period_data["education"]
                + 0.04 * period_data["experience"]
                - 0.0005 * period_data["experience"] ** 2
                + -0.15 * period_data["female"]
                + 0.3 * period_data["ability"]
            )

            # Time trend
            log_earnings += 0.02 * period

            # Treatment effect (heterogeneous)
            # Effect varies by position in skill distribution
            # Proxy for skill: education + ability
            skill = period_data["education"] / 20 + (period_data["ability"] + 2) / 4

            # Treatment effect structure:
            # - Larger effects for low-skill workers
            # - Diminishing returns for high-skill workers
            treatment_effect = period_data["treated"] * (
                0.15 * (1 - skill) + 0.05 * skill  # Larger for low skill  # Smaller for high skill
            )

            log_earnings += treatment_effect

            # Add noise (heteroskedastic)
            noise_scale = 0.3 * (1 + 0.5 * period_data["female"])
            log_earnings += np.random.normal(0, noise_scale, n_persons)

            period_data["log_earnings"] = log_earnings
            period_data["earnings"] = np.exp(log_earnings) * 1000  # Scale to dollars

            panel_data.append(period_data)

        return pd.concat(panel_data, ignore_index=True)

    def estimate_unconditional_qte(self, tau_list=None):
        """
        Estimate unconditional QTE.

        Simply compares quantiles of treated vs control groups.
        """
        if tau_list is None:
            tau_list = [0.10, 0.25, 0.50, 0.75, 0.90]

        print("\n" + "=" * 70)
        print("UNCONDITIONAL QUANTILE TREATMENT EFFECTS")
        print("=" * 70)

        # Post-treatment period only
        post_data = self.data[self.data["period"] == 1]

        treated = post_data[post_data["treated"] == 1]["log_earnings"]
        control = post_data[post_data["treated"] == 0]["log_earnings"]

        results = []

        print(f"\n{'Quantile':<10} {'Treated':<12} {'Control':<12} {'QTE':<12} {'% Effect':<12}")
        print("-" * 70)

        for tau in tau_list:
            q_treated = treated.quantile(tau)
            q_control = control.quantile(tau)
            qte = q_treated - q_control
            pct_effect = (np.exp(qte) - 1) * 100

            results.append(
                {
                    "tau": tau,
                    "treated": q_treated,
                    "control": q_control,
                    "qte": qte,
                    "pct_effect": pct_effect,
                }
            )

            print(
                f"{tau:<10.2f} {q_treated:>11.4f} {q_control:>11.4f} "
                f"{qte:>11.4f} {pct_effect:>11.2f}%"
            )

        df_results = pd.DataFrame(results)

        # Visualization
        self._plot_unconditional_qte(df_results)

        return df_results

    def estimate_conditional_qte(self, tau_list=None):
        """
        Estimate conditional QTE controlling for covariates.

        Uses quantile regression to control for confounders.
        """
        if tau_list is None:
            tau_list = [0.10, 0.25, 0.50, 0.75, 0.90]

        print("\n" + "=" * 70)
        print("CONDITIONAL QUANTILE TREATMENT EFFECTS")
        print("=" * 70)

        # Post-treatment period
        post_data = self.data[self.data["period"] == 1]

        # Model with treatment and covariates
        formula = "log_earnings ~ treated + education + experience + I(experience**2) + female"

        model = PooledQuantile(post_data, formula, tau=tau_list)
        result = model.fit(se_type="robust")

        # Extract treatment effects
        treatment_idx = 1  # Index of 'treated' variable

        print(f"\n{'Quantile':<10} {'QTE':<12} {'Std Error':<12} {'t-stat':<12} {'p-value':<12}")
        print("-" * 70)

        qte_results = []

        for tau in tau_list:
            qte = result.results[tau].params[treatment_idx]
            se = result.results[tau].bse[treatment_idx]
            t_stat = qte / se
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

            qte_results.append(
                {
                    "tau": tau,
                    "qte": qte,
                    "se": se,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "pct_effect": (np.exp(qte) - 1) * 100,
                }
            )

            significance = (
                "***"
                if p_value < 0.01
                else ("**" if p_value < 0.05 else ("*" if p_value < 0.10 else ""))
            )

            print(
                f"{tau:<10.2f} {qte:>11.4f} {se:>11.4f} "
                f"{t_stat:>11.3f} {p_value:>11.4f} {significance}"
            )

        df_qte = pd.DataFrame(qte_results)

        # Test for heterogeneity
        print("\n" + "-" * 70)
        print("HETEROGENEITY TEST")
        print("-" * 70)

        qte_low = df_qte[df_qte["tau"] == 0.10]["qte"].values[0]
        qte_high = df_qte[df_qte["tau"] == 0.90]["qte"].values[0]

        print(f"\nQTE at 10th percentile: {qte_low:.4f} ({(np.exp(qte_low)-1)*100:.2f}%)")
        print(f"QTE at 90th percentile: {qte_high:.4f} ({(np.exp(qte_high)-1)*100:.2f}%)")
        print(f"Difference: {qte_low - qte_high:.4f}")

        if abs(qte_low) > abs(qte_high) * 1.5:
            print("\n→ Strong evidence of heterogeneous treatment effects")
            print("→ Treatment is more effective for low earners")

        # Visualization
        self._plot_conditional_qte(df_qte)

        return df_qte, result

    def estimate_did_qte(self, tau_list=None):
        """
        Estimate QTE using Difference-in-Differences.

        Accounts for time-invariant unobservables using panel structure.
        """
        if tau_list is None:
            tau_list = [0.10, 0.25, 0.50, 0.75, 0.90]

        print("\n" + "=" * 70)
        print("DIFFERENCE-IN-DIFFERENCES QUANTILE TREATMENT EFFECTS")
        print("=" * 70)

        # Create interaction term
        self.data["post"] = (self.data["period"] == 1).astype(int)
        self.data["treated_x_post"] = self.data["treated"] * self.data["post"]

        # DiD specification
        formula = "log_earnings ~ treated + post + treated_x_post + education + experience + female"

        # Method 1: Pooled QR (doesn't fully control for fixed effects)
        print("\nPooled QR DiD:")
        print("-" * 70)

        model_pooled = PooledQuantile(self.data, formula, tau=tau_list)
        result_pooled = model_pooled.fit(se_type="cluster", cluster="person_id")

        did_idx = 3  # Index of interaction term

        pooled_did = []

        for tau in tau_list:
            did_effect = result_pooled.results[tau].params[did_idx]
            se = result_pooled.results[tau].bse[did_idx]

            pooled_did.append(
                {
                    "tau": tau,
                    "did_qte": did_effect,
                    "se": se,
                    "pct_effect": (np.exp(did_effect) - 1) * 100,
                }
            )

            print(f"τ={tau:.2f}: DiD-QTE = {did_effect:.4f} ({(np.exp(did_effect)-1)*100:+.2f}%)")

        # Method 2: Canay (2011) two-step with fixed effects
        print("\n\nCanay Two-Step DiD:")
        print("-" * 70)

        if self.panel is not None:
            formula_fe = "log_earnings ~ treated_x_post + post + education + experience + female"

            model_canay = CanayTwoStep(self.panel, formula_fe, tau=tau_list)
            result_canay = model_canay.fit()

            canay_did = []

            for tau in tau_list:
                did_effect = result_canay.results[tau].params[1]  # treated_x_post

                canay_did.append(
                    {
                        "tau": tau,
                        "did_qte": did_effect,
                        "pct_effect": (np.exp(did_effect) - 1) * 100,
                    }
                )

                print(
                    f"τ={tau:.2f}: DiD-QTE = {did_effect:.4f} ({(np.exp(did_effect)-1)*100:+.2f}%)"
                )

            # Comparison
            df_comparison = pd.DataFrame(
                {
                    "tau": tau_list,
                    "pooled": [x["did_qte"] for x in pooled_did],
                    "canay": [x["did_qte"] for x in canay_did],
                }
            )

            self._plot_did_comparison(df_comparison)

            return pooled_did, canay_did
        else:
            return pooled_did, None

    def policy_simulation(self, n_simulations=1000):
        """
        Simulate policy impacts on earnings distribution.
        """
        print("\n" + "=" * 70)
        print("POLICY IMPACT SIMULATION")
        print("=" * 70)

        # Get post-treatment data
        post_data = self.data[self.data["period"] == 1]

        # Actual distribution
        actual_earnings = post_data["log_earnings"]

        # Counterfactual: What if everyone was treated?
        # Use conditional QTE estimates
        _, qr_result = self.estimate_conditional_qte()

        # For simplicity, use median QTE
        median_qte = qr_result.results[0.50].params[1]

        # Simulate counterfactual
        counterfactual_earnings = actual_earnings + median_qte * (1 - post_data["treated"])

        # Compare distributions
        print("\nDistributional Impact:")
        print("-" * 70)

        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

        print(f"{'Quantile':<10} {'Actual':<12} {'Universal':<12} {'Change':<12}")
        print("-" * 70)

        for q in quantiles:
            actual_q = actual_earnings.quantile(q)
            counter_q = counterfactual_earnings.quantile(q)
            change = counter_q - actual_q

            print(
                f"{q:<10.2f} {actual_q:>11.4f} {counter_q:>11.4f} "
                f"{change:>11.4f} ({(np.exp(change)-1)*100:+.2f}%)"
            )

        # Inequality measures
        print("\n\nInequality Impact:")
        print("-" * 70)

        # Interquartile range
        iqr_actual = actual_earnings.quantile(0.75) - actual_earnings.quantile(0.25)
        iqr_counter = counterfactual_earnings.quantile(0.75) - counterfactual_earnings.quantile(
            0.25
        )

        print(f"Interquartile Range:")
        print(f"  Actual: {iqr_actual:.4f}")
        print(f"  Universal Treatment: {iqr_counter:.4f}")
        print(
            f"  Change: {iqr_counter - iqr_actual:.4f} ({((iqr_counter/iqr_actual - 1)*100):+.2f}%)"
        )

        # 90-10 ratio
        ratio_actual = actual_earnings.quantile(0.90) - actual_earnings.quantile(0.10)
        ratio_counter = counterfactual_earnings.quantile(0.90) - counterfactual_earnings.quantile(
            0.10
        )

        print(f"\n90-10 Range:")
        print(f"  Actual: {ratio_actual:.4f}")
        print(f"  Universal Treatment: {ratio_counter:.4f}")
        print(
            f"  Change: {ratio_counter - ratio_actual:.4f} ({((ratio_counter/ratio_actual - 1)*100):+.2f}%)"
        )

        # Visualization
        self._plot_policy_simulation(actual_earnings, counterfactual_earnings)

    def _plot_unconditional_qte(self, df_results):
        """Plot unconditional QTE."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            df_results["tau"],
            df_results["qte"],
            "o-",
            linewidth=2.5,
            markersize=8,
            color="darkblue",
        )

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel("Quantile", fontsize=12, fontweight="bold")
        ax.set_ylabel("Quantile Treatment Effect (log points)", fontsize=12, fontweight="bold")
        ax.set_title("Unconditional Quantile Treatment Effects", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("unconditional_qte.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_conditional_qte(self, df_qte):
        """Plot conditional QTE with confidence intervals."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Point estimates
        ax.plot(
            df_qte["tau"],
            df_qte["qte"],
            "o-",
            linewidth=2.5,
            markersize=8,
            color="darkred",
            label="QTE",
        )

        # Confidence intervals
        ci_lower = df_qte["qte"] - 1.96 * df_qte["se"]
        ci_upper = df_qte["qte"] + 1.96 * df_qte["se"]

        ax.fill_between(df_qte["tau"], ci_lower, ci_upper, alpha=0.3, color="red", label="95% CI")

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel("Quantile", fontsize=12, fontweight="bold")
        ax.set_ylabel("Conditional QTE (log points)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Heterogeneous Treatment Effects Across Earnings Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("conditional_qte.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_did_comparison(self, df_comparison):
        """Plot DiD QTE comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            df_comparison["tau"],
            df_comparison["pooled"],
            "o-",
            linewidth=2,
            markersize=8,
            label="Pooled QR",
            color="blue",
        )
        ax.plot(
            df_comparison["tau"],
            df_comparison["canay"],
            "s--",
            linewidth=2,
            markersize=8,
            label="Canay FE",
            color="red",
        )

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel("Quantile", fontsize=12, fontweight="bold")
        ax.set_ylabel("DiD-QTE (log points)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Difference-in-Differences Quantile Treatment Effects", fontsize=14, fontweight="bold"
        )
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("did_qte_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_policy_simulation(self, actual, counterfactual):
        """Plot policy simulation results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Distribution comparison
        ax1.hist(actual, bins=50, alpha=0.5, label="Actual", color="blue", density=True)
        ax1.hist(
            counterfactual,
            bins=50,
            alpha=0.5,
            label="Universal Treatment",
            color="red",
            density=True,
        )

        ax1.set_xlabel("Log Earnings", fontweight="bold")
        ax1.set_ylabel("Density", fontweight="bold")
        ax1.set_title("Earnings Distribution Comparison", fontweight="bold")
        ax1.legend(frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)

        # Quantile-Quantile plot
        quantiles = np.linspace(0.01, 0.99, 99)
        q_actual = [actual.quantile(q) for q in quantiles]
        q_counter = [counterfactual.quantile(q) for q in quantiles]

        ax2.scatter(q_actual, q_counter, alpha=0.5, s=20)
        ax2.plot(
            [min(q_actual), max(q_actual)],
            [min(q_actual), max(q_actual)],
            "r--",
            linewidth=2,
            label="45° line",
        )

        ax2.set_xlabel("Actual Distribution", fontweight="bold")
        ax2.set_ylabel("Universal Treatment Distribution", fontweight="bold")
        ax2.set_title("Quantile-Quantile Plot", fontweight="bold")
        ax2.legend(frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("policy_simulation.png", dpi=300, bbox_inches="tight")
        plt.show()


# Example usage
if __name__ == "__main__":
    from scipy import stats

    # Initialize analysis
    print("=" * 70)
    print("QUANTILE TREATMENT EFFECTS ANALYSIS")
    print("Job Training Program Evaluation")
    print("=" * 70)

    analysis = QuantileTreatmentEffectsAnalysis()

    # 1. Unconditional QTE
    print("\n### STEP 1: Unconditional QTE ###")
    uncond_qte = analysis.estimate_unconditional_qte()

    # 2. Conditional QTE
    print("\n### STEP 2: Conditional QTE ###")
    cond_qte, qr_result = analysis.estimate_conditional_qte()

    # 3. DiD-QTE
    print("\n### STEP 3: Difference-in-Differences QTE ###")
    did_results = analysis.estimate_did_qte()

    # 4. Policy simulation
    print("\n### STEP 4: Policy Impact Simulation ###")
    analysis.policy_simulation()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
