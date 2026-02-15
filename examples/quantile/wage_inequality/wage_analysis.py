# examples/quantile/wage_inequality/wage_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from panelbox import PanelData
from panelbox.models.quantile import (
    CanayTwoStep,
    LocationScale,
    PooledQuantile,
    QuantileTreatmentEffects,
)
from panelbox.visualization.quantile import QuantileVisualizer


class WageInequalityAnalysis:
    """
    Complete wage inequality analysis using quantile regression.

    This example demonstrates:
    1. Returns to education across the wage distribution
    2. Gender wage gap decomposition
    3. Union effects
    4. Temporal evolution of inequality
    """

    def __init__(self, data_path=None):
        """Load or simulate wage panel data."""
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self.data = self._simulate_wage_data()

        # Create panel structure
        self.panel = PanelData(self.data, entity="person_id", time="year")

    def _simulate_wage_data(self, n_persons=1000, n_years=10):
        """
        Simulate realistic wage panel data.

        Features:
        - Education returns vary by quantile
        - Gender wage gap
        - Experience effects (quadratic)
        - Union premium
        - Individual fixed effects
        """
        np.random.seed(42)

        # Generate person characteristics
        persons = pd.DataFrame(
            {
                "person_id": range(n_persons),
                "female": np.random.binomial(1, 0.5, n_persons),
                "education": np.random.normal(12, 3, n_persons).clip(0, 20),
                "ability": np.random.normal(0, 1, n_persons),  # Unobserved
                "union_propensity": np.random.uniform(0, 1, n_persons),
            }
        )

        # Panel structure
        panel_data = []

        for year in range(n_years):
            year_data = persons.copy()
            year_data["year"] = 2010 + year
            year_data["experience"] = (
                year_data["year"] - 2010 + np.maximum(0, year_data["education"] - 6)
            )

            # Time-varying union status
            year_data["union"] = (
                year_data["union_propensity"] + 0.1 * year > np.random.uniform(0, 1, n_persons)
            ).astype(int)

            panel_data.append(year_data)

        data = pd.concat(panel_data, ignore_index=True)

        # Generate wages with heterogeneous effects
        # Base wage equation
        data["log_wage"] = (
            2.0
            + 0.05 * data["education"]  # Intercept
            + 0.02 * data["experience"]  # Base return to education
            - 0.0005 * data["experience"] ** 2  # Experience
            + 0.15 * data["union"]  # Experience squared
            + -0.20 * data["female"]  # Union premium
            + 0.3 * data["ability"]  # Gender gap  # Ability (fixed effect)
        )

        # Add quantile-specific effects
        # Higher returns to education at top of distribution
        quantile_rank = np.random.uniform(0, 1, len(data))
        data["log_wage"] += 0.03 * data["education"] * (quantile_rank - 0.5)

        # Larger gender gap at bottom of distribution
        data["log_wage"] -= 0.10 * data["female"] * (0.5 - quantile_rank).clip(0, None)

        # Add random shock
        data["log_wage"] += np.random.normal(0, 0.3, len(data))

        # Convert to levels
        data["wage"] = np.exp(data["log_wage"]) * 20  # Scale to realistic range

        return data

    def analyze_education_returns(self, tau_list=None):
        """
        Analyze heterogeneous returns to education.
        """
        if tau_list is None:
            tau_list = [0.10, 0.25, 0.50, 0.75, 0.90]

        print("\n" + "=" * 60)
        print("RETURNS TO EDUCATION ACROSS WAGE DISTRIBUTION")
        print("=" * 60)

        # Estimate quantile regression
        formula = "log_wage ~ education + experience + I(experience**2) + female + union"

        # 1. Pooled QR
        pooled_model = PooledQuantile(self.panel, formula, tau=tau_list)
        pooled_result = pooled_model.fit(se_type="cluster")

        # 2. Fixed Effects QR (Canay)
        canay_model = CanayTwoStep(self.panel, formula, tau=tau_list)
        canay_result = canay_model.fit()

        # Extract education coefficients
        edu_returns = {"Quantile": tau_list, "Pooled QR": [], "FE QR (Canay)": []}

        edu_idx = 1  # Index of education variable

        for tau in tau_list:
            edu_returns["Pooled QR"].append(pooled_result.results[tau].params[edu_idx])
            edu_returns["FE QR (Canay)"].append(canay_result.results[tau].params[edu_idx])

        # Display results
        df_returns = pd.DataFrame(edu_returns)
        print("\nEducation Returns by Quantile:")
        print(df_returns.to_string(index=False))

        # Statistical test for heterogeneity
        from panelbox.diagnostics.quantile import test_slope_equality

        test_result = test_slope_equality(
            pooled_result.results[0.10], pooled_result.results[0.90], var_names=["education"]
        )

        print(f"\nTest for equal returns (τ=0.10 vs τ=0.90):")
        print(f"  Statistic: {test_result.statistic:.4f}")
        print(f"  P-value: {test_result.pvalue:.4f}")

        if test_result.pvalue < 0.05:
            print("  → Significant evidence of heterogeneous returns")

        # Visualization
        self._plot_education_returns(pooled_result, canay_result, tau_list)

        return pooled_result, canay_result

    def analyze_gender_gap(self):
        """
        Decompose gender wage gap using QR.
        """
        print("\n" + "=" * 60)
        print("GENDER WAGE GAP DECOMPOSITION")
        print("=" * 60)

        tau_list = np.arange(0.05, 1.0, 0.05)

        # Separate models by gender
        male_data = self.panel[self.panel.data["female"] == 0]
        female_data = self.panel[self.panel.data["female"] == 1]

        formula = "log_wage ~ education + experience + I(experience**2) + union"

        # Estimate for each gender
        male_results = {}
        female_results = {}

        for tau in tau_list:
            # Male wages
            male_model = PooledQuantile(male_data, formula, tau=tau)
            male_results[tau] = male_model.fit(verbose=False).results[tau]

            # Female wages
            female_model = PooledQuantile(female_data, formula, tau=tau)
            female_results[tau] = female_model.fit(verbose=False).results[tau]

        # Machado-Mata decomposition
        decomposition = self._machado_mata_decomposition(male_results, female_results, tau_list)

        # Display results
        print("\nGender Wage Gap by Quantile:")
        print(f"{'Quantile':<10} {'Total Gap':<12} {'Explained':<12} {'Unexplained':<12}")
        print("-" * 50)

        for i, tau in enumerate([0.10, 0.25, 0.50, 0.75, 0.90]):
            idx = list(tau_list).index(tau)
            print(
                f"{tau:<10.2f} {decomposition['total'][idx]:>11.3f} "
                f"{decomposition['explained'][idx]:>11.3f} "
                f"{decomposition['unexplained'][idx]:>11.3f}"
            )

        # Visualization
        self._plot_gender_gap_decomposition(decomposition, tau_list)

        return decomposition

    def _machado_mata_decomposition(self, male_results, female_results, tau_list):
        """
        Machado-Mata decomposition of wage gap.

        Decomposes gap into:
        - Characteristics effect (explained)
        - Coefficients effect (unexplained/discrimination)
        """
        male_data = self.panel[self.panel.data["female"] == 0]
        female_data = self.panel[self.panel.data["female"] == 1]

        # Get average characteristics
        X_vars = ["education", "experience", "union"]
        X_male_mean = male_data.data[X_vars].mean()
        X_female_mean = female_data.data[X_vars].mean()

        total_gap = []
        explained = []
        unexplained = []

        for tau in tau_list:
            # Coefficients
            beta_m = male_results[tau].params[1:4]  # Skip intercept
            beta_f = female_results[tau].params[1:4]

            # Total gap
            gap = (X_male_mean @ beta_m) - (X_female_mean @ beta_f)

            # Explained (characteristics)
            exp = X_female_mean @ (beta_m - beta_f)

            # Unexplained (coefficients)
            unexp = (X_male_mean - X_female_mean) @ beta_m

            total_gap.append(gap)
            explained.append(exp)
            unexplained.append(unexp)

        return {
            "total": np.array(total_gap),
            "explained": np.array(explained),
            "unexplained": np.array(unexplained),
        }

    def analyze_union_effects(self):
        """
        Analyze heterogeneous union wage premium.
        """
        print("\n" + "=" * 60)
        print("UNION WAGE PREMIUM ANALYSIS")
        print("=" * 60)

        # Quantile treatment effects
        qte = QuantileTreatmentEffects(
            data=self.panel.data,
            outcome="log_wage",
            treatment="union",
            covariates=["education", "experience", "female"],
        )

        tau_list = [0.10, 0.25, 0.50, 0.75, 0.90]
        qte_result = qte.estimate_qte(tau=tau_list, method="standard", bootstrap=True, n_boot=500)

        # Display results
        qte_result.summary()

        # Test for heterogeneous effects
        qte_values = [qte_result.qte_results[tau]["qte"] for tau in tau_list]
        heterogeneity = np.std(qte_values)

        print(f"\nHeterogeneity in union effects (std): {heterogeneity:.4f}")

        if heterogeneity > 0.05:
            print("→ Substantial heterogeneity in union wage premium")
            print("→ Premium is larger at lower quantiles")

        # Visualization
        fig = qte.plot_qte(qte_result)
        plt.show()

        return qte_result

    def _plot_education_returns(self, pooled_result, canay_result, tau_list):
        """Plot education returns across quantiles."""
        viz = QuantileVisualizer(style="academic")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract coefficients
        edu_idx = 1
        pooled_coefs = [pooled_result.results[tau].params[edu_idx] for tau in tau_list]
        canay_coefs = [canay_result.results[tau].params[edu_idx] for tau in tau_list]

        # Plot
        ax.plot(tau_list, pooled_coefs, "o-", linewidth=2, markersize=8, label="Pooled QR")
        ax.plot(tau_list, canay_coefs, "s--", linewidth=2, markersize=8, label="Fixed Effects QR")

        # OLS comparison
        from panelbox.models.linear import FixedEffects, PooledOLS

        ols_model = PooledOLS(
            self.panel, "log_wage ~ education + experience + I(experience**2) + female + union"
        )
        ols_result = ols_model.fit()
        ols_return = ols_result.params[edu_idx]

        ax.axhline(ols_return, color="red", linestyle=":", linewidth=2, label="OLS")

        # Formatting
        ax.set_xlabel("Quantile", fontsize=12, fontweight="bold")
        ax.set_ylabel("Return to Education (log points)", fontsize=12, fontweight="bold")
        ax.set_title("Heterogeneous Returns to Education", fontsize=14, fontweight="bold")
        ax.legend(frameon=True, shadow=True, loc="best")
        ax.grid(True, alpha=0.3)

        # Add annotations
        ax.text(0.10, pooled_coefs[0], f"{pooled_coefs[0]:.3f}", ha="center", va="bottom")
        ax.text(0.90, pooled_coefs[-1], f"{pooled_coefs[-1]:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.show()

    def _plot_gender_gap_decomposition(self, decomposition, tau_list):
        """Plot gender gap decomposition."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Total decomposition
        ax1.plot(tau_list, decomposition["total"], "k-", linewidth=2.5, label="Total Gap")
        ax1.plot(tau_list, decomposition["explained"], "b--", linewidth=2, label="Explained")
        ax1.plot(tau_list, decomposition["unexplained"], "r--", linewidth=2, label="Unexplained")

        ax1.fill_between(tau_list, 0, decomposition["unexplained"], alpha=0.3, color="red")
        ax1.fill_between(
            tau_list, decomposition["unexplained"], decomposition["total"], alpha=0.3, color="blue"
        )

        ax1.set_xlabel("Quantile", fontweight="bold")
        ax1.set_ylabel("Log Wage Gap", fontweight="bold")
        ax1.set_title("Gender Wage Gap Decomposition", fontweight="bold")
        ax1.legend(frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="black", linewidth=0.5)

        # Percentage unexplained
        pct_unexplained = 100 * decomposition["unexplained"] / decomposition["total"]
        ax2.plot(tau_list, pct_unexplained, "r-", linewidth=2.5)
        ax2.fill_between(tau_list, 0, pct_unexplained, alpha=0.3, color="red")

        ax2.set_xlabel("Quantile", fontweight="bold")
        ax2.set_ylabel("Unexplained Gap (%)", fontweight="bold")
        ax2.set_title("Discrimination Component", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(50, color="black", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.show()

    def generate_report(self, output_path="wage_inequality_report.html"):
        """Generate complete analysis report."""
        from panelbox.reports.quantile import QuantileReportGenerator

        # Run all analyses
        edu_results = self.analyze_education_returns()
        gender_decomp = self.analyze_gender_gap()
        union_effects = self.analyze_union_effects()

        # Create custom report
        generator = QuantileReportGenerator(edu_results[0])

        custom_sections = {
            "wage_analysis": True,
            "gender_decomposition": gender_decomp,
            "union_effects": union_effects,
        }

        html = generator.generate_html(custom_sections=custom_sections)

        with open(output_path, "w") as f:
            f.write(html)

        print(f"\nReport saved to {output_path}")


# Run the analysis
if __name__ == "__main__":
    analysis = WageInequalityAnalysis()

    # Full analysis
    analysis.analyze_education_returns()
    analysis.analyze_gender_gap()
    analysis.analyze_union_effects()

    # Generate report
    analysis.generate_report()
