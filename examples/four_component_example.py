"""
Four-Component SFA Model Example (Kumbhakar et al., 2014)

This example demonstrates the revolutionary four-component model that
separates inefficiency into:
    - Persistent inefficiency (structural, long-run) vs
    - Transient inefficiency (managerial, short-run)

This decomposition is CRITICAL for policy-making!

Example Application: Hospital Efficiency
    Hospital A has low overall efficiency (60%)

    Traditional SFA:
        "Hospital A is 60% efficient" → Unclear what to do

    Four-Component SFA:
        Persistent TE = 95% (structural factors)
        Transient TE = 63% (managerial factors)

        POLICY IMPLICATION:
        → Don't invest in new buildings (persistent is good!)
        → Focus on training and management (transient is poor!)

    Hospital B also has 60% overall efficiency, but:
        Persistent TE = 65% (structural factors)
        Transient TE = 92% (managerial factors)

        POLICY IMPLICATION:
        → Invest in new equipment, better location (persistent is poor!)
        → Management is already good (transient is excellent!)

This is why the four-component model is revolutionary!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.frontier.advanced import FourComponentSFA


def generate_hospital_data(n_hospitals=50, n_years=8, seed=42):
    """Generate realistic hospital efficiency data.

    Some hospitals have structural problems (bad location, old equipment)
    → High persistent inefficiency

    Some hospitals have management problems (poor organization)
    → High transient inefficiency
    """
    np.random.seed(seed)

    data = []

    # True parameters
    beta_doctors = 0.35
    beta_nurses = 0.25
    beta_beds = 0.20

    for hospital_id in range(n_hospitals):
        # Hospital type determines inefficiency profile
        hospital_type = np.random.choice(["A", "B", "C"], p=[0.3, 0.4, 0.3])

        if hospital_type == "A":
            # Type A: Good structure, poor management
            eta_i = 0.05  # Low persistent inefficiency
            u_mean = 0.40  # High transient inefficiency mean
        elif hospital_type == "B":
            # Type B: Poor structure, good management
            eta_i = 0.50  # High persistent inefficiency
            u_mean = 0.10  # Low transient inefficiency mean
        else:
            # Type C: Balanced
            eta_i = 0.20
            u_mean = 0.20

        # Random heterogeneity
        mu_i = np.random.normal(0, 0.15)

        for year in range(n_years):
            # Inputs (log scale)
            log_doctors = np.random.normal(3.5, 0.3)
            log_nurses = np.random.normal(4.0, 0.3)
            log_beds = np.random.normal(4.5, 0.3)

            # Noise
            v_it = np.random.normal(0, 0.12)

            # Transient inefficiency (varies over time)
            u_it = np.abs(np.random.normal(u_mean, 0.15))

            # Output: log(patients treated)
            log_output = (
                5.0  # Base productivity
                + beta_doctors * log_doctors
                + beta_nurses * log_nurses
                + beta_beds * log_beds
                + mu_i  # Heterogeneity
                - eta_i  # Persistent inefficiency
                + v_it  # Noise
                - u_it  # Transient inefficiency
            )

            data.append(
                {
                    "hospital_id": hospital_id,
                    "year": 2015 + year,
                    "log_patients": log_output,
                    "log_doctors": log_doctors,
                    "log_nurses": log_nurses,
                    "log_beds": log_beds,
                    "hospital_type": hospital_type,
                    "true_persistent_te": np.exp(-eta_i),
                    "true_transient_te": np.exp(-u_it),
                }
            )

    return pd.DataFrame(data)


def main():
    """Main example demonstrating four-component SFA."""

    print("=" * 80)
    print("FOUR-COMPONENT STOCHASTIC FRONTIER MODEL")
    print("Example: Hospital Efficiency Analysis")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic hospital data...")
    df = generate_hospital_data(n_hospitals=50, n_years=8)

    print(f"   Sample: {len(df)} observations")
    print(f"   Hospitals: {df['hospital_id'].nunique()}")
    print(f"   Years: {df['year'].nunique()}")
    print(f"   Hospital types: {df['hospital_type'].value_counts().to_dict()}")

    # Estimate four-component model
    print("\n2. Estimating Four-Component SFA Model...")
    print("   (This separates persistent from transient inefficiency)")

    model = FourComponentSFA(
        data=df,
        depvar="log_patients",
        exog=["log_doctors", "log_nurses", "log_beds"],
        entity="hospital_id",
        time="year",
        frontier_type="production",
    )

    result = model.fit(verbose=True)

    # Get efficiency estimates
    print("\n3. Extracting Efficiency Estimates...")

    te_persistent = result.persistent_efficiency()
    te_transient = result.transient_efficiency()
    te_overall = result.overall_efficiency()

    print("\nPersistent Efficiency (structural, time-invariant):")
    print(f"   Mean: {te_persistent['persistent_efficiency'].mean():.4f}")
    print(f"   Std:  {te_persistent['persistent_efficiency'].std():.4f}")
    print(f"   Min:  {te_persistent['persistent_efficiency'].min():.4f}")
    print(f"   Max:  {te_persistent['persistent_efficiency'].max():.4f}")

    print("\nTransient Efficiency (managerial, time-varying):")
    print(f"   Mean: {te_transient['transient_efficiency'].mean():.4f}")
    print(f"   Std:  {te_transient['transient_efficiency'].std():.4f}")
    print(f"   Min:  {te_transient['transient_efficiency'].min():.4f}")
    print(f"   Max:  {te_transient['transient_efficiency'].max():.4f}")

    print("\nOverall Efficiency (product of both):")
    print(f"   Mean: {te_overall['overall_efficiency'].mean():.4f}")
    print(f"   Std:  {te_overall['overall_efficiency'].std():.4f}")
    print(f"   Min:  {te_overall['overall_efficiency'].min():.4f}")
    print(f"   Max:  {te_overall['overall_efficiency'].max():.4f}")

    # Policy recommendations
    print("\n4. Policy Recommendations for Selected Hospitals...")
    print("=" * 80)

    # Find hospitals with different efficiency profiles
    te_summary = (
        te_overall.groupby("entity")
        .agg(
            {
                "overall_efficiency": "mean",
                "persistent_efficiency": "first",  # Time-invariant
                "transient_efficiency": "mean",
            }
        )
        .round(4)
    )

    # Hospital with low persistent, high transient
    low_persistent = te_summary.nsmallest(5, "persistent_efficiency")
    print("\nHospitals with LOW persistent efficiency (structural problems):")
    print(low_persistent)
    print("\nRECOMMENDATION: Invest in infrastructure, equipment, location improvements")

    # Hospital with high persistent, low transient
    low_transient = te_summary.nsmallest(5, "transient_efficiency")
    print("\nHospitals with LOW transient efficiency (managerial problems):")
    print(low_transient)
    print("\nRECOMMENDATION: Improve management, training, organization, incentives")

    # Create visualization
    print("\n5. Creating Visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distribution of persistent efficiency
    axes[0, 0].hist(
        te_persistent["persistent_efficiency"],
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    axes[0, 0].set_xlabel("Persistent Efficiency")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Persistent Efficiency\n(Structural, Long-run)")
    axes[0, 0].axvline(
        te_persistent["persistent_efficiency"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean = {te_persistent["persistent_efficiency"].mean():.3f}',
    )
    axes[0, 0].legend()

    # Plot 2: Distribution of transient efficiency
    axes[0, 1].hist(
        te_transient["transient_efficiency"], bins=20, edgecolor="black", alpha=0.7, color="coral"
    )
    axes[0, 1].set_xlabel("Transient Efficiency")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Transient Efficiency\n(Managerial, Short-run)")
    axes[0, 1].axvline(
        te_transient["transient_efficiency"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean = {te_transient["transient_efficiency"].mean():.3f}',
    )
    axes[0, 1].legend()

    # Plot 3: Scatter plot - persistent vs transient
    sample_summary = te_overall.groupby("entity").agg(
        {
            "persistent_efficiency": "first",
            "transient_efficiency": "mean",
        }
    )

    axes[1, 0].scatter(
        sample_summary["persistent_efficiency"],
        sample_summary["transient_efficiency"],
        alpha=0.6,
        s=100,
        color="purple",
    )
    axes[1, 0].set_xlabel("Persistent Efficiency")
    axes[1, 0].set_ylabel("Transient Efficiency (mean)")
    axes[1, 0].set_title("Persistent vs Transient Efficiency\n(Each point = one hospital)")
    axes[1, 0].grid(True, alpha=0.3)

    # Add quadrant lines
    axes[1, 0].axvline(
        sample_summary["persistent_efficiency"].median(), color="gray", linestyle=":", alpha=0.5
    )
    axes[1, 0].axhline(
        sample_summary["transient_efficiency"].median(), color="gray", linestyle=":", alpha=0.5
    )

    # Annotate quadrants
    axes[1, 0].text(
        0.02,
        0.98,
        "Low Both\n→ Major reforms needed",
        transform=axes[1, 0].transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1, 0].text(
        0.98,
        0.98,
        "High Persistent\nLow Transient\n→ Improve management",
        transform=axes[1, 0].transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # Plot 4: Variance decomposition
    variances = {
        "Noise\n(v_it)": result.sigma_v**2,
        "Transient\nIneff.\n(u_it)": result.sigma_u**2,
        "Heterogeneity\n(μ_i)": result.sigma_mu**2,
        "Persistent\nIneff.\n(η_i)": result.sigma_eta**2,
    }

    colors = ["lightblue", "coral", "lightgreen", "gold"]
    axes[1, 1].bar(variances.keys(), variances.values(), color=colors, edgecolor="black", alpha=0.7)
    axes[1, 1].set_ylabel("Variance Component")
    axes[1, 1].set_title("Variance Decomposition")
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig("four_component_analysis.png", dpi=300, bbox_inches="tight")
    print("   Saved visualization to: four_component_analysis.png")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. Persistent inefficiency reflects structural factors (location, equipment)")
    print("2. Transient inefficiency reflects managerial factors (organization, training)")
    print("3. Different hospitals need DIFFERENT policies based on their profile!")
    print("\nThis is why the four-component model is revolutionary for policy-making.")
    print("=" * 80)


if __name__ == "__main__":
    main()
