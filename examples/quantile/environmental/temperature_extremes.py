"""
Environmental Analysis: Temperature Extremes using Quantile Regression

This example demonstrates:
1. Analysis of climate extremes using panel QR
2. Heterogeneous effects of climate change across distribution
3. Regional differences in temperature trends
4. Extreme weather event prediction
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from panelbox import PanelData
from panelbox.models.quantile import FixedEffectsQuantile, PooledQuantile
from panelbox.visualization.quantile import QuantileVisualizer


class TemperatureExtremesAnalysis:
    """
    Analysis of temperature extremes using quantile regression.

    Demonstrates:
    1. Trends in extreme temperatures (cold and hot)
    2. Regional heterogeneity
    3. Climate change impacts on distribution tails
    4. Extreme event probability
    """

    def __init__(self, data_path=None):
        """Load or simulate temperature panel data."""
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self.data = self._simulate_temperature_data()

        # Create panel structure
        self.panel = PanelData(self.data, entity="station_id", time="date")

    def _simulate_temperature_data(self, n_stations=50, n_years=30):
        """
        Simulate realistic temperature panel data.

        Features:
        - Seasonal patterns
        - Long-term warming trend
        - Increasing variance (climate change)
        - Regional differences
        - Extreme events
        """
        np.random.seed(42)

        # Generate stations
        stations = pd.DataFrame(
            {
                "station_id": range(n_stations),
                "latitude": np.random.uniform(25, 50, n_stations),
                "longitude": np.random.uniform(-120, -70, n_stations),
                "elevation": np.random.uniform(0, 2000, n_stations),
                "region": np.random.choice(["North", "South", "East", "West"], n_stations),
            }
        )

        # Panel structure (daily data)
        start_date = datetime(1990, 1, 1)
        n_days = n_years * 365

        panel_data = []

        for station_idx, station in stations.iterrows():
            for day in range(n_days):
                date = start_date + timedelta(days=day)
                year = date.year - 1990  # Years since start
                day_of_year = date.timetuple().tm_yday

                # Base temperature (depends on latitude and season)
                base_temp = (
                    60
                    - 0.5 * (station["latitude"] - 37.5)  # Base
                    + 20 * np.sin(2 * np.pi * day_of_year / 365)  # Latitude effect
                    + -0.1 * station["elevation"] / 100  # Seasonal  # Elevation effect
                )

                # Climate change trend (warming)
                # Asymmetric: more warming at high temperatures
                warming_trend = 0.03 * year  # ~0.3°C per decade at mean

                # Increasing variance (climate change effect)
                # More variance in extremes
                base_variance = 5 + 0.05 * year

                # Generate temperature with fat tails
                epsilon = np.random.standard_t(df=4) * base_variance

                # Add asymmetric warming (more at high quantiles)
                if epsilon > 0:
                    epsilon += warming_trend * (1 + epsilon / base_variance)
                else:
                    epsilon += warming_trend * (1 - 0.5 * abs(epsilon) / base_variance)

                temperature = base_temp + epsilon

                panel_data.append(
                    {
                        "station_id": station["station_id"],
                        "date": date,
                        "temperature": temperature,
                        "year": year,
                        "day_of_year": day_of_year,
                        "latitude": station["latitude"],
                        "elevation": station["elevation"],
                        "region": station["region"],
                        # Climate variables
                        "co2_ppm": 350 + 2 * year,  # Rising CO2
                        "solar_radiation": 250 + 10 * np.sin(2 * np.pi * day_of_year / 365),
                    }
                )

        return pd.DataFrame(panel_data)

    def analyze_extreme_trends(self, tau_list=None):
        """
        Analyze trends in extreme temperatures.
        """
        if tau_list is None:
            tau_list = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        print("\n" + "=" * 70)
        print("TEMPERATURE EXTREMES TREND ANALYSIS")
        print("=" * 70)

        # Model: temperature ~ year + season + latitude + elevation
        formula = "temperature ~ year + np.sin(2*np.pi*day_of_year/365) + np.cos(2*np.pi*day_of_year/365) + latitude + elevation"

        # Pooled QR for multiple quantiles
        model = PooledQuantile(self.panel, formula, tau=tau_list)
        result = model.fit(se_type="cluster", cluster="station_id")

        # Extract year coefficients (warming trends)
        trends = {}
        for tau in tau_list:
            year_coef = result.results[tau].params[1]  # year coefficient
            year_se = result.results[tau].bse[1]

            trends[tau] = {
                "coefficient": year_coef,
                "se": year_se,
                "ci_lower": year_coef - 1.96 * year_se,
                "ci_upper": year_coef + 1.96 * year_se,
            }

            # Convert to per-decade trend
            per_decade = year_coef * 10

            print(f"\nQuantile τ={tau:.2f} (Temperature Trend):")
            print(f"  Trend: {per_decade:.4f}°F per decade")
            print(f"  95% CI: [{trends[tau]['ci_lower']*10:.4f}, {trends[tau]['ci_upper']*10:.4f}]")
            print(f"  Significant: {'Yes' if abs(year_coef) > 1.96 * year_se else 'No'}")

        # Test for heterogeneous trends
        print("\n" + "-" * 70)
        print("HETEROGENEITY TEST")
        print("-" * 70)

        cold_extreme_trend = trends[0.05]["coefficient"]
        hot_extreme_trend = trends[0.95]["coefficient"]

        print(f"\nCold extreme (5th percentile): {cold_extreme_trend*10:.4f}°F/decade")
        print(f"Hot extreme (95th percentile): {hot_extreme_trend*10:.4f}°F/decade")
        print(f"Difference: {(hot_extreme_trend - cold_extreme_trend)*10:.4f}°F/decade")

        if hot_extreme_trend > cold_extreme_trend * 1.5:
            print("\n→ Strong evidence of asymmetric warming")
            print("→ Hot extremes warming faster than cold extremes")

        # Visualization
        self._plot_extreme_trends(trends, tau_list)

        return result, trends

    def analyze_regional_differences(self):
        """
        Analyze regional differences in extreme temperatures.
        """
        print("\n" + "=" * 70)
        print("REGIONAL ANALYSIS OF TEMPERATURE EXTREMES")
        print("=" * 70)

        regions = self.data["region"].unique()
        tau_list = [0.05, 0.50, 0.95]

        regional_results = {}

        for region in regions:
            print(f"\n{region} Region:")
            print("-" * 40)

            # Filter data for region
            region_data = self.panel[self.panel.data["region"] == region]

            # Estimate model
            formula = "temperature ~ year + np.sin(2*np.pi*day_of_year/365) + np.cos(2*np.pi*day_of_year/365)"

            model = PooledQuantile(region_data, formula, tau=tau_list)
            result = model.fit()

            regional_results[region] = result

            # Extract trends
            for tau in tau_list:
                year_coef = result.results[tau].params[1]
                trend_per_decade = year_coef * 10

                quantile_name = {0.05: "Cold", 0.50: "Median", 0.95: "Hot"}[tau]
                print(f"  {quantile_name} (τ={tau}): {trend_per_decade:.4f}°F/decade")

        # Compare regions
        self._plot_regional_comparison(regional_results, regions, tau_list)

        return regional_results

    def estimate_extreme_probabilities(self):
        """
        Estimate probability of extreme events.
        """
        print("\n" + "=" * 70)
        print("EXTREME EVENT PROBABILITY ANALYSIS")
        print("=" * 70)

        # Define extreme thresholds
        hot_threshold = self.data["temperature"].quantile(0.95)
        cold_threshold = self.data["temperature"].quantile(0.05)

        print(f"\nHistorical Extremes:")
        print(f"  Hot extreme threshold: {hot_threshold:.1f}°F")
        print(f"  Cold extreme threshold: {cold_threshold:.1f}°F")

        # Estimate probability over time
        years = sorted(self.data["year"].unique())
        prob_hot = []
        prob_cold = []

        for year in years:
            year_data = self.data[self.data["year"] == year]

            p_hot = (year_data["temperature"] > hot_threshold).mean()
            p_cold = (year_data["temperature"] < cold_threshold).mean()

            prob_hot.append(p_hot)
            prob_cold.append(p_cold)

        # Trend in probabilities
        from scipy.stats import linregress

        slope_hot, intercept_hot, r_hot, p_hot, se_hot = linregress(years, prob_hot)
        slope_cold, intercept_cold, r_cold, p_cold, se_cold = linregress(years, prob_cold)

        print(f"\nTrends in Extreme Event Probability:")
        print(f"  Hot extremes: {slope_hot*100:.4f} percentage points/year (p={p_hot:.4f})")
        print(f"  Cold extremes: {slope_cold*100:.4f} percentage points/year (p={p_cold:.4f})")

        # Future projections
        future_years = [30, 40, 50]  # Years from start

        print(f"\nProjected Probabilities:")
        for future_year in future_years:
            proj_hot = intercept_hot + slope_hot * future_year
            proj_cold = intercept_cold + slope_cold * future_year

            print(f"\n  Year {1990 + future_year}:")
            print(f"    Hot extreme probability: {proj_hot*100:.2f}%")
            print(f"    Cold extreme probability: {proj_cold*100:.2f}%")

        # Visualization
        self._plot_extreme_probabilities(years, prob_hot, prob_cold)

        return {
            "years": years,
            "prob_hot": prob_hot,
            "prob_cold": prob_cold,
            "slope_hot": slope_hot,
            "slope_cold": slope_cold,
        }

    def _plot_extreme_trends(self, trends, tau_list):
        """Plot warming trends across quantiles."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        coefficients = [trends[tau]["coefficient"] * 10 for tau in tau_list]
        ci_lower = [trends[tau]["ci_lower"] * 10 for tau in tau_list]
        ci_upper = [trends[tau]["ci_upper"] * 10 for tau in tau_list]

        # Plot
        ax.plot(
            tau_list,
            coefficients,
            "o-",
            linewidth=2.5,
            markersize=8,
            color="darkred",
            label="Temperature Trend",
        )
        ax.fill_between(tau_list, ci_lower, ci_upper, alpha=0.3, color="red", label="95% CI")

        # Reference line
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        # Formatting
        ax.set_xlabel("Temperature Quantile", fontsize=12, fontweight="bold")
        ax.set_ylabel("Warming Trend (°F per decade)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Heterogeneous Climate Change Effects Across Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Annotations
        ax.text(
            0.05,
            coefficients[0],
            "Cold\nExtreme",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            0.95,
            coefficients[-1],
            "Hot\nExtreme",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig("temperature_trends.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_regional_comparison(self, regional_results, regions, tau_list):
        """Plot regional comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ["blue", "green", "red", "orange"]

        for region, color in zip(regions, colors):
            result = regional_results[region]
            trends = [result.results[tau].params[1] * 10 for tau in tau_list]

            ax.plot(tau_list, trends, "o-", linewidth=2, markersize=8, color=color, label=region)

        ax.set_xlabel("Quantile", fontsize=12, fontweight="bold")
        ax.set_ylabel("Warming Trend (°F per decade)", fontsize=12, fontweight="bold")
        ax.set_title("Regional Differences in Temperature Trends", fontsize=14, fontweight="bold")
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        plt.tight_layout()
        plt.savefig("regional_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_extreme_probabilities(self, years, prob_hot, prob_cold):
        """Plot evolution of extreme event probabilities."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert to calendar years
        calendar_years = [1990 + y for y in years]

        # Plot
        ax.plot(
            calendar_years,
            np.array(prob_hot) * 100,
            "o-",
            linewidth=2,
            markersize=6,
            color="red",
            label="Hot Extremes (>95th percentile)",
        )
        ax.plot(
            calendar_years,
            np.array(prob_cold) * 100,
            "o-",
            linewidth=2,
            markersize=6,
            color="blue",
            label="Cold Extremes (<5th percentile)",
        )

        # Reference line
        ax.axhline(5, color="black", linestyle="--", linewidth=1, label="Historical frequency (5%)")

        # Formatting
        ax.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax.set_ylabel("Probability (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Evolution of Extreme Temperature Event Probability", fontsize=14, fontweight="bold"
        )
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("extreme_probabilities.png", dpi=300, bbox_inches="tight")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize analysis
    analysis = TemperatureExtremesAnalysis()

    # Analyze extreme trends
    print("\n1. EXTREME TEMPERATURE TRENDS")
    result, trends = analysis.analyze_extreme_trends()

    # Regional analysis
    print("\n2. REGIONAL ANALYSIS")
    regional_results = analysis.analyze_regional_differences()

    # Extreme probabilities
    print("\n3. EXTREME EVENT PROBABILITIES")
    prob_results = analysis.estimate_extreme_probabilities()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print("1. Temperature warming is asymmetric across the distribution")
    print("2. Hot extremes are warming faster than cold extremes")
    print("3. Regional differences in warming patterns exist")
    print("4. Probability of extreme events is changing over time")
