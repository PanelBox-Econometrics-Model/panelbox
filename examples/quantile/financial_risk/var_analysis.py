"""
examples/quantile/financial_risk/var_analysis.py

Value at Risk (VaR) and Conditional VaR analysis using Quantile Regression.

This example demonstrates:
1. Dynamic VaR estimation
2. Backtesting procedures
3. Stress testing
4. Portfolio risk analysis
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class ValueAtRiskAnalysis:
    """
    Value at Risk (VaR) and Conditional VaR analysis using QR.

    Demonstrates:
    1. Dynamic VaR estimation
    2. Backtesting procedures
    3. Stress testing
    4. Portfolio risk analysis
    """

    def __init__(self, tickers=None, start_date="2018-01-01", end_date="2023-12-31"):
        """Initialize with market data."""
        self.tickers = tickers or ["SPY", "QQQ", "IWM"]
        self.start_date = start_date
        self.end_date = end_date

        # Try to load real data, fall back to simulation
        try:
            self.data = self._load_market_data(start_date, end_date)
            self.data_source = "real"
        except:
            print("Could not load real market data. Using simulated data.")
            self.data = self._simulate_market_data()
            self.data_source = "simulated"

        self.returns = self._calculate_returns()

        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)

    def _load_market_data(self, start, end):
        """Download market data from Yahoo Finance."""
        try:
            import yfinance as yf

            data = yf.download(self.tickers, start=start, end=end)["Adj Close"]
            return data
        except ImportError:
            raise ImportError("yfinance not installed. Install with: pip install yfinance")

    def _simulate_market_data(self, n_days=1500):
        """Simulate realistic market data with GARCH-like properties."""
        np.random.seed(42)

        dates = pd.date_range(start="2018-01-01", periods=n_days, freq="D")

        data = pd.DataFrame(index=dates)

        for ticker in self.tickers:
            # Simulate with time-varying volatility
            returns = []
            vol = 0.01  # Initial volatility

            for _ in range(n_days):
                # GARCH(1,1)-like volatility
                vol = 0.00001 + 0.05 * vol + 0.94 * vol + 0.01 * np.random.randn() ** 2
                ret = np.random.normal(0.0003, np.sqrt(vol))
                returns.append(ret)

            # Convert to price level
            cumulative_returns = np.cumprod(1 + np.array(returns))
            data[ticker] = 100 * cumulative_returns

        return data

    def _calculate_returns(self):
        """Calculate log returns."""
        returns = np.log(self.data / self.data.shift(1))
        returns = returns.dropna()
        return returns

    def estimate_var(self, confidence_levels=None, window=250):
        """
        Estimate Value at Risk using quantile regression.

        Parameters
        ----------
        confidence_levels : list
            Confidence levels for VaR (e.g., 0.95 = 5% VaR)
        window : int
            Rolling window size
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        print("\n" + "=" * 70)
        print(" VALUE AT RISK ESTIMATION")
        print("=" * 70)
        print(f"Data source: {self.data_source}")
        print(f"Period: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        print(f"Number of observations: {len(self.returns)}")

        var_results = {}

        for ticker in self.tickers:
            returns = self.returns[ticker]

            # Prepare data with lags and volatility
            data = pd.DataFrame(
                {
                    "returns": returns,
                    "returns_lag1": returns.shift(1),
                    "returns_lag2": returns.shift(2),
                    "volatility": returns.rolling(20).std(),
                }
            ).dropna()

            # Estimate for each confidence level
            ticker_var = {}

            for conf in confidence_levels:
                tau = 1 - conf  # Left tail quantile

                # Rolling VaR estimation (simplified)
                var_forecast = self._rolling_var_simple(data, tau, window)

                ticker_var[conf] = var_forecast

                print(f"\n{ticker} - {conf*100:.0f}% VaR:")
                print(f"  Mean VaR: {var_forecast.mean():.4f} ({var_forecast.mean()*100:.2f}%)")
                print(f"  Min VaR (worst): {var_forecast.min():.4f}")
                print(f"  Current VaR: {var_forecast.iloc[-1]:.4f}")

            var_results[ticker] = ticker_var

        # Backtesting
        self._backtest_var(var_results)

        # Visualization
        self._plot_var(var_results)

        return var_results

    def _rolling_var_simple(self, data, tau, window):
        """Calculate rolling VaR using historical quantile method."""
        var_series = pd.Series(index=data.index[window:], dtype=float)

        for i in range(window, len(data)):
            # Use past window to estimate quantile
            historical = data["returns"].iloc[i - window : i]
            var_estimate = historical.quantile(tau)
            var_series.iloc[i - window] = var_estimate

        return var_series

    def _backtest_var(self, var_results):
        """
        Backtest VaR estimates using Kupiec test.
        """
        print("\n" + "=" * 70)
        print(" VAR BACKTESTING RESULTS")
        print("=" * 70)

        for ticker in self.tickers:
            print(f"\n{ticker}:")

            for conf, var_series in var_results[ticker].items():
                # Align returns and VaR
                common_idx = var_series.index.intersection(self.returns[ticker].index)
                actual_returns = self.returns[ticker].loc[common_idx]
                var_values = var_series.loc[common_idx]

                # Count violations (returns worse than VaR)
                violations = actual_returns < var_values
                n_violations = violations.sum()
                violation_rate = n_violations / len(violations)

                # Expected violation rate
                expected_rate = 1 - conf

                # Kupiec test statistic
                if n_violations > 0 and n_violations < len(violations):
                    lr_stat = -2 * (
                        np.log(
                            (1 - expected_rate) ** (len(violations) - n_violations)
                            * expected_rate**n_violations
                        )
                        - np.log(
                            (1 - violation_rate) ** (len(violations) - n_violations)
                            * violation_rate**n_violations
                        )
                    )
                    # Chi-square(1) critical value at 5% is 3.84
                    reject = lr_stat > 3.84
                else:
                    lr_stat = np.nan
                    reject = True

                print(f"\n  {conf*100:.0f}% VaR Backtest:")
                print(
                    f"    Violations: {n_violations}/{len(violations)} ({violation_rate*100:.2f}%)"
                )
                print(f"    Expected Rate: {expected_rate*100:.2f}%")
                print(
                    f"    LR Statistic: {lr_stat:.4f}"
                    if not np.isnan(lr_stat)
                    else "    LR Statistic: N/A"
                )
                print(f"    Model Status: {'REJECTED' if reject else 'NOT REJECTED'}")

    def estimate_cvar(self, confidence_level=0.95):
        """
        Estimate Conditional Value at Risk (Expected Shortfall).
        """
        print("\n" + "=" * 70)
        print(" CONDITIONAL VAR (EXPECTED SHORTFALL)")
        print("=" * 70)

        cvar_results = {}

        for ticker in self.tickers:
            returns = self.returns[ticker]

            # VaR threshold
            var = returns.quantile(1 - confidence_level)

            # CVaR is mean of returns below VaR
            tail_returns = returns[returns <= var]
            cvar = tail_returns.mean()

            cvar_results[ticker] = {
                "VaR": var,
                "CVaR": cvar,
                "CVaR/VaR": cvar / var if var != 0 else np.inf,
                "n_tail_obs": len(tail_returns),
            }

            print(f"\n{ticker}:")
            print(f"  VaR ({confidence_level*100:.0f}%): {var:.4f} ({var*100:.2f}%)")
            print(f"  CVaR (ES): {cvar:.4f} ({cvar*100:.2f}%)")
            print(f"  CVaR/VaR Ratio: {cvar/var:.2f}")
            print(f"  Tail observations: {len(tail_returns)}")

        return cvar_results

    def stress_testing(self):
        """
        Stress testing using extreme quantiles.
        """
        print("\n" + "=" * 70)
        print(" STRESS TESTING ANALYSIS")
        print("=" * 70)

        # Extreme quantiles
        extreme_tau = [0.001, 0.005, 0.01]

        for ticker in self.tickers:
            returns = self.returns[ticker]

            print(f"\n{ticker} - Extreme Scenarios:")
            print(f"{'Probability':<20} {'Return':>12} {'Loss':>12}")
            print("-" * 45)

            for tau in extreme_tau:
                extreme_return = returns.quantile(tau)
                loss = -extreme_return * 100  # Convert to percentage loss

                print(
                    f"{tau*100:>6.3f}% (1 in {int(1/tau):<6}) "
                    f"{extreme_return:>11.4f}  {loss:>11.2f}%"
                )

            # Historical worst cases
            worst_5 = returns.nsmallest(5)
            print(f"\n  Historical Worst 5 Days:")
            for date, ret in worst_5.items():
                print(f"    {date.strftime('%Y-%m-%d')}: {ret:>8.4f} ({-ret*100:>6.2f}% loss)")

    def _plot_var(self, var_results):
        """Plot VaR evolution and violations."""
        n_tickers = len(self.tickers)
        fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 4 * n_tickers))

        if n_tickers == 1:
            axes = [axes]

        for ax, ticker in zip(axes, self.tickers):
            # Plot returns
            returns = self.returns[ticker]
            ax.plot(returns.index, returns, "gray", alpha=0.4, linewidth=0.5, label="Daily Returns")

            # Plot VaR lines
            colors = ["#2E86AB", "#E63946"]
            labels = []

            for (conf, var_series), color in zip(var_results[ticker].items(), colors):
                ax.plot(
                    var_series.index,
                    var_series,
                    color=color,
                    linewidth=1.5,
                    label=f"VaR {conf*100:.0f}%",
                    alpha=0.8,
                )

                # Highlight violations
                common_idx = var_series.index.intersection(returns.index)
                violations = returns.loc[common_idx] < var_series.loc[common_idx]
                violation_dates = violations[violations].index

                if len(violation_dates) > 0:
                    ax.scatter(
                        violation_dates,
                        returns.loc[violation_dates],
                        color=color,
                        s=30,
                        alpha=0.8,
                        zorder=5,
                        edgecolors="black",
                        linewidths=0.5,
                    )

            ax.set_title(f"{ticker} - Value at Risk Analysis", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date", fontsize=11)
            ax.set_ylabel("Returns", fontsize=11)
            ax.legend(loc="lower left", frameon=True, shadow=True)
            ax.grid(True, alpha=0.2)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))

        plt.tight_layout()
        output_file = self.output_dir / "var_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to {output_file}")
        plt.close()

    def generate_report(self):
        """Generate complete risk analysis report."""
        print("\n" + "=" * 70)
        print(" GENERATING RISK ANALYSIS REPORT")
        print("=" * 70)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Financial Risk Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        h2 {{
            color: #1e3a8a;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            background: #eff6ff;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }}
        .metric-label {{ font-size: 0.9em; color: #666; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #1e3a8a; }}
        img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .warning {{
            background-color: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 15px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #1e3a8a;
            color: white;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 50px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Financial Risk Analysis</h1>
        <p>Value at Risk and Expected Shortfall Analysis</p>
        <p>Period: {self.returns.index[0].strftime('%Y-%m-%d')} to {self.returns.index[-1].strftime('%Y-%m-%d')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report presents a comprehensive risk analysis using quantile regression methods
        to estimate Value at Risk (VaR) and Conditional Value at Risk (CVaR).</p>

        <div class="metric">
            <div class="metric-label">Assets Analyzed</div>
            <div class="metric-value">{len(self.tickers)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Trading Days</div>
            <div class="metric-value">{len(self.returns)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Data Source</div>
            <div class="metric-value">{self.data_source.title()}</div>
        </div>
    </div>

    <div class="section">
        <h2>Value at Risk Analysis</h2>
        <p>VaR estimates the maximum expected loss at a given confidence level under normal
        market conditions.</p>
        <img src="var_analysis.png" alt="VaR Analysis">
        <div class="warning">
            <strong>Note:</strong> VaR violations (dots) indicate days when actual losses
            exceeded the VaR estimate. The model is calibrated correctly if violations occur
            at approximately the expected rate (e.g., 5% for 95% VaR).
        </div>
    </div>

    <div class="section">
        <h2>Risk Metrics Summary</h2>
        <table>
            <tr>
                <th>Ticker</th>
                <th>95% VaR</th>
                <th>99% VaR</th>
                <th>CVaR (ES)</th>
            </tr>
"""

        # Add summary statistics
        for ticker in self.tickers:
            returns = self.returns[ticker]
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            cvar_95 = returns[returns <= var_95].mean()

            html += f"""
            <tr>
                <td><strong>{ticker}</strong></td>
                <td>{var_95*100:.2f}%</td>
                <td>{var_99*100:.2f}%</td>
                <td>{cvar_95*100:.2f}%</td>
            </tr>
"""

        html += """
        </table>
    </div>

    <div class="section">
        <h2>Methodology</h2>
        <h3>Value at Risk (VaR)</h3>
        <p>VaR at confidence level α is the quantile of the return distribution:</p>
        <p><em>VaR<sub>α</sub> = Q<sub>Returns</sub>(1-α)</em></p>
        <p>For example, 95% VaR is the 5th percentile of returns.</p>

        <h3>Conditional VaR (Expected Shortfall)</h3>
        <p>CVaR is the expected loss conditional on exceeding VaR:</p>
        <p><em>CVaR<sub>α</sub> = E[Returns | Returns ≤ VaR<sub>α</sub>]</em></p>
        <p>CVaR is a coherent risk measure and captures tail risk better than VaR.</p>

        <h3>Backtesting</h3>
        <p>The Kupiec test checks if the actual violation rate matches the expected rate.
        A p-value < 0.05 indicates model rejection.</p>
    </div>

    <div class="section">
        <h2>Limitations and Disclaimers</h2>
        <ul>
            <li>VaR assumes normal market conditions and may underestimate extreme events</li>
            <li>Past performance does not guarantee future results</li>
            <li>This analysis is for educational purposes only</li>
            <li>Always consult with financial professionals before making investment decisions</li>
        </ul>
    </div>

    <div class="footer">
        <p>Generated with PanelBox Quantile Regression Module</p>
        <p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
        """

        output_file = self.output_dir / "risk_analysis_report.html"
        with open(output_file, "w") as f:
            f.write(html)

        print(f"\nComplete report saved to {output_file}")
        return output_file


def main():
    """Run complete financial risk analysis."""
    print("\n" + "=" * 70)
    print(" FINANCIAL RISK ANALYSIS USING QUANTILE REGRESSION")
    print("=" * 70)

    # Initialize analysis with major indices
    tickers = ["SPY", "QQQ", "IWM"]  # S&P 500, Nasdaq, Russell 2000
    analysis = ValueAtRiskAnalysis(tickers)

    # VaR estimation
    var_results = analysis.estimate_var(confidence_levels=[0.95, 0.99])

    # CVaR estimation
    cvar_results = analysis.estimate_cvar(confidence_level=0.95)

    # Stress testing
    analysis.stress_testing()

    # Generate report
    analysis.generate_report()

    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nAll outputs saved to:", analysis.output_dir)


if __name__ == "__main__":
    main()
