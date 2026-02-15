# examples/quantile/financial_risk/var_analysis.py
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from panelbox.models.quantile import DynamicQuantile

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

    def __init__(self, tickers, start_date="2018-01-01", end_date="2023-12-31"):
        """Load financial data."""
        self.tickers = tickers
        self.data = self._load_market_data(start_date, end_date)
        self.returns = self._calculate_returns()

    def _load_market_data(self, start, end):
        """Download market data from Yahoo Finance."""
        data = yf.download(self.tickers, start=start, end=end)["Adj Close"]
        return data

    def _calculate_returns(self):
        """Calculate log returns."""
        returns = np.log(self.data / self.data.shift(1))
        returns = returns.dropna()
        return returns

    def estimate_var(self, confidence_levels=[0.95, 0.99], window=250):
        """
        Estimate Value at Risk using quantile regression.

        Parameters
        ----------
        confidence_levels : list
            Confidence levels for VaR (e.g., 0.95 = 5% VaR)
        window : int
            Rolling window size
        """
        print("\n" + "=" * 60)
        print("VALUE AT RISK ESTIMATION")
        print("=" * 60)

        var_results = {}

        for ticker in self.tickers:
            returns = self.returns[ticker]

            # Prepare data with lags
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
                tau = 1 - conf  # Left tail

                # Dynamic quantile regression
                model = DynamicQuantile(
                    data=data,
                    formula="returns ~ returns_lag1 + returns_lag2 + volatility",
                    tau=tau,
                    lags=0,  # Lags already in formula
                )

                result = model.fit(verbose=False)

                # Rolling VaR forecast
                var_forecast = self._rolling_var(data, result, window)

                ticker_var[conf] = var_forecast

                print(f"\n{ticker} - {conf*100:.0f}% VaR:")
                print(f"  Mean VaR: {var_forecast.mean():.4f}")
                print(f"  Max VaR: {var_forecast.min():.4f}")
                print(f"  Current VaR: {var_forecast.iloc[-1]:.4f}")

            var_results[ticker] = ticker_var

        # Backtesting
        self._backtest_var(var_results)

        # Visualization
        self._plot_var(var_results)

        return var_results

    def _rolling_var(self, data, model_result, window):
        """Calculate rolling VaR forecasts."""
        var_series = pd.Series(index=data.index[window:])

        for i in range(window, len(data)):
            # Use past window for prediction
            X_pred = data.iloc[i - 1 : i][["returns_lag1", "returns_lag2", "volatility"]]
            var_pred = model_result.predict(X_pred)
            var_series.iloc[i - window] = var_pred[0]

        return var_series

    def _backtest_var(self, var_results):
        """
        Backtest VaR estimates using Kupiec test.
        """
        print("\n" + "=" * 60)
        print("VAR BACKTESTING RESULTS")
        print("=" * 60)

        for ticker in self.tickers:
            print(f"\n{ticker}:")

            for conf, var_series in var_results[ticker].items():
                # Align returns and VaR
                common_idx = var_series.index.intersection(self.returns[ticker].index)
                actual_returns = self.returns[ticker].loc[common_idx]
                var_values = var_series.loc[common_idx]

                # Count violations
                violations = actual_returns < var_values
                n_violations = violations.sum()
                violation_rate = n_violations / len(violations)

                # Expected violation rate
                expected_rate = 1 - conf

                # Kupiec test (Proportion of Failures)
                from scipy.stats import binom

                p_value = binom.test(n_violations, len(violations), expected_rate)

                print(f"\n  {conf*100:.0f}% VaR Backtest:")
                print(f"    Violations: {n_violations}/{len(violations)}")
                print(f"    Violation Rate: {violation_rate:.4f}")
                print(f"    Expected Rate: {expected_rate:.4f}")
                print(f"    Kupiec Test p-value: {p_value:.4f}")

                if p_value < 0.05:
                    print("    → Model rejected at 5% level")
                else:
                    print("    → Model not rejected")

    def estimate_cvar(self, confidence_level=0.95):
        """
        Estimate Conditional Value at Risk (Expected Shortfall).
        """
        print("\n" + "=" * 60)
        print("CONDITIONAL VAR (EXPECTED SHORTFALL)")
        print("=" * 60)

        cvar_results = {}

        for ticker in self.tickers:
            returns = self.returns[ticker]

            # Estimate quantiles below VaR level
            tau_list = np.linspace(0.001, 1 - confidence_level, 50)

            quantiles = []
            for tau in tau_list:
                q = returns.quantile(tau)
                quantiles.append(q)

            # CVaR is average of quantiles below VaR
            cvar = np.mean(quantiles)
            var = returns.quantile(1 - confidence_level)

            cvar_results[ticker] = {
                "VaR": var,
                "CVaR": cvar,
                "CVaR/VaR": cvar / var if var != 0 else np.inf,
            }

            print(f"\n{ticker}:")
            print(f"  VaR ({confidence_level*100:.0f}%): {var:.4f}")
            print(f"  CVaR: {cvar:.4f}")
            print(f"  CVaR/VaR Ratio: {cvar/var:.2f}")

        return cvar_results

    def stress_testing(self):
        """
        Stress testing using extreme quantiles.
        """
        print("\n" + "=" * 60)
        print("STRESS TESTING ANALYSIS")
        print("=" * 60)

        # Extreme quantiles
        extreme_tau = [0.001, 0.005, 0.01]

        for ticker in self.tickers:
            returns = self.returns[ticker]

            print(f"\n{ticker} - Extreme Scenarios:")
            print(f"{'Probability':<15} {'Return':<10} {'Loss':>10}")
            print("-" * 35)

            for tau in extreme_tau:
                extreme_return = returns.quantile(tau)
                loss = -extreme_return * 100  # Convert to percentage loss

                print(
                    f"{tau*100:>6.2f}% (1 in {int(1/tau):<4}) "
                    f"{extreme_return:>9.4f} {loss:>9.2f}%"
                )

            # Historical worst cases
            worst_5 = returns.nsmallest(5)
            print(f"\nHistorical Worst Days:")
            for date, ret in worst_5.items():
                print(f"  {date.strftime('%Y-%m-%d')}: {ret:.4f} ({-ret*100:.2f}% loss)")

    def _plot_var(self, var_results):
        """Plot VaR evolution."""
        import matplotlib.pyplot as plt

        n_tickers = len(self.tickers)
        fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 4 * n_tickers))

        if n_tickers == 1:
            axes = [axes]

        for ax, ticker in zip(axes, self.tickers):
            # Plot returns
            returns = self.returns[ticker]
            ax.plot(returns.index, returns, "gray", alpha=0.5, linewidth=0.5, label="Returns")

            # Plot VaR lines
            colors = ["blue", "red"]
            for (conf, var_series), color in zip(var_results[ticker].items(), colors):
                ax.plot(
                    var_series.index,
                    var_series,
                    color=color,
                    linewidth=1.5,
                    label=f"VaR {conf*100:.0f}%",
                )

                # Highlight violations
                common_idx = var_series.index.intersection(returns.index)
                violations = returns.loc[common_idx] < var_series.loc[common_idx]
                violation_dates = violations[violations].index

                ax.scatter(
                    violation_dates,
                    returns.loc[violation_dates],
                    color=color,
                    s=20,
                    alpha=0.7,
                    zorder=5,
                )

            ax.set_title(f"{ticker} - Value at Risk", fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Returns")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Analyze major indices
    tickers = ["SPY", "QQQ", "IWM"]  # S&P 500, Nasdaq, Russell 2000

    analysis = ValueAtRiskAnalysis(tickers)

    # VaR estimation
    var_results = analysis.estimate_var()

    # CVaR estimation
    cvar_results = analysis.estimate_cvar()

    # Stress testing
    analysis.stress_testing()
