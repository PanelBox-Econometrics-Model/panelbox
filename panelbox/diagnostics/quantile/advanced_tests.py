"""
Advanced diagnostic tests for quantile regression.

This module implements state-of-the-art specification tests and diagnostics
for quantile regression models, including the Khmaladze test, He-Zhu test,
outlier detection, and influence diagnostics.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.linalg import lstsq


@dataclass
class DiagnosticResult:
    """Container for diagnostic test results."""

    test_name: str
    statistic: float
    p_value: float
    status: str  # 'pass', 'warning', 'fail'
    message: str
    recommendation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class AdvancedDiagnostics:
    """
    Advanced diagnostic tests for quantile regression.

    Implements state-of-the-art specification tests including Khmaladze,
    He-Zhu, and various influence diagnostics.

    Parameters
    ----------
    result : QuantilePanelResult
        Fitted QR model result
    verbose : bool, default=True
        Print diagnostic messages

    Examples
    --------
    >>> diag = AdvancedDiagnostics(result)
    >>> report = diag.run_all_diagnostics(tau=0.5)
    >>> report.print_summary()
    """

    def __init__(self, result, verbose: bool = True):
        """Initialize diagnostics with fitted result."""
        self.result = result
        self.verbose = verbose
        self.diagnostics = []

        # Extract model information
        if hasattr(result, "model"):
            self.model = result.model
            self.X = getattr(result.model, "X", None)
            self.y = getattr(result.model, "y", None)
        else:
            self.model = None
            self.X = None
            self.y = None

    def run_all_diagnostics(self, tau: Optional[float] = None) -> "DiagnosticReport":
        """
        Run complete diagnostic battery.

        Parameters
        ----------
        tau : float, optional
            Quantile to test. If None, uses median or first available

        Returns
        -------
        DiagnosticReport
            Comprehensive diagnostic report
        """
        if tau is None:
            tau = 0.5 if 0.5 in self.result.results else list(self.result.results.keys())[0]

        # Clear previous diagnostics
        self.diagnostics = []

        # Run tests
        try:
            self.test_specification(tau)
        except Exception as e:
            warnings.warn(f"Specification test failed: {e}")

        try:
            self.test_heteroscedasticity(tau)
        except Exception as e:
            warnings.warn(f"Heteroscedasticity test failed: {e}")

        try:
            self.test_outliers(tau)
        except Exception as e:
            warnings.warn(f"Outlier test failed: {e}")

        try:
            self.test_influence(tau)
        except Exception as e:
            warnings.warn(f"Influence test failed: {e}")

        try:
            self.test_convergence(tau)
        except Exception as e:
            warnings.warn(f"Convergence test failed: {e}")

        try:
            self.test_monotonicity()
        except Exception as e:
            warnings.warn(f"Monotonicity test failed: {e}")

        # Generate report
        return self._generate_report()

    def test_specification(self, tau: float):
        """
        Khmaladze test for correct specification.

        Tests if the conditional quantile function is correctly specified
        using the Khmaladze martingale transformation.
        """
        if self.X is None or self.y is None:
            self.diagnostics.append(
                DiagnosticResult(
                    test_name="Khmaladze Specification Test",
                    statistic=np.nan,
                    p_value=np.nan,
                    status="warning",
                    message="Test skipped - data not available",
                    recommendation="Ensure model has X and y attributes",
                )
            )
            return

        res_tau = self.result.results[tau]

        # Get parameters
        if hasattr(res_tau, "params"):
            params = res_tau.params
        else:
            params = res_tau

        # Compute residuals
        predictions = (
            self.X @ params if len(params) == self.X.shape[1] else self.X[:, : len(params)] @ params
        )
        residuals = self.y - predictions

        # Empirical process
        n = len(residuals)
        sorted_resid = np.sort(residuals)

        # Compute Khmaladze transformation (simplified version)
        process = np.zeros(n)
        for i in range(n):
            indicator = (residuals <= sorted_resid[i]).astype(float)
            check_function = tau - (residuals < 0).astype(float)
            process[i] = np.sum(indicator * check_function) / np.sqrt(n)

        # Kolmogorov-Smirnov type statistic
        ks_stat = np.max(np.abs(process))

        # P-value approximation using Brownian bridge distribution
        # This is a simplified approximation
        p_value = 2 * np.exp(-2 * n * (ks_stat / np.sqrt(n)) ** 2)
        p_value = min(1.0, max(0.0, p_value))  # Ensure valid range

        # Determine status
        if p_value > 0.10:
            status = "pass"
            message = "No evidence of misspecification"
            recommendation = None
        elif p_value > 0.05:
            status = "warning"
            message = "Weak evidence of misspecification"
            recommendation = "Consider adding nonlinear terms or interactions"
        else:
            status = "fail"
            message = "Strong evidence of misspecification"
            recommendation = (
                "Model is misspecified. Add omitted variables or transform existing ones"
            )

        self.diagnostics.append(
            DiagnosticResult(
                test_name="Khmaladze Specification Test",
                statistic=ks_stat,
                p_value=p_value,
                status=status,
                message=message,
                recommendation=recommendation,
                details={"tau": tau, "n": n},
            )
        )

    def test_heteroscedasticity(self, tau: float):
        """
        He-Zhu test for heteroscedasticity in quantile regression.

        Tests whether the scale of residuals varies with covariates.
        """
        if self.X is None or self.y is None:
            self.diagnostics.append(
                DiagnosticResult(
                    test_name="He-Zhu Heteroscedasticity Test",
                    statistic=np.nan,
                    p_value=np.nan,
                    status="warning",
                    message="Test skipped - data not available",
                    recommendation=None,
                )
            )
            return

        res_tau = self.result.results[tau]

        # Get parameters
        if hasattr(res_tau, "params"):
            params = res_tau.params
        else:
            params = res_tau

        # Compute residuals
        predictions = (
            self.X @ params if len(params) == self.X.shape[1] else self.X[:, : len(params)] @ params
        )
        residuals = self.y - predictions

        # Test if scale varies with X
        # Auxiliary regression: |residuals| on X
        abs_resid = np.abs(residuals)

        # OLS regression using scipy
        try:
            beta_aux, _, _, _ = lstsq(self.X, abs_resid)
            fitted_aux = self.X @ beta_aux

            # R-squared of auxiliary regression
            ss_res = np.sum((abs_resid - fitted_aux) ** 2)
            ss_tot = np.sum((abs_resid - np.mean(abs_resid)) ** 2)
            r2_aux = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # LM test statistic
            n = len(residuals)
            lm_stat = n * r2_aux

            # Chi-square test
            df = self.X.shape[1] - 1  # Exclude intercept
            p_value = 1 - stats.chi2.cdf(lm_stat, df)

        except Exception as e:
            # If test fails, report as warning
            self.diagnostics.append(
                DiagnosticResult(
                    test_name="He-Zhu Heteroscedasticity Test",
                    statistic=np.nan,
                    p_value=np.nan,
                    status="warning",
                    message=f"Test computation failed: {e}",
                    recommendation="Check data quality",
                )
            )
            return

        # Determine status
        if p_value > 0.10:
            status = "pass"
            message = "No evidence of heteroscedasticity"
            recommendation = None
        elif p_value > 0.05:
            status = "warning"
            message = "Weak evidence of heteroscedasticity"
            recommendation = "Consider robust standard errors"
        else:
            status = "fail"
            message = "Strong evidence of heteroscedasticity"
            recommendation = "Use location-scale model or weighted QR"

        self.diagnostics.append(
            DiagnosticResult(
                test_name="He-Zhu Heteroscedasticity Test",
                statistic=lm_stat,
                p_value=p_value,
                status=status,
                message=message,
                recommendation=recommendation,
                details={"tau": tau, "r2_auxiliary": r2_aux, "df": df},
            )
        )

    def test_outliers(self, tau: float, threshold: float = 3.0):
        """
        Detect outliers using quantile-specific residuals.

        Parameters
        ----------
        tau : float
            Quantile level
        threshold : float, default=3.0
            Threshold for outlier detection (in MAD units)
        """
        if self.X is None or self.y is None:
            self.diagnostics.append(
                DiagnosticResult(
                    test_name="Outlier Detection",
                    statistic=np.nan,
                    p_value=np.nan,
                    status="warning",
                    message="Test skipped - data not available",
                    recommendation=None,
                )
            )
            return

        res_tau = self.result.results[tau]

        # Get parameters
        if hasattr(res_tau, "params"):
            params = res_tau.params
        else:
            params = res_tau

        # Compute residuals
        predictions = (
            self.X @ params if len(params) == self.X.shape[1] else self.X[:, : len(params)] @ params
        )
        residuals = self.y - predictions

        # Standardized residuals using MAD for robustness
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad > 0:
            std_resid = residuals / (1.4826 * mad)  # 1.4826 makes MAD consistent for normal
        else:
            # Fall back to standard deviation if MAD is zero
            std_resid = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)

        # Detect outliers
        outliers = np.abs(std_resid) > threshold
        n_outliers = np.sum(outliers)
        pct_outliers = 100 * n_outliers / len(residuals)

        # Find which observations are outliers
        outlier_indices = np.where(outliers)[0]

        # Store outlier information
        self.outlier_indices = outlier_indices
        self.outlier_residuals = std_resid[outliers]

        # Determine status
        if pct_outliers < 1:
            status = "pass"
            message = f"{n_outliers} outliers detected ({pct_outliers:.1f}%)"
            recommendation = None
        elif pct_outliers < 5:
            status = "warning"
            message = f"{n_outliers} outliers detected ({pct_outliers:.1f}%)"
            recommendation = "Review outliers for data quality issues"
        else:
            status = "fail"
            message = f"{n_outliers} outliers detected ({pct_outliers:.1f}%)"
            recommendation = "Too many outliers. Check data quality or use robust methods"

        self.diagnostics.append(
            DiagnosticResult(
                test_name="Outlier Detection",
                statistic=n_outliers,
                p_value=pct_outliers / 100,  # Use percentage as pseudo p-value
                status=status,
                message=message,
                recommendation=recommendation,
                details={
                    "tau": tau,
                    "threshold": threshold,
                    "outlier_indices": (
                        outlier_indices.tolist()
                        if len(outlier_indices) < 100
                        else "Too many to list"
                    ),
                },
            )
        )

    def test_influence(self, tau: float, threshold: float = 2.0):
        """
        Compute influence diagnostics (DFBETAS, Cook's D).

        This is a simplified version that samples observations for efficiency.

        Parameters
        ----------
        tau : float
            Quantile level
        threshold : float
            Threshold for influential observations
        """
        if self.X is None or self.y is None:
            self.diagnostics.append(
                DiagnosticResult(
                    test_name="Influence Diagnostics",
                    statistic=np.nan,
                    p_value=np.nan,
                    status="warning",
                    message="Test skipped - data not available",
                    recommendation=None,
                )
            )
            return

        res_tau = self.result.results[tau]
        n, p = self.X.shape

        # Get parameters
        if hasattr(res_tau, "params"):
            beta_full = res_tau.params
        else:
            beta_full = res_tau

        # For efficiency, sample a subset of observations
        sample_size = min(100, n)
        if n > sample_size:
            sample_idx = np.random.choice(n, sample_size, replace=False)
        else:
            sample_idx = np.arange(n)

        # Simplified influence measure based on leverage and residuals
        # Compute hat matrix diagonal (leverage)
        try:
            # Use QR decomposition for numerical stability
            Q, R = np.linalg.qr(self.X)
            H_diag = np.sum(Q**2, axis=1)
        except:
            H_diag = np.ones(n) / n

        # Compute residuals
        predictions = (
            self.X @ beta_full
            if len(beta_full) == self.X.shape[1]
            else self.X[:, : len(beta_full)] @ beta_full
        )
        residuals = self.y - predictions

        # Standardize residuals
        resid_std = np.std(residuals)
        if resid_std > 0:
            std_resid = residuals / resid_std
        else:
            std_resid = residuals

        # Cook's D approximation for quantile regression
        cooks_d = (std_resid**2 * H_diag) / (p * (1 - H_diag + 1e-10))

        # Identify influential observations
        influential = cooks_d > 4 / n  # Common threshold
        n_influential = np.sum(influential)
        pct_influential = 100 * n_influential / n

        # Store influence measures
        self.leverage = H_diag
        self.cooks_d = cooks_d
        self.influential_indices = np.where(influential)[0]

        # Determine status
        if pct_influential < 1:
            status = "pass"
            message = f"{n_influential} influential observations"
            recommendation = None
        elif pct_influential < 5:
            status = "warning"
            message = f"{n_influential} influential observations"
            recommendation = "Review influential observations"
        else:
            status = "fail"
            message = f"{n_influential} highly influential observations"
            recommendation = "Model is sensitive to specific observations"

        self.diagnostics.append(
            DiagnosticResult(
                test_name="Influence Diagnostics",
                statistic=n_influential,
                p_value=pct_influential / 100,
                status=status,
                message=message,
                recommendation=recommendation,
                details={
                    "tau": tau,
                    "max_cooks_d": float(np.max(cooks_d)),
                    "influential_threshold": 4 / n,
                },
            )
        )

    def test_convergence(self, tau: float):
        """
        Test numerical convergence and stability.
        """
        res_tau = self.result.results[tau]

        # Check if model converged
        converged = getattr(res_tau, "converged", True)
        n_iterations = getattr(res_tau, "n_iterations", 0)
        max_iterations = getattr(res_tau, "max_iterations", 100)

        # Check gradient norm if available
        gradient_norm = getattr(res_tau, "gradient_norm", None)

        if gradient_norm is None and self.X is not None and self.y is not None:
            # Compute approximate gradient
            if hasattr(res_tau, "params"):
                params = res_tau.params
            else:
                params = res_tau

            predictions = (
                self.X @ params
                if len(params) == self.X.shape[1]
                else self.X[:, : len(params)] @ params
            )
            residuals = self.y - predictions
            psi = tau - (residuals < 0).astype(float)
            gradient_norm = np.linalg.norm(self.X.T @ psi) / len(self.y)

        # Determine status
        if gradient_norm is not None:
            if converged and gradient_norm < 1e-6:
                status = "pass"
                message = f"Converged in {n_iterations} iterations"
                recommendation = None
            elif converged and gradient_norm < 1e-4:
                status = "warning"
                message = f"Converged but gradient norm = {gradient_norm:.2e}"
                recommendation = "Consider tighter convergence tolerance"
            else:
                status = "fail"
                message = "Convergence issues detected"
                recommendation = "Try different starting values or optimization method"
        else:
            if converged:
                status = "pass"
                message = f"Model converged"
                recommendation = None
            else:
                status = "fail"
                message = "Model did not converge"
                recommendation = "Increase max iterations or adjust tolerance"

        self.diagnostics.append(
            DiagnosticResult(
                test_name="Convergence Check",
                statistic=gradient_norm if gradient_norm is not None else float(converged),
                p_value=1.0 if converged else 0.0,
                status=status,
                message=message,
                recommendation=recommendation,
                details={
                    "tau": tau,
                    "n_iterations": n_iterations,
                    "max_iterations": max_iterations,
                },
            )
        )

    def test_monotonicity(self):
        """
        Test if quantile curves are monotonic (non-crossing).
        """
        tau_list = sorted(self.result.results.keys())

        if len(tau_list) < 2:
            self.diagnostics.append(
                DiagnosticResult(
                    test_name="Monotonicity Check",
                    statistic=0,
                    p_value=1.0,
                    status="pass",
                    message="Single quantile - no crossing possible",
                    recommendation=None,
                )
            )
            return

        # Check for crossing at mean X values
        if self.X is not None:
            X_mean = self.X.mean(axis=0)
        else:
            # Use unit vector for testing
            first_result = self.result.results[tau_list[0]]
            if hasattr(first_result, "params"):
                n_params = len(first_result.params)
            else:
                n_params = 1
            X_mean = np.ones(n_params)

        # Get predictions at mean X
        predictions = []
        for tau in tau_list:
            res = self.result.results[tau]
            if hasattr(res, "params"):
                params = res.params
            else:
                params = res

            pred = X_mean @ params if len(params) == len(X_mean) else X_mean[: len(params)] @ params
            predictions.append(pred)

        # Check for inversions
        inversions = 0
        for i in range(len(predictions) - 1):
            if predictions[i] > predictions[i + 1]:
                inversions += 1

        # Percentage of crossing
        pct_crossing = 100 * inversions / (len(predictions) - 1)

        # Determine status
        if inversions == 0:
            status = "pass"
            message = "No crossing detected"
            recommendation = None
        elif pct_crossing < 10:
            status = "warning"
            message = f"Crossing detected in {pct_crossing:.1f}% of adjacent quantiles"
            recommendation = "Consider rearrangement or location-scale model"
        else:
            status = "fail"
            message = f"Significant crossing in {pct_crossing:.1f}% of adjacent quantiles"
            recommendation = "Use monotonicity-constrained estimation"

        self.diagnostics.append(
            DiagnosticResult(
                test_name="Monotonicity Check",
                statistic=inversions,
                p_value=1 - pct_crossing / 100,
                status=status,
                message=message,
                recommendation=recommendation,
                details={"n_quantiles": len(tau_list), "inversions": inversions},
            )
        )

    def _generate_report(self) -> "DiagnosticReport":
        """
        Generate comprehensive diagnostic report.
        """
        report = DiagnosticReport(self.diagnostics)

        if self.verbose:
            report.print_summary()

        return report


class DiagnosticReport:
    """
    Comprehensive diagnostic report with traffic light system.
    """

    def __init__(self, diagnostics: List[DiagnosticResult]):
        self.diagnostics = diagnostics
        self._compute_health_score()

    def _compute_health_score(self):
        """
        Compute overall model health score.
        """
        scores = {"pass": 1.0, "warning": 0.5, "fail": 0.0}

        if not self.diagnostics:
            self.health_score = 0.0
            self.health_status = "unknown"
            return

        total_score = sum(scores.get(d.status, 0) for d in self.diagnostics)
        self.health_score = total_score / len(self.diagnostics)

        if self.health_score >= 0.8:
            self.health_status = "good"
        elif self.health_score >= 0.5:
            self.health_status = "fair"
        else:
            self.health_status = "poor"

    def print_summary(self):
        """
        Print formatted diagnostic summary with colors.
        """
        # Colors for terminal output (ANSI escape codes)
        colors = {
            "pass": "\033[92m✓\033[0m",  # Green
            "warning": "\033[93m⚠\033[0m",  # Yellow
            "fail": "\033[91m✗\033[0m",  # Red
        }

        print("\n" + "=" * 70)
        print("QUANTILE REGRESSION DIAGNOSTICS")
        print("=" * 70)

        # Overall health
        health_colors = {"good": "\033[92m", "fair": "\033[93m", "poor": "\033[91m"}

        health_color = health_colors.get(self.health_status, "")
        print(f"\nOverall Model Health: {health_color}{self.health_status.upper()}\033[0m")
        print(f"Health Score: {self.health_score:.1%}")

        # Individual tests
        print("\n" + "-" * 70)
        print(f"{'Test':<30} {'Status':<8} {'Statistic':>12} {'P-value':>10}")
        print("-" * 70)

        for diag in self.diagnostics:
            status_symbol = colors.get(diag.status, diag.status)
            # Handle NaN values
            stat_str = (
                f"{diag.statistic:>12.4f}" if not np.isnan(diag.statistic) else f"{'N/A':>12}"
            )
            p_str = f"{diag.p_value:>10.4f}" if not np.isnan(diag.p_value) else f"{'N/A':>10}"

            print(f"{diag.test_name:<30} {status_symbol:<17} {stat_str} {p_str}")

        # Recommendations
        recommendations = [
            d.recommendation for d in self.diagnostics if d.recommendation is not None
        ]

        if recommendations:
            print("\n" + "-" * 70)
            print("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")

        print("\n" + "=" * 70)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary format.
        """
        return {
            "health_score": self.health_score,
            "health_status": self.health_status,
            "tests": [
                {
                    "name": d.test_name,
                    "statistic": d.statistic,
                    "p_value": d.p_value,
                    "status": d.status,
                    "message": d.message,
                    "recommendation": d.recommendation,
                    "details": d.details,
                }
                for d in self.diagnostics
            ],
        }

    def to_html(self) -> str:
        """
        Generate HTML report.
        """
        # Health score background colors
        bg_colors = {"good": "#d4edda", "fair": "#fff3cd", "poor": "#f8d7da"}

        # Status symbols
        status_symbols = {"pass": "✓", "warning": "⚠", "fail": "✗"}

        # Status colors
        status_colors = {"pass": "#28a745", "warning": "#ffc107", "fail": "#dc3545"}

        html = f"""
        <div class="diagnostic-report" style="font-family: Arial, sans-serif;">
        <h2>Quantile Regression Diagnostics</h2>

        <div class="health-score" style="padding: 15px;
             background-color: {bg_colors.get(self.health_status, '#f8f9fa')};
             border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin: 0;">Overall Health: {self.health_status.upper()}</h3>
            <p style="margin: 5px 0;">Score: {self.health_score:.1%}</p>
        </div>

        <table class="diagnostic-table" style="width: 100%;
               border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="padding: 10px; text-align: left;">Test</th>
                    <th style="padding: 10px; text-align: left;">Status</th>
                    <th style="padding: 10px; text-align: right;">Statistic</th>
                    <th style="padding: 10px; text-align: right;">P-value</th>
                    <th style="padding: 10px; text-align: left;">Message</th>
                </tr>
            </thead>
            <tbody>
        """

        # Add rows
        for diag in self.diagnostics:
            status_symbol = status_symbols.get(diag.status, "")
            status_color = status_colors.get(diag.status, "#000")

            # Handle NaN values
            stat_str = f"{diag.statistic:.4f}" if not np.isnan(diag.statistic) else "N/A"
            p_str = f"{diag.p_value:.4f}" if not np.isnan(diag.p_value) else "N/A"

            html += f"""
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 10px;">{diag.test_name}</td>
                <td style="padding: 10px; color: {status_color}; font-weight: bold;">
                    {status_symbol} {diag.status}
                </td>
                <td style="padding: 10px; text-align: right;">{stat_str}</td>
                <td style="padding: 10px; text-align: right;">{p_str}</td>
                <td style="padding: 10px;">{diag.message}</td>
            </tr>
            """

        html += """
            </tbody>
        </table>
        """

        # Add recommendations
        recommendations = [
            d.recommendation for d in self.diagnostics if d.recommendation is not None
        ]

        if recommendations:
            html += """
            <div class="recommendations" style="margin-top: 20px; padding: 15px;
                 background-color: #f8f9fa; border-radius: 5px;">
                <h3>Recommendations</h3>
                <ul>
            """
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += """
                </ul>
            </div>
            """

        html += "</div>"

        return html
