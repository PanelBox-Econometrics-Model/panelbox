"""
Quantile Treatment Effects for Panel Data.

Implements various methods for estimating quantile treatment effects (QTE)
including standard QTE, unconditional QTE via RIF regression, difference-in-differences
QR, and changes-in-changes methods.

References:
    Firpo, S., Fortin, N. M., & Lemieux, T. (2009). Unconditional quantile
    regressions. Econometrica, 77(3), 953-973.

    Athey, S., & Imbens, G. W. (2006). Identification and inference in nonlinear
    difference‐in‐differences models. Econometrica, 74(2), 431-497.

    Callaway, B., & Li, T. (2019). Quantile treatment effects in difference in
    differences models with panel data. Quantitative Economics, 10(4), 1579-1618.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq

from .base import QuantilePanelModel


class QuantileTreatmentEffects:
    """
    Quantile Treatment Effects (QTE) estimation.

    Implements various QTE methods:
    - Standard QTE for binary treatment
    - Difference-in-differences QR
    - Changes-in-changes (Athey & Imbens)
    - Unconditional QTE via RIF regression
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, "PanelData"],
        outcome: str,
        treatment: str,
        covariates: Optional[List[str]] = None,
        entity_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        data : PanelData or DataFrame
            Panel or cross-sectional data
        outcome : str
            Outcome variable name
        treatment : str
            Treatment variable name (binary)
        covariates : list, optional
            Control variables
        entity_col : str, optional
            Entity identifier for panel data
        time_col : str, optional
            Time identifier for panel data
        """
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
        self.covariates = covariates or []
        self.entity_col = entity_col
        self.time_col = time_col

        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for QTE estimation."""
        # Handle PanelData vs DataFrame
        if hasattr(self.data, "_data"):
            # PanelData object
            df = self.data._data
            self.entity_col = self.entity_col or self.data.entity_col
            self.time_col = self.time_col or self.data.time_col
        else:
            # Regular DataFrame
            df = self.data

        self.y = df[self.outcome].values
        self.d = df[self.treatment].values

        if self.covariates:
            self.X = df[self.covariates].values
        else:
            self.X = np.ones((len(self.y), 1))  # Intercept only

        # Store entity and time if available
        if self.entity_col:
            self.entities = df[self.entity_col].values
        else:
            self.entities = None

        if self.time_col:
            self.time = df[self.time_col].values
        else:
            self.time = None

        # Check treatment is binary
        unique_d = np.unique(self.d)
        if len(unique_d) != 2:
            if not all(d in [0, 1] for d in unique_d):
                # Try to convert to binary
                self.d = (self.d > np.median(self.d)).astype(int)
                warnings.warn("Treatment variable converted to binary")

        self.n_treated = np.sum(self.d == 1)
        self.n_control = np.sum(self.d == 0)

    def estimate_qte(
        self, tau: Union[float, np.ndarray] = 0.5, method: str = "standard", **kwargs
    ) -> "QTEResult":
        """
        Estimate Quantile Treatment Effects.

        Parameters
        ----------
        tau : float or array
            Quantile(s)
        method : str
            'standard': QTE with covariates
            'unconditional': Unconditional QTE via RIF
            'did': Difference-in-differences QR
            'cic': Changes-in-changes

        Returns
        -------
        QTEResult
            Estimated QTE results
        """
        tau = np.atleast_1d(tau)

        if method == "standard":
            return self._qte_standard(tau, **kwargs)
        elif method == "unconditional":
            return self._qte_unconditional(tau, **kwargs)
        elif method == "did":
            return self._qte_did(tau, **kwargs)
        elif method == "cic":
            return self._qte_cic(tau, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _qte_standard(
        self,
        tau_list: np.ndarray,
        bootstrap: bool = True,
        n_boot: int = 999,
        cluster_bootstrap: bool = False,
    ) -> "QTEResult":
        """
        Standard QTE: difference in conditional quantiles.

        QTE(τ) = Q_Y(τ|D=1,X) - Q_Y(τ|D=0,X)
        """
        from ...optimization.quantile.interior_point import frisch_newton_qr

        qte_results = {}

        for tau in tau_list:
            # Include treatment and interaction with covariates
            X_full = np.column_stack([self.X, self.d[:, np.newaxis]])

            # Add interactions if covariates present
            if self.X.shape[1] > 1:  # More than just intercept
                X_interactions = self.X * self.d[:, np.newaxis]
                X_full = np.column_stack([X_full, X_interactions])

            # Estimate QR
            beta, info = frisch_newton_qr(X_full, self.y, tau)

            # QTE is coefficient on treatment (plus average interaction effect)
            qte = beta[self.X.shape[1]]  # Main treatment effect

            if self.X.shape[1] > 1 and len(beta) > self.X.shape[1] + 1:
                # Add average interaction effects
                X_mean = np.mean(self.X[:, 1:], axis=0)  # Exclude intercept
                interaction_effects = beta[self.X.shape[1] + 1 :] * X_mean
                qte += np.sum(interaction_effects)

            # Bootstrap for inference
            if bootstrap:
                qte_boot = self._bootstrap_qte(tau, n_boot, cluster_bootstrap)
                se_qte = np.std(qte_boot)
                ci_lower = np.percentile(qte_boot, 2.5)
                ci_upper = np.percentile(qte_boot, 97.5)
            else:
                se_qte = None
                ci_lower = None
                ci_upper = None

            qte_results[tau] = {
                "qte": qte,
                "se": se_qte,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "beta_full": beta,
                "n_treated": self.n_treated,
                "n_control": self.n_control,
            }

        return QTEResult(qte_results, method="standard")

    def _qte_unconditional(
        self, tau_list: np.ndarray, bandwidth: Optional[float] = None
    ) -> "QTEResult":
        """
        Unconditional QTE via Recentered Influence Function (RIF).

        Firpo, Fortin, Lemieux (2009) approach.
        """
        qte_results = {}

        for tau in tau_list:
            # Compute unconditional quantile
            q_tau = np.quantile(self.y, tau)

            # Estimate density at quantile
            f_q = self._density_at_quantile(q_tau, bandwidth)

            # RIF for quantile
            rif = q_tau + (tau - (self.y <= q_tau)) / f_q

            # Regress RIF on treatment and covariates
            X_rif = np.column_stack([self.X, self.d[:, np.newaxis]])

            # OLS regression
            beta_rif = lstsq(X_rif, rif, rcond=None)[0]

            # QTE is coefficient on treatment
            qte = beta_rif[-1]

            # Standard error via heteroskedasticity-robust formula
            residuals = rif - X_rif @ beta_rif
            meat = X_rif.T @ np.diag(residuals**2) @ X_rif
            bread = np.linalg.inv(X_rif.T @ X_rif)
            vcov = bread @ meat @ bread / len(rif)
            se_qte = np.sqrt(vcov[-1, -1])

            qte_results[tau] = {
                "qte": qte,
                "se": se_qte,
                "ci_lower": qte - 1.96 * se_qte,
                "ci_upper": qte + 1.96 * se_qte,
                "unconditional_quantile": q_tau,
                "beta_rif": beta_rif,
                "density": f_q,
            }

        return QTEResult(qte_results, method="unconditional")

    def _density_at_quantile(self, q: float, bandwidth: Optional[float] = None) -> float:
        """Estimate density at quantile using kernel method."""
        if bandwidth is None:
            # Silverman's rule of thumb
            bandwidth = 1.06 * np.std(self.y) * len(self.y) ** (-1 / 5)

        # Gaussian kernel density at q
        kernel = np.exp(-0.5 * ((self.y - q) / bandwidth) ** 2)
        density = np.mean(kernel) / (bandwidth * np.sqrt(2 * np.pi))

        return max(density, 1e-6)  # Bound away from zero

    def _qte_did(
        self,
        tau_list: np.ndarray,
        pre_post_cutoff: Optional[Any] = None,
        bootstrap: bool = True,
        n_boot: int = 999,
    ) -> "QTEResult":
        """
        Difference-in-differences QR.

        QTE_DiD(τ) = [Q_Y(τ|D=1,T=1) - Q_Y(τ|D=1,T=0)] -
                     [Q_Y(τ|D=0,T=1) - Q_Y(τ|D=0,T=0)]
        """
        if self.time is None:
            raise ValueError("Time variable required for DiD")

        # Determine pre/post periods
        if pre_post_cutoff is None:
            pre_post_cutoff = np.median(self.time)

        post_period = self.time > pre_post_cutoff
        pre_period = ~post_period

        qte_results = {}

        for tau in tau_list:
            # Compute quantiles for each group-period
            # Treated-Post
            mask_11 = (self.d == 1) & post_period
            q11 = np.quantile(self.y[mask_11], tau) if np.sum(mask_11) > 0 else np.nan

            # Treated-Pre
            mask_10 = (self.d == 1) & pre_period
            q10 = np.quantile(self.y[mask_10], tau) if np.sum(mask_10) > 0 else np.nan

            # Control-Post
            mask_01 = (self.d == 0) & post_period
            q01 = np.quantile(self.y[mask_01], tau) if np.sum(mask_01) > 0 else np.nan

            # Control-Pre
            mask_00 = (self.d == 0) & pre_period
            q00 = np.quantile(self.y[mask_00], tau) if np.sum(mask_00) > 0 else np.nan

            # DiD estimate
            qte_did = (q11 - q10) - (q01 - q00)

            # Bootstrap for inference
            if bootstrap:
                qte_boot = self._bootstrap_did(tau, pre_post_cutoff, n_boot)
                se_qte = np.std(qte_boot) if len(qte_boot) > 0 else np.nan
                ci_lower = np.percentile(qte_boot, 2.5) if len(qte_boot) > 0 else np.nan
                ci_upper = np.percentile(qte_boot, 97.5) if len(qte_boot) > 0 else np.nan
            else:
                se_qte = None
                ci_lower = None
                ci_upper = None

            qte_results[tau] = {
                "qte": qte_did,
                "se": se_qte,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "treated_change": q11 - q10,
                "control_change": q01 - q00,
                "q11": q11,
                "q10": q10,
                "q01": q01,
                "q00": q00,
            }

        return QTEResult(qte_results, method="did")

    def _qte_cic(self, tau_list: np.ndarray, n_grid: int = 100) -> "QTEResult":
        """
        Changes-in-Changes (Athey & Imbens 2006).

        Allows for nonlinear effects of unobserved heterogeneity.
        """
        if self.time is None:
            raise ValueError("Time variable required for CiC")

        # Determine pre/post periods
        pre_post_cutoff = np.median(self.time)
        post_period = self.time > pre_post_cutoff
        pre_period = ~post_period

        qte_results = {}

        # Get outcome distributions for each group-period
        y_11 = self.y[(self.d == 1) & post_period]  # Treated-Post
        y_10 = self.y[(self.d == 1) & pre_period]  # Treated-Pre
        y_01 = self.y[(self.d == 0) & post_period]  # Control-Post
        y_00 = self.y[(self.d == 0) & pre_period]  # Control-Pre

        for tau in tau_list:
            # CiC estimator
            # Step 1: Get quantile of control group in period 0
            q_00 = np.quantile(y_00, tau)

            # Step 2: Get corresponding quantile rank in period 1 control
            rank_01 = np.mean(y_01 <= q_00)

            # Step 3: Get quantile at this rank for period 1 control
            q_01_adjusted = np.quantile(y_01, rank_01)

            # Step 4: Compute counterfactual change for treated
            change_control = q_01_adjusted - q_00

            # Step 5: Apply to treated group
            q_10 = np.quantile(y_10, tau)
            q_11_counterfactual = q_10 + change_control

            # Step 6: Get actual treated outcome
            q_11 = np.quantile(y_11, tau)

            # CiC estimate
            qte_cic = q_11 - q_11_counterfactual

            qte_results[tau] = {
                "qte": qte_cic,
                "se": None,  # Complex to compute
                "ci_lower": None,
                "ci_upper": None,
                "q_11": q_11,
                "q_11_counterfactual": q_11_counterfactual,
                "change_control": change_control,
            }

        return QTEResult(qte_results, method="cic")

    def _bootstrap_qte(self, tau: float, n_boot: int, cluster: bool = False) -> np.ndarray:
        """Bootstrap QTE for inference."""
        from ...optimization.quantile.interior_point import frisch_newton_qr

        qte_boot = []

        for _ in range(n_boot):
            # Resample
            if cluster and self.entities is not None:
                # Cluster bootstrap by entity
                unique_entities = np.unique(self.entities)
                boot_entities = np.random.choice(
                    unique_entities, len(unique_entities), replace=True
                )
                idx = np.concatenate([np.where(self.entities == e)[0] for e in boot_entities])
            else:
                # Simple bootstrap
                idx = np.random.choice(len(self.y), len(self.y), replace=True)

            y_boot = self.y[idx]
            d_boot = self.d[idx]
            X_boot = self.X[idx]

            # Estimate QTE on bootstrap sample
            X_full = np.column_stack([X_boot, d_boot[:, np.newaxis]])

            try:
                beta_boot, _ = frisch_newton_qr(X_full, y_boot, tau, max_iter=50, verbose=False)
                qte_boot.append(beta_boot[self.X.shape[1]])
            except:
                # Skip if optimization fails
                pass

        return np.array(qte_boot)

    def _bootstrap_did(self, tau: float, pre_post_cutoff: Any, n_boot: int) -> np.ndarray:
        """Bootstrap DiD QTE."""
        qte_boot = []

        post_period = self.time > pre_post_cutoff
        pre_period = ~post_period

        for _ in range(n_boot):
            # Cluster bootstrap by entity if panel data
            if self.entities is not None:
                unique_entities = np.unique(self.entities)
                boot_entities = np.random.choice(
                    unique_entities, len(unique_entities), replace=True
                )
                idx = np.concatenate([np.where(self.entities == e)[0] for e in boot_entities])
            else:
                idx = np.random.choice(len(self.y), len(self.y), replace=True)

            y_b = self.y[idx]
            d_b = self.d[idx]
            post_b = post_period[idx]
            pre_b = pre_period[idx]

            try:
                # Compute quantiles for each group-period
                q11 = np.quantile(y_b[(d_b == 1) & post_b], tau)
                q10 = np.quantile(y_b[(d_b == 1) & pre_b], tau)
                q01 = np.quantile(y_b[(d_b == 0) & post_b], tau)
                q00 = np.quantile(y_b[(d_b == 0) & pre_b], tau)

                # DiD estimate
                qte_did_b = (q11 - q10) - (q01 - q00)
                qte_boot.append(qte_did_b)
            except:
                pass

        return np.array(qte_boot)

    def plot_qte(
        self, results: "QTEResult", tau_grid: Optional[np.ndarray] = None, show_ate: bool = True
    ):
        """
        Plot Quantile Treatment Effects across τ.
        """
        import matplotlib.pyplot as plt

        if tau_grid is None:
            tau_grid = sorted(results.qte_results.keys())

        # Extract QTE and CI
        qte_values = [results.qte_results[tau]["qte"] for tau in tau_grid]

        has_ci = results.qte_results[tau_grid[0]]["ci_lower"] is not None
        if has_ci:
            ci_lower = [results.qte_results[tau]["ci_lower"] for tau in tau_grid]
            ci_upper = [results.qte_results[tau]["ci_upper"] for tau in tau_grid]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(tau_grid, qte_values, "b-", linewidth=2, label="QTE")

        if has_ci:
            ax.fill_between(tau_grid, ci_lower, ci_upper, alpha=0.3, label="95% CI")

        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Quantile (τ)")
        ax.set_ylabel("Quantile Treatment Effect")
        ax.set_title(f"Treatment Effects Across the Distribution ({results.method})")

        # Add average treatment effect for comparison
        if show_ate:
            ate = np.mean(self.y[self.d == 1]) - np.mean(self.y[self.d == 0])
            ax.axhline(ate, color="red", linestyle="--", alpha=0.5, label=f"ATE = {ate:.3f}")

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class QTEResult:
    """Results container for QTE estimation."""

    def __init__(self, qte_results: Dict, method: str):
        self.qte_results = qte_results
        self.method = method
        self.tau_list = sorted(qte_results.keys())

    def summary(self):
        """Print QTE summary."""
        print(f"\nQuantile Treatment Effects ({self.method})")
        print("=" * 50)

        # Determine column widths
        has_se = self.qte_results[self.tau_list[0]]["se"] is not None

        if has_se:
            print(f"{'Quantile':<10} {'QTE':>10} {'Std Error':>10} {'95% CI':>20}")
            print("-" * 50)

            for tau in self.tau_list:
                res = self.qte_results[tau]
                qte = res["qte"]
                se = res["se"] if res["se"] is not None else np.nan
                ci = (
                    f"[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]"
                    if res["ci_lower"] is not None
                    else "N/A"
                )

                print(f"{tau:<10.2f} {qte:>10.4f} {se:>10.4f} {ci:>20}")
        else:
            print(f"{'Quantile':<10} {'QTE':>10}")
            print("-" * 25)

            for tau in self.tau_list:
                res = self.qte_results[tau]
                qte = res["qte"]
                print(f"{tau:<10.2f} {qte:>10.4f}")

        # Test for heterogeneity
        qte_values = [res["qte"] for res in self.qte_results.values() if not np.isnan(res["qte"])]
        if len(qte_values) > 1:
            heterogeneity = np.std(qte_values)
            mean_qte = np.mean(qte_values)
            print(f"\nMean QTE: {mean_qte:.4f}")
            print(f"Heterogeneity (std of QTE): {heterogeneity:.4f}")

            if heterogeneity > 0.1 * abs(mean_qte):  # 10% of mean as threshold
                print("✓ Substantial heterogeneity detected across quantiles")
            else:
                print("✗ Treatment effect appears relatively homogeneous")

        # Additional statistics for specific methods
        if self.method == "did":
            print("\nDifference-in-Differences Components:")
            for tau in self.tau_list[:3]:  # Show first 3
                res = self.qte_results[tau]
                print(f"  τ={tau:.2f}:")
                print(f"    Treated change: {res['treated_change']:.4f}")
                print(f"    Control change: {res['control_change']:.4f}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for tau in self.tau_list:
            res = self.qte_results[tau]
            row = {
                "tau": tau,
                "qte": res["qte"],
                "se": res.get("se", np.nan),
                "ci_lower": res.get("ci_lower", np.nan),
                "ci_upper": res.get("ci_upper", np.nan),
            }

            # Add method-specific info
            if self.method == "unconditional":
                row["unconditional_quantile"] = res.get("unconditional_quantile", np.nan)
            elif self.method == "did":
                row["treated_change"] = res.get("treated_change", np.nan)
                row["control_change"] = res.get("control_change", np.nan)

            rows.append(row)

        return pd.DataFrame(rows)

    def test_constant_effects(self) -> Dict:
        """
        Test hypothesis of constant treatment effects across quantiles.

        Returns
        -------
        dict
            Test statistics and p-values
        """
        qte_values = np.array(
            [res["qte"] for res in self.qte_results.values() if not np.isnan(res["qte"])]
        )

        if len(qte_values) < 2:
            return {"test_statistic": np.nan, "p_value": np.nan}

        # Simple F-test for constant effects
        mean_qte = np.mean(qte_values)
        ssr = np.sum((qte_values - mean_qte) ** 2)
        k = len(qte_values) - 1

        # Under null, QTE is constant
        # This is simplified - proper test would account for correlation
        if self.qte_results[self.tau_list[0]]["se"] is not None:
            # Use average SE
            avg_se = np.mean(
                [res["se"] for res in self.qte_results.values() if res["se"] is not None]
            )
            test_stat = ssr / (k * avg_se**2)
            p_value = 1 - stats.chi2.cdf(test_stat, k)
        else:
            test_stat = np.nan
            p_value = np.nan

        return {
            "test_statistic": test_stat,
            "p_value": p_value,
            "reject_constant": p_value < 0.05 if not np.isnan(p_value) else None,
        }
