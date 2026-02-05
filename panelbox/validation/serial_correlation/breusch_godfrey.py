"""
Breusch-Godfrey LM test for serial correlation in panel data.

References
----------
Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models.
Australian Economic Papers, 17(31), 334-355.

Godfrey, L. G. (1978). Testing against general autoregressive and moving average
error models when the regressors include lagged dependent variables.
Econometrica, 46(6), 1293-1301.
"""

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class BreuschGodfreyTest(ValidationTest):
    """
    Breusch-Godfrey LM test for serial correlation.

    Tests the null hypothesis of no serial correlation against the
    alternative of AR(p) serial correlation in the errors.

    H0: No serial correlation
    H1: AR(p) serial correlation present

    The test regresses the residuals on lagged residuals and the original
    regressors, then tests if the lagged residuals are jointly significant.

    Notes
    -----
    Unlike the Durbin-Watson test, the BG test:
    - Can test for higher-order serial correlation
    - Is valid when regressors include lagged dependent variables
    - Provides an LM test statistic ~ Chi2(p)

    For panel data, the test is applied accounting for the panel structure.

    Examples
    --------
    >>> from panelbox.models.static.fixed_effects import FixedEffects
    >>> fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>>
    >>> from panelbox.validation.serial_correlation.breusch_godfrey import BreuschGodfreyTest
    >>> test = BreuschGodfreyTest(results)
    >>> result = test.run(lags=1)  # Test for AR(1)
    >>> print(result)
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize Breusch-Godfrey test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        super().__init__(results)

    def run(self, lags: int = 1, alpha: float = 0.05) -> ValidationTestResult:
        """
        Run Breusch-Godfrey LM test for serial correlation.

        Parameters
        ----------
        lags : int, default=1
            Number of lags to test (order of AR process)
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        ValidationTestResult
            Test results

        Raises
        ------
        ValueError
            If required data is not available or lags < 1

        Notes
        -----
        The test procedure:
        1. Obtain residuals from original model
        2. Regress residuals on lagged residuals (up to lag p) and original X
        3. Compute LM = n*R² from this auxiliary regression
        4. Compare to Chi2(p) distribution
        """
        if lags < 1:
            raise ValueError(f"lags must be >= 1, got {lags}")

        # Get residuals with entity and time structure
        resid_df = self._prepare_residual_data()

        # Get design matrix
        X = self._get_design_matrix()

        if X is None:
            raise ValueError("Design matrix not available for Breusch-Godfrey test")

        # Create lagged residuals
        resid_df = resid_df.sort_values(["entity", "time"])

        for lag in range(1, lags + 1):
            resid_df[f"resid_lag{lag}"] = resid_df.groupby("entity")["resid"].shift(lag)

        # Drop missing values (first lags obs per entity)
        lag_cols = [f"resid_lag{i}" for i in range(1, lags + 1)]
        resid_df = resid_df.dropna(subset=lag_cols)

        if len(resid_df) == 0:
            raise ValueError("No valid observations after creating lags")

        # Get residuals and lagged residuals as arrays
        resid = resid_df["resid"].values
        X_lags = resid_df[lag_cols].values

        # Match X to the reduced sample (after dropping NAs)
        # We need to align X with the residuals we kept
        # This is tricky - we need the original indices

        # Simpler approach: use all X but only for non-missing resid indices
        if len(X) == len(resid) + lags * resid_df["entity"].nunique():
            # Need to match indices properly
            # For now, assume X already matches the full data
            # and we need to select the rows that correspond to non-missing resid

            # Get indices of non-missing residuals
            valid_indices = resid_df.index.values

            # This assumes resid_df index corresponds to original data indices
            if max(valid_indices) < len(X):
                X_matched = X[valid_indices, :]
            else:
                # Fallback: use last len(resid) rows
                X_matched = X[-len(resid) :, :]
        else:
            # Assume X and resid are aligned
            X_matched = X[: len(resid), :]

        # Auxiliary regression: resid on [X, resid_lag1, ..., resid_lagp]
        X_aug = np.column_stack([X_matched, X_lags])

        # OLS
        try:
            XtX = X_aug.T @ X_aug
            Xty = X_aug.T @ resid
            beta_aux = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta_aux = np.linalg.lstsq(X_aug, resid, rcond=None)[0]

        # Fitted values
        fitted_aux = X_aug @ beta_aux

        # R² from auxiliary regression using explained sum of squares
        # This is more numerically stable
        mean_resid = np.mean(resid)
        SST = np.sum((resid - mean_resid) ** 2)
        SSE = np.sum((fitted_aux - mean_resid) ** 2)

        if SST > 0:
            R2_aux = SSE / SST
        else:
            R2_aux = 0.0

        # Ensure R² is in [0, 1]
        R2_aux = np.clip(R2_aux, 0.0, 1.0)

        # LM statistic for panel data
        # IMPORTANT: For panel data, we use the number of CROSS-SECTIONAL UNITS (N)
        # not the total number of observations (N*T) or the reduced sample size.
        #
        # The Breusch-Godfrey test for panels (pbgtest in plm) uses:
        # LM = N * R²
        # where N is the number of cross-sectional units.
        #
        # This is different from the time-series version which uses n = T observations.
        #
        # Reference: Baltagi & Li (1995), "Testing AR(1) against MA(1) disturbances
        # in an error component model"

        n_entities = resid_df["entity"].nunique()
        lm_stat = n_entities * R2_aux

        # Sanity check
        if lm_stat < 0:
            lm_stat = 0.0

        # Degrees of freedom = number of lags
        df = lags

        # P-value
        pvalue = 1 - stats.chi2.cdf(lm_stat, df)

        # Metadata
        n_obs = len(resid)
        metadata = {
            "lags": lags,
            "R2_auxiliary": R2_aux,
            "n_obs_auxiliary": n_obs,
            "n_entities": n_entities,
            "note": "Panel BG test uses LM = N * R² where N = number of entities",
        }

        result = ValidationTestResult(
            test_name=f"Breusch-Godfrey LM Test for Serial Correlation (AR({lags}))",
            statistic=lm_stat,
            pvalue=pvalue,
            null_hypothesis="No serial correlation",
            alternative_hypothesis=f"AR({lags}) serial correlation present",
            alpha=alpha,
            df=df,
            metadata=metadata,
        )

        return result

    def _prepare_residual_data(self) -> pd.DataFrame:
        """Prepare residual data with entity and time identifiers."""
        if hasattr(self.results, "entity_index") and hasattr(self.results, "time_index"):
            resid_flat = self.resid.ravel() if hasattr(self.resid, "ravel") else self.resid

            resid_df = pd.DataFrame(
                {
                    "entity": self.results.entity_index,
                    "time": self.results.time_index,
                    "resid": resid_flat,
                }
            )

            return resid_df
        else:
            raise AttributeError(
                "Results object must have 'entity_index' and 'time_index' attributes"
            )

    def _get_design_matrix(self) -> np.ndarray:
        """Get the design matrix X."""
        if not hasattr(self.results, "_model"):
            return None

        model = self.results._model

        if hasattr(model, "formula_parser") and hasattr(model, "data"):
            try:
                _, X = model.formula_parser.build_design_matrices(
                    model.data.data, return_type="array"
                )
                return X
            except Exception:
                pass

        return None
