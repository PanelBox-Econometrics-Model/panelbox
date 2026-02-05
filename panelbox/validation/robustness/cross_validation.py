"""
Time-series cross-validation for panel data models.

This module implements cross-validation methods that respect the temporal
structure of panel data, essential for evaluating out-of-sample predictive
performance.

References
----------
Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for
    time series predictor evaluation. Information Sciences, 191, 192-213.
Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an
    analysis and review. International Journal of Forecasting, 16(4), 437-450.
"""

from typing import Optional, Union, Literal, Dict, Any, Tuple, List
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

from panelbox.core.results import PanelResults


@dataclass
class CVResults:
    """
    Container for cross-validation results.

    Attributes
    ----------
    predictions : pd.DataFrame
        Out-of-sample predictions with columns ['actual', 'predicted', 'fold']
    metrics : Dict[str, float]
        Dictionary of evaluation metrics (MSE, RMSE, MAE, R²)
    fold_metrics : pd.DataFrame
        Per-fold metrics
    method : str
        CV method used ('expanding' or 'rolling')
    n_folds : int
        Number of CV folds
    window_size : Optional[int]
        Window size for rolling CV
    """
    predictions: pd.DataFrame
    metrics: Dict[str, float]
    fold_metrics: pd.DataFrame
    method: str
    n_folds: int
    window_size: Optional[int] = None

    def summary(self) -> str:
        """Generate summary of CV results."""
        lines = []
        lines.append("Cross-Validation Results")
        lines.append("=" * 70)
        lines.append(f"Method: {self.method.capitalize()} Window")
        lines.append(f"Number of folds: {self.n_folds}")
        if self.window_size is not None:
            lines.append(f"Window size: {self.window_size}")
        lines.append("")

        lines.append("Overall Metrics:")
        lines.append("-" * 70)
        lines.append(f"  MSE:         {self.metrics['mse']:>12.6f}")
        lines.append(f"  RMSE:        {self.metrics['rmse']:>12.6f}")
        lines.append(f"  MAE:         {self.metrics['mae']:>12.6f}")
        lines.append(f"  R² (OOS):    {self.metrics['r2_oos']:>12.6f}")
        lines.append("")

        lines.append("Per-Fold Metrics:")
        lines.append("-" * 70)
        lines.append(self.fold_metrics.to_string())

        return "\n".join(lines)


class TimeSeriesCV:
    """
    Time-series cross-validation for panel data models.

    This class implements cross-validation methods that respect the temporal
    ordering of panel data. Two main methods are supported:

    1. Expanding window: Train on periods [1, t], predict period t+1
    2. Rolling window: Train on periods [t-w, t], predict period t+1

    Parameters
    ----------
    results : PanelResults
        Fitted model results containing the model and data
    method : {'expanding', 'rolling'}, default='expanding'
        Cross-validation method:

        - 'expanding': Expanding window (cumulative training)
        - 'rolling': Rolling window (fixed-size training)
    window_size : int, optional
        Window size for rolling CV. Required if method='rolling'.
        Recommended: at least 0.5 * total_periods
    min_train_periods : int, default=3
        Minimum number of periods for training set
    verbose : bool, default=True
        Whether to print progress information

    Attributes
    ----------
    cv_results_ : CVResults
        Cross-validation results after calling cross_validate()
    predictions_ : pd.DataFrame
        Out-of-sample predictions
    metrics_ : Dict[str, float]
        Overall evaluation metrics

    Examples
    --------
    >>> import panelbox as pb
    >>> import pandas as pd
    >>>
    >>> # Fit model
    >>> data = pd.read_csv('panel_data.csv')
    >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "entity_id", "time")
    >>> results = fe.fit()
    >>>
    >>> # Expanding window CV
    >>> cv = pb.TimeSeriesCV(results, method='expanding')
    >>> cv_results = cv.cross_validate()
    >>> print(f"Out-of-sample R²: {cv_results.metrics['r2_oos']:.3f}")
    >>>
    >>> # Rolling window CV
    >>> cv_roll = pb.TimeSeriesCV(results, method='rolling', window_size=5)
    >>> cv_results_roll = cv_roll.cross_validate()
    >>>
    >>> # Plot predictions
    >>> cv.plot_predictions()

    Notes
    -----
    - Cross-validation is performed at the time-period level
    - All entities are included in each fold
    - Models are re-estimated for each fold
    - This can be computationally expensive for large datasets
    """

    def __init__(
        self,
        results: PanelResults,
        method: Literal['expanding', 'rolling'] = 'expanding',
        window_size: Optional[int] = None,
        min_train_periods: int = 3,
        verbose: bool = True
    ):
        self.results = results
        self.method = method
        self.window_size = window_size
        self.min_train_periods = min_train_periods
        self.verbose = verbose

        # Validate inputs
        self._validate_inputs()

        # Extract model information
        self.model = results._model
        self.formula = results.formula
        self.entity_col = self.model.data.entity_col
        self.time_col = self.model.data.time_col

        # Get original data
        self.data = self.model.data.data  # Full dataset

        # Get unique time periods
        self.time_periods = sorted(self.data[self.time_col].unique())
        self.n_periods = len(self.time_periods)

        # Results storage
        self.cv_results_: Optional[CVResults] = None
        self.predictions_: Optional[pd.DataFrame] = None
        self.metrics_: Optional[Dict[str, float]] = None

    def _validate_inputs(self):
        """Validate input parameters."""
        if self.method not in ['expanding', 'rolling']:
            raise ValueError(f"method must be 'expanding' or 'rolling', got '{self.method}'")

        if self.method == 'rolling' and self.window_size is None:
            raise ValueError("window_size must be specified for rolling window CV")

        if self.min_train_periods < 2:
            raise ValueError("min_train_periods must be at least 2")

    def cross_validate(self) -> CVResults:
        """
        Perform time-series cross-validation.

        Returns
        -------
        cv_results : CVResults
            Cross-validation results containing predictions and metrics

        Notes
        -----
        The cross-validation procedure:

        1. For each time period t (starting from min_train_periods):
           - Define training window based on method
           - Fit model on training data
           - Predict on period t
           - Store predictions and compute metrics

        2. Aggregate results across all folds

        The number of folds depends on the method and parameters:
        - Expanding: n_periods - min_train_periods
        - Rolling: n_periods - min_train_periods
        """
        if self.verbose:
            print(f"Starting {self.method} window cross-validation...")
            print(f"Total periods: {self.n_periods}")
            print(f"Min train periods: {self.min_train_periods}")
            if self.method == 'rolling':
                print(f"Window size: {self.window_size}")

        # Storage for predictions and metrics
        all_predictions = []
        fold_metrics_list = []

        # Determine CV folds
        folds = self._get_cv_folds()
        n_folds = len(folds)

        if self.verbose:
            print(f"Number of CV folds: {n_folds}")
            print("")

        # Perform CV
        for fold_idx, (train_periods, test_period) in enumerate(folds, 1):
            if self.verbose:
                print(f"Fold {fold_idx}/{n_folds}: Training on {len(train_periods)} periods, "
                      f"testing on period {test_period}")

            # Split data
            train_data = self.data[self.data[self.time_col].isin(train_periods)]
            test_data = self.data[self.data[self.time_col] == test_period]

            # Fit model on training data
            try:
                model_class = type(self.model)
                train_model = model_class(
                    self.formula,
                    train_data,
                    self.entity_col,
                    self.time_col
                )
                train_results = train_model.fit(cov_type=self.results.cov_type)

                # Predict on test data
                predictions = self._predict_fold(train_results, test_data)

                # Store predictions
                predictions['fold'] = fold_idx
                predictions['test_period'] = test_period
                all_predictions.append(predictions)

                # Compute fold metrics
                fold_metrics = self._compute_metrics(
                    predictions['actual'].values,
                    predictions['predicted'].values
                )
                fold_metrics['fold'] = fold_idx
                fold_metrics['test_period'] = test_period
                fold_metrics_list.append(fold_metrics)

                if self.verbose:
                    print(f"  Fold {fold_idx} R²: {fold_metrics['r2_oos']:.4f}, "
                          f"RMSE: {fold_metrics['rmse']:.4f}")

            except Exception as e:
                warnings.warn(f"Fold {fold_idx} failed: {str(e)}")
                continue

        # Combine all predictions
        if not all_predictions:
            raise RuntimeError("All CV folds failed")

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        fold_metrics_df = pd.DataFrame(fold_metrics_list)

        # Compute overall metrics
        overall_metrics = self._compute_metrics(
            predictions_df['actual'].values,
            predictions_df['predicted'].values
        )

        # Create results object
        self.cv_results_ = CVResults(
            predictions=predictions_df,
            metrics=overall_metrics,
            fold_metrics=fold_metrics_df,
            method=self.method,
            n_folds=n_folds,
            window_size=self.window_size
        )

        self.predictions_ = predictions_df
        self.metrics_ = overall_metrics

        if self.verbose:
            print("\nCross-Validation Complete!")
            print(f"Overall Out-of-Sample R²: {overall_metrics['r2_oos']:.4f}")
            print(f"Overall RMSE: {overall_metrics['rmse']:.4f}")

        return self.cv_results_

    def _get_cv_folds(self) -> List[Tuple[List, Any]]:
        """
        Generate CV folds based on method.

        Returns
        -------
        folds : List[Tuple[List, Any]]
            List of (train_periods, test_period) tuples
        """
        folds = []

        if self.method == 'expanding':
            # Expanding window: train on [1, t], test on t+1
            for t in range(self.min_train_periods, self.n_periods):
                train_periods = self.time_periods[:t]
                test_period = self.time_periods[t]
                folds.append((train_periods, test_period))

        elif self.method == 'rolling':
            # Rolling window: train on [t-w, t], test on t+1
            for t in range(self.min_train_periods, self.n_periods):
                # Determine window start
                window_start = max(0, t - self.window_size)
                train_periods = self.time_periods[window_start:t]
                test_period = self.time_periods[t]

                # Ensure minimum training size
                if len(train_periods) >= self.min_train_periods:
                    folds.append((train_periods, test_period))

        return folds

    def _predict_fold(self, train_results: PanelResults, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for a CV fold.

        Parameters
        ----------
        train_results : PanelResults
            Results from model trained on training data
        test_data : pd.DataFrame
            Test data for prediction

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame with columns ['actual', 'predicted', 'entity', 'time']
        """
        # Extract dependent variable name from formula
        dependent_var = train_results.formula.split('~')[0].strip()

        # Get X matrix for test data using patsy
        from patsy import dmatrix
        formula_rhs = train_results.formula.split('~')[1].strip()

        # Build design matrix
        X_test = dmatrix(formula_rhs, test_data, return_type='dataframe')

        # Get parameter estimates
        params = train_results.params

        # Match columns between training and test
        # Get only the columns that are in params
        param_names = params.index.tolist()

        # Ensure X_test has same columns as training parameters
        X_test_aligned = X_test[param_names] if all(col in X_test.columns for col in param_names) else X_test

        # Make predictions
        predictions_raw = X_test_aligned.values @ params.values

        # Create results dataframe
        predictions_df = pd.DataFrame({
            'actual': test_data[dependent_var].values,
            'predicted': predictions_raw,
            'entity': test_data[self.entity_col].values,
            'time': test_data[self.time_col].values
        })

        return predictions_df

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary of metrics
        """
        # Residuals
        residuals = y_true - y_pred

        # MSE and RMSE
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(residuals))

        # R² (out-of-sample)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_oos = 1 - (ss_res / ss_tot)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_oos': r2_oos
        }

    def plot_predictions(
        self,
        entity: Optional[Union[int, str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot actual vs predicted values.

        Parameters
        ----------
        entity : int or str, optional
            Specific entity to plot. If None, plots all entities.
        save_path : str, optional
            Path to save the plot. If None, displays the plot.

        Raises
        ------
        RuntimeError
            If cross_validate() has not been called yet
        ImportError
            If matplotlib is not installed
        """
        if self.cv_results_ is None:
            raise RuntimeError("Must call cross_validate() before plotting")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. "
                            "Install with: pip install matplotlib")

        predictions = self.cv_results_.predictions

        # Filter by entity if specified
        if entity is not None:
            predictions = predictions[predictions['entity'] == entity]
            if len(predictions) == 0:
                raise ValueError(f"No predictions found for entity {entity}")

        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Actual vs Predicted
        ax1 = axes[0]
        ax1.scatter(predictions['actual'], predictions['predicted'],
                   alpha=0.5, s=30)

        # Add diagonal line
        min_val = min(predictions['actual'].min(), predictions['predicted'].min())
        max_val = max(predictions['actual'].max(), predictions['predicted'].max())
        ax1.plot([min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='Perfect prediction')

        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'Out-of-Sample Predictions: Actual vs Predicted\n'
                     f'R² = {self.cv_results_.metrics["r2_oos"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Time series of predictions
        ax2 = axes[1]

        # Group by time period and compute means
        time_means = predictions.groupby('time').agg({
            'actual': 'mean',
            'predicted': 'mean'
        }).reset_index()

        ax2.plot(time_means['time'], time_means['actual'],
                'o-', label='Actual', linewidth=2, markersize=6)
        ax2.plot(time_means['time'], time_means['predicted'],
                's--', label='Predicted', linewidth=2, markersize=6)

        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Mean Value')
        ax2.set_title(f'{self.method.capitalize()} Window CV: Mean Predictions Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def summary(self) -> str:
        """
        Generate summary of cross-validation results.

        Returns
        -------
        summary_str : str
            Formatted summary string

        Raises
        ------
        RuntimeError
            If cross_validate() has not been called yet
        """
        if self.cv_results_ is None:
            raise RuntimeError("Must call cross_validate() before summary()")

        return self.cv_results_.summary()
