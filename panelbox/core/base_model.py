"""
Base model class for panel econometric models.

This module provides the abstract base class that all panel models inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

from panelbox.core.formula_parser import FormulaParser
from panelbox.core.panel_data import PanelData


class PanelModel(ABC):
    """
    Abstract base class for panel econometric models.

    All panel models (PooledOLS, FixedEffects, RandomEffects, GMM, etc.)
    inherit from this class and must implement the abstract methods.

    Parameters
    ----------
    formula : str
        Model formula in R-style syntax (e.g., "y ~ x1 + x2")
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of the column identifying entities
    time_col : str
        Name of the column identifying time periods
    weights : np.ndarray, optional
        Observation weights

    Attributes
    ----------
    formula : str
        Model formula
    data : PanelData
        Panel data container
    weights : np.ndarray, optional
        Observation weights
    formula_parser : FormulaParser
        Parsed formula object
    _fitted : bool
        Whether model has been fitted
    _results : PanelResults, optional
        Fitted model results

    Examples
    --------
    This is an abstract class. See concrete implementations like
    PooledOLS, FixedEffects, etc. for usage examples.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        weights: Optional[np.ndarray] = None,
    ):
        # Store formula
        self.formula = formula

        # Create PanelData container
        if not isinstance(data, PanelData):
            self.data = PanelData(data, entity_col, time_col)
        else:
            self.data = data

        # Store weights
        self.weights = weights
        if weights is not None:
            if len(weights) != self.data.n_obs:
                raise ValueError(f"weights must have length {self.data.n_obs}, got {len(weights)}")

        # Parse formula
        self.formula_parser = FormulaParser(formula).parse()

        # Model state
        self._fitted = False
        self._results: Optional[Any] = None

    @abstractmethod
    def fit(self, **kwargs) -> "PanelResults":
        """
        Fit the model.

        This method must be implemented by subclasses.

        Parameters
        ----------
        **kwargs
            Model-specific fitting options

        Returns
        -------
        PanelResults
            Fitted model results
        """

    @abstractmethod
    def _estimate_coefficients(self) -> np.ndarray:
        """
        Estimate model coefficients.

        This method contains the core estimation logic and must be
        implemented by subclasses.

        Returns
        -------
        np.ndarray
            Estimated coefficients
        """

    def validate(self, tests: Optional[list] = None, verbose: bool = True) -> "ValidationReport":
        """
        Run validation suite on fitted model.

        Parameters
        ----------
        tests : list, optional
            Specific tests to run. If None, runs all applicable tests.
        verbose : bool, default=True
            Print progress during validation

        Returns
        -------
        ValidationReport
            Validation results

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before validation. Call fit() first.")

        # Import here to avoid circular dependency
        from panelbox.validation.validation_suite import ValidationSuite

        suite = ValidationSuite(self._results)
        return suite.run(tests=tests, verbose=verbose)

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"formula='{self.formula}', "
            f"n_entities={self.data.n_entities}, "
            f"n_obs={self.data.n_obs}, "
            f"status={status})"
        )
