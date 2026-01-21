"""
Base classes for validation tests.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class ValidationTestResult:
    """
    Container for validation test results.
    
    Attributes
    ----------
    test_name : str
        Name of the test
    statistic : float
        Test statistic value
    pvalue : float
        P-value
    df : int or tuple, optional
        Degrees of freedom
    alpha : float
        Significance level used
    conclusion : str
        Interpretation of test result
    null_hypothesis : str
        Description of H0
    alternative_hypothesis : str
        Description of H1
    metadata : dict
        Additional test-specific information
    """
    
    def __init__(
        self,
        test_name: str,
        statistic: float,
        pvalue: float,
        null_hypothesis: str,
        alternative_hypothesis: str,
        alpha: float = 0.05,
        df: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.test_name = test_name
        self.statistic = statistic
        self.pvalue = pvalue
        self.df = df
        self.alpha = alpha
        self.null_hypothesis = null_hypothesis
        self.alternative_hypothesis = alternative_hypothesis
        self.metadata = metadata or {}
        # Add 'df' to metadata for convenience if provided
        if df is not None and 'df' not in self.metadata:
            self.metadata['df'] = df

        # Determine conclusion
        if pvalue < alpha:
            self.reject_null = True
            self.conclusion = (
                f"Reject H0 at {alpha*100:.0f}% level. {alternative_hypothesis}"
            )
        else:
            self.reject_null = False
            self.conclusion = (
                f"Fail to reject H0 at {alpha*100:.0f}% level. {null_hypothesis}"
            )

    @property
    def details(self):
        """Alias for metadata for backwards compatibility."""
        return self.metadata
    
    def __str__(self) -> str:
        """String representation."""
        return self.summary()
    
    def __repr__(self) -> str:
        """Repr."""
        return f"{self.test_name}Result(statistic={self.statistic:.3f}, pvalue={self.pvalue:.4f})"
    
    def summary(self) -> str:
        """
        Generate formatted summary.
        
        Returns
        -------
        str
            Formatted test results
        """
        lines = []
        lines.append("=" * 70)
        lines.append(self.test_name.upper())
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"H0: {self.null_hypothesis}")
        lines.append(f"H1: {self.alternative_hypothesis}")
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"{'Test Statistic':<30} {self.statistic:>15.4f}")
        lines.append(f"{'P-value':<30} {self.pvalue:>15.4f}")
        
        if self.df is not None:
            if isinstance(self.df, tuple):
                df_str = f"({self.df[0]}, {self.df[1]})"
            else:
                df_str = str(self.df)
            lines.append(f"{'Degrees of Freedom':<30} {df_str:>15}")
        
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"Conclusion: {self.conclusion}")
        
        # Add metadata if present
        if self.metadata:
            lines.append("")
            lines.append("Additional Information:")
            for key, value in self.metadata.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        lines.append("=" * 70)
        lines.append("")
        
        return "\n".join(lines)


class ValidationTest(ABC):
    """
    Abstract base class for validation tests.
    
    All validation tests should inherit from this class and implement
    the `run()` method.
    """
    
    def __init__(self, results: 'PanelResults'):
        """
        Initialize validation test.
        
        Parameters
        ----------
        results : PanelResults
            Results object from panel model estimation
        """
        self.results = results
        self.resid = results.resid
        self.fittedvalues = results.fittedvalues
        self.params = results.params
        self.nobs = results.nobs
        self.n_entities = results.n_entities
        self.n_periods = results.n_periods
        self.model_type = results.model_type
    
    @abstractmethod
    def run(self, alpha: float = 0.05, **kwargs) -> ValidationTestResult:
        """
        Run the validation test.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Significance level
        **kwargs
            Test-specific keyword arguments
        
        Returns
        -------
        ValidationTestResult
            Test results
        """
        pass
