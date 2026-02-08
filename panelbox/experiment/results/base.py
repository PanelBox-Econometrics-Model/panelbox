"""
Base class for all result containers.

This module defines the abstract base class that all result containers
(ValidationResult, ComparisonResult, etc.) must inherit from.
"""

import json
import webbrowser
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class BaseResult(ABC):
    """
    Abstract base class for all result containers.

    All result classes (ValidationResult, ComparisonResult, etc.) should
    inherit from this class and implement the abstract methods.

    This class provides common functionality:
    - Timestamp tracking
    - Metadata storage
    - HTML report generation via ReportManager
    - JSON serialization

    Parameters
    ----------
    timestamp : datetime, optional
        Timestamp of result creation. If None, uses current time.
    metadata : dict, optional
        Additional metadata to store with result

    Examples
    --------
    >>> class MyResult(BaseResult):
    ...     def to_dict(self):
    ...         return {'my_data': self.my_data}
    ...
    ...     def summary(self):
    ...         return "My result summary"

    >>> result = MyResult()
    >>> result.save_html('report.html', test_type='validation', theme='professional')
    >>> result.save_json('result.json')
    """

    def __init__(
        self, timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BaseResult.

        Parameters
        ----------
        timestamp : datetime, optional
            Timestamp of result creation
        metadata : dict, optional
            Additional metadata
        """
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        This method must be implemented by subclasses.

        Returns
        -------
        dict
            Result data as dictionary

        Examples
        --------
        >>> result.to_dict()
        {'model_name': 'OLS', 'tests': [...], ...}
        """
        pass

    @abstractmethod
    def summary(self) -> str:
        """
        Generate text summary of result.

        This method must be implemented by subclasses.

        Returns
        -------
        str
            Text summary

        Examples
        --------
        >>> print(result.summary())
        Model Validation Report
        =======================
        Tests passed: 8/10
        ...
        """
        pass

    def save_html(
        self,
        file_path: str,
        test_type: str,
        template: Optional[str] = None,
        report_type: str = "interactive",
        theme: str = "professional",
        title: Optional[str] = None,
        open_browser: bool = False,
    ) -> Path:
        """
        Save result as HTML report.

        This method integrates with ReportManager to generate
        professional HTML reports.

        Parameters
        ----------
        file_path : str
            Path to save HTML file
        test_type : str
            Type of test ('validation', 'comparison', 'residuals')
        template : str, optional
            Template path. If None, uses default for test_type.
        report_type : str, default 'interactive'
            Report type ('interactive' or 'static')
        theme : str, default 'professional'
            Theme name ('professional', 'academic', 'presentation')
        title : str, optional
            Custom report title
        open_browser : bool, default False
            Whether to open report in browser after saving

        Returns
        -------
        Path
            Path to saved HTML file

        Examples
        --------
        >>> result.save_html(
        ...     'validation_report.html',
        ...     test_type='validation',
        ...     theme='professional'
        ... )
        PosixPath('/path/to/validation_report.html')

        >>> # Open in browser automatically
        >>> result.save_html(
        ...     'report.html',
        ...     test_type='validation',
        ...     open_browser=True
        ... )
        """
        # Import ReportManager
        from panelbox.report.report_manager import ReportManager

        # Convert result to dict
        context = self.to_dict()

        # Add title if provided
        if title:
            context["report_title"] = title

        # Determine template if not provided
        if template is None:
            template = f"{test_type}/interactive/index.html"

        # Create ReportManager
        report_mgr = ReportManager()

        # Generate HTML
        html = report_mgr.generate_report(
            report_type=test_type,
            template=template,
            context=context,
            embed_assets=True,
            include_plotly=True,
        )

        # Save to file
        output_path = Path(file_path)
        output_path.write_text(html, encoding="utf-8")

        # Open in browser if requested
        if open_browser:
            webbrowser.open(f"file://{output_path.absolute()}")

        return output_path

    def save_json(self, file_path: str, indent: int = 2) -> Path:
        """
        Save result as JSON file.

        Parameters
        ----------
        file_path : str
            Path to save JSON file
        indent : int, default 2
            JSON indentation level

        Returns
        -------
        Path
            Path to saved JSON file

        Examples
        --------
        >>> result.save_json('result.json')
        PosixPath('/path/to/result.json')

        >>> # Compact JSON (no indentation)
        >>> result.save_json('result.json', indent=None)
        """
        output_path = Path(file_path)

        # Convert to dict
        data = self.to_dict()

        # Add metadata
        data["_metadata"] = {
            "timestamp": self.timestamp.isoformat(),
            "class": self.__class__.__name__,
            **self.metadata,
        }

        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(data, f, indent=indent, default=str)

        return output_path

    def _get_report_manager(self):
        """
        Get ReportManager instance.

        Returns
        -------
        ReportManager
            ReportManager instance
        """
        from panelbox.report.report_manager import ReportManager

        return ReportManager()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  timestamp={self.timestamp.isoformat()},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )
