"""
Serialization mixin for model results persistence.

Provides save/load functionality that can be added to any Results class
via multiple inheritance, enabling model deployment in production.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SerializableMixin:
    """
    Mixin class providing save/load functionality for model results.

    Can be added to any Results class (GMMResults, PanelVARResult, etc.)
    to enable model persistence for production deployment.

    Examples
    --------
    >>> # Add to existing Results class:
    >>> class GMMResults(SerializableMixin): ...
    >>>
    >>> # Save model
    >>> results = model.fit()
    >>> results.save("model.pkl")
    >>>
    >>> # Load model (in production)
    >>> loaded = GMMResults.load("model.pkl")
    >>> predictions = loaded.predict(new_data)
    """

    def save(self, filepath: str | Path, format: str = "pickle") -> None:
        """
        Save results to file.

        Parameters
        ----------
        filepath : str or Path
            Path to save file
        format : str, default='pickle'
            Format: 'pickle' (recommended) or 'json'
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "pickle":
            # Add version metadata
            self._panelbox_version = self._get_version()
            self._save_timestamp = pd.Timestamp.now().isoformat()

            with open(filepath, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == "json":
            json_data = self._to_json_dict()
            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=2, default=self._json_serializer)
        else:
            raise ValueError(f"Format '{format}' not supported. Use 'pickle' or 'json'.")

    @classmethod
    def load(cls, filepath: str | Path) -> SerializableMixin:
        """
        Load results from pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to pickle file

        Returns
        -------
        Results object
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "rb") as f:
            results = pickle.load(f)  # noqa: S301 — intentional deserialization of user's own saved results

        return results

    @staticmethod
    def _get_version():
        """Get panelbox version string."""
        try:
            import panelbox

            return getattr(panelbox, "__version__", "unknown")
        except Exception:
            return "unknown"

    def _to_json_dict(self) -> dict[str, Any]:
        """Convert results to JSON-serializable dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            result[key] = value
        # Also include dataclass fields if applicable
        if hasattr(self, "__dataclass_fields__"):
            for key in self.__dataclass_fields__:
                result[key] = getattr(self, key)
        return result

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy/pandas objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_model(filepath: str | Path) -> Any:
    """
    Load any panelbox model results from a pickle file.

    This is a convenience function that works with any Results class
    (PanelResults, GMMResults, PanelVARResult, etc.)

    Parameters
    ----------
    filepath : str or Path
        Path to pickle file

    Returns
    -------
    Results object (type depends on what was saved)

    Examples
    --------
    >>> from panelbox import load_model
    >>> results = load_model("my_model.pkl")
    >>> type(results)  # GMMResults, PanelResults, etc.
    >>> predictions = results.predict(new_data)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        results = pickle.load(f)  # noqa: S301 — intentional deserialization of user's own saved results

    return results
