"""
Spatial HAC covariance matrix estimator (Conley 1999).

This module implements the Spatial Heteroskedasticity and Autocorrelation
Consistent (HAC) covariance matrix estimator following Conley (1999).

The estimator is robust to:
- Spatial autocorrelation up to a specified distance
- Temporal autocorrelation up to a specified lag
- Heteroskedasticity

References
----------
Conley, T.G. (1999). "GMM Estimation with Cross Sectional Dependence."
    Journal of Econometrics, 92(1), 1-45.
Hsiang, S.M. (2010). "Temperatures and cyclones strongly associated with
    economic production in the Caribbean and Central America."
    PNAS, 107(35), 15367-15372.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)
from typing import Annotated, Callable, ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"]  # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg=None):  # type: ignore
    """Forward call to original or mutated function, depending on the environment."""
    import os  # type: ignore

    mutant_under_test = os.environ["MUTANT_UNDER_TEST"]  # type: ignore
    if mutant_under_test == "fail":  # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException  # type: ignore

        raise MutmutProgrammaticFailException("Failed programmatically")  # type: ignore
    elif mutant_under_test == "stats":  # type: ignore
        from mutmut.__main__ import record_trampoline_hit  # type: ignore

        record_trampoline_hit(orig.__module__ + "." + orig.__name__)  # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs)  # type: ignore
        return result  # type: ignore
    prefix = orig.__module__ + "." + orig.__name__ + "__mutmut_"  # type: ignore
    if not mutant_under_test.startswith(prefix):  # type: ignore
        result = orig(*call_args, **call_kwargs)  # type: ignore
        return result  # type: ignore
    mutant_name = mutant_under_test.rpartition(".")[-1]  # type: ignore
    if self_arg is not None:  # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)  # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)  # type: ignore
    return result  # type: ignore


class SpatialHAC:
    """
    Spatial HAC covariance matrix estimator (Conley 1999).

    This class implements heteroskedasticity and autocorrelation consistent
    standard errors for panel data with spatial and temporal correlation.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Geographic distances between entities (N×N)
    spatial_cutoff : float
        Maximum distance for spatial correlation (in same units as distance_matrix)
    temporal_cutoff : int
        Maximum lag for temporal correlation (default: 0)
    spatial_kernel : {'bartlett', 'uniform', 'triangular', 'epanechnikov'}
        Spatial kernel function
    temporal_kernel : {'bartlett', 'uniform', 'parzen', 'quadratic_spectral'}
        Temporal kernel function

    Attributes
    ----------
    distance_matrix : np.ndarray
        Stored distance matrix
    spatial_cutoff : float
        Spatial correlation cutoff
    temporal_cutoff : int
        Temporal correlation cutoff
    spatial_kernel : str
        Selected spatial kernel
    temporal_kernel : str
        Selected temporal kernel
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        args = [distance_matrix, spatial_cutoff, temporal_cutoff, spatial_kernel, temporal_kernel]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁSpatialHACǁ__init____mutmut_orig"),
            object.__getattribute__(self, "xǁSpatialHACǁ__init____mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁSpatialHACǁ__init____mutmut_orig(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_1(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 1,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_2(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "XXbartlettXX",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_3(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "BARTLETT",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_4(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "XXbartlettXX",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_5(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "BARTLETT",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_6(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = None
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_7(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = None
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_8(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = None
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_9(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = None
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_10(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.upper()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_11(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = None

        # Validate inputs
        self._validate_inputs()

    def xǁSpatialHACǁ__init____mutmut_12(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""
        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.upper()

        # Validate inputs
        self._validate_inputs()

    xǁSpatialHACǁ__init____mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁSpatialHACǁ__init____mutmut_1": xǁSpatialHACǁ__init____mutmut_1,
        "xǁSpatialHACǁ__init____mutmut_2": xǁSpatialHACǁ__init____mutmut_2,
        "xǁSpatialHACǁ__init____mutmut_3": xǁSpatialHACǁ__init____mutmut_3,
        "xǁSpatialHACǁ__init____mutmut_4": xǁSpatialHACǁ__init____mutmut_4,
        "xǁSpatialHACǁ__init____mutmut_5": xǁSpatialHACǁ__init____mutmut_5,
        "xǁSpatialHACǁ__init____mutmut_6": xǁSpatialHACǁ__init____mutmut_6,
        "xǁSpatialHACǁ__init____mutmut_7": xǁSpatialHACǁ__init____mutmut_7,
        "xǁSpatialHACǁ__init____mutmut_8": xǁSpatialHACǁ__init____mutmut_8,
        "xǁSpatialHACǁ__init____mutmut_9": xǁSpatialHACǁ__init____mutmut_9,
        "xǁSpatialHACǁ__init____mutmut_10": xǁSpatialHACǁ__init____mutmut_10,
        "xǁSpatialHACǁ__init____mutmut_11": xǁSpatialHACǁ__init____mutmut_11,
        "xǁSpatialHACǁ__init____mutmut_12": xǁSpatialHACǁ__init____mutmut_12,
    }
    xǁSpatialHACǁ__init____mutmut_orig.__name__ = "xǁSpatialHACǁ__init__"

    def _validate_inputs(self):
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁSpatialHACǁ_validate_inputs__mutmut_orig"),
            object.__getattribute__(self, "xǁSpatialHACǁ_validate_inputs__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁSpatialHACǁ_validate_inputs__mutmut_orig(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_1(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim == 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_2(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 3:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_3(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError(None)

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_4(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("XXdistance_matrix must be 2-dimensionalXX")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_5(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("DISTANCE_MATRIX MUST BE 2-DIMENSIONAL")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_6(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[1] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_7(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] == self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_8(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[2]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_9(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError(None)

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_10(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("XXdistance_matrix must be squareXX")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_11(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("DISTANCE_MATRIX MUST BE SQUARE")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_12(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff < 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_13(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 1:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_14(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError(None)

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_15(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("XXspatial_cutoff must be positiveXX")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_16(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("SPATIAL_CUTOFF MUST BE POSITIVE")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_17(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff <= 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_18(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 1:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_19(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError(None)

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_20(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("XXtemporal_cutoff must be non-negativeXX")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_21(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("TEMPORAL_CUTOFF MUST BE NON-NEGATIVE")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_22(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = None
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_23(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["XXbartlettXX", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_24(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["BARTLETT", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_25(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "XXuniformXX", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_26(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "UNIFORM", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_27(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "XXtriangularXX", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_28(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "TRIANGULAR", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_29(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "XXepanechnikovXX"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_30(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "EPANECHNIKOV"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_31(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_32(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(None)

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_33(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = None
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_34(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["XXbartlettXX", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_35(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["BARTLETT", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_36(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "XXuniformXX", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_37(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "UNIFORM", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_38(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "XXparzenXX", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_39(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "PARZEN", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_40(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "XXquadratic_spectralXX"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_41(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "QUADRATIC_SPECTRAL"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_42(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def xǁSpatialHACǁ_validate_inputs__mutmut_43(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(None)

    xǁSpatialHACǁ_validate_inputs__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁSpatialHACǁ_validate_inputs__mutmut_1": xǁSpatialHACǁ_validate_inputs__mutmut_1,
        "xǁSpatialHACǁ_validate_inputs__mutmut_2": xǁSpatialHACǁ_validate_inputs__mutmut_2,
        "xǁSpatialHACǁ_validate_inputs__mutmut_3": xǁSpatialHACǁ_validate_inputs__mutmut_3,
        "xǁSpatialHACǁ_validate_inputs__mutmut_4": xǁSpatialHACǁ_validate_inputs__mutmut_4,
        "xǁSpatialHACǁ_validate_inputs__mutmut_5": xǁSpatialHACǁ_validate_inputs__mutmut_5,
        "xǁSpatialHACǁ_validate_inputs__mutmut_6": xǁSpatialHACǁ_validate_inputs__mutmut_6,
        "xǁSpatialHACǁ_validate_inputs__mutmut_7": xǁSpatialHACǁ_validate_inputs__mutmut_7,
        "xǁSpatialHACǁ_validate_inputs__mutmut_8": xǁSpatialHACǁ_validate_inputs__mutmut_8,
        "xǁSpatialHACǁ_validate_inputs__mutmut_9": xǁSpatialHACǁ_validate_inputs__mutmut_9,
        "xǁSpatialHACǁ_validate_inputs__mutmut_10": xǁSpatialHACǁ_validate_inputs__mutmut_10,
        "xǁSpatialHACǁ_validate_inputs__mutmut_11": xǁSpatialHACǁ_validate_inputs__mutmut_11,
        "xǁSpatialHACǁ_validate_inputs__mutmut_12": xǁSpatialHACǁ_validate_inputs__mutmut_12,
        "xǁSpatialHACǁ_validate_inputs__mutmut_13": xǁSpatialHACǁ_validate_inputs__mutmut_13,
        "xǁSpatialHACǁ_validate_inputs__mutmut_14": xǁSpatialHACǁ_validate_inputs__mutmut_14,
        "xǁSpatialHACǁ_validate_inputs__mutmut_15": xǁSpatialHACǁ_validate_inputs__mutmut_15,
        "xǁSpatialHACǁ_validate_inputs__mutmut_16": xǁSpatialHACǁ_validate_inputs__mutmut_16,
        "xǁSpatialHACǁ_validate_inputs__mutmut_17": xǁSpatialHACǁ_validate_inputs__mutmut_17,
        "xǁSpatialHACǁ_validate_inputs__mutmut_18": xǁSpatialHACǁ_validate_inputs__mutmut_18,
        "xǁSpatialHACǁ_validate_inputs__mutmut_19": xǁSpatialHACǁ_validate_inputs__mutmut_19,
        "xǁSpatialHACǁ_validate_inputs__mutmut_20": xǁSpatialHACǁ_validate_inputs__mutmut_20,
        "xǁSpatialHACǁ_validate_inputs__mutmut_21": xǁSpatialHACǁ_validate_inputs__mutmut_21,
        "xǁSpatialHACǁ_validate_inputs__mutmut_22": xǁSpatialHACǁ_validate_inputs__mutmut_22,
        "xǁSpatialHACǁ_validate_inputs__mutmut_23": xǁSpatialHACǁ_validate_inputs__mutmut_23,
        "xǁSpatialHACǁ_validate_inputs__mutmut_24": xǁSpatialHACǁ_validate_inputs__mutmut_24,
        "xǁSpatialHACǁ_validate_inputs__mutmut_25": xǁSpatialHACǁ_validate_inputs__mutmut_25,
        "xǁSpatialHACǁ_validate_inputs__mutmut_26": xǁSpatialHACǁ_validate_inputs__mutmut_26,
        "xǁSpatialHACǁ_validate_inputs__mutmut_27": xǁSpatialHACǁ_validate_inputs__mutmut_27,
        "xǁSpatialHACǁ_validate_inputs__mutmut_28": xǁSpatialHACǁ_validate_inputs__mutmut_28,
        "xǁSpatialHACǁ_validate_inputs__mutmut_29": xǁSpatialHACǁ_validate_inputs__mutmut_29,
        "xǁSpatialHACǁ_validate_inputs__mutmut_30": xǁSpatialHACǁ_validate_inputs__mutmut_30,
        "xǁSpatialHACǁ_validate_inputs__mutmut_31": xǁSpatialHACǁ_validate_inputs__mutmut_31,
        "xǁSpatialHACǁ_validate_inputs__mutmut_32": xǁSpatialHACǁ_validate_inputs__mutmut_32,
        "xǁSpatialHACǁ_validate_inputs__mutmut_33": xǁSpatialHACǁ_validate_inputs__mutmut_33,
        "xǁSpatialHACǁ_validate_inputs__mutmut_34": xǁSpatialHACǁ_validate_inputs__mutmut_34,
        "xǁSpatialHACǁ_validate_inputs__mutmut_35": xǁSpatialHACǁ_validate_inputs__mutmut_35,
        "xǁSpatialHACǁ_validate_inputs__mutmut_36": xǁSpatialHACǁ_validate_inputs__mutmut_36,
        "xǁSpatialHACǁ_validate_inputs__mutmut_37": xǁSpatialHACǁ_validate_inputs__mutmut_37,
        "xǁSpatialHACǁ_validate_inputs__mutmut_38": xǁSpatialHACǁ_validate_inputs__mutmut_38,
        "xǁSpatialHACǁ_validate_inputs__mutmut_39": xǁSpatialHACǁ_validate_inputs__mutmut_39,
        "xǁSpatialHACǁ_validate_inputs__mutmut_40": xǁSpatialHACǁ_validate_inputs__mutmut_40,
        "xǁSpatialHACǁ_validate_inputs__mutmut_41": xǁSpatialHACǁ_validate_inputs__mutmut_41,
        "xǁSpatialHACǁ_validate_inputs__mutmut_42": xǁSpatialHACǁ_validate_inputs__mutmut_42,
        "xǁSpatialHACǁ_validate_inputs__mutmut_43": xǁSpatialHACǁ_validate_inputs__mutmut_43,
    }
    xǁSpatialHACǁ_validate_inputs__mutmut_orig.__name__ = "xǁSpatialHACǁ_validate_inputs"

    def _spatial_kernel_weight(self, distance: np.ndarray) -> np.ndarray:
        args = [distance]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_orig"),
            object.__getattribute__(self, "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_orig(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_1(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = None

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_2(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance * self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_3(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel != "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_4(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "XXbartlettXX":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_5(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "BARTLETT":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_6(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = None

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_7(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(None, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_8(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, None)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_9(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_10(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(
                1 - u,
            )

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_11(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 + u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_12(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(2 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_13(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 1)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_14(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel != "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_15(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "XXuniformXX":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_16(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "UNIFORM":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_17(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = None

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_18(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(None)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_19(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u < 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_20(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 2).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_21(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel != "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_22(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "XXtriangularXX":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_23(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "TRIANGULAR":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_24(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = None

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_25(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(None, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_26(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, None)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_27(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_28(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(
                1 - u,
            )

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_29(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 + u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_30(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(2 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_31(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 1)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_32(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel != "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_33(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "XXepanechnikovXX":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_34(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "EPANECHNIKOV":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_35(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = None

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_36(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(None, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_37(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, None, 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_38(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), None)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_39(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_40(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_41(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(
                u <= 1,
                0.75 * (1 - u**2),
            )

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_42(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u < 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_43(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 2, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_44(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 / (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_45(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 1.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_46(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 + u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_47(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (2 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_48(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u * 2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_49(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**3), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_50(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 1)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def xǁSpatialHACǁ_spatial_kernel_weight__mutmut_51(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(None)

        return weight

    xǁSpatialHACǁ_spatial_kernel_weight__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_1": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_1,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_2": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_2,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_3": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_3,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_4": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_4,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_5": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_5,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_6": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_6,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_7": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_7,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_8": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_8,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_9": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_9,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_10": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_10,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_11": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_11,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_12": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_12,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_13": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_13,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_14": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_14,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_15": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_15,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_16": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_16,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_17": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_17,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_18": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_18,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_19": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_19,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_20": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_20,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_21": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_21,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_22": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_22,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_23": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_23,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_24": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_24,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_25": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_25,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_26": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_26,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_27": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_27,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_28": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_28,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_29": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_29,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_30": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_30,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_31": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_31,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_32": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_32,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_33": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_33,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_34": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_34,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_35": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_35,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_36": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_36,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_37": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_37,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_38": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_38,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_39": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_39,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_40": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_40,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_41": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_41,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_42": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_42,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_43": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_43,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_44": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_44,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_45": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_45,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_46": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_46,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_47": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_47,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_48": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_48,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_49": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_49,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_50": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_50,
        "xǁSpatialHACǁ_spatial_kernel_weight__mutmut_51": xǁSpatialHACǁ_spatial_kernel_weight__mutmut_51,
    }
    xǁSpatialHACǁ_spatial_kernel_weight__mutmut_orig.__name__ = (
        "xǁSpatialHACǁ_spatial_kernel_weight"
    )

    def _temporal_kernel_weight(self, lag: int | np.ndarray) -> float | np.ndarray:
        args = [lag]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_orig"),
            object.__getattribute__(self, "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_orig(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_1(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff != 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_2(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 1:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_3(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(None) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_4(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag != 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_5(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 1).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_6(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(None)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_7(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag != 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_8(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 1)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_9(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = None

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_10(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) * (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_11(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(None) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_12(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff - 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_13(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 2)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_14(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel != "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_15(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "XXbartlettXX":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_16(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "BARTLETT":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_17(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = None

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_18(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(None, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_19(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, None)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_20(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_21(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(
                1 - u,
            )

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_22(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 + u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_23(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(2 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_24(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 1)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_25(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel != "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_26(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "XXuniformXX":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_27(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "UNIFORM":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_28(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = None

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_29(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(None)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_30(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(None) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_31(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) < self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_32(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel != "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_33(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "XXparzenXX":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_34(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "PARZEN":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_35(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = None
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_36(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(None)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_37(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            np.abs(u)
            weight = None

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_38(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                None,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_39(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                None,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_40(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                None,
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_41(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_42(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_43(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_44(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs < 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_45(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 1.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_46(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 - 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_47(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 + 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_48(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                2 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_49(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 / u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_50(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 7 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_51(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs * 2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_52(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**3 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_53(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 / u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_54(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 7 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_55(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs * 3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_56(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**4,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_57(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(None, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_58(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, None, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_59(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, None),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_60(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_61(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_62(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(
                    u_abs <= 1,
                    2 * (1 - u_abs) ** 3,
                ),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_63(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs < 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_64(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 2, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_65(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 / (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_66(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 3 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_67(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) * 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_68(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 + u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_69(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (2 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_70(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 4, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_71(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 1),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_72(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel != "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_73(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "XXquadratic_spectralXX":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_74(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "QUADRATIC_SPECTRAL":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_75(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = None
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_76(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u * 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_77(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi / u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_78(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 / np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_79(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 7 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_80(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 6
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_81(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide=None, invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_82(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid=None):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_83(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_84(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(
                divide="ignore",
            ):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_85(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="XXignoreXX", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_86(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="IGNORE", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_87(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="XXignoreXX"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_88(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="IGNORE"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_89(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = None
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_90(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 / (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_91(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 * x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_92(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 4 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_93(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x * 2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_94(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**3 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_95(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x + np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_96(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) * x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_97(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(None) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_98(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(None))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_99(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = None  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_100(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(None, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_101(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, None, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_102(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, None)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_103(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_104(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_105(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(
                    u == 0,
                    1,
                )  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_106(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u != 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_107(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 1, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_108(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 2, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def xǁSpatialHACǁ_temporal_kernel_weight__mutmut_109(
        self, lag: int | np.ndarray
    ) -> float | np.ndarray:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(None)

        return weight

    xǁSpatialHACǁ_temporal_kernel_weight__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_1": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_1,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_2": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_2,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_3": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_3,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_4": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_4,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_5": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_5,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_6": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_6,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_7": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_7,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_8": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_8,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_9": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_9,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_10": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_10,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_11": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_11,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_12": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_12,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_13": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_13,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_14": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_14,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_15": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_15,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_16": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_16,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_17": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_17,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_18": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_18,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_19": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_19,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_20": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_20,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_21": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_21,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_22": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_22,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_23": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_23,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_24": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_24,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_25": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_25,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_26": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_26,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_27": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_27,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_28": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_28,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_29": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_29,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_30": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_30,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_31": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_31,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_32": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_32,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_33": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_33,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_34": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_34,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_35": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_35,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_36": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_36,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_37": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_37,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_38": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_38,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_39": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_39,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_40": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_40,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_41": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_41,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_42": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_42,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_43": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_43,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_44": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_44,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_45": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_45,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_46": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_46,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_47": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_47,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_48": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_48,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_49": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_49,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_50": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_50,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_51": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_51,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_52": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_52,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_53": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_53,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_54": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_54,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_55": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_55,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_56": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_56,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_57": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_57,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_58": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_58,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_59": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_59,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_60": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_60,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_61": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_61,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_62": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_62,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_63": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_63,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_64": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_64,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_65": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_65,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_66": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_66,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_67": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_67,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_68": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_68,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_69": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_69,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_70": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_70,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_71": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_71,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_72": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_72,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_73": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_73,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_74": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_74,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_75": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_75,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_76": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_76,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_77": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_77,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_78": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_78,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_79": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_79,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_80": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_80,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_81": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_81,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_82": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_82,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_83": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_83,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_84": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_84,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_85": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_85,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_86": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_86,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_87": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_87,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_88": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_88,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_89": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_89,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_90": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_90,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_91": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_91,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_92": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_92,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_93": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_93,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_94": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_94,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_95": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_95,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_96": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_96,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_97": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_97,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_98": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_98,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_99": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_99,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_100": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_100,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_101": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_101,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_102": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_102,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_103": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_103,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_104": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_104,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_105": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_105,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_106": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_106,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_107": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_107,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_108": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_108,
        "xǁSpatialHACǁ_temporal_kernel_weight__mutmut_109": xǁSpatialHACǁ_temporal_kernel_weight__mutmut_109,
    }
    xǁSpatialHACǁ_temporal_kernel_weight__mutmut_orig.__name__ = (
        "xǁSpatialHACǁ_temporal_kernel_weight"
    )

    def compute(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        args = [X, residuals, entity_index, time_index, small_sample_correction]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁSpatialHACǁcompute__mutmut_orig"),
            object.__getattribute__(self, "xǁSpatialHACǁcompute__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁSpatialHACǁcompute__mutmut_orig(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_1(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = False,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_2(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = None
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_3(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = None
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_4(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(None)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_5(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = None
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_6(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(None)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_7(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = None
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_8(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = None

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_9(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs == N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_10(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N / T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_11(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(None, stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_12(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=None)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_13(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_14(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(
                f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2,
            )

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_15(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=3)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_16(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = None
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_17(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(None):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_18(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(None, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_19(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, None)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_20(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_21(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(
            zip(
                entity_index,
            )
        ):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_22(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for _idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = None

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_23(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = None

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_24(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros(None)

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_25(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(None):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_26(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(None):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_27(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for _t1_idx, t1 in enumerate(times):
            for _t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = None
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_28(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for _t1_idx, t1 in enumerate(times):
            for _t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(None)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_29(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx + t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_30(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag >= self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_31(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    break
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_32(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = None

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_33(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(None)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_34(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal != 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_35(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 1:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_36(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    break

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_37(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(None):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_38(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_39(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        break
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_40(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = None

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_41(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(None):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_42(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_43(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            break
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_44(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = None

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_45(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for _i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for _j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = None
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_46(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance >= self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_47(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            break
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_48(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = None

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_49(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(None)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_50(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial != 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_51(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 1:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_52(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            break

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_53(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = None

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_54(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial / w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_55(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega = weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_56(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega -= weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_57(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight / (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_58(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            / np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_59(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            / residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_60(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1] * residuals[obs_j_t2] * np.outer(None, X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_61(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1] * residuals[obs_j_t2] * np.outer(X[obs_i_t1], None)
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_62(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1] * residuals[obs_j_t2] * np.outer(X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_63(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(
                                X[obs_i_t1],
                            )
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_64(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = None

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_65(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = None
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_66(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(None)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_67(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn(None, stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_68(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=None)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_69(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn(stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_70(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn(
                "X'X is singular, using pseudo-inverse", stacklevel=2,
            )
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_71(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("XXX'X is singular, using pseudo-inverseXX", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_72(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("x'x is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_73(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X IS SINGULAR, USING PSEUDO-INVERSE", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_74(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=3)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_75(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = None

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_76(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(None)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_77(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = None

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_78(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = None
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_79(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs * (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_80(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs + k_vars)
            V_hac *= correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_81(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac = correction_factor

        return V_hac

    def xǁSpatialHACǁcompute__mutmut_82(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}", stacklevel=2)

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse", stacklevel=2)
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac /= correction_factor

        return V_hac

    xǁSpatialHACǁcompute__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁSpatialHACǁcompute__mutmut_1": xǁSpatialHACǁcompute__mutmut_1,
        "xǁSpatialHACǁcompute__mutmut_2": xǁSpatialHACǁcompute__mutmut_2,
        "xǁSpatialHACǁcompute__mutmut_3": xǁSpatialHACǁcompute__mutmut_3,
        "xǁSpatialHACǁcompute__mutmut_4": xǁSpatialHACǁcompute__mutmut_4,
        "xǁSpatialHACǁcompute__mutmut_5": xǁSpatialHACǁcompute__mutmut_5,
        "xǁSpatialHACǁcompute__mutmut_6": xǁSpatialHACǁcompute__mutmut_6,
        "xǁSpatialHACǁcompute__mutmut_7": xǁSpatialHACǁcompute__mutmut_7,
        "xǁSpatialHACǁcompute__mutmut_8": xǁSpatialHACǁcompute__mutmut_8,
        "xǁSpatialHACǁcompute__mutmut_9": xǁSpatialHACǁcompute__mutmut_9,
        "xǁSpatialHACǁcompute__mutmut_10": xǁSpatialHACǁcompute__mutmut_10,
        "xǁSpatialHACǁcompute__mutmut_11": xǁSpatialHACǁcompute__mutmut_11,
        "xǁSpatialHACǁcompute__mutmut_12": xǁSpatialHACǁcompute__mutmut_12,
        "xǁSpatialHACǁcompute__mutmut_13": xǁSpatialHACǁcompute__mutmut_13,
        "xǁSpatialHACǁcompute__mutmut_14": xǁSpatialHACǁcompute__mutmut_14,
        "xǁSpatialHACǁcompute__mutmut_15": xǁSpatialHACǁcompute__mutmut_15,
        "xǁSpatialHACǁcompute__mutmut_16": xǁSpatialHACǁcompute__mutmut_16,
        "xǁSpatialHACǁcompute__mutmut_17": xǁSpatialHACǁcompute__mutmut_17,
        "xǁSpatialHACǁcompute__mutmut_18": xǁSpatialHACǁcompute__mutmut_18,
        "xǁSpatialHACǁcompute__mutmut_19": xǁSpatialHACǁcompute__mutmut_19,
        "xǁSpatialHACǁcompute__mutmut_20": xǁSpatialHACǁcompute__mutmut_20,
        "xǁSpatialHACǁcompute__mutmut_21": xǁSpatialHACǁcompute__mutmut_21,
        "xǁSpatialHACǁcompute__mutmut_22": xǁSpatialHACǁcompute__mutmut_22,
        "xǁSpatialHACǁcompute__mutmut_23": xǁSpatialHACǁcompute__mutmut_23,
        "xǁSpatialHACǁcompute__mutmut_24": xǁSpatialHACǁcompute__mutmut_24,
        "xǁSpatialHACǁcompute__mutmut_25": xǁSpatialHACǁcompute__mutmut_25,
        "xǁSpatialHACǁcompute__mutmut_26": xǁSpatialHACǁcompute__mutmut_26,
        "xǁSpatialHACǁcompute__mutmut_27": xǁSpatialHACǁcompute__mutmut_27,
        "xǁSpatialHACǁcompute__mutmut_28": xǁSpatialHACǁcompute__mutmut_28,
        "xǁSpatialHACǁcompute__mutmut_29": xǁSpatialHACǁcompute__mutmut_29,
        "xǁSpatialHACǁcompute__mutmut_30": xǁSpatialHACǁcompute__mutmut_30,
        "xǁSpatialHACǁcompute__mutmut_31": xǁSpatialHACǁcompute__mutmut_31,
        "xǁSpatialHACǁcompute__mutmut_32": xǁSpatialHACǁcompute__mutmut_32,
        "xǁSpatialHACǁcompute__mutmut_33": xǁSpatialHACǁcompute__mutmut_33,
        "xǁSpatialHACǁcompute__mutmut_34": xǁSpatialHACǁcompute__mutmut_34,
        "xǁSpatialHACǁcompute__mutmut_35": xǁSpatialHACǁcompute__mutmut_35,
        "xǁSpatialHACǁcompute__mutmut_36": xǁSpatialHACǁcompute__mutmut_36,
        "xǁSpatialHACǁcompute__mutmut_37": xǁSpatialHACǁcompute__mutmut_37,
        "xǁSpatialHACǁcompute__mutmut_38": xǁSpatialHACǁcompute__mutmut_38,
        "xǁSpatialHACǁcompute__mutmut_39": xǁSpatialHACǁcompute__mutmut_39,
        "xǁSpatialHACǁcompute__mutmut_40": xǁSpatialHACǁcompute__mutmut_40,
        "xǁSpatialHACǁcompute__mutmut_41": xǁSpatialHACǁcompute__mutmut_41,
        "xǁSpatialHACǁcompute__mutmut_42": xǁSpatialHACǁcompute__mutmut_42,
        "xǁSpatialHACǁcompute__mutmut_43": xǁSpatialHACǁcompute__mutmut_43,
        "xǁSpatialHACǁcompute__mutmut_44": xǁSpatialHACǁcompute__mutmut_44,
        "xǁSpatialHACǁcompute__mutmut_45": xǁSpatialHACǁcompute__mutmut_45,
        "xǁSpatialHACǁcompute__mutmut_46": xǁSpatialHACǁcompute__mutmut_46,
        "xǁSpatialHACǁcompute__mutmut_47": xǁSpatialHACǁcompute__mutmut_47,
        "xǁSpatialHACǁcompute__mutmut_48": xǁSpatialHACǁcompute__mutmut_48,
        "xǁSpatialHACǁcompute__mutmut_49": xǁSpatialHACǁcompute__mutmut_49,
        "xǁSpatialHACǁcompute__mutmut_50": xǁSpatialHACǁcompute__mutmut_50,
        "xǁSpatialHACǁcompute__mutmut_51": xǁSpatialHACǁcompute__mutmut_51,
        "xǁSpatialHACǁcompute__mutmut_52": xǁSpatialHACǁcompute__mutmut_52,
        "xǁSpatialHACǁcompute__mutmut_53": xǁSpatialHACǁcompute__mutmut_53,
        "xǁSpatialHACǁcompute__mutmut_54": xǁSpatialHACǁcompute__mutmut_54,
        "xǁSpatialHACǁcompute__mutmut_55": xǁSpatialHACǁcompute__mutmut_55,
        "xǁSpatialHACǁcompute__mutmut_56": xǁSpatialHACǁcompute__mutmut_56,
        "xǁSpatialHACǁcompute__mutmut_57": xǁSpatialHACǁcompute__mutmut_57,
        "xǁSpatialHACǁcompute__mutmut_58": xǁSpatialHACǁcompute__mutmut_58,
        "xǁSpatialHACǁcompute__mutmut_59": xǁSpatialHACǁcompute__mutmut_59,
        "xǁSpatialHACǁcompute__mutmut_60": xǁSpatialHACǁcompute__mutmut_60,
        "xǁSpatialHACǁcompute__mutmut_61": xǁSpatialHACǁcompute__mutmut_61,
        "xǁSpatialHACǁcompute__mutmut_62": xǁSpatialHACǁcompute__mutmut_62,
        "xǁSpatialHACǁcompute__mutmut_63": xǁSpatialHACǁcompute__mutmut_63,
        "xǁSpatialHACǁcompute__mutmut_64": xǁSpatialHACǁcompute__mutmut_64,
        "xǁSpatialHACǁcompute__mutmut_65": xǁSpatialHACǁcompute__mutmut_65,
        "xǁSpatialHACǁcompute__mutmut_66": xǁSpatialHACǁcompute__mutmut_66,
        "xǁSpatialHACǁcompute__mutmut_67": xǁSpatialHACǁcompute__mutmut_67,
        "xǁSpatialHACǁcompute__mutmut_68": xǁSpatialHACǁcompute__mutmut_68,
        "xǁSpatialHACǁcompute__mutmut_69": xǁSpatialHACǁcompute__mutmut_69,
        "xǁSpatialHACǁcompute__mutmut_70": xǁSpatialHACǁcompute__mutmut_70,
        "xǁSpatialHACǁcompute__mutmut_71": xǁSpatialHACǁcompute__mutmut_71,
        "xǁSpatialHACǁcompute__mutmut_72": xǁSpatialHACǁcompute__mutmut_72,
        "xǁSpatialHACǁcompute__mutmut_73": xǁSpatialHACǁcompute__mutmut_73,
        "xǁSpatialHACǁcompute__mutmut_74": xǁSpatialHACǁcompute__mutmut_74,
        "xǁSpatialHACǁcompute__mutmut_75": xǁSpatialHACǁcompute__mutmut_75,
        "xǁSpatialHACǁcompute__mutmut_76": xǁSpatialHACǁcompute__mutmut_76,
        "xǁSpatialHACǁcompute__mutmut_77": xǁSpatialHACǁcompute__mutmut_77,
        "xǁSpatialHACǁcompute__mutmut_78": xǁSpatialHACǁcompute__mutmut_78,
        "xǁSpatialHACǁcompute__mutmut_79": xǁSpatialHACǁcompute__mutmut_79,
        "xǁSpatialHACǁcompute__mutmut_80": xǁSpatialHACǁcompute__mutmut_80,
        "xǁSpatialHACǁcompute__mutmut_81": xǁSpatialHACǁcompute__mutmut_81,
        "xǁSpatialHACǁcompute__mutmut_82": xǁSpatialHACǁcompute__mutmut_82,
    }
    xǁSpatialHACǁcompute__mutmut_orig.__name__ = "xǁSpatialHACǁcompute"

    @classmethod
    def from_coordinates(
        cls,
        coords: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        distance_metric: str = "haversine",
        **kwargs,
    ):
        """
        Create SpatialHAC from geographic coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Geographic coordinates (N × 2), typically (latitude, longitude)
        spatial_cutoff : float
            Maximum distance for spatial correlation
        temporal_cutoff : int
            Maximum lag for temporal correlation
        distance_metric : {'haversine', 'euclidean', 'manhattan'}
            Distance calculation method
        **kwargs
            Additional arguments passed to SpatialHAC constructor

        Returns
        -------
        SpatialHAC
            Initialized SpatialHAC object
        """
        if coords.shape[1] != 2:
            raise ValueError("coords must have shape (N, 2)")

        if distance_metric == "haversine":
            distance_matrix = cls._haversine_distance_matrix(coords)
        elif distance_metric in ["euclidean", "manhattan", "cityblock"]:
            distance_matrix = cdist(coords, coords, metric=distance_metric)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        return cls(distance_matrix, spatial_cutoff, temporal_cutoff, **kwargs)

    @staticmethod
    def _haversine_distance_matrix(coords: np.ndarray) -> np.ndarray:
        """
        Compute great circle distances between coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates in degrees (N × 2): [latitude, longitude]

        Returns
        -------
        np.ndarray
            Distance matrix in kilometers (N × N)
        """
        # Convert to radians
        coords_rad = np.radians(coords)
        lat = coords_rad[:, 0]
        lon = coords_rad[:, 1]

        # Haversine formula
        N = len(coords)
        distance_matrix = np.zeros((N, N))

        for i in range(N):
            # Vectorized computation for all j
            dlat = lat - lat[i]
            dlon = lon - lon[i]

            a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))

            # Earth radius in kilometers
            R = 6371.0
            distance_matrix[i] = R * c

        return distance_matrix

    def compare_with_standard_errors(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        args = [X, residuals, entity_index, time_index]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁSpatialHACǁcompare_with_standard_errors__mutmut_orig"),
            object.__getattribute__(
                self, "xǁSpatialHACǁcompare_with_standard_errors__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_orig(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_1(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = None

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_2(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        _n_obs, _k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = None
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_3(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) * (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_4(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(None) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_5(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals * 2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_6(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**3) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_7(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs + k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_8(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = None
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_9(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(None)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_10(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = None

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_11(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 / XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_12(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = None
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_13(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(None) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_14(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() * 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_15(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 3) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_16(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = None

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_17(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = None

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_18(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(None, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_19(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, None, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_20(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, None, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_21(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, None)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_22(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_23(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_24(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_25(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(
            X,
            residuals,
            entity_index,
        )

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_26(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = None
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_27(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(None)
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_28(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(None))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_29(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = None
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_30(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(None)
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_31(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(None))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_32(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = None

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_33(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(None)

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_34(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(None))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_35(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "XXV_olsXX": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_36(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "v_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_37(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_OLS": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_38(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "XXV_whiteXX": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_39(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "v_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_40(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_WHITE": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_41(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "XXV_hacXX": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_42(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "v_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_43(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_HAC": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_44(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "XXse_olsXX": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_45(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "SE_OLS": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_46(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "XXse_whiteXX": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_47(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "SE_WHITE": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_48(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "XXse_hacXX": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_49(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "SE_HAC": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_50(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "XXse_ratio_hac_olsXX": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_51(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "SE_RATIO_HAC_OLS": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_52(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac * se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_53(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "XXse_ratio_hac_whiteXX": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_54(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "SE_RATIO_HAC_WHITE": se_hac / se_white,
        }

    def xǁSpatialHACǁcompare_with_standard_errors__mutmut_55(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac * se_white,
        }

    xǁSpatialHACǁcompare_with_standard_errors__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_1": xǁSpatialHACǁcompare_with_standard_errors__mutmut_1,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_2": xǁSpatialHACǁcompare_with_standard_errors__mutmut_2,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_3": xǁSpatialHACǁcompare_with_standard_errors__mutmut_3,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_4": xǁSpatialHACǁcompare_with_standard_errors__mutmut_4,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_5": xǁSpatialHACǁcompare_with_standard_errors__mutmut_5,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_6": xǁSpatialHACǁcompare_with_standard_errors__mutmut_6,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_7": xǁSpatialHACǁcompare_with_standard_errors__mutmut_7,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_8": xǁSpatialHACǁcompare_with_standard_errors__mutmut_8,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_9": xǁSpatialHACǁcompare_with_standard_errors__mutmut_9,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_10": xǁSpatialHACǁcompare_with_standard_errors__mutmut_10,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_11": xǁSpatialHACǁcompare_with_standard_errors__mutmut_11,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_12": xǁSpatialHACǁcompare_with_standard_errors__mutmut_12,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_13": xǁSpatialHACǁcompare_with_standard_errors__mutmut_13,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_14": xǁSpatialHACǁcompare_with_standard_errors__mutmut_14,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_15": xǁSpatialHACǁcompare_with_standard_errors__mutmut_15,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_16": xǁSpatialHACǁcompare_with_standard_errors__mutmut_16,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_17": xǁSpatialHACǁcompare_with_standard_errors__mutmut_17,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_18": xǁSpatialHACǁcompare_with_standard_errors__mutmut_18,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_19": xǁSpatialHACǁcompare_with_standard_errors__mutmut_19,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_20": xǁSpatialHACǁcompare_with_standard_errors__mutmut_20,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_21": xǁSpatialHACǁcompare_with_standard_errors__mutmut_21,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_22": xǁSpatialHACǁcompare_with_standard_errors__mutmut_22,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_23": xǁSpatialHACǁcompare_with_standard_errors__mutmut_23,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_24": xǁSpatialHACǁcompare_with_standard_errors__mutmut_24,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_25": xǁSpatialHACǁcompare_with_standard_errors__mutmut_25,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_26": xǁSpatialHACǁcompare_with_standard_errors__mutmut_26,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_27": xǁSpatialHACǁcompare_with_standard_errors__mutmut_27,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_28": xǁSpatialHACǁcompare_with_standard_errors__mutmut_28,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_29": xǁSpatialHACǁcompare_with_standard_errors__mutmut_29,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_30": xǁSpatialHACǁcompare_with_standard_errors__mutmut_30,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_31": xǁSpatialHACǁcompare_with_standard_errors__mutmut_31,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_32": xǁSpatialHACǁcompare_with_standard_errors__mutmut_32,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_33": xǁSpatialHACǁcompare_with_standard_errors__mutmut_33,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_34": xǁSpatialHACǁcompare_with_standard_errors__mutmut_34,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_35": xǁSpatialHACǁcompare_with_standard_errors__mutmut_35,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_36": xǁSpatialHACǁcompare_with_standard_errors__mutmut_36,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_37": xǁSpatialHACǁcompare_with_standard_errors__mutmut_37,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_38": xǁSpatialHACǁcompare_with_standard_errors__mutmut_38,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_39": xǁSpatialHACǁcompare_with_standard_errors__mutmut_39,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_40": xǁSpatialHACǁcompare_with_standard_errors__mutmut_40,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_41": xǁSpatialHACǁcompare_with_standard_errors__mutmut_41,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_42": xǁSpatialHACǁcompare_with_standard_errors__mutmut_42,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_43": xǁSpatialHACǁcompare_with_standard_errors__mutmut_43,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_44": xǁSpatialHACǁcompare_with_standard_errors__mutmut_44,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_45": xǁSpatialHACǁcompare_with_standard_errors__mutmut_45,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_46": xǁSpatialHACǁcompare_with_standard_errors__mutmut_46,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_47": xǁSpatialHACǁcompare_with_standard_errors__mutmut_47,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_48": xǁSpatialHACǁcompare_with_standard_errors__mutmut_48,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_49": xǁSpatialHACǁcompare_with_standard_errors__mutmut_49,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_50": xǁSpatialHACǁcompare_with_standard_errors__mutmut_50,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_51": xǁSpatialHACǁcompare_with_standard_errors__mutmut_51,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_52": xǁSpatialHACǁcompare_with_standard_errors__mutmut_52,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_53": xǁSpatialHACǁcompare_with_standard_errors__mutmut_53,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_54": xǁSpatialHACǁcompare_with_standard_errors__mutmut_54,
        "xǁSpatialHACǁcompare_with_standard_errors__mutmut_55": xǁSpatialHACǁcompare_with_standard_errors__mutmut_55,
    }
    xǁSpatialHACǁcompare_with_standard_errors__mutmut_orig.__name__ = (
        "xǁSpatialHACǁcompare_with_standard_errors"
    )


class DriscollKraayComparison:
    """
    Compare Spatial HAC with Driscoll-Kraay standard errors.

    Both methods are robust to spatial correlation, but:
    - Spatial HAC uses explicit geographic distance
    - Driscoll-Kraay assumes uniform spatial correlation
    """

    @staticmethod
    def compare(
        spatial_hac_se: np.ndarray,
        driscoll_kraay_se: np.ndarray,
        param_names: list | None = None,
    ) -> pd.DataFrame:
        """
        Compare Spatial HAC and Driscoll-Kraay standard errors.

        Parameters
        ----------
        spatial_hac_se : np.ndarray
            Standard errors from Spatial HAC
        driscoll_kraay_se : np.ndarray
            Standard errors from Driscoll-Kraay
        param_names : list, optional
            Parameter names for the DataFrame

        Returns
        -------
        pd.DataFrame
            Comparison of standard errors
        """
        if param_names is None:
            param_names = [f"param_{i}" for i in range(len(spatial_hac_se))]

        comparison_df = pd.DataFrame(
            {
                "parameter": param_names,
                "Spatial_HAC": spatial_hac_se,
                "Driscoll_Kraay": driscoll_kraay_se,
                "ratio": spatial_hac_se / driscoll_kraay_se,
                "pct_diff": 100 * (spatial_hac_se - driscoll_kraay_se) / driscoll_kraay_se,
            }
        )

        # Add summary statistics
        comparison_df.loc["mean"] = comparison_df.mean(numeric_only=True)
        comparison_df.loc["median"] = comparison_df.median(numeric_only=True)
        comparison_df.loc["std"] = comparison_df.std(numeric_only=True)

        return comparison_df
