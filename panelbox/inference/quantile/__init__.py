"""
Inference methods for quantile regression models.

This module provides bootstrap and analytic inference methods for
quantile regression, including cluster bootstrap, pairs bootstrap,
wild bootstrap, and subsampling bootstrap.
"""

from .bootstrap import BootstrapResult, QuantileBootstrap, bootstrap_qr

__all__ = [
    "bootstrap_qr",
    "QuantileBootstrap",
    "BootstrapResult",
]
