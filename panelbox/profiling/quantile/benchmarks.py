"""Performance benchmarks for quantile regression estimators.

This module provides timing and profiling utilities for comparing the
performance of different quantile regression optimization backends.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
