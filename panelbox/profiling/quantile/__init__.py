"""
Performance profiling for quantile regression.

This module provides tools for profiling and benchmarking quantile regression
implementations to identify performance bottlenecks and optimize for large datasets.
"""

from __future__ import annotations

from .performance import BenchmarkReport, PerformanceProfiler

__all__ = ["BenchmarkReport", "PerformanceProfiler"]
