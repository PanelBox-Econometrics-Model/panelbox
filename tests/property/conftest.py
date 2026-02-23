"""Hypothesis profiles for property-based testing."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

# Profile for CI (faster)
settings.register_profile(
    "ci",
    max_examples=30,
    deadline=60000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# Profile for local development (more thorough)
settings.register_profile(
    "dev",
    max_examples=100,
    deadline=120000,
)

# Profile for deep validation (periodic)
settings.register_profile(
    "thorough",
    max_examples=500,
    deadline=None,
)

# Load profile from environment variable, defaulting to ci
profile = os.environ.get("HYPOTHESIS_PROFILE", "ci")
settings.load_profile(profile)
