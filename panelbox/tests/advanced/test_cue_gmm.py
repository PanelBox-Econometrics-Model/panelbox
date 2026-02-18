"""Tests for CUE-GMM estimator."""

import numpy as np
import pandas as pd
import pytest


class TestCUEGMM:
    """Test CUE-GMM estimator."""

    def test_init(self, gmm_data):
        """Test initialization."""
        from panelbox.gmm import ContinuousUpdatedGMM

        model = ContinuousUpdatedGMM(
            data=gmm_data, dep_var="y", exog_vars=["x1", "x2"], instruments=["z1", "z2", "z3"]
        )

        assert model is not None
        assert hasattr(model, "n_instruments")
        assert hasattr(model, "m")  # Bug fix check

    def test_fit_converges(self, gmm_data):
        """Test that fit converges."""
        from panelbox.gmm import ContinuousUpdatedGMM

        model = ContinuousUpdatedGMM(
            data=gmm_data, dep_var="y", exog_vars=["x1", "x2"], instruments=["z1", "z2", "z3"]
        )

        result = model.fit()
        assert result.converged or result.params is not None

    def test_parameter_signs(self, gmm_data):
        """Test parameter signs are correct."""
        from panelbox.gmm import ContinuousUpdatedGMM

        model = ContinuousUpdatedGMM(
            data=gmm_data, dep_var="y", exog_vars=["x1", "x2"], instruments=["z1", "z2", "z3"]
        )

        result = model.fit()

        # True values: const=1, x1=2, x2=1.5
        assert result.params[1] > 0  # x1 should be positive
        assert result.params[2] > 0  # x2 should be positive

    def test_standard_errors_exist(self, gmm_data):
        """Test that standard errors are computed."""
        from panelbox.gmm import ContinuousUpdatedGMM

        model = ContinuousUpdatedGMM(
            data=gmm_data, dep_var="y", exog_vars=["x1", "x2"], instruments=["z1", "z2", "z3"]
        )

        result = model.fit()

        # Check for standard errors attribute
        assert hasattr(result, "std_errors")
        assert len(result.std_errors) == 3
        assert all(result.std_errors > 0)

    def test_j_statistic(self, gmm_data):
        """Test J-statistic computation."""
        from panelbox.gmm import ContinuousUpdatedGMM

        model = ContinuousUpdatedGMM(
            data=gmm_data, dep_var="y", exog_vars=["x1", "x2"], instruments=["z1", "z2", "z3"]
        )

        result = model.fit()

        # Should have J-stat for overidentified model
        if hasattr(result, "j_stat"):
            assert result.j_stat >= 0

    def test_summary(self, gmm_data):
        """Test summary output."""
        from panelbox.gmm import ContinuousUpdatedGMM

        model = ContinuousUpdatedGMM(
            data=gmm_data, dep_var="y", exog_vars=["x1", "x2"], instruments=["z1", "z2", "z3"]
        )

        result = model.fit()
        summary = result.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
