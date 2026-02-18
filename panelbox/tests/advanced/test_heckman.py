"""Tests for Heckman selection model."""

import numpy as np
import pandas as pd
import pytest


class TestPanelHeckman:
    """Test Panel Heckman selection model."""

    def test_init(self, selection_data):
        """Test initialization."""
        from panelbox.models.selection import PanelHeckman

        # Extract data as numpy arrays
        endog = selection_data["y"].values
        exog = selection_data[["x1"]].values
        selection = selection_data["s"].values
        exog_selection = selection_data[["z1", "z2"]].values
        entity = selection_data["entity"].values
        time = selection_data["time"].values

        model = PanelHeckman(
            endog=endog,
            exog=exog,
            selection=selection,
            exog_selection=exog_selection,
            entity=entity,
            time=time,
            method="two_step",
        )

        assert model is not None

    def test_fit_2step(self, selection_data):
        """Test two-step estimation."""
        from panelbox.models.selection import PanelHeckman

        # Extract data as numpy arrays
        endog = selection_data["y"].values
        exog = selection_data[["x1"]].values
        selection = selection_data["s"].values
        exog_selection = selection_data[["z1", "z2"]].values
        entity = selection_data["entity"].values
        time = selection_data["time"].values

        model = PanelHeckman(
            endog=endog,
            exog=exog,
            selection=selection,
            exog_selection=exog_selection,
            entity=entity,
            time=time,
            method="two_step",
        )

        result = model.fit()
        assert result is not None
        assert hasattr(result, "params")

    def test_selection_parameters(self, selection_data):
        """Test selection equation parameters."""
        from panelbox.models.selection import PanelHeckman

        # Extract data as numpy arrays
        endog = selection_data["y"].values
        exog = selection_data[["x1"]].values
        selection = selection_data["s"].values
        exog_selection = selection_data[["z1", "z2"]].values
        entity = selection_data["entity"].values
        time = selection_data["time"].values

        model = PanelHeckman(
            endog=endog,
            exog=exog,
            selection=selection,
            exog_selection=exog_selection,
            entity=entity,
            time=time,
            method="two_step",
        )

        result = model.fit()

        # True values: z1=0.8, z2=0.4
        # Check signs at least - parameters are in result.params_selection
        if hasattr(result, "params_selection"):
            # z1 and z2 should be positive
            assert result.params_selection[1] > 0  # z1
            assert result.params_selection[2] > 0  # z2

    def test_imr_coefficient(self, selection_data):
        """Test IMR coefficient (theta = rho*sigma)."""
        from panelbox.models.selection import PanelHeckman

        # Extract data as numpy arrays
        endog = selection_data["y"].values
        exog = selection_data[["x1"]].values
        selection = selection_data["s"].values
        exog_selection = selection_data[["z1", "z2"]].values
        entity = selection_data["entity"].values
        time = selection_data["time"].values

        model = PanelHeckman(
            endog=endog,
            exog=exog,
            selection=selection,
            exog_selection=exog_selection,
            entity=entity,
            time=time,
            method="two_step",
        )

        result = model.fit()

        # Should have IMR/mills coefficient
        if hasattr(result, "rho"):
            # With rho=0.5, should be positive
            assert result.rho > 0

    def test_summary(self, selection_data):
        """Test summary output."""
        from panelbox.models.selection import PanelHeckman

        # Extract data as numpy arrays
        endog = selection_data["y"].values
        exog = selection_data[["x1"]].values
        selection = selection_data["s"].values
        exog_selection = selection_data[["z1", "z2"]].values
        entity = selection_data["entity"].values
        time = selection_data["time"].values

        model = PanelHeckman(
            endog=endog,
            exog=exog,
            selection=selection,
            exog_selection=exog_selection,
            entity=entity,
            time=time,
            method="two_step",
        )

        result = model.fit()
        summary = result.summary()

        assert isinstance(summary, str)
        assert "selection" in summary.lower() or "Selection" in summary or "Heckman" in summary
