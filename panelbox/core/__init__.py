"""Core infrastructure for panel data handling.

This subpackage provides:

- ``PanelData`` : Container for panel data with entity/time structure.
- ``SerializableMixin`` : Mixin for save/load serialization support.
"""

from __future__ import annotations

from panelbox.core.panel_data import PanelData
from panelbox.core.serialization import SerializableMixin

__all__ = ["PanelData", "SerializableMixin"]
