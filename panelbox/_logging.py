"""Logging configuration for PanelBox library."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Parameters
    ----------
    name : str
        Module name, typically ``__name__``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    return logging.getLogger(name)


def set_verbosity(level: int = logging.WARNING) -> None:
    """Set the verbosity level for all PanelBox loggers.

    Parameters
    ----------
    level : int, default logging.WARNING
        Logging level. Use logging.DEBUG, logging.INFO,
        logging.WARNING, logging.ERROR, or logging.CRITICAL.

    Examples
    --------
    >>> import logging
    >>> import panelbox
    >>> panelbox.set_verbosity(logging.DEBUG)  # see all messages
    >>> panelbox.set_verbosity(logging.WARNING)  # only warnings (default)
    """
    logging.getLogger("panelbox").setLevel(level)
