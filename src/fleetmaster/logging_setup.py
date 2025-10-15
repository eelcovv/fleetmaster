"""Initialize the  logger with a rich console handler."""

import logging
from logging import Logger

from rich.logging import RichHandler

from . import LOGGER_NAME


def initialize_logging() -> Logger:
    """Initialize the central logger with a rich console handler.

    Returns
    -------
        Logger: The configured logger instance with RichHandler.

    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.WARNING)
    logger.propagate = True  # Allow logs to propagate to parent loggers

    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        console_handler = RichHandler(
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)

    return logger
