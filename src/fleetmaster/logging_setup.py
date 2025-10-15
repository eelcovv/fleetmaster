"""Initialize the  logger with a rich console handler."""

import logging
from logging import Logger

from rich.logging import RichHandler


def setup_general_logger() -> Logger:
    """Initialize the central logger .

    Returns
    -------
        Logger: The configured logger.

    """
    logger = logging.getLogger(__name__)
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
