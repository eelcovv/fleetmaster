"""Initialize the  logger with a rich console handler."""

import logging
from logging import Logger


def setup_general_logger(level: int = logging.INFO) -> Logger:
    """Initialize the central logger .

    Returns
    -------
        Logger: The configured logger.

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = True  # Allow logs to propagate to parent loggers

    if logger.hasHandlers():
        logger.handlers.clear()

    return logger
