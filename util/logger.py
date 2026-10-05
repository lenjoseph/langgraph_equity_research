import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: The name of the logger (usually __name__)

    Returns:
        A configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers, assume it's already configured
    if logger.hasHandlers():
        return logger

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
