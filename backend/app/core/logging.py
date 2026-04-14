"""Structured logging setup."""

import logging
import sys

from backend.app.core.config import settings


def setup_logging() -> logging.Logger:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level, format=fmt, force=True)
    logger = logging.getLogger("raf")
    logger.setLevel(level)
    return logger


logger = setup_logging()
