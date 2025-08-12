"""Simple logging configuration.

Use func:`setup_logging` at the start of your scripts to configure a
consistent logging format across the project. You can extend this helper to
add file handlers, adjust log levels, or integrate with other logging frameworks.
"""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt)