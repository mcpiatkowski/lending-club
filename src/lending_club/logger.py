"""Logging module."""

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d"
)

log = logging.getLogger("lending-club")
