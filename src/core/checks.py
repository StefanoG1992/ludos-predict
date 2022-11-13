"""Integrity checks

This module contains all integrity/validation checks, boolean flags, present in the script.
"""

from pathlib import Path

import click
import logging


# retrieve logger
logger = logging.getLogger(__name__)


def path_callback(
        ctx: click.core.Context,  # only needed for click
        param: click.core.Option,  # only needed for click
        path: str | None) -> Path:
    """
    Callback function to validate paths.

    Path existence is checked, if path does not exist returns error.
    None path is accepted as option; in this case, returns None.

    :param ctx: click context (needed for click purposes)
    :param param: click param (needed for click purposes)
    :param path: path to validate
    :return: path
    """
    # if path is None, keeps None
    if path is None:
        logger.debug("No path has been passed.")
        return None
    # read as path
    logger.debug(f"Validate path {path}")
    path = Path(path)

    if not path.exists():
        msg = f"Path {path} does not exist."
        logger.error(msg)
        raise click.BadParameter(msg)
    return path

