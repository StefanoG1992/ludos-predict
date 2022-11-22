"""Integrity checks.

This module contains all integrity/validation checks, boolean flags,
present in the script.g
"""

import logging

import pandas as pd
import numpy.typing as npt

# retrieve logger
logger = logging.getLogger(__name__)


def response_labels_are_int(y: pd.Series) -> None:
    """Check whether response labels are labelled 0, 1, ...

    n
    """
    if list(y.unique()) != [i for i in range(len(y.unique()))]:
        msg = "Response must have classes labelled as integers 0, 1, 2, ...n"
        raise ValueError(msg)


def label_in_range(label: int, output: npt.NDArray) -> None:
    """Check whether label is in y.unique()"""
    if label >= len(output):
        msg = f"Label index out of bound. Max label can be {len(output) - 1}."
        raise ValueError(msg)
