"""Scores module.

Contain all scoring functions that are passed in the .evaluate model.
The y response classes are assumed to be ordered integers, starting with
0
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

import core.checks as check

from sklearn.metrics import recall_score, fbeta_score


# L1 fbeta score
def f_score(
    y: pd.Series, y_pred: pd.Series, label: int, beta: float | np.float64
) -> np.float64:
    """Compute fbeta score for a given label.

    Our scoring methods assume response classes are ordered integers
    starting from 0.
    :param y: response
    :param y_pred: predicted response
    :param label: label. Must be int in 0, ..., len(y.unique())-1
    :param beta: beta score
    :return: f beta score for given label
    """
    output: npt.NDArray[np.float64] = fbeta_score(
        y_true=y, y_pred=y_pred, average=None, beta=beta
    )
    check.label_in_range(label, output)
    return output[label]


# recall
def recall(y: pd.Series, y_pred: pd.Series, label: int) -> np.float64:
    """Compute recall for a given label.

    Our scoring methods assume response classes are ordered integers
    starting from 0.
    :param y: response
    :param y_pred: predicted response
    :param label: label. Must be int in 0, ... len(y.unique())-1
    :return: f beta score for given label
    """
    output: npt.NDArray[np.float64] = recall_score(
        y_true=y, y_pred=y_pred, average=None
    )
    check.label_in_range(label, output)
    return output[label]
