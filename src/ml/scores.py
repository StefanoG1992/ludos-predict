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


# fbeta score
def f_score(
    y: pd.Series,
    y_pred: pd.Series,
    beta: float | np.float64,
    label: int | None = None,
) -> np.float64:
    """Compute fbeta score for a given label.

    Our scoring methods assume response classes are ordered integers
    starting from 0.
    :param y: response
    :param y_pred: predicted response
    :param beta: beta score
    :param label: label. If passed, return the score for the specific label.
    If not passed (default), returns the average over labels.
    Must be int in 0, ..., len(y.unique())-1.
    :return: f beta score for given label
    """
    output: npt.NDArray[np.float64] = fbeta_score(
        y_true=y, y_pred=y_pred, average=None, beta=beta
    )
    # if label is passed, return output for specific label
    if label is not None:
        check.label_in_range(label, output)
        return output[label]
    return np.mean(output)


# recall
def recall(y: pd.Series, y_pred: pd.Series, label: int | None = None) -> np.float64:
    """Compute recall for a given label.

    Our scoring methods assume response classes are ordered integers
    starting from 0.
    :param y: response
    :param y_pred: predicted response
    :param label: label. If passed, return the score for the specific label.
    If not passed (default), returns the average over labels.
    Must be int in 0, ..., len(y.unique())-1.
    :return: f beta score for given label
    """
    output: npt.NDArray[np.float64] = recall_score(
        y_true=y, y_pred=y_pred, average=None
    )
    # if label is passed, return output for specific label
    if label is not None:
        check.label_in_range(label, output)
        return output[label]
    return np.mean(output)
