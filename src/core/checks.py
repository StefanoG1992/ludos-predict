"""Integrity checks.

This module contains all integrity/validation checks, boolean flags,
present in the script
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
        logger.error(msg)
        raise ValueError(msg)


def label_in_range(label: int, output: npt.NDArray) -> None:
    """Check whether label is in y.unique()"""
    if label >= len(output):
        msg = f"Label index out of bound. Max label can be {len(output) - 1}."
        logger.error(msg)
        raise ValueError(msg)


def models_scores(all_scores: dict[str, dict[str, float]]):
    """Check integrity of model scores.

    Given the dictionary containing all scores results per model, check:
    1. All models have the same scores computed - raise an error otherwise
    2. All scores have been computed - raise a warning otherwise
    :param all_scores: model scores dictionary computed in main challenge
    """
    if not len(all_scores):
        msg = "Empty scores dictionary"
        logger.error(msg)
        raise Exception(msg)

    # instance scores names for first model - they have to be all the same
    first_model_name: str = list(all_scores.keys())[0]
    scores_first_model: set[str] = set(all_scores[first_model_name])

    # check all models have the same score
    for model_name, model_scores in all_scores.items():
        scores_for_model: set[str] = set(model_scores.keys())
        if scores_first_model != scores_for_model:
            msg = (
                f"Model {first_model_name} has scores: {scores_first_model}\n.",
                f"Model {model_name} has different scores: {scores_for_model}",
            )
            logger.error(msg)
            raise ValueError(msg)
        for score_name, score_value in model_scores.items():
            if pd.isna(score_value):
                msg = f"Score {score_name} for model {model_name} is null."
                logger.warning(msg)
