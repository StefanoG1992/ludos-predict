"""Base model.

Contain a base model class to be used for predictive purposes.
A predictive model must contain following methods:

Abstract methods
- build: define model structure (e.g. simple class, pipeline, keras layers...)
The final model is expected to be instanced as self.model
- fit: training the model. Require training explanatory data X and response y
- predict: predicting values. Require test response y

Following methods are not mandatory instead:
- get_scores: fit the model via cross-validation and compute main metrics
Note, the model most get built before
"""

import logging

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score

from typing import TypeVar

# get logger if any
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract predictive model."""

    def __init__(self):
        self.mdl = None
        self.params = None

    def __str__(self):
        return "Abstract model class"

    @abstractmethod
    def build(self) -> None:
        """Building model structure.

        This step defines the model structure by instancing the model
        object in self.mdl. Model parameters are encoded in the
        function.
        """
        raise NotImplementedError("Missing building method")

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Training the model.

        This step trains the model with training data.
        :param X_train: training data
        :param y_train: training response
        """
        raise NotImplementedError("Missing training method")

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Prediction step.

        :param X_test: explanatory data to predict
        :return: predicted response
        """
        raise NotImplementedError("Missing prediction method")

    def run(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Executes build and fit methods."""
        self.build()
        self.fit(X, y)

    def get_scores(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict[str, np.float64]:
        """Evaluate the model via cross validation over several metrics.

        Train-test split is performed during cross validation.
        Output is a dictionary containing the results for each
        metric. As cross validation returns a result tuple for each run,
        we take the average over the runs.

        :param X: explanatory data
        :param y: response
        :param cv: number of folds. Default is 5
        :return: Metrics results {metric_name: avg_metric_result}
        """
        # check if self.mdl is instanced - if not, exit the function
        assert self.mdl is not None, "Model not built"

        # define metrics dictionary
        scores: dict = {}

        logger.info(f"Compute metrics for {self}")

        # accuracy
        scores["accuracy"] = np.mean(
            cross_val_score(estimator=self.mdl, X=X, y=y, cv=cv, scoring="accuracy")
        )

        # balanced accuracy
        scores["balanced_accuracy"] = np.mean(
            cross_val_score(
                estimator=self.mdl, X=X, y=y, cv=cv, scoring="balanced_accuracy"
            )
        )

        return scores

    def optimize(self):
        """Optimization step.

        Do not raise Exception is not implemented
        """
        pass


# define class model
Model = TypeVar("Model", bound=BaseModel)
