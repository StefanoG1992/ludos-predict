"""Abstract class model.

Contain a base model class to be used to build predictive models.
An abstract model must contain following methods:
- .fit: fit training data
- .predict: predict test data
- .evaluate: evaluate the model

The only abstract method is .build, which defines the model structure and
instances it on self.mdl. This depends on specific model
"""

import logging

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer

import core.checks as check

from ml.scores import f_score, recall

from typing import TypeVar

# get logger if any
logger = logging.getLogger(__name__)

# define score name
Score = TypeVar("Score", bound=str)


class BaseModel(ABC):
    """Abstract predictive model."""

    def __init__(self):
        self._model = None
        self._scorers = [
            "accuracy",
            "f1_score",
            "f2_score",
            "recall",
        ]

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

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Training the model.

        This step trains the model with training data.
        The training logic mimics scikit-learn .fit logic.
        If the model has a different logic, it should be overriden
        :param X_train: training data
        :param y_train: training response
        """
        self._model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Prediction step.

        The prediction logic mimics scikit-learn .predict logic.
        If the model has a different logic, it should be overriden
        :param X_test: explanatory data to predict
        :return: predicted response
        """
        y_pred = self._model.predict(X_test)
        return y_pred

    def run(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Executes build and fit methods."""
        self.build()
        self.fit(X, y)

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict[Score, float]:
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
        # model must be already built
        if self._model is None:
            raise NotImplementedError("Model not built")

        # define metrics dictionary
        scores: dict[Score, float] = {}

        logger.info(f"Compute metrics for {self}")

        # accuracy
        scores["accuracy"] = np.mean(
            cross_val_score(estimator=self._model, X=X, y=y, cv=cv, scoring="accuracy")
        ).round(4)

        # balanced accuracy
        scores["balanced_accuracy"] = np.mean(
            cross_val_score(
                estimator=self._model,
                X=X,
                y=y,
                cv=cv,
                scoring="balanced_accuracy",
            )
        ).round(4)

        # check that y has integer labels
        check.response_labels_are_int(y)

        # compute specific response class scores
        # Passing None computes the average
        labels = list(y.unique()) + [None]
        for i in labels:
            # compute f1 score
            # order names
            label_name = f"label_{i}" if i is not None else "avg"

            scores[f"f1_score_{label_name}"] = np.mean(
                cross_val_score(
                    estimator=self._model,
                    X=X,
                    y=y,
                    cv=cv,
                    scoring=make_scorer(f_score, beta=1.0, label=i),
                )
            ).round(4)

            # compute f2 score
            scores[f"f2_score_{label_name}"] = np.mean(
                cross_val_score(
                    estimator=self._model,
                    X=X,
                    y=y,
                    cv=cv,
                    scoring=make_scorer(f_score, beta=2.0, label=i),
                )
            ).round(4)

            # compute recall
            scores[f"recall_{label_name}"] = np.mean(
                cross_val_score(
                    estimator=self._model,
                    X=X,
                    y=y,
                    cv=cv,
                    scoring=make_scorer(recall, label=i),
                )
            ).round(4)

        # float conversion - needed for yaml.dump
        scores = {k: float(v) for k, v in scores.items()}

        return scores

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict[str, list],
        scoring: str | None,
        cv: int = 5,
    ):
        """Optimization step.

        Take a dictionary of model parameters and pass it through a grid search.
        Once computed, override the self._model parameters present in .build()
        method.

        Optimize step may be used *only* during testing phase to better finetune
        the self._model in .build().
        Final parameters are meant to be written explicitly in the model design,
        not computed at every run.

        :param X: explanatory data
        :param y: response
        :param param_grid: parameters dictionary. Must be a dictionary with
        structure {param_name: [param_val_0, param_val_1, ...]}
        or {param_name: math.distribution}
        :param scoring: scoring metric
        :param cv: cross validation n-folds. Default is 5
        """
        # model must be already built
        if self._model is None:
            raise NotImplementedError("Model not built")

        if scoring not in self._scorers:
            raise NotImplementedError("Score not implemented")

        scoring_map = {
            "accuracy": None,  # None is default for GridSearchCV accuracy
            "f1_score": make_scorer(f_score, beta=1.0),
            "f2_score": make_scorer(f_score, beta=2.0),
            "recall": make_scorer(recall),
        }

        clf = GridSearchCV(
            estimator=self._model,
            param_grid=param_grid,
            cv=cv,
            refit=True,
            scoring=scoring_map[scoring],
        )

        clf.fit(X, y)

        logger.debug(f"Best parameters for {self}:\n{clf.best_params_}")

        # reassign model
        self._model = clf.best_estimator_


# define class model
Model = TypeVar("Model", bound=BaseModel)

# define model name
ModelName = TypeVar("ModelName", bound=str)
