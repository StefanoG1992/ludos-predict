"""Base model.

Contain a base model class to be used for predictive purposes.
A predictive model must contain following methods:

- build: define model structure (e.g. simple class, pipeline, keras layers...)
The final model is expected to be instanced as self.model
- train: training the model. Require training explanatory data X and response y
- predict: predicting values. Require test response y

Following methods are not mandatory instead:
- crossval: perform cross validation check of the model
- optimize: performing model parameters optimization
"""

import pandas as pd

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract predictive model."""

    @abstractmethod
    def build(self) -> None:
        """Building model structure.

        Initialize self.mdl
        """
        raise NotImplementedError("Missing building method")

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Training the model.

        :param X: training data
        :param y: training response
        """
        raise NotImplementedError("Missing training method")

    @abstractmethod
    def predict(self, y: pd.Series) -> pd.Series:
        """Prediction step.

        :param y: test response
        :return: predicted response
        """
        raise NotImplementedError("Missing prediction method")

    def crossval(self):
        """Cross validation step.

        Do not raise Exception is not implemented
        """
        pass

    def optimize(self):
        """Optimization step.

        Do not raise Exception is not implemented
        """
        pass
