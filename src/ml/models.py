"""Models module.

This module defines all models used in the challenge. Models are defined
as child classes of an BaseModel class
"""

import random

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


from ml.base import BaseModel


class MostFrequentClassifier(BaseModel):
    """Simple model that predict most frequent class in training set."""

    def __init__(self):
        super().__init__()
        self.most_frequent_class = None
        self._estimator_type = "classifier"  # align with other models

    def __str__(self):
        return "MostFrequentClassifier"

    def build(self):
        """Step not needed for this classifier."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        assert len(y), "No values passed"
        self.most_frequent_class = y.mode().values[0]

    def predict(self, y: pd.Series) -> pd.Series:
        y_pred = pd.Series(self.most_frequent_class)
        return y_pred

    def get_scores(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        pass


class SmartRandomClassifier(BaseModel):
    """Simple model that predict random class.

    Probability to predict a class is proportional to class frequency in
    training set
    """

    def __init__(self):
        super().__init__()
        self.class_list = None
        self.class_weights = None

    def __str__(self):
        return "SmartRandomClassifier"

    def build(self):
        """Step not needed for this classifier."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        assert len(y), "No values passed"

        # Store  class freq
        self.class_list = y.unique()
        self.class_weights = y.value_counts()[self.class_list].tolist()

    def predict(self, y: pd.DataFrame) -> pd.Series:
        extractions = random.choices(
            population=self.class_list, weights=self.class_weights, k=len(y)
        )
        y_pred = pd.Series(extractions)
        return y_pred

    def get_scores(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        pass


class SimpleRegressionClassifier(BaseModel):
    """Simple regression classifier."""

    def __init__(self):
        super().__init__()
        self.mdl = None

    def __str__(self):
        return "SimpleRegressionClassifier"

    def build(self):
        self.mdl = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("predictor", LogisticRegression(penalty="l2")),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.mdl.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.mdl.predict(X)
        return y_pred
