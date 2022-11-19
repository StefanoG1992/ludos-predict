"""Models module.

This module defines all models used in the challenge. Models are defined
as child classes of an BaseModel class
"""

import random

import pandas as pd

from ml.base import BaseModel


class MostFrequentClassifier(BaseModel):
    """Simple model that predict most frequent class in training set."""

    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None
        self.most_frequent_class = None

    def build(self):
        """Step not needed for this classifier."""
        pass

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        assert len(y), "No values passed"
        self.most_frequent_class = y.mode().values[0]

    def predict(self, y: pd.Series) -> pd.Series:
        y_pred = pd.DataFrame(
            self.most_frequent_class,
            index=self.X.index,
        )
        return y_pred


class SmartRandomClassifier(BaseModel):
    """Simple model that predict random class.

    Probability to predict a class is proportional to class frequency in
    training set
    """

    def __init__(self):
        self.class_list = None
        self.class_weights = None

    def build(self):
        """Step not needed for this classifier."""
        pass

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
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
