"""Machine learning module.

Define all models used in the challenge.
Models are defined through an abstract BaseModel class which implements basic ml
methods (.fit, .predict, .evaluate) unless there is need of overriding.
See module ml.base for details.

The model-specific method is .build, which defines its structure and instances
it on self.mdl.
"""

import random

import pandas as pd

from ml import scores as sc

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline as PipelineSMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

from ml.base import BaseModel, Score


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
        y_pred = pd.Series([self.most_frequent_class for _ in range(len(y))])
        return y_pred

    def evaluate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """Evaluation step.

        Cross validation not implemented for this classifier.
        """
        # build defined this way for this classifier
        self.fit(X, y)

        # predict values
        y_pred = self.predict(y)

        # add scores
        scores: dict[Score, float] = {"accuracy": accuracy_score(y, y_pred)}

        # add label specific scores
        for i in y.unique():
            # f1 score
            scores[f"f1_score_label_{i}"] = sc.f_score(y, y_pred, label=i, beta=1.0)

            # f2 score
            scores[f"f2_score_label_{i}"] = sc.f_score(y, y_pred, label=i, beta=2.0)

            # recall
            scores[f"recall_label_{i}"] = sc.recall(
                y,
                y_pred,
                label=i,
            )

        return scores


class SmartRandomClassifier(BaseModel):
    """Simple model that predict random class.

    Probability to predict a class is proportional to class frequency in
    training set
    """

    def __init__(self):
        super().__init__()
        self.class_list = None
        self.class_weights = None
        self._estimator_type = "classifier"  # align with other models

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

    def evaluate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """Evaluation step.

        Cross validation not implemented for this classifier.
        """
        # build defined this way for this classifier
        self.fit(X, y)

        # predict values
        y_pred: pd.Series = self.predict(y)

        # add scores
        scores: dict[Score, float] = {"accuracy": accuracy_score(y, y_pred)}

        # add label specific scores
        for i in y.unique():
            # f1 score
            scores[f"f1_score_label_{i}"] = sc.f_score(y, y_pred, label=i, beta=1.0)

            # f2 score
            scores[f"f2_score_label_{i}"] = sc.f_score(y, y_pred, label=i, beta=2.0)

            # recall
            scores[f"recall_label_{i}"] = sc.recall(
                y,
                y_pred,
                label=i,
            )

        return scores


class SimpleRegressionClassifier(BaseModel):
    """Simple regression classifier."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SimpleRegressionClassifier"

    def build(self):
        self._model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "predictor",
                    LogisticRegression(
                        penalty="l2",
                        C=1.5,
                        class_weight="balanced",
                        max_iter=500,
                    ),
                ),
            ]
        )


class CatBoost(BaseModel):
    """Cat boost classifier."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "CatBoostClassifier"

    def build(self):
        self._model = CatBoostClassifier(
            silent=True,
            early_stopping_rounds=200,
            num_trees=500,
        )


class RandomForest(BaseModel):
    """Random Tree forest classifier."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "RandomForestClassifier"

    def build(self):
        self._model = RandomForestClassifier(
            n_estimators=500,
        )


class XGB(BaseModel):
    """XGB Classifier."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "XGBoostClassifier"

    def build(self):
        self._model = XGBClassifier(
            n_estimators=500,
            n_jobs=10,
            max_depth=10,
            subsample=0.8,
            gpu_id=0,
            alpha=1.0,
            objective="multi:softmax",
            num_class=3,
            min_child_weight=1,
        )


class CatBoostSmote(BaseModel):
    """Cat boost classifier with SMOTE algorithm."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "CatBoostSmoteClassifier"

    def build(self):
        self._model = PipelineSMOTE(
            [
                ("SMOTE", SMOTE()),
                ("under", RandomUnderSampler()),
                (
                    "model",
                    CatBoostClassifier(
                        early_stopping_rounds=50,
                        num_trees=500,
                        verbose=0,
                    ),
                ),
            ]
        )


class XGBSmote(BaseModel):
    """XGB with SMOTE algorithm."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "XGBSmoteClassifier"

    def build(self):
        self._model = PipelineSMOTE(
            [
                ("SMOTE", SMOTE()),
                ("under", RandomUnderSampler()),
                (
                    "model",
                    XGBClassifier(
                        n_jobs=10,
                        max_depth=5,
                        subsample=0.8,
                    ),
                ),
            ]
        )
