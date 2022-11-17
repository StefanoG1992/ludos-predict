"""Models module.

This module contains all self-made ML models for data science purposes
"""

import logging
import shap

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier


# initialize logger
logger = logging.getLogger(__name__)


def get_shap_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 10,
) -> (list, list[str]):
    """Compute shapley features for a model using CatBoostClassifier as model.

    :param X: Explanable data
    :param y: Classes
    :param n_top: top n features to return. Default = 10
    :return: Tuple (Explainer.shap_values, list of sorted best features)
    """
    logger.info(f"Computing {n_top} best features")

    # split values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.7, shuffle=True, stratify=y
    )

    # rescale values as CatBoost is scaling-sensitive
    scl = StandardScaler()
    X_train_scl = scl.fit_transform(X_train)
    X_val_scl = scl.transform(X_val)

    logger.info("Training model")
    # simple model
    mdl = CatBoostClassifier(allow_writing_files=False)
    mdl.fit(
        X=X_train_scl,
        y=y_train,
        verbose=0,
        eval_set=(X_val_scl, y_val),
        use_best_model=True,
    )

    logger.info("Explaining data")

    # shap explainer
    explainer = shap.TreeExplainer(mdl)
    # shap values are a list
    shap_values: list = explainer.shap_values(X_train_scl)

    # generate feature importances df
    shap_array = np.array(shap_values)  # shape = (# classes, # data, # features)
    logger.debug(f"Shapley coefficients array has shape: {shap_array.shape}")

    # retrieve absolute importance
    shap_array = np.abs(shap_array)  # get absolute importance
    shap_array = shap_array.mean(1)  # shape (# classes, # features)

    # get it into a df
    features_df = pd.DataFrame(shap_array)
    features_df.columns = X.columns

    # find best features - summing on all classes and sorting
    best_features: pd.Series = features_df.sum().sort_values(ascending=False)

    # taking first n values
    best_n_features = best_features.iloc[:n_top]

    return shap_values, best_n_features
