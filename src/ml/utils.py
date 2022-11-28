"""Utils module.

This module stores utility functions to support ml models
"""

import logging
import shap

import numpy as np
import numpy.typing as npt
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostClassifier


# initialize logger
logger = logging.getLogger(__name__)


def get_shap_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 10,
) -> (list[list[npt.NDArray]], list[str]):
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
    # shap_array.shape = # classes, # data, # features
    shap_array: npt.NDArray = np.array(shap_values)
    logger.debug(f"Shapley coefficients array has shape: {shap_array.shape}")

    # retrieve absolute importance
    shap_array: npt.NDArray = np.abs(shap_array)  # get absolute importance
    shap_array: npt.NDArray = shap_array.mean(1)  # shape (# classes, # features)

    # get it into a df
    features_df = pd.DataFrame(shap_array)
    features_df.columns = X.columns

    # find best features - summing on all classes and sorting
    best_features: pd.Series = features_df.sum().sort_values(ascending=False)

    # taking first n values
    best_n_features: list[str] = list(best_features.iloc[:n_top].index)

    return shap_values, best_n_features


def encoder(
    X: pd.DataFrame | npt.NDArray | list, reg: MLPRegressor, steps: int
) -> npt.NDArray:
    """Encode data in 2D.

    It assumes layers: (n_1, n_2, ... n_k, 2, n_k, ... n_1)
    :param X: explanatory data
    :param reg: trained model
    :param steps: step to reach 2D layer
    :return: encoded array of 2D data
    """
    encoded = np.asmatrix(X)  # at step 0, encoded is X

    for i in range(steps):
        linear = encoded * reg.coefs_[i] + reg.intercepts_[i]
        encoded = (np.exp(linear) - np.exp(-linear)) / (
            np.exp(linear) + np.exp(-linear)
        )

    return np.asarray(encoded)


def autoencode_precision(X: npt.NDArray, reg: MLPRegressor) -> np.float64:
    """Accuracy metric for autoencoders.

    Compute X_enc as the autoencoded X, then measure their difference as follows:
    - compute the percentage array X_pctg = abs(X-X_enc)/abs(X_enc) [no division by 0]
    - return avg(X_pctg).

    :param X: array to test
    :param reg: trained autoencoder
    :return: autoencoder accuracy
    """
    # from here just math
    X_enc = reg.predict(X)
    X_diff = np.abs(X - X_enc)
    X_pctg = np.divide(
        X_diff, np.abs(X), out=np.copy(X_diff), where=(np.abs(X) >= 0.1)
    ).round(4)

    avg_error_pctg = np.mean(np.abs(X_pctg))

    # precision = 1 - error
    return 1 - avg_error_pctg
