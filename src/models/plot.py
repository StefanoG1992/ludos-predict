"""Plot module.

This module contains all data visualization functions used in the script
"""

from pathlib import Path

import logging
import shap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

from models.models import get_shap_features


# initialize logger
logger = logging.getLogger(__name__)


def heatmap_correlation(
    X: pd.DataFrame,
    save_path: Path,
) -> None:
    """Compute correlation matrix as plot as heatmap."""
    # compute correlation matrix
    corr_matrix = X.corr()

    # title
    title = "correlation_matrix"

    # initialize figure
    fig = plt.figure(figsize=(10, 10))
    plt.title(title, fontweight="bold")
    sns.heatmap(
        corr_matrix.round(2),
        square=True,
        linewidths=0.5,
        annot=True,
        annot_kws={"size": 8},
    )

    save_path = save_path / f"{title}.png"

    plt.savefig(save_path)
    plt.close()


def class_frequency(y: pd.Series, save_path: Path) -> None:
    """Simple bar plot of classes frequency."""
    # title
    title = "Class frequency"

    # generate class list
    class_list = y.unique()

    ax = sns.countplot(x=y)
    plt.grid(linestyle=":")
    plt.title("Class frequency", fontweight="bold")

    save_path = save_path / f"{title}.png"

    plt.savefig(save_path)
    plt.close()


def print_shap_values(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: Path,
) -> None:
    """Plot Shapley feature importance using tree model.

    :param X: Explanatory data as df
    :param y: Classes as series
    :param save_dir: directory where to save results
    """
    # start
    logger.info("Computing Shapley values.")
    title = "shapley_values"

    # generate shap_values
    shap_values, _ = get_shap_features(X, y, n_top=len(X.columns))

    # Generate and save shapley summary plot
    shap.summary_plot(
        shap_values=shap_values,
        feature_names=X.columns,
        plot_type="bar",
        show=False,
        plot_size=(10, 20),
    )
    plt.grid(linestyle=":")

    # save image
    image_path = save_dir / f"{title}.png"
    plt.savefig(image_path)
    plt.close()

    # end
    logger.info(f"Shapley summary info saved in {save_dir}")


def encoder(X: pd.DataFrame | np.ndarray | list, reg: MLPRegressor) -> np.ndarray:
    """Encode data in 2D.

    :param X: explanatory data
    :param reg: trained model
    :return:
    """
    X = np.asmatrix(X)

    encoder1 = X * reg.coefs_[0] + reg.intercepts_[0]
    encoder1 = (np.exp(encoder1) - np.exp(-encoder1)) / (
        np.exp(encoder1) + np.exp(-encoder1)
    )

    encoder2 = encoder1 * reg.coefs_[1] + reg.intercepts_[1]
    encoder2 = (np.exp(encoder2) - np.exp(-encoder2)) / (
        np.exp(encoder2) + np.exp(-encoder2)
    )

    latent = encoder2 * reg.coefs_[2] + reg.intercepts_[2]
    latent = (np.exp(latent) - np.exp(-latent)) / (np.exp(latent) + np.exp(-latent))

    return np.asarray(latent)


def plot_2d(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: Path,
) -> None:
    """Transform data in 2D and plot them.

    Data are transformed in 2D via autoencoding.
    The autoencoder is defined as a MLPRegressor with dimensions:
        - n_input = 21
        - encoding_layers = (10, 5)
        - plot_dim = 2
        - decoding_layers (10, 5)
        - n_input = 21
    To plot data, only encoding part is needed.

    :param X: explanatory data as df
    :param y: classes
    :param save_dir: directory where to save results
    """
    logger.info("Plot data in 2D")

    title = "2D encoded data"

    # creating train test split
    # y_train not used
    X_train, X_test, _, y_test = train_test_split(
        X,
        train_size=0.7,
        shuffle=True,
    )

    logger.info("Define autoencoding model.")
    reg = MLPRegressor(
        hidden_layer_sizes=(10, 5, 2, 5, 10),
        activation="tanh",
        solver="adam",
        learning_rate_init=0.0001,
        max_iter=20,
    )

    logger.info("Fitting model")
    reg.fit(X_train, X_train)  # for autoencoders, same matrix is train & test

    # define encoded variables
    test_latent: np.ndarray = encoder(X_test, reg)

    plt.figure(figsize=(10, 10))
    for y_class in y_test.unique:
        plt.scatter(
            test_latent[np.argmax(y_test, axis=1) == y_class, 0],
            test_latent[np.argmax(y_test, axis=1) == y_class, 1],
            label=f"Class {y_class}",
        )
    plt.title(title)
    plt.legend(fontsize=15)
    plt.axis("equal")

    # save image
    image_path = save_dir / f"{title}.png"
    plt.savefig(image_path)
    plt.close()

    # end
    logger.info(f"Shapley summary info saved in {save_dir}")
