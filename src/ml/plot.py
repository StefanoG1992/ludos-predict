"""Plot module.

This module contains all data visualization functions used in the script
"""

from pathlib import Path

import logging
import shap

import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

from ml.utils import get_shap_features, encoder, autoencode_precision


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
        y,
        train_size=0.7,
        shuffle=True,
    )

    logger.info("Define autoencoding model.")

    # define layers
    hidden_layers = (18, 14, 10, 7, 5, 2, 5, 7, 10, 14, 18)
    steps_to_2d = int((len(hidden_layers) + 1) / 2)  # half size hidden layers

    # scale the result - autoencoder is very sensitive to scaling
    scaler = MinMaxScaler()

    # fit results on X_train, transform X_test as well
    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl = scaler.transform(X_test)

    # define model
    reg = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="tanh",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=200,
    )

    logger.info("Fitting model...")
    reg.fit(X_train_scl, X_train_scl)  # for autoencoders, same matrix is train & test

    # due to MinMax scaling & precision linearity, no need to inverse_transform
    avg_precision = autoencode_precision(X_test_scl, reg)

    logger.info(f"Model fitted. Average difference: {avg_precision:.2%}")

    # define encoded variables
    X_encoded: npt.NDArray = encoder(X_test, reg, steps_to_2d)

    # reset y_test index - needed as X_encoded has lost X_test indexing
    y_test.reset_index(inplace=True, drop=True)

    plt.figure(figsize=(10, 10))
    for y_class in y_test.unique():
        # for any class, plot scatterplot and label
        class_index = y_test[y_test == y_class].index
        logger.debug(f"Class n. {y_class} has {len(class_index)} elements.")
        plt.scatter(
            x=X_encoded[class_index, 0],
            y=X_encoded[class_index, 1],
            label=f"{y_class}",
        )
    plt.title(title)
    plt.legend(fontsize=15)
    plt.axis("equal")

    # save image
    image_path = save_dir / f"{title}.png"
    plt.savefig(image_path)
    plt.close()

    # end
    logger.info(f"2D data saved in {image_path}")
