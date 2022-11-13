"""Plot module.

This module contains all data visualization functions used in the script
"""

from pathlib import Path

import logging
import shap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier


# initialize logger
logger = logging.getLogger(__name__)


def heatmap_correlation(
        X: pd.DataFrame,
        save_path: Path,
) -> None:
    """
    Compute correlation matrix as plot as heatmap
    """
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


def class_frequency(
        y: pd.Series,
        save_path: Path
) -> None:
    """Simple bar plot of classes frequency"""
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
        save_dir: Path
) -> None:
    """
    Plot Shapley feature importance using tree model.
    """
    # start
    logger.info("Computing Shapley values.")
    title = "Shapley Values"

    # split values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.7, shuffle=True, stratify=y
    )

    # rescale values
    scl = StandardScaler()
    X_train_scl = scl.fit_transform(X_train)
    X_val_scl = scl.transform(X_val)

    # simple model
    mdl = CatBoostClassifier()
    mdl.fit(
        X=X_train_scl,
        y=y_train,
        verbose=0,
        eval_set=(X_val_scl, y_val),
        use_best_model=True
    )

    explainer = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(X_train_scl)

    # Generate and save shapley summary plot
    shap.summary_plot(
        shap_values=shap_values,
        features=X_train_scl,
        feature_names=X.columns,
        class_names=mdl.classes_,
        plot_type="bar",
        show=False,
        plot_size=(7, 8),
    )
    plt.grid(linestyle=":")

    # save
    save_path = save_dir / f"{title}.png"
    plt.savefig(save_path)
    plt.close()

    # end
    logger.info(f"Shapley summary plot saved in {save_path}")
