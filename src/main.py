"""Main module, the only one to be executed."""

from pathlib import Path

import click

import pandas as pd

from utils.log import config_logger
from core.checks import path_callback


@click.group()
def main():
    pass


@click.command
@click.option(
    "-p",
    "--plot-graph",
    required=True,
    type=click.Choice(["shapley", "2d"]),
    help="Which graph to plot. Implemented graphs: 'shapley', '2d'",
)
@click.option(
    "-i",
    "--input-path",
    required=True,
    type=click.UNPROCESSED,
    callback=path_callback,
    help="Path to csv data",
)
@click.option(
    "-s",
    "--save-dir",
    required=True,
    type=click.UNPROCESSED,
    callback=path_callback,
    help="Path where to save images",
)
def plot(
    plot_graph: str,
    input_path: Path,
    save_dir: Path,
) -> None:
    """Plot command to test.

    :param: plot_graph: which graph to plot. Current choices are:
        - shapley: print shapley values
        - 2d: encode data in 2d and plot distribution
    :param input_path: path to csv data
    :param save_dir: path where to save images
    """
    # initialize logger
    logger = config_logger()
    logger.info("Saving pictures")

    # read csv as dataframe, split as X, y
    df = pd.read_csv(input_path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1].astype(int)

    # choose plot
    if plot_graph == "shapley":
        # plot shapley
        logger.info("Plot shapley values.")
        from models.plot import print_shap_values

        print_shap_values(X, y, save_dir)

    elif plot_graph == "2d":
        # plot graph
        logger.info("Plot 2D graph.")
        from models.plot import plot_2d

        plot_2d(X, y, save_dir)


if __name__ == "__main__":
    main.add_command(plot)
    main()
