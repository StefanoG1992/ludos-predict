from pathlib import Path

import click
import yaml

import pandas as pd

from ml import plot as mlplot
from ml import models

from ml.base import Model

from utils.log import config_logger

_PLOT_CHOICES = [
    "shapley",
    "plot_2d",
]

_MODEL_CHOICES = [
    "most_frequent",
    "smart_random",
    "logistic",
]


@click.group()
@click.option(
    "-i",
    "--input-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to csv data",
)
@click.option(
    "-s",
    "--save-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path where to save outputs",
)
@click.pass_context
def main(ctx: click.core.Context, input_path: Path, save_dir: Path):
    """Main function. Instance paths.

    :param ctx: click context, to be shared among commands
    :param input_path: path to csv data
    :param save_dir: path where to save outputs
    """
    ctx.obj = {"input_path": input_path, "save_dir": save_dir}
    pass


@main.command
@click.option(
    "-p",
    "--plot-graph",
    required=True,
    type=click.Choice(_PLOT_CHOICES),
    help=f"Which graph to plot. Implemented choices are:\n{_PLOT_CHOICES}",
)
@click.pass_context
def plot(
    ctx: click.core.Context,
    plot_graph: str,
) -> None:
    """Plot command to test.

    :param ctx: click context inherited from main
    :param plot_graph: plot_graph: which graph to plot. Current choices are:
        - shapley: print shapley values
        - 2d: encode data in 2d and plot distribution
    """
    # initialize logger
    logger = config_logger()
    logger.info("Saving pictures")

    # initialize context variables from main
    input_path: Path = ctx.obj["input_path"]
    save_dir: Path = ctx.obj["save_dir"] / "plot"
    save_dir.mkdir(exist_ok=True, parents=True)

    # read csv as dataframe, split as X, y
    df = pd.read_csv(input_path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1].astype(int)

    # plot functions dictionary
    # remap plot_graph to corresponding function
    # we can execute commands in a compact way avoiding multiple ifs
    plot_map = {
        "shapley": mlplot.print_shap_values,
        "2d": mlplot.plot_2d,
    }
    # assigning function
    plot_func = plot_map[plot_graph]

    # compact plot
    logger.info(f"Plotting. Graph choice: {plot_graph}")
    plot_func(X, y, save_dir)


@main.command
@click.option(
    "-m",
    "--model-name",
    required=True,
    type=click.Choice(_MODEL_CHOICES),
    help=f"Which model to test. Implemented choices are {_MODEL_CHOICES}",
)
@click.option(
    "-O",
    "--optimize",
    is_flag=True,
    show_default=True,
    default=False,
    help="Boolean flag. If passed, optimize the method through a grid",
)
@click.pass_context
def test(
    ctx: click.core.Context,
    model_name: str,
    optimize: bool,
):
    """Single tester for a ml model.

    Use to test a model and optimize it.
    To see model choices see the --help click function

    :param ctx: click context inherited from main
    :param model_name: model to be tested
    :return: None
    """
    # initialize logger
    logger = config_logger()
    logger.info("Execution step.")

    # initialize context variables from main
    input_path: Path = ctx.obj["input_path"]
    save_dir: Path = ctx.obj["save_dir"] / "outputs"
    save_dir.mkdir(exist_ok=True)

    logger.info(f"Reading data from {input_path}")
    # read csv as dataframe, split as X, y
    df = pd.read_csv(input_path)

    X, y = df.iloc[:, :-1], df.iloc[:, -1].astype(int)

    # remap response as integers starting with 0
    classes = y.unique()
    remap_classes = {cl: i for i, cl in enumerate(classes)}
    y = y.map(remap_classes)

    logger.info("Data instanced. Initializing model.")
    # model dictionary
    # we can execute commands in a compact way avoiding multiple ifs
    mdls = {
        "most_frequent": models.MostFrequentClassifier,
        "smart_random": models.SmartRandomClassifier,
        "logistic": models.SimpleRegressionClassifier,
    }

    # initialize model
    model: Model = mdls[model_name]()
    logger.info(f"Model {model} initialized. Building:")

    # build model - define its internal structure
    model.build()

    if optimize:
        logger.info("Optimizing the model:")
        # find root directory
        root_dir = Path(__file__).parent.parent
        params_path = root_dir / "optim-params.yaml"
        assert params_path.exists(), "optim-params.yaml not found in root"

        # load params
        with open(params_path, "rt") as params_file:
            params: dict[str, dict[str, list]] = yaml.load(
                params_file, Loader=yaml.FullLoader
            )

        # define params for specific model
        model_params = params[model_name]

        # optimize
        model.optimize(X, y, model_params)
    logger.info("Building step done. Getting scores:")

    # get model scores - model fit is done internally
    scores: dict = model.evaluate(X, y)

    logger.info("Scores computed. Saving as csv:")
    save_path = save_dir / f"{model}.yaml"

    with open(save_path, "wt") as f:
        yaml.dump(scores, f)

    logger.info(f"Scores saved to path {save_path}")
    logger.info("Finished")


if __name__ == "__main__":
    main.add_command(plot)
    main.add_command(test)
    main()
