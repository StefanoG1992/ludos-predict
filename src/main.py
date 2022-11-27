import logging
from pathlib import Path

import click
import yaml

import pandas as pd

from ml import plot as mlplot
from ml import models

from ml.base import Model, ModelName, Score
from ml.utils import get_shap_features

from core import checks as check

from utils.log import config_logger

_PLOT_CHOICES: list[str] = [
    "shapley",
    "plot_2d",
]

_MODEL_CHOICES: list[ModelName] = [
    "most_frequent",
    "smart_random",
    "logistic",
    "catboost",
]

_SCORE_CHOICES: list[Score] = [
    "f1_score",
    "f2_score",
    "recall",
    "accuracy",
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
@click.option(
    "-t",
    "--use-top-cols",
    is_flag=True,
    show_default=True,
    default=False,
    help="""Boolean flag. If passed, use only top 10 explanatory columns.
    Explainability is computed via shapley algorithm, see ml.utils module.
    """,
)
@click.option(
    "-l",
    "--log-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory where to save logs. If None, logs are printed in stdout",
)
@click.pass_context
def main(
    ctx: click.core.Context,
    input_path: Path,
    save_dir: Path,
    use_top_cols: bool,
    log_dir: Path,
):
    """Main function. Instance paths.

    :param ctx: click context, to be shared among commands
    :param input_path: path to csv data
    :param save_dir: path where to save outputs
    :param use_top_cols: If true, use only top 10 explanatory columns.
    Columns explainability is computed via shapley algorithm.
    :param log_dir: "Directory where to save logs.
    If None, logs are only printed in stdout"
    """
    # initialize logger
    logger = config_logger(log_dir)
    logger.info("Process started.")

    logger.info(f"Reading data from {input_path}")
    # read csv as dataframe, split as X, y
    df = pd.read_csv(input_path)

    X, y = df.iloc[:, :-1], df.iloc[:, -1].astype(int)

    if use_top_cols is not None:
        _, best_n_features = get_shap_features(X=X, y=y, n_top=10)
        X = X.loc[:, best_n_features]

    # remap response as integers starting with 0
    classes = y.unique()
    remap_classes = {cl: i for i, cl in enumerate(classes)}
    y = y.map(remap_classes)

    logger.info("Data instanced. Passing to step:")

    # pass variables through context
    ctx.obj = {
        "save_dir": save_dir,
        "X": X,
        "y": y,
    }
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
    logger = logging.getLogger(__name__)
    logger.info("Command plot: saving pictures.")

    # initialize context variables from main
    save_dir: Path = ctx.obj["save_dir"] / "plot"
    save_dir.mkdir(exist_ok=True, parents=False)  # parent must exist

    X: pd.DataFrame = ctx.obj["X"]
    y: pd.Series = ctx.obj["y"]

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
@click.option(
    "-s",
    "--scoring",
    show_default=True,
    type=click.Choice(_SCORE_CHOICES),
    default="f1_score",
    help="Boolean flag. If passed, optimize the method through a grid",
)
@click.pass_context
def test(
    ctx: click.core.Context,
    model_name: str,
    optimize: bool,
    scoring: str,
):
    """Single tester for a ml model.

    Use to test a model and optimize it.
    To see model choices see the --help click function

    :param ctx: click context inherited from main
    :param model_name: model to be tested
    :param optimize: boolean. If true, optimize the method through a grid
    :param scoring: scoring function. Metric choice for optimization step
    :return: None
    """
    # initialize logger
    logger = logging.getLogger(__name__)
    logger.info("Command test: testing a single model")

    # initialize context variables from main
    save_dir: Path = ctx.obj["save_dir"] / "outputs"
    save_dir.mkdir(exist_ok=True, parents=False)  # parent must exist

    X: pd.DataFrame = ctx.obj["X"]
    y: pd.Series = ctx.obj["y"]

    logger.info("Data instanced. Initializing model.")
    # model dictionary
    # we can execute commands in a compact way avoiding multiple ifs
    mdls = {
        "most_frequent": models.MostFrequentClassifier,
        "smart_random": models.SmartRandomClassifier,
        "logistic": models.SimpleRegressionClassifier,
        "catboost": models.CatBoost,
    }

    # initialize model
    model: Model = mdls[model_name]()  # instance class here
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
            params: dict[ModelName, dict[str, list]] = yaml.load(
                params_file, Loader=yaml.FullLoader
            )

        # define params for specific model
        model_params = params[model_name]

        # optimize
        model.optimize(X, y, model_params, scoring)
        logger.info("Model optimized.")
    logger.info("Building step done. Getting scores:")

    # get model scores - model fit is done internally
    scores: dict[Score, float] = model.evaluate(X, y)

    logger.info("Scores computed. Saving as yaml:")
    save_path = save_dir / f"{model}.yaml"

    with open(save_path, "wt") as f:
        yaml.dump(scores, f)

    logger.info(f"Scores saved to path {save_path}")
    logger.info("Finished")


@main.command
@click.pass_context
def challenge(ctx: click.core.Context):
    """Perform a challenge between all non-trivial models in scope.

    Instance non-trivial models (all except SmartRandom, MostFrequent) and
    compute f1, f2 score, recall and accuracy for all of them.
    Results are modelled as dictionary with structure
    {metric_1: {model_1: value_1, ... model_n: value_n}, metric_2: {...}}
    and saved as yaml in output dir
    """
    # initialize logger
    logger = logging.getLogger(__name__)
    logger.info("Command test: model challenge.")

    # initialize context variables from main
    save_dir: Path = ctx.obj["save_dir"] / "outputs"
    save_dir.mkdir(exist_ok=True, parents=False)  # parent must exist

    X: pd.DataFrame = ctx.obj["X"]
    y: pd.Series = ctx.obj["y"]

    logger.info("Data instanced. Starting challenge:")

    # better hard-coded dict than other solutions as eval, getattr...
    all_models: dict[str, Model] = {
        "logistic": models.SimpleRegressionClassifier(),  # already instanced
    }

    # define global scores as dictionaries {model_name: model_scores}}}
    all_scores: dict[ModelName, dict[Score, float]] = {}

    logger.info("Challenge finished. Saving results")

    for model_name, model in all_models.items():
        model.build()
        model_scores: dict[Score, float] = model.evaluate(X, y)
        all_scores[model_name] = model_scores

    # check model scores are proper
    check.models_scores(all_scores)

    # rearrange all_scores to have structure {score: {model_1: metric_1 ...}}
    # get scores_list: if checks are passed, one random list is ok
    first_model_name: ModelName = list(all_scores.keys())[0]
    scores_list: list[Score] = list(all_scores[first_model_name].keys())

    challenge_result: dict[Score, dict[ModelName, float]] = {
        score: {
            mdl_name: all_scores[mdl_name][score]
            for mdl_name in all_scores
            if score in all_scores[mdl_name]
        }
        for score in scores_list
    }

    save_path = save_dir / "challenge_result.yaml"

    with open(save_path, "wt") as f:
        yaml.dump(challenge_result, f)

    logger.info(f"Change results saved to path {save_path}")
    logger.info("Finished.")


if __name__ == "__main__":
    main.add_command(plot)
    main.add_command(test)
    main.add_command(challenge)
    main()
