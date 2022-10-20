"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycopt2d/train_and_log_models.py
"""
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field
from wasabi import Printer

from psycopt2d.evaluate_saved_model_predictions import (
    infer_look_distance,
    infer_outcome_col_name,
    infer_predictor_col_name,
)
from psycopt2d.load import DataLoader, DatasetSpecification, DatasetTimeSpecification


class PossibleLookDistanceDays(BaseModel):
    """Possible look distances."""

    ahead: list[str]
    behind: list[str]


class MetaConf(BaseModel):
    """Meta configuration for the script."""

    conf_name: str = Field("integration_testing.yaml")
    data_dir: Path = Path(
        "/Users/au484925/Desktop/psycop-t2d/tests/test_data/synth_splits/",
    )
    overtaci: str = Field(
        default="false",
        description="Change to 'true' if running on overtaci",
    )


class WatcherConf(BaseModel):
    """Configuration for the watcher."""

    archive_all: str = Field(
        default="false",
        description="Whether to archive all runs in the wandb folder before starting model training. Change to 't' to archive all wandb runs",
    )
    n_runs_before_first_eval: int = Field(
        default="1",
        description="The number of runs to upload to wandb before evaluating the best runs.",
    )
    keep_alive_after_training_minutes: int = Field(
        default=5,
        description="minutes to wait for the wandb watcher after training has finished. Will kill the watcher after this time.",
    )


class WandbConf(BaseModel):
    """Configuration for wandb."""

    project_name: str = "psycopt2d-testing"
    entity: str = Field(
        default="psycop",
        description="The wandb entity to upload to (e.g. 'psycop' or your user name)",
    )
    mode: str = Field(default="online", description="The wandb mode to use")


class TrainConf(BaseModel):
    """Configuration for model training."""

    gpu: bool = Field(default="false", description="Whether to use GPU")

    n_trials_per_cell_in_grid: int = Field(
        default=50,
        description="Number of trials per cell in the lookahead/lookbehind grid",
    )

    model_conf: str = Field(
        default="xgboost",
        description="The model conf to open. For example, 'xgboost' or 'logistic_regression'.",
    )

    conf_name: str = Field(default="integration_testing.yaml")

    multirun: bool = Field(
        default=False,
        description="Whether to use Hydra to run multiple models.",
    )

    possible_look_distances: PossibleLookDistanceDays


def load_data(dataset_spec):
    """Load the data."""
    loader = DataLoader(dataset_spec)
    return loader.load_dataset_from_dir(split_names="train")


def infer_possible_look_directions(train):
    """Infer the possible values for min_lookahead_days and
    min_lookbehind_days."""
    # Get potential lookaheads from outc_ columns
    outcome_col_names = infer_outcome_col_name(df=train, allow_multiple=True)

    possible_lookahead_days = infer_look_distance(col_name=outcome_col_names)

    # Get potential lookbehinds from pred_ columns
    pred_col_names = infer_predictor_col_name(df=train, allow_multiple=True)
    possible_lookbehind_days = list(set(infer_look_distance(col_name=pred_col_names)))

    return PossibleLookDistanceDays(
        ahead=possible_lookahead_days,
        behind=possible_lookbehind_days,
    )


def get_dataset_spec(data_dir_path: Path):
    """Get dataset specification."""
    time_spec = DatasetTimeSpecification(
        drop_patient_if_outcome_before_date=None,
        min_prediction_time_date="1979-01-01",
        min_lookbehind_days=0,
        min_lookahead_days=0,
    )

    return DatasetSpecification(
        file_suffix="csv",
        time=time_spec,
        pred_col_name_prefix="pred_",
        pred_time_colname="timestamp",
        split_dir_path=data_dir_path,
    )


def train_models_for_each_cell_in_grid(
    train_conf: TrainConf,
):
    """Train a model for each cell in the grid of possible look distances."""
    from random_word import RandomWords

    random_word = RandomWords()

    for lookbehind in train_conf.possible_look_distances.behind:
        for lookahead in train_conf.possible_look_distances.ahead:
            wandb_group = f"{random_word.get_random_word()}-{random_word.get_random_word()}-beh-{lookbehind}-ahead-{lookahead}"

            subprocess_args: list[str] = [
                "python",
                "src/psycopt2d/train_model.py",
                f"+model={train_conf.model_conf}",
                f"data.min_lookbehind_days={lookbehind}",
                f"data.min_lookahead_days={lookahead}",
                f"project.wandb_group='{wandb_group}'",
                f"hydra.sweeper.n_trials={train_conf.n_trials_per_cell_in_grid}",
                "--config-name",
                f"{meta_conf.conf_name}",
            ]

            if train_conf.multirun:
                subprocess_args.insert(2, "--multirun")

            if train_conf.model_conf == "xgboost" and not train_conf.gpu:
                subprocess_args.insert(3, "++model.args.tree_method='auto'")

            msg.info("Starting trainer with command")
            msg.info(f'{" ".join(subprocess_args)}')

            trainer = subprocess.Popen(  # pylint: disable=consider-using-with
                args=subprocess_args,
            )

            while trainer.poll() is None:
                time.sleep(1)


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    meta_conf = MetaConf(
        conf_name="integration_testing.yaml",
        overtaci="false",
        data_dir=Path(
            "/Users/au484925/Desktop/psycop-t2d/tests/test_data/synth_splits/",
        ),
    )

    wandb_conf = WandbConf(
        entity="psycop",
        project_name="psycopt2d-testing",
        mode="offline",
    )

    watcher_conf = WatcherConf(archive_all="true", keep_alive_after_training_minutes=5)
    dataset_spec = get_dataset_spec(data_dir_path=meta_conf.data_dir)
    train = load_data(dataset_spec=dataset_spec)

    possible_look_distances = infer_possible_look_directions(train)

    train_conf = TrainConf(
        conf_name=meta_conf.conf_name,
        multirun=False,
        model_conf="xgboost",
        n_trials_per_cell_in_grid=1,
        possible_look_distances=possible_look_distances,
        gpu=False,
    )

    if not train_conf.gpu:
        msg.warn("Not using GPU for training")

    # watcher = subprocess.Popen(  # pylint: disable=consider-using-with
    #     [
    #         "python",
    #         "src/psycopt2d/model_training_watcher.py",
    #         "--entity",
    #         wandb_conf.entity,
    #         "--project_name",
    #         wandb_conf.project_name,
    #         "--n_runs_before_eval",
    #         str(watcher_conf.n_runs_before_first_eval),
    #         "--overtaci",
    #         meta_conf.overtaci,
    #         "--timeout",
    #         "None",
    #         "--clean_wandb_dir",
    #         watcher_conf.archive_all,
    #     ],
    # )

    train_models_for_each_cell_in_grid(
        train_conf=train_conf,
    )

    msg.good(
        f"Training finished. Stopping the watcher in {watcher_conf.keep_alive_after_training_minutes} minutes...",
    )

    # time.sleep(60 * watcher_conf.keep_alive_after_training_minutes)
    # watcher.kill()
    msg.good("Watcher stopped.")
