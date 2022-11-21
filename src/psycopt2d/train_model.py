"""Training script for training a single model for predicting t2d."""
import os
import time
from collections.abc import Iterable
from typing import Any, Optional

import hydra
import numpy as np
import pandas as pd
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wasabi import Printer

from psycopt2d.evaluate_model import run_full_evaluation
from psycopt2d.evaluation_dataclasses import EvalDataset, PipeMetadata

# from psycopt2d.evaluation import evaluate_model
from psycopt2d.load import load_train_and_val_from_cfg
from psycopt2d.models import MODELS
from psycopt2d.preprocessing.feature_selectors import DropDateTimeColumns
from psycopt2d.preprocessing.feature_transformers import (
    ConvertToBoolean,
    DateTimeConverter,
)
from psycopt2d.utils.config_schemas import (
    FullConfigSchema,
    convert_omegaconf_to_pydantic_object,
)
from psycopt2d.utils.utils import (
    PROJECT_ROOT,
    create_wandb_folders,
    eval_ds_cfg_pipe_to_disk,
    flatten_nested_dict,
    get_feature_importance_dict,
)

CONFIG_PATH = PROJECT_ROOT / "src" / "psycopt2d" / "config"

# Handle wandb not playing nice with joblib
os.environ["WANDB_START_METHOD"] = "thread"


def get_feature_selection_steps(cfg):
    """Add feature selection steps to the preprocessing pipeline."""
    new_steps = []

    if cfg.preprocessing.feature_selection.name:
        if cfg.preprocessing.feature_selection.name == "f_classif":
            new_steps.append(
                (
                    "feature_selection",
                    SelectPercentile(
                        f_classif,
                        percentile=cfg.preprocessing.feature_selection.params[
                            "percentile"
                        ],
                    ),
                ),
            )
        elif cfg.preprocessing.feature_selection.name == "chi2":
            new_steps.append(
                (
                    "feature_selection",
                    SelectPercentile(
                        chi2,
                        percentile=cfg.preprocessing.feature_selection.params[
                            "percentile"
                        ],
                    ),
                ),
            )
        else:
            raise ValueError(
                f"Unknown feature selection method {cfg.preprocessing.feature_selection.name}",
            )

    return new_steps


def create_preprocessing_pipeline(cfg: FullConfigSchema):
    """Create preprocessing pipeline based on config."""
    steps = []
    # Conversion
    if cfg.preprocessing.drop_datetime_predictor_columns:
        steps.append(
            (
                "DropDateTimeColumns",
                DropDateTimeColumns(pred_prefix=cfg.data.pred_prefix),
            ),
        )

    if cfg.preprocessing.convert_datetimes_to_ordinal:
        dtconverter = DateTimeConverter()
        steps.append(("DateTimeConverter", dtconverter))

    if cfg.preprocessing.convert_to_boolean:
        steps.append(("ConvertToBoolean", ConvertToBoolean()))

    # Imputation
    if cfg.model.require_imputation and not cfg.preprocessing.imputation_method:
        raise ValueError(
            f"{cfg.model.name} requires imputation, but no imputation method was specified in the config file.",
        )

    if cfg.preprocessing.imputation_method:
        steps.append(
            (
                "Imputation",
                SimpleImputer(strategy=cfg.preprocessing.imputation_method),
            ),
        )

    # Feature selection
    # Important to do this before scaling, since chi2
    # requires non-negative values
    steps += get_feature_selection_steps(cfg)

    # Feature scaling
    # Important to do this after feature selection, since
    # half of the values in z-score normalisation will be negative,
    # which is not allowed for chi2
    if cfg.preprocessing.scaling:
        if cfg.preprocessing.scaling in {
            "z-score-normalization",
            "z-score-normalisation",
        }:
            steps.append(
                ("z-score-normalization", StandardScaler()),
            )
        else:
            raise ValueError(
                f"{cfg.preprocessing.scaling} is not implemented. See above",
            )

    return Pipeline(steps)


def create_model(cfg: FullConfigSchema):
    """Instantiate and return a model object based on settings in the config
    file."""
    model_dict = MODELS.get(cfg.model.name)

    model_args = model_dict["static_hyperparameters"]

    training_arguments = getattr(cfg.model, "args")
    model_args.update(training_arguments)

    return model_dict["model"](**model_args)


def stratified_cross_validation(  # pylint: disable=too-many-locals
    cfg: FullConfigSchema,
    pipe: Pipeline,
    train_df: pd.DataFrame,
    train_col_names: Iterable[str],
    outcome_col_name: str,
    n_splits: int,
):
    """Performs stratified and grouped cross validation using the pipeline."""
    msg = Printer(timestamp=True)

    X = train_df[train_col_names]  # pylint: disable=invalid-name
    y = train_df[outcome_col_name]  # pylint: disable=invalid-name

    # Create folds
    msg.info("Creating folds")
    msg.info(f"Training on {X.shape[1]} columns and {X.shape[0]} rows")

    folds = StratifiedGroupKFold(n_splits=n_splits).split(
        X=X,
        y=y,
        groups=train_df[cfg.data.col_name.id],
    )

    # Perform CV and get out of fold predictions
    train_df["oof_y_hat"] = np.nan

    for i, (train_idxs, val_idxs) in enumerate(folds):
        msg_prefix = f"Fold {i + 1}"

        msg.info(f"{msg_prefix}: Training fold")

        X_train, y_train = (  # pylint: disable=invalid-name
            X.loc[train_idxs],
            y.loc[train_idxs],
        )  # pylint: disable=invalid-name
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_train)[:, 1]

        msg.info(f"{msg_prefix}: AUC = {round(roc_auc_score(y_train,y_pred), 3)}")

        train_df.loc[val_idxs, "oof_y_hat"] = pipe.predict_proba(X.loc[val_idxs])[
            :,
            1,
        ]

    return train_df


def create_eval_dataset(cfg: FullConfigSchema, outcome_col_name: str, df: pd.DataFrame):
    """Create an evaluation dataset object from a dataframe and
    FullConfigSchema."""

    eval_dataset = EvalDataset(
        ids=df[cfg.data.col_name.id],
        y=df[outcome_col_name],
        y_hat_probs=df["y_hat_prob"],
        y_hat_int=df["y_hat_prob"].round(),
        pred_timestamps=df[cfg.data.col_name.pred_timestamp],
        outcome_timestamps=df[cfg.data.col_name.outcome_timestamp],
        age=df[cfg.data.col_name.age],
        exclusion_timestamps=df[cfg.data.col_name.exclusion_timestamp],
    )

    if cfg.data.col_name.custom:
        if cfg.data.col_name.custom.n_hba1c:
            eval_dataset.custom.n_hba1c = df[cfg.data.col_name.custom.n_hba1c]

    return eval_dataset


def train_and_eval_on_crossvalidation(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: Iterable[str],
    n_splits: int,
) -> EvalDataset:
    """Train model on cross validation folds and return evaluation dataset.

    Args:
        cfg (DictConfig): Config object
        train: Training dataset
        val: Validation dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training
        n_splits: Number of folds for cross validation.

    Returns:
        Evaluation dataset
    """
    msg = Printer(timestamp=True)

    msg.info("Concatenating train and val for crossvalidation")
    train_val = pd.concat([train, val], ignore_index=True)

    df = stratified_cross_validation(
        cfg=cfg,
        pipe=pipe,
        train_df=train_val,
        train_col_names=train_col_names,
        outcome_col_name=outcome_col_name,
        n_splits=n_splits,
    )

    df.rename(columns={"oof_y_hat": "y_hat_prob"}, inplace=True)

    return create_eval_dataset(cfg=cfg, outcome_col_name=outcome_col_name, df=df)


def train_and_eval_on_val_split(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
) -> EvalDataset:
    """Train model on pre-defined train and validation split and return
    evaluation dataset.

    Args:
        cfg (FullConfig): Config object
        train: Training dataset
        val: Validation dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training

    Returns:
        Evaluation dataset
    """

    X_train = train[train_col_names]  # pylint: disable=invalid-name
    y_train = train[outcome_col_name]
    X_val = val[train_col_names]  # pylint: disable=invalid-name

    pipe.fit(X_train, y_train)

    y_train_hat_prob = pipe.predict_proba(X_train)[:, 1]
    y_val_hat_prob = pipe.predict_proba(X_val)[:, 1]

    print(
        f"Performance on train: {round(roc_auc_score(y_train, y_train_hat_prob), 3)}",
    )

    df = val
    df["y_hat_prob"] = y_val_hat_prob

    return create_eval_dataset(cfg=cfg, outcome_col_name=outcome_col_name, df=df)


def train_and_get_model_eval_df(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
    n_splits: Optional[int],
) -> EvalDataset:
    """Train model and return evaluation dataset.

    Args:
        cfg (FullConfigSchema): Config object
        train: Training dataset
        val: Validation dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training
        n_splits: Number of folds for cross validation. If None, no cross validation is performed.

    Returns:
        Evaluation dataset
    """
    # Set feature names if model is EBM to get interpretable feature importance
    # output
    if cfg.model.name in ("ebm", "xgboost"):
        pipe["model"].feature_names = train_col_names

    if n_splits is None:  # train on pre-defined splits
        eval_dataset = train_and_eval_on_val_split(
            cfg=cfg,
            train=train,
            val=val,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
        )
    else:
        eval_dataset = train_and_eval_on_crossvalidation(
            cfg=cfg,
            train=train,
            val=val,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
            n_splits=n_splits,
        )

    return eval_dataset


def create_pipeline(cfg):
    """Create pipeline.

    Args:
        cfg (DictConfig): Config object

    Returns:
        Pipeline
    """
    steps = []
    preprocessing_pipe = create_preprocessing_pipeline(cfg)
    if len(preprocessing_pipe.steps) != 0:
        steps.append(("preprocessing", preprocessing_pipe))

    mdl = create_model(cfg)
    steps.append(("model", mdl))
    return Pipeline(steps)


def get_col_names(cfg: DictConfig, train: pd.DataFrame) -> tuple[str, list[str]]:
    """Get column names for outcome and features.

    Args:
        cfg (DictConfig): Config object
        train: Training dataset

    Returns:
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training
    """

    potential_outcome_col_names = [
        c
        for c in train.columns
        if cfg.data.outc_prefix in c and str(cfg.data.min_lookahead_days) in c
    ]

    if len(potential_outcome_col_names) != 1:
        raise ValueError(
            "More than one outcome column found. Please make outcome column names unambiguous.",
        )

    outcome_col_name = potential_outcome_col_names[0]

    train_col_names = [  # pylint: disable=invalid-name
        c for c in train.columns if c.startswith(cfg.data.pred_prefix)
    ]

    return outcome_col_name, train_col_names


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """Main function for training a single model."""
    # Save dictconfig for easier logging
    if isinstance(cfg, DictConfig):
        # Create flattened dict for logging to wandb
        # Wandb doesn't allow configs to be nested, so we
        # flatten it.
        dict_config_to_log: dict[str, Any] = flatten_nested_dict(OmegaConf.to_container(cfg), sep=".")  # type: ignore
    else:
        # For testing, we can take a FullConfig object instead. Simplifies boilerplate.
        dict_config_to_log = cfg.__dict__

    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)

    msg = Printer(timestamp=True)

    create_wandb_folders()

    run = wandb.init(
        project=cfg.project.name,
        reinit=True,
        config=dict_config_to_log,
        mode=cfg.project.wandb.mode,
        group=cfg.project.wandb.group,
        entity=cfg.project.wandb.entity,
    )

    if run is None:
        raise ValueError("Failed to initialise Wandb")

    # Add random delay based on cfg.train.random_delay_per_job to avoid
    # each job needing the same resources (GPU, disk, network) at the same time
    if cfg.train.random_delay_per_job_seconds:
        delay = np.random.randint(0, cfg.train.random_delay_per_job_seconds)
        msg.info(f"Delaying job by {delay} seconds to avoid resource competition")
        time.sleep(delay)

    dataset = load_train_and_val_from_cfg(cfg)

    msg.info("Creating pipeline")
    pipe = create_pipeline(cfg)

    outcome_col_name, train_col_names = get_col_names(cfg, dataset.train)

    msg.info("Training model")
    eval_ds = train_and_get_model_eval_df(
        cfg=cfg,
        train=dataset.train,
        val=dataset.val,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
        n_splits=cfg.train.n_splits,
    )

    pipe_metadata = PipeMetadata()

    if hasattr(pipe["model"], "feature_importances_"):
        pipe_metadata.feature_importances = get_feature_importance_dict(pipe=pipe)

    # Save model predictions, feature importance, and config to disk
    eval_ds_cfg_pipe_to_disk(
        eval_dataset=eval_ds,
        cfg=cfg,
        pipe_metadata=pipe_metadata,
        run=run,
    )

    if cfg.project.wandb.mode == "run" or cfg.eval.force:
        msg.info("Evaluating model.")

        upload_to_wandb = cfg.project.wandb.mode == "run"

        run_full_evaluation(
            cfg=cfg,
            eval_dataset=eval_ds,
            run=run,
            pipe_metadata=pipe_metadata,
            save_dir=PROJECT_ROOT / "wandb" / "plots" / run.name,
            upload_to_wandb=upload_to_wandb,
        )

    roc_auc = roc_auc_score(
        eval_ds.y,
        eval_ds.y_hat_probs,
    )

    msg.info(f"ROC AUC: {roc_auc}")
    run.log(
        {
            "roc_auc_unweighted": roc_auc,
            "lookbehind": max(cfg.data.lookbehind_combination),
            "lookahead": cfg.data.min_lookahead_days,
        },
    )
    run.finish()
    return roc_auc


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
