"""Training script for training a single model for predicting t2d."""
import os
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from wasabi import Printer

# from psycop_model_training.evaluation import evaluate_model
from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.training.model_specs import MODELS
from psycop_model_training.training.utils import create_eval_dataset
from psycop_model_training.utils.config_schemas import FullConfigSchema
from psycop_model_training.utils.utils import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "src" / "psycop_model_training" / "config"

# Handle wandb not playing nice with joblib
os.environ["WANDB_START_METHOD"] = "thread"


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
