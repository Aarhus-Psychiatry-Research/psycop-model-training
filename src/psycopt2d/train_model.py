"""Training script for training a single model for predicting t2d."""
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import wandb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from psycopt2d.evaluation import evaluate_model
from psycopt2d.feature_transformers import ConvertToBoolean, DateTimeConverter
from psycopt2d.load import load_dataset
from psycopt2d.models import MODELS
from psycopt2d.utils import flatten_nested_dict, prediction_df_with_metadata_to_disk

CONFIG_PATH = Path(__file__).parent / "config"
TRAINING_COL_NAME_PREFIX = "pred_"

# Handle wandb not playing nice with joblib
os.environ["WANDB_START_METHOD"] = "thread"


def create_preprocessing_pipeline(cfg):
    """Create preprocessing pipeline based on config."""
    steps = []

    if cfg.preprocessing.convert_datetimes_to:
        dtconverter = DateTimeConverter(
            convert_to=cfg.preprocessing.convert_datetimes_to,
        )
        steps.append(("DateTimeConverter", dtconverter))

    if cfg.preprocessing.convert_to_boolean:
        steps.append(("ConvertToBoolean", ConvertToBoolean()))

    if cfg.model.require_imputation:
        steps.append(
            ("Imputation", SimpleImputer(strategy=cfg.preprocessing.imputation_method)),
        )
    if cfg.preprocessing.transform in {
        "z-score-normalization",
        "z-score-normalisation",
    }:
        steps.append(
            ("z-score-normalization", StandardScaler()),
        )

    return Pipeline(steps)


def create_model(cfg):
    """Instantiate and return a model object based on settings in the config
    file."""
    model_dict = MODELS.get(cfg.model.model_name)

    model_args = model_dict["static_hyperparameters"]

    training_arguments = getattr(cfg.model, "args")
    model_args.update(training_arguments)

    mdl = model_dict["model"](**model_args)
    return mdl


def load_dataset_from_config(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset based on settings in the config file."""

    allowed_data_sources = {"csv", "synthetic"}

    if cfg.data.source.lower() == "csv":
        path = Path(cfg.data.dir)

        train = load_dataset(
            split_names="train",
            dir=path,
            n_training_samples=cfg.data.n_training_samples,
            drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
            min_lookahead_days=cfg.data.min_lookahead_days,
        )
        val = load_dataset(
            split_names="val",
            dir=path,
            n_training_samples=cfg.data.n_training_samples,
            drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
            min_lookahead_days=cfg.data.min_lookahead_days,
        )
    elif cfg.data.source.lower() == "synthetic":
        repo_dir = Path(__file__).parent.parent.parent
        dataset = pd.read_csv(
            repo_dir / "tests" / "test_data" / "synth_prediction_data.csv",
        )

        # Convert all timestamp cols to datetime
        for col in [col for col in dataset.columns if "timestamp" in col]:
            dataset[col] = pd.to_datetime(dataset[col])

        # Get 75% of dataset for train
        train, val = train_test_split(
            dataset,
            test_size=0.25,
            random_state=cfg.project.seed,
        )
    else:
        raise ValueError(
            f"The config data.source is {cfg.data.source}, allowed are {allowed_data_sources}",
        )
    return train, val


def stratified_cross_validation(
    cfg,
    pipe: Pipeline,
    dataset: pd.DataFrame,
    train_col_names: list[str],
    outcome_col_name: str,
):
    """Performs stratified and grouped cross validation using the pipeline."""
    X = dataset[train_col_names]  # pylint: disable=invalid-name
    y = dataset[outcome_col_name]  # pylint: disable=invalid-name

    # Create folds
    folds = StratifiedGroupKFold(n_splits=cfg.training.n_splits).split(
        X=X,
        y=y,
        groups=dataset[cfg.data.id_col_name],
    )

    # Perform CV and get out of fold predictions
    dataset["oof_y_hat"] = np.nan
    for train_idxs, val_idxs in folds:
        X_, y_ = X.loc[train_idxs], y.loc[train_idxs]  # pylint: disable=invalid-name
        pipe.fit(X_, y_)

        y_hat = pipe.predict_proba(X_)[:, 1]
        print(f"Within-fold performance: {round(roc_auc_score(y_,y_hat), 3)}")
        dataset["oof_y_hat_prob"].loc[val_idxs] = pipe.predict_proba(X.loc[val_idxs])[
            :,
            1,
        ]

    return dataset


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg):
    """Main function for training a single model."""
    run = wandb.init(
        project=cfg.project.name,
        reinit=True,
        config=flatten_nested_dict(cfg, sep="."),
        mode=cfg.project.wandb_mode,
    )

    # load dataset
    train, val = load_dataset_from_config(cfg)

    # creating pipeline
    steps = []
    preprocessing_pipe = create_preprocessing_pipeline(cfg)
    if len(preprocessing_pipe.steps) != 0:
        steps.append(("preprocessing", preprocessing_pipe))

    mdl = create_model(cfg)
    steps.append(("model", mdl))
    pipe = Pipeline(steps)

    # train
    # define columns
    OUTCOME_COL_NAME = (  # pylint: disable=invalid-name
        f"outc_dichotomous_t2d_within_{cfg.data.lookahead_days}_days_max_fallback_0"
    )

    TRAIN_COL_NAMES = [  # pylint: disable=invalid-name
        c for c in train.columns if c.startswith(cfg.data.pred_col_name_prefix)
    ]

    # Set feature names if model is EBM to get interpretable feature importance
    # output
    if cfg.model.model_name == "ebm":
        pipe["model"].feature_names = TRAIN_COL_NAMES

    if cfg.training.n_splits is None:  # train on pre-defined splits
        X_train = train[TRAIN_COL_NAMES]  # pylint: disable=invalid-name
        y_train = train[OUTCOME_COL_NAME]
        X_val = val[TRAIN_COL_NAMES]  # pylint: disable=invalid-name

        pipe.fit(X_train, y_train)

        y_train_hat_prob = pipe.predict_proba(X_train)[:, 1]
        y_val_hat_prob = pipe.predict_proba(X_val)[:, 1]

        print(
            f"Performance on train: {round(roc_auc_score(y_train, y_train_hat_prob), 3)}",
        )  # TODO log to wandb

        eval_dataset = val
        eval_dataset["y_hat_prob"] = y_val_hat_prob
        y_hat_prob_col_name = "y_hat_prob"
    else:
        train_val = pd.concat([train, val], ignore_index=True)
        eval_dataset = stratified_cross_validation(
            cfg,
            pipe,
            dataset=train_val,
            train_col_names=TRAIN_COL_NAMES,
            outcome_col_name=OUTCOME_COL_NAME,
        )
        y_hat_prob_col_name = "oof_y_hat_prob"

    # Evaluate: Calculate performance metrics and log to wandb
    evaluate_model(
        cfg=cfg,
        pipe=pipe,
        eval_dataset=eval_dataset,
        y_col_name=OUTCOME_COL_NAME,
        train_col_names=TRAIN_COL_NAMES,
        y_hat_prob_col_name=y_hat_prob_col_name,
        run=run,
    )
    # Save results to disk
    prediction_df_with_metadata_to_disk(df=eval_dataset, cfg=cfg)

    # Log metadata to wandb
    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")

    run.finish()

    return roc_auc_score(
        eval_dataset[OUTCOME_COL_NAME],
        eval_dataset[y_hat_prob_col_name],
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
