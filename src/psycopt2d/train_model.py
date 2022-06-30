"""Training script for training a single model for predicting t2d.

TODO:
add split for using pre-defined train-val split
add dynamic hyperparams for hydra optimisation

Features:
# fix impute
# move filter to compute
"""

from pathlib import Path
from typing import Tuple

import hydra
import numpy as np

# import wandb
from pandas import Series
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from psycopt2d.load import load_dataset

# from psycopt2d.models import model_catalogue
# from psycopt2d.utils import flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "config"
TRAINING_COL_NAME_PREFIX = "pred_"


def create_model(cfg):

    mdl = XGBClassifier(missing=np.nan, verbose=True)
    return mdl


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="train_config",
)
def main(cfg):

    OUTCOME_COL_NAME = (
        f"outc_dichotomous_t2d_within_{cfg.data.lookahead_days}_days_max_fallback_0"
    )

    mdl = create_model(cfg)
    pipe = Pipeline([("mdl", mdl)])  # ("preprocessing", preprocessing_pipe),

    y, y_hat_prob = pre_defined_split_performance(cfg, OUTCOME_COL_NAME, pipe)

    print(f"Performance on val: {roc_auc_score(y, y_hat_prob)}")


def pre_defined_split_performance(cfg, OUTCOME_COL_NAME, pipe) -> Tuple[Series, Series]:
    """Loads dataset and fits a model on the pre-defined split.

    Args:
        cfg (_type_): _description_
        OUTCOME_COL_NAME (_type_): _description_

    Returns:
        Tuple(Series, Series): Two series: True labels and predicted labels for the validation set.
    """
    # Train set
    train = load_dataset(
        split_names="train",
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
    )
    X_train = train[
        [c for c in train.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]
    y_train = train[[OUTCOME_COL_NAME]]

    # Val set
    val = load_dataset(
        split_names="val",
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
    )
    X_val = val[[c for c in val.columns if c.startswith(cfg.data.pred_col_name_prefix)]]
    y_val = val[[OUTCOME_COL_NAME]]

    pipe.fit(X_train, y_train)
    y_train_hat = pipe.predict(X_train)
    y_val_hat = pipe.predict(X_val)

    print(f"Performance on train: {roc_auc_score(y_train, y_train_hat)}")
    print(f"Performance on val: {roc_auc_score(y_val, y_val_hat)}")
    return y_val, y_val_hat


if __name__ == "__main__":
    main()
