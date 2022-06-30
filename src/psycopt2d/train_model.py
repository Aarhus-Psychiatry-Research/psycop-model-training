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
from xgboost import XGBClassifier

from psycopt2d.load import load_dataset

# from psycopt2d.models import model_catalogue
# from psycopt2d.utils import flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "config"
TRAINING_COL_NAME_PREFIX = "pred_"


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="train_config",
)
def main(cfg):

    OUTCOME_COL_NAME = (
        f"outc_dichotomous_t2d_within_{cfg.data.lookahead_days}_days_max_fallback_0"
    )
    pipe = XGBClassifier(missing=np.nan)

    y, y_hat_prob = pre_defined_split_performance(cfg, OUTCOME_COL_NAME, pipe)

    print(f"Performance on val: {roc_auc_score(y, y_hat_prob)}")


def pre_defined_split_performance(cfg, OUTCOME_COL_NAME, pipe) -> Tuple[Series, Series]:

    train = load_dataset(
        split_names="train",
    )
    X_train = train[
        [c for c in train.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]
    y_train = train[[OUTCOME_COL_NAME]]
    print(f"Train columns: {X_train.columns}")

    # Val set
    val = load_dataset(
        split_names="val",
    )
    X_val = val[[c for c in val.columns if c.startswith(cfg.data.pred_col_name_prefix)]]
    y_val = val[[OUTCOME_COL_NAME]]
    print(f"Val columns: {X_val.columns}")

    pipe.fit(X_train, y_train, verbose=True)
    y_train_hat = pipe.predict(X_train)
    y_val_hat = pipe.predict(X_val)

    print(f"Performance on train: {roc_auc_score(y_train, y_train_hat)}")
    print(f"Performance on val: {roc_auc_score(y_val, y_val_hat)}")
    return y_val, y_val_hat


if __name__ == "__main__":
    main()
