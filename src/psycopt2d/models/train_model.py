from pathlib import Path

import hydra
from sklearn.metrics import roc_auc_score
from wasabi import Printer

import psycopt2d.features.post_process as post_process
from psycopt2d.features.load_features import load_dataset
from psycopt2d.utils import generate_predictions


@hydra.main(
    version_base=None,
    config_path=Path(".") / "conf",
    config_name="train_config",
)
def main(cfg):
    OUTCOME_COL_NAME = (
        f"outc_dichotomous_t2d_within_{cfg.training.lookahead_days}_days_max_fallback_0"
    )

    # Val set
    X_val, y_val = load_dataset(
        split_name="val",
        n_to_load=cfg.post_processing.n_to_load,
        outcome_col_name=OUTCOME_COL_NAME,
    )

    X_val, y_val = post_process.combined(
        X=X_val,
        y=y_val,
        outcome_col_name=OUTCOME_COL_NAME,
    )

    # Train set
    X_train, y_train = load_dataset(
        split_name="train",
        outcome_col_name=OUTCOME_COL_NAME,
        n_to_load=cfg.post_processing.n_to_load,
    )

    X_train, y_train = post_process.combined(
        X=X_train,
        y=y_train,
        outcome_col_name=OUTCOME_COL_NAME,
    )

    # Keep only cols that start with pred_
    X_train = X_train.loc[:, X_train.columns.str.startswith("pred_")]
    print(f"Training cols: {X_train.columns}")

    X_val = X_val.loc[:, X_val.columns.str.startswith("pred_")]
    print(f"Val cols: {X_val.columns}")

    y_val_probas, model, y_train_pred_probs = generate_predictions(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
    )

    msg.info(f"Performance on train: {roc_auc_score(y_val, y_train_pred_probs)}")
    msg.info(f"Performance on val: {roc_auc_score(y_val, y_val_probas)}")


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    main()
