"""Helpers and example code for evaluating an already trained model based on a
pickle file with model predictions and hydra config.

Possible extensions (JIT when needed):
- Load most recent pickle file from 'evaluation_results' folder
- Evaluate all models in 'evaluation_results' folder
- CLI for evaluating a model
"""
from pathlib import Path

import pandas as pd
from omegaconf.dictconfig import DictConfig

from psycopt2d.utils import PROJECT_ROOT, read_pickle
from psycopt2d.visualization import plot_auc_by_time_from_first_visit


def get_colname_from_prefix(df: pd.DataFrame, prefix: str = "outc") -> str:
    """Infer the outcome col_name in a dataframe based on prefix."""
    outcome_names = [c for c in df.columns if c.startswith(prefix)]
    if len(outcome_names) == 1:
        return outcome_names[0]
    else:
        raise ValueError("More than one outcome inferred")


def infer_predictor_col_names(df: pd.DataFrame, cfg: DictConfig) -> list[str]:
    """Get the predictors that are used in the model.

    Args:
        df (pd.Dataframe): Dataframe with model predictions
        cfg (DictConfig): Config file

    Returns:
        list[str]: List of predictors
    """
    return [c for c in df.columns if c.startswith(cfg.data.pred_col_name_prefix)]


def load_model_predictions_and_cfg(path: Path) -> tuple[pd.DataFrame, DictConfig]:
    """Load model predictions and config file from a pickle file.

    Args:
        path (Path): Path to pickle file
    """
    obj = read_pickle(path)
    return obj["df"], obj["cfg"]


if __name__ == "__main__":

    eval_path = PROJECT_ROOT / "evaluation_results"
    eval_df, cfg = load_model_predictions_and_cfg(
        eval_path
        # insert your own model path here
        / "eval_model_name-xgboost_require_imputation-True_args-n_estimators-100_tree_method-auto_2022_09_22_10_52.pkl",
    )

    train_col_names = infer_predictor_col_names(eval_df, cfg)
    y_col_name = get_colname_from_prefix(df=eval_df, prefix="outc_")
    Y_HAT_PROB_COL_NAME = "y_hat_prob"  # change to 'y_hat_prob_oof' if using cv
    first_visit_timestamp = eval_df.groupby(cfg.data.id_col_name)[
        cfg.data.pred_timestamp_col_name
    ].transform("min")

    # Do whatever extra evaluation you want to here
    p = plot_auc_by_time_from_first_visit(
        labels=eval_df[y_col_name],
        y_hat_probs=eval_df[Y_HAT_PROB_COL_NAME],
        first_visit_timestamps=first_visit_timestamp,
        prediction_timestamps=eval_df[cfg.data.pred_timestamp_col_name],
    )
