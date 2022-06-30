from typing import Tuple

import pandas as pd


def combined(
    X: pd.DataFrame,
    y: pd.Series,
    outcome_col_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_combined = X
    df_combined[outcome_col_name] = y

    df_combined["pred_sex_female"] = df_combined["pred_sex_female"].astype(bool)

    _X = df_combined.drop(outcome_col_name, axis=1)
    _y = df_combined[outcome_col_name]

    return _X, _y
