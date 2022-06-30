from typing import Tuple, Union

import pandas as pd
from psycopmlutils.loaders import sql_load
from wasabi import Printer

msg = Printer(timestamp=True)


def load_dataset(
    split_name: str,
    outcome_col_name: str,
    n_to_load: Union[None, int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    sql_table_name = f"psycop_t2d_{split_name}"

    if n_to_load is not None:
        msg.info(f"{sql_table_name}: Loading {n_to_load} rows from")
        select = f"SELECT TOP {n_to_load}"
    else:
        msg.info(f"{sql_table_name}: Loading all rows")
        select = "SELECT"

    df_combined = sql_load(
        query=f"{select} * FROM [fct].[{sql_table_name}]",
        format_timestamp_cols_to_datetime=False,
    )

    # Convert sex to bool
    df_combined["pred_sex_female"] = df_combined["pred_sex_female"].astype("bool")

    _X = df_combined.drop(outcome_col_name, axis=1)
    _y = df_combined[outcome_col_name]

    msg.good(f"{split_name}: Returning!")
    return _X, _y
