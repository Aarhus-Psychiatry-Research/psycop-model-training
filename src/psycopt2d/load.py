# from datetime import date, datetime, timedelta
from typing import List, Union

import pandas as pd
from psycopmlutils.loaders import sql_load
from wasabi import Printer

msg = Printer(timestamp=True)


def load_dataset(
    split_names: Union[List[str], str],
) -> pd.DataFrame:

    sql_table_name = f"psycop_t2d_{split_names}"

    select = "SELECT"

    dataset = sql_load(
        query=f"{select} * FROM [fct].[{sql_table_name}]",
        format_timestamp_cols_to_datetime=False,
    )

    msg.good(f"{split_names}: Returning!")
    return dataset
