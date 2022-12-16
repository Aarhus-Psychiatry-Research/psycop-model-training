import numpy as np
import pandas as pd

from psycop_model_training.utils.decorators import print_df_dimensions_diff
from psycop_model_training.utils.utils import infer_predictor_col_name


class PresSplitColTransformer():

    @staticmethod
    @print_df_dimensions_diff
    def convert_timestamp_dtype_and_nat(dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert columns with `timestamp`in their name to datetime, and
        convert 0's to NaT."""
        timestamp_colnames = [col for col in dataset.columns if "timestamp" in col]

        for colname in timestamp_colnames:
            if dataset[colname].dtype != "datetime64[ns]":
                # Convert all 0s in colname to NaT
                dataset[colname] = dataset[colname].apply(
                    lambda x: pd.NaT if x == "0" else x,
                )
                dataset[colname] = pd.to_datetime(dataset[colname])

        return dataset

    @print_df_dimensions_diff
    def _convert_boolean_dtypes_to_int(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert boolean dtypes to int."""
        for col in dataset.columns:
            if dataset[col].dtype == bool:
                dataset[col] = dataset[col].astype(int)

        return dataset

    @print_df_dimensions_diff
    def _negative_values_to_nan(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert negative values to NaN."""
        preds = dataset[infer_predictor_col_name(df=dataset)]

        # Get all columns with negative values
        cols_with_numerical_values = preds.select_dtypes(include=["number"]).columns

        numerical_columns_with_negative_values = [
            c for c in cols_with_numerical_values if preds[c].min() < 0
        ]

        df_to_replace = dataset[numerical_columns_with_negative_values]

        # Convert to NaN
        df_to_replace[df_to_replace < 0] = np.nan
        dataset[numerical_columns_with_negative_values] = df_to_replace

        return dataset
