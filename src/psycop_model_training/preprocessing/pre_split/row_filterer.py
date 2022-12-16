from datetime import timedelta
from typing import Union

import pandas as pd

from psycop_model_training.data_loader.data_loader import msg
from psycop_model_training.utils.decorators import print_df_dimensions_diff
from psycop_model_training.utils.utils import get_percent_lost


class PreSplitRowFilterer():
    def __init__(self):
        raise NotImplementedError

    def _drop_rows_if_datasets_ends_within_days(
        self,
        n_days: Union[int, float],
        dataset: pd.DataFrame,
        direction: str,
    ) -> pd.DataFrame:
        """Drop visits that lie within certain amount of days from end of
        dataset.

        Args:
            n_days (Union[float, int]): Number of days.
            dataset (pd.DataFrame): Dataset.
            direction (str): Direction to look. Allowed are ["before", "after"].

        Returns:
            pd.DataFrame: Dataset with dropped rows.
        """
        if not isinstance(n_days, timedelta):
            n_days_timedelt: timedelta = timedelta(days=n_days)  # type: ignore

        if direction not in ("ahead", "behind"):
            raise ValueError(f"Direction {direction} not supported.")

        n_rows_before_modification = dataset.shape[0]

        if direction == "ahead":
            max_datetime = (
                dataset[self.cfg.data.col_name.pred_timestamp].max() - n_days_timedelt
            )
            before_max_dt = (
                dataset[self.cfg.data.col_name.pred_timestamp] < max_datetime
            )
            dataset = dataset[before_max_dt]
        elif direction == "behind":
            min_datetime = (
                dataset[self.cfg.data.col_name.pred_timestamp].min() + n_days_timedelt
            )
            after_min_dt = dataset[self.cfg.data.col_name.pred_timestamp] > min_datetime
            dataset = dataset[after_min_dt]

        n_rows_after_modification = dataset.shape[0]
        percent_dropped = get_percent_lost(
            n_before=n_rows_before_modification,
            n_after=n_rows_after_modification,
        )

        if n_rows_before_modification - n_rows_after_modification != 0:
            msg.info(
                f"Dropped {n_rows_before_modification - n_rows_after_modification} ({percent_dropped}%) rows because the end of the dataset was within {n_days} of their prediction time when looking {direction} from their prediction time",
            )

        return dataset

    @print_df_dimensions_diff
    def _drop_patient_if_excluded(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop patients that have an exclusion event within the washin
        period."""

        n_rows_before_modification = dataset.shape[0]

        outcome_before_date = (
            dataset[self.cfg.data.col_name.exclusion_timestamp]
            < self.cfg.data.drop_patient_if_exclusion_before_date
        )

        patients_to_drop = set(
            dataset[self.cfg.data.col_name.id][outcome_before_date].unique(),
        )
        dataset = dataset[~dataset[self.cfg.data.col_name.id].isin(patients_to_drop)]

        n_rows_after_modification = dataset.shape[0]

        percent_dropped = get_percent_lost(
            n_before=n_rows_after_modification,
            n_after=n_rows_after_modification,
        )

        if n_rows_before_modification - n_rows_after_modification != 0:
            msg.info(
                f"Dropped {n_rows_before_modification - n_rows_after_modification} ({percent_dropped}%) rows because they met exclusion criteria before {self.cfg.data.drop_patient_if_exclusion_before_date}.",
            )
        else:
            msg.info(
                f"No rows met exclusion criteria before {self.cfg.data.drop_patient_if_exclusion_before_date}. Didn't drop any.",
            )

        return dataset

    def _keep_only_if_older_than_min_age(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows that are older than the minimum age specified in the
        config."""
        return dataset[dataset[self.cfg.data.col_name.age] >= self.cfg.data.min_age]

    @print_df_dimensions_diff
    def _drop_rows_after_event_time(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Drop all rows where prediction timestamp is after the outcome."""

        rows_to_drop = (
            dataset[self.cfg.data.col_name.pred_timestamp]
            > dataset[self.cfg.data.col_name.outcome_timestamp]
        )

        return dataset[~rows_to_drop]
