"""Misc.

utilities.
"""
import sys
import tempfile
from collections.abc import Iterable, MutableMapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Union

import dill as pkl
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from psycop_model_training.model_eval.dataclasses import ModelEvalData
from psycop_model_training.model_eval.model_performance import ModelPerformance

SHARED_RESOURCES_PATH = Path(r"E:\shared_resources")
FEATURE_SETS_PATH = SHARED_RESOURCES_PATH / "feature_sets"
OUTCOME_DATA_PATH = SHARED_RESOURCES_PATH / "outcome_data"
RAW_DATA_VALIDATION_PATH = SHARED_RESOURCES_PATH / "raw_data_validation"
FEATURIZERS_PATH = SHARED_RESOURCES_PATH / "featurizers"
MODEL_PREDICTIONS_PATH = SHARED_RESOURCES_PATH / "model_predictions"

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def format_dict_for_printing(d: dict) -> str:
    """Format a dictionary for printing. Removes extra apostrophes, formats
    colon to dashes, separates items with underscores and removes curly
    brackets.

    Args:
        d (dict): dictionary to format.

    Returns:
        str: Formatted dictionary.

    Example:
        >>> d = {"a": 1, "b": 2}
        >>> print(format_dict_for_printing(d))
        >>> "a-1_b-2"
    """
    return (
        str(d)
        .replace("'", "")
        .replace(": ", "-")
        .replace("{", "")
        .replace("}", "")
        .replace(", ", "_")
    )


def flatten_nested_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Recursively flatten an infinitely nested dict.

    E.g. {"level1": {"level2": "level3": {"level4": 5}}}} becomes
    {"level1.level2.level3.level4": 5}.

    Args:
        d (dict): dict to flatten.
        parent_key (str): The parent key for the current dict, e.g. "level1" for the first iteration.
        sep (str): How to separate each level in the dict. Defaults to ".".

    Returns:
        dict: The flattened dict.
    """

    items: list[dict[str, Any]] = []

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(
                flatten_nested_dict(d=v, parent_key=new_key, sep=sep).items(),  # type: ignore
            )  # typing: ignore
        else:
            items.append((new_key, v))  # type: ignore

    return dict(items)  # type: ignore


def drop_records_if_datediff_days_smaller_than(  # pylint: disable=inconsistent-return-statements
    df: pd.DataFrame,
    t2_col_name: str,
    t1_col_name: str,
    threshold_days: Union[float, int],
    inplace: bool = True,
) -> pd.DataFrame:
    """Drop rows where datediff is smaller than threshold_days. datediff = t2 - t1.

    Args:
        df (pd.DataFrame): Dataframe.
        t2_col_name (str): Column name of a time column
        t1_col_name (str): Column name of a time column
        threshold_days (Union[float, int]): Drop if datediff is smaller than this.
        inplace (bool, optional): Defaults to True.

    Returns:
        A pandas dataframe without the records where datadiff was smaller than threshold_days.
    """
    if inplace:
        df.drop(
            df[
                (df[t2_col_name] - df[t1_col_name]) / np.timedelta64(1, "D")
                < threshold_days
            ].index,
            inplace=True,
        )
    else:
        return df[
            (df[t2_col_name] - df[t1_col_name]) / np.timedelta64(1, "D")
            < threshold_days
        ]


def round_floats_to_edge(series: pd.Series, bins: list[float]) -> np.ndarray:
    """Rounds a float to the lowest value it is larger than. E.g. if bins = [0, 1, 2, 3],
    0.9 will be rounded to 0, 1.8 will be rounded to 1, etc.

    Args:
        series (pd.Series): The series of floats to round to bin edges.
        bins (list[floats]): Values to round to.

    Returns:
        A numpy ndarray with the borders.
    """
    _, edges = pd.cut(series, bins=bins, retbins=True, duplicates="drop")
    labels = [  # pylint: disable=unsubscriptable-object
        f"({abs(edges[i]):.0f}, {edges[i+1]:.0f}]"  # pylint: disable=unsubscriptable-object
        for i in range(len(bins) - 1)
    ]

    return pd.cut(series, bins=bins, labels=labels)


def calculate_performance_metrics(
    eval_df: pd.DataFrame,
    outcome_col_name: str,
    prediction_probabilities_col_name: str,
    id_col_name: str = "dw_ek_borger",
) -> pd.DataFrame:
    """Log performance metrics to WandB.

    Args:
        eval_df (pd.DataFrame): DataFrame with predictions, labels, and id
        outcome_col_name (str): Name of the column containing the outcome (label)
        prediction_probabilities_col_name (str): Name of the column containing predicted
            probabilities
        id_col_name (str): Name of the id column

    Returns:
        A pandas dataframe with the performance metrics.
    """
    performance_metrics = ModelPerformance.performance_metrics_from_df(
        prediction_df=eval_df,
        prediction_col_name=prediction_probabilities_col_name,
        label_col_name=outcome_col_name,
        id_col_name=id_col_name,
        metadata_col_names=None,
        to_wide=True,
    )

    performance_metrics = performance_metrics.to_dict("records")[0]
    return performance_metrics


def bin_continuous_data(
    series: pd.Series,
    bins: Sequence[int],
    min_n_in_bin: int = 5,
    use_min_as_label: bool = False,
) -> pd.Series:
    """For prettier formatting of continuous binned data such as age.

    Args:
        series (pd.Series): Series with continuous data such as age
        bins (list[int]): Desired bins. Last value creates a bin from the last value to infinity.
        min_n_in_bin (int, optional): Minimum number of observations in a bin. If fewer than this, the bin is dropped. Defaults to 5.
        use_min_as_label (bool, optional): If True, the minimum value in the bin is used as the label. If False, the maximum value is used. Defaults to False.

    Returns:
        pd.Series: Binned data

    Example:
    >>> ages = pd.Series([15, 18, 20, 30, 32, 40, 50, 60, 61])
    >>> age_bins = [0, 18, 30, 50, 110]
    >>> bin_Age(ages, age_bins)
    0     0-18
    1     0-18
    2    19-30
    3    19-30
    4    31-50
    5    31-50
    6    31-50
    7      51+
    8      51+
    """
    labels = []

    if not isinstance(bins, list):
        bins = list(bins)

    # Apend maximum value from series ot bins set upper cut-off if larger than maximum bins value
    if int(series.max()) > max(bins):
        bins.append(int(series.max()))

    # Create bin labels
    for i, bin_v in enumerate(bins):
        # If not the final bin
        if i < len(bins) - 2:
            # If the difference between the current bin and the next bin is 1, the bin label is a single value and not an interval
            if (bins[i + 1] - bin_v) == 1 or use_min_as_label:
                labels.append(f"{bin_v}")
            # Else generate bin labels as intervals
            elif i == 0:
                labels.append(f"{bin_v}-{bins[i+1]}")
            else:
                labels.append(f"{bin_v+1}-{bins[i+1]}")
        elif i == len(bins) - 2:
            labels.append(f"{bin_v+1}+")
        else:
            continue

    df = pd.DataFrame(
        {
            "series": series,
            "bin": pd.cut(
                series,
                bins=bins,
                labels=labels,
                duplicates="drop",
                include_lowest=True,
            ),
        },
    )

    # Drop any category in the series where the bin has fewer than 5 observations
    return df.bin[df.groupby("bin")["series"].transform("size") >= min_n_in_bin]


def positive_rate_to_pred_probs(
    pred_probs: pd.Series,
    positive_rate_thresholds: Iterable,
) -> pd.Series:
    """Get thresholds for a set of percentiles. E.g. if one
    positive_rate_threshold == 1, return the value where 1% of predicted
    probabilities lie above.

    Args:
        pred_probs (pd.Sereis): Predicted probabilities.
        positive_rate_thresholds (Iterable): positive_rate_thresholds

    Returns:
        pd.Series: Thresholds for each percentile
    """

    # Check if percentiles provided as whole numbers, e.g. 99, 98 etc.
    # If so, convert to float.
    if max(positive_rate_thresholds) > 1:
        positive_rate_thresholds = [x / 100 for x in positive_rate_thresholds]

    # Invert thresholds for quantile calculation
    thresholds = [1 - threshold for threshold in positive_rate_thresholds]

    return pd.Series(pred_probs).quantile(thresholds).tolist()


def read_pickle(path: Union[str, Path]) -> Any:
    """Reads a pickled object from a file.

    Args:
        path (str): Path to pickle file.

    Returns:
        Any: Pickled object.
    """
    with open(path, "rb") as f:
        return pkl.load(f)


def write_df_to_file(
    df: pd.DataFrame,
    file_path: Path,
):
    """Write dataset to file. Handles csv and parquet files based on suffix.

    Args:
        df: Dataset
        file_path (str): File path. Infers file type from suffix.
    """

    file_suffix = file_path.suffix

    if file_suffix == ".csv":
        df.to_csv(file_path, index=False)
    elif file_suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Invalid file suffix {file_suffix}")


def get_feature_importance_dict(pipe: Pipeline) -> Union[None, dict[str, float]]:
    """Returns feature importances as a dict.

    Args:
        pipe (Pipeline): Sklearn pipeline.

    Returns:
        Union[None, dict[str, float]]: Dictionary of feature importances.
    """
    return dict(
        zip(pipe["model"].feature_names, pipe["model"].feature_importances_),
    )


def get_selected_features_dict(
    pipe: Pipeline,
    train_col_names: list[str],
) -> Union[None, dict[str, bool]]:
    """Returns results from feature selection as a dict.

    Args:
        pipe (Pipeline): Sklearn pipeline.
        train_col_names (list[str]): List of column names in the training set.

    Returns:
        Union[None, dict[str, int]]: Dictionary of selected features. 0 if not selected, 1 if selected.
    """
    is_selected = [
        int(i) for i in pipe["preprocessing"]["feature_selection"].get_support()
    ]
    return dict(
        zip(train_col_names, is_selected),
    )


def create_wandb_folders():
    """Creates folders to store logs on Overtaci."""
    if sys.platform == "win32":
        (Path(tempfile.gettempdir()) / "debug-cli.onerm").mkdir(
            exist_ok=True,
            parents=True,
        )
        (PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)


def coerce_to_datetime(date_repr: Union[str, date]) -> datetime:
    """Coerce date or str to datetime."""
    if isinstance(date_repr, str):
        date_repr = date.fromisoformat(
            date_repr,
        )

    if isinstance(date_repr, date):
        date_repr = datetime.combine(
            date_repr,
            datetime.min.time(),
        )

    return date_repr


def load_evaluation_data(model_data_dir: Path) -> ModelEvalData:
    """Get evaluation data from a directory.

    Args:
        model_data_dir (Path): Path to model data directory.

    Returns:
        ModelEvalData: Evaluation data.
    """
    eval_dataset = read_pickle(model_data_dir / "evaluation_dataset.pkl")
    cfg = read_pickle(model_data_dir / "cfg.pkl")
    pipe_metadata = read_pickle(model_data_dir / "pipe_metadata.pkl")

    return ModelEvalData(
        eval_dataset=eval_dataset,
        cfg=cfg,
        pipe_metadata=pipe_metadata,
    )


def get_percent_lost(n_before: Union[int, float], n_after: Union[int, float]) -> float:
    """Get the percent lost."""
    return round((100 * (1 - n_after / n_before)), 2)
