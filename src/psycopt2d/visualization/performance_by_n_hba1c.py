"""Plotting function for performance by number of HbA1c measurements."""
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Optional, Union

from sklearn.metrics import roc_auc_score

from psycopt2d.visualization.base_charts import plot_basic_chart
from psycopt2d.visualization.utils import create_performance_by_input


def plot_performance_by_n_hba1c(
    labels: Iterable[int],
    y_hat: Iterable[int, float],
    n_hba1c: Iterable[int],
    save_path: Optional[Path] = None,
    bins: tuple = (0, 1, 2, 5, 10, 100),
    pretty_bins: Optional[bool] = True,
    metric_fn: Callable = roc_auc_score,
) -> Union[None, Path]:
    """Plot bar plot of performance (default AUC) by number of HbA1c
    measurements.

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int]): Predicted label or probability depending on metric
        n_hba1c (Iterable[int]): Number of HbA1c measurements
        bins (Iterable[float]): Bins to group by. Defaults to (0, 1, 2, 5, 10, 100).
        pretty_bins (bool, optional): Whether to prettify bin names. I.e. make
            bins look like "1-7" instead of "[1-7)". Defaults to True.
        metric_fn (Callable): Callable which returns the metric to calculate
        save_path (Path, optional): Path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """

    df = create_performance_by_input(
        labels=labels,
        y_hat=y_hat,
        input=n_hba1c,
        input_name="n_hba1c",
        metric_fn=metric_fn,
        bins=bins,
        pretty_bins=pretty_bins,
    )

    sort_order = sorted(df["n_hba1c_binned"].unique())
    return plot_basic_chart(
        x_values=df["n_hba1c_binned"],
        y_values=df["metric"],
        x_title="Number of HbA1c measurements",
        y_title="AUC",
        sort_x=sort_order,
        plot_type=["bar"],
        save_path=save_path,
    )
