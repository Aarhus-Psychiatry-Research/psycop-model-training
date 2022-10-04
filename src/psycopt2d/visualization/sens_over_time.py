from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psycopt2d.utils import PROJECT_ROOT, round_floats_to_edge


def create_sensitivity_by_time_to_outcome_df(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    pred_proba_threshold: float,
    outcome_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
) -> pd.DataFrame:
    """Calculate sensitivity by time to outcome.

    Args:
        labels (Iterable[int]): True labels of the data.
        y_hat_probs (Iterable[int]): Predicted label probability.
        pred_proba_threshold (float): The pred_proba threshold above which predictions are classified as positive.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].

    Returns:
        pd.DataFrame
    """

    # Modify pandas series to 1 if y_hat is larger than threshold, otherwise 0
    y_hat = pd.Series(y_hat_probs).apply(
        lambda x: 1 if x > pred_proba_threshold else 0,
    )

    df = pd.DataFrame(
        {
            "y": labels,
            "y_hat": y_hat,
            "outcome_timestamp": outcome_timestamps,
            "prediction_timestamp": prediction_timestamps,
        },
    )

    # Get proportion of y_hat == 1, which is equal to the positive rate
    threshold_percentile = round(
        df[df["y_hat"] == 1].shape[0] / df.shape[0] * 100,
        2,
    )

    df = df[df["y"] == 1]

    # Calculate difference in days between columns
    df["days_to_outcome"] = (
        df["outcome_timestamp"] - df["prediction_timestamp"]
    ) / np.timedelta64(1, "D")

    df["true_positive"] = (df["y"] == 1) & (df["y_hat"] == 1)
    df["false_negative"] = (df["y"] == 1) & (df["y_hat"] == 0)

    df["days_to_outcome_binned"] = round_floats_to_edge(
        df["days_to_outcome"],
        bins=bins,
    )

    output_df = (
        df[["days_to_outcome_binned", "true_positive", "false_negative"]]
        .groupby("days_to_outcome_binned")
        .sum()
    )

    output_df["sens"] = round(
        output_df["true_positive"]
        / (output_df["true_positive"] + output_df["false_negative"]),
        2,
    )

    # Prep for plotting
    ## Save the threshold for each bin
    output_df["threshold"] = pred_proba_threshold

    output_df["threshold_percentile"] = threshold_percentile

    output_df = output_df.reset_index()

    # Convert days_to_outcome_binned to string for plotting
    output_df["days_to_outcome_binned"] = output_df["days_to_outcome_binned"].astype(
        str,
    )

    return output_df


def _generate_sensitivity_array(
    df: pd.DataFrame,
    y_n_decimals: int,
) -> np.ndarray:
    """Generate sensitivity array for plotting heatmap.

    Args:
        df (pd.DataFrame): Dataframe with columns "sens", "days_to_outcome_binned" and "threshold".

    Returns:
        np.ndarray
    """
    x_labels = df["days_to_outcome_binned"].unique().tolist()
    y_labels = df["threshold"].unique().tolist()

    sens_array = []
    for threshold in y_labels:
        sens_for_threshold = []
        small_df = df[df.threshold == threshold]
        for days_interval in x_labels:
            smaller_df = small_df[small_df["days_to_outcome_binned"] == days_interval]
            if len(smaller_df["sens"].unique().tolist()) > 1:
                raise ValueError(
                    "More than one sensitivity value for this threshold and days interval",
                )
            else:
                sens_for_threshold.append(smaller_df["sens"].unique().tolist()[0])

        sens_array.append(sens_for_threshold)

    return (
        np.array(sens_array),
        x_labels,
        [round(y_labels[value], y_n_decimals) for value in range(len(y_labels))],
    )


def _annotate_heatmap(
    im: matplotlib.image.AxesImage,
    data: Optional[np.array] = None,
    valfmt: Optional[str] = "{x:.2f}",
    textcolors: Optional[tuple] = ("black", "white"),
    threshold: Optional[float] = None,
    **textkw
):
    """A function to annotate a heatmap. Adapted from matplotlib documentation.

    Args:
        im (matplotlib.image.AxesImage): The AxesImage to be labeled.
        data (np.ndarray): Data used to annotate. If None, the image's data is used. Defaults to None.
        valfmt (str, optional): The format of the annotations inside the heatmap. This should either use the string format method, e.g. "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`. Defaults to "{x:.2f}".
        textcolors (tuple, optional): A pair of colors. The first is used for values below a threshold, the second for those above. Defaults to ("black", "white").
        threshold (float, optional): Value in data units according to which the colors from textcolors are applied. If None (the default) uses the middle of the colormap as separation. Defaults to None.
        **kwargs (dict, optional): All other arguments are forwarded to each call to `text` used to create the text labels. Defaults to {}.

    Returns:
        texts (list): A list of matplotlib.text.Text instances for each label.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(
        horizontalalignment="center",
        verticalalignment="center",
    )
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_sensitivity_by_time_to_outcome(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    pred_proba_thresholds: List[float],
    outcome_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins: List[int] = [0, 28, 182, 365, 730, 1825],
    cmap: Optional[str] = "PuBu",
    cbarlabel: Optional[str] = "Sensitivity",
    x_title: Optional[str] = "Days to outcome",
    y_title: Optional[str] = "Positive rate",
    y_n_decimals: Optional[int] = 4,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot heatmap of sensitivity by time to outcome according to different
    positive rate thresholds.

    Args:
        labels (Iterable[int]): True labels of the data.
        y_hat_probs (Iterable[int]): Predicted probability of class 1.
        pred_proba_thresholds (Iterable[float]): List of pred_proba thresholds to plot, above which predictions are classified as positive.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].
        cmap (str, optional): Colormap to use. Defaults to "PuBu".
        cbarlabel (str, optional): Colorbar label. Defaults to "Sensitivity".
        x_title (str, optional): X axis title. Defaults to "Days to outcome".
        y_title (str, optional): Y axis title. Defaults to "Positive rate".
        y_n_decimals (int, optional): Number of decimals to round y axis labels. Defaults to 4.
        save_path (Optional[Path], optional): Path to save the plot. Defaults to None.

    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure

    Examples:
        >>> from pathlib import Path
        >>> from psycopt2d.utils import positive_rate_to_pred_probs

        >>> repo_path = Path(__file__).parent.parent.parent.parent
        >>> path = repo_path / "tests" / "test_data" / "synth_eval_data.csv"
        >>> df = pd.read_csv(path)

        >>> positive_rate_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        >>> pred_proba_thresholds = positive_rate_to_pred_probs(
        >>>     pred_probs=df["pred_prob"],
        >>>     positive_rate_thresholds=positive_rate_thresholds,
        >>> )

        >>> plot_sensitivity_by_time_to_outcome(
        >>>     labels=df["label"],
        >>>     y_hat_probs=df["pred_prob"],
        >>>     pred_proba_thresholds=pred_proba_thresholds,
        >>>     outcome_timestamps=df["timestamp_t2d_diag"],
        >>>     prediction_timestamps=df["timestamp"],
        >>>     bins=[0, 28, 182, 365, 730, 1825],
        >>> )
    """
    # Construct sensitivity dataframe
    func = partial(
        create_sensitivity_by_time_to_outcome_df,
        labels=labels,
        y_hat_probs=y_hat_probs,
        outcome_timestamps=outcome_timestamps,
        prediction_timestamps=prediction_timestamps,
        bins=bins,
    )

    df = pd.concat(
        [
            func(
                pred_proba_threshold=pred_proba_thresholds[i],
            )
            for i in range(len(pred_proba_thresholds))
        ],
        axis=0,
    )

    # Prepare data for plotting
    data, x_labels, y_labels = _generate_sensitivity_array(df, y_n_decimals)

    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(data, cmap=cmap)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=x_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=y_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="right",
        rotation_mode="anchor",
    )

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add annotations
    _ = _annotate_heatmap(im, valfmt="{x:.1f}")

    # Set axis labels and title
    ax.set(
        xlabel=x_title,
        ylabel=y_title,
    )

    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()
    return save_path


if __name__ == "__main__":
    from pathlib import Path

    from psycopt2d.utils import positive_rate_to_pred_probs

    path = PROJECT_ROOT / "tests" / "test_data" / "synth_eval_data.csv"
    df = pd.read_csv(path)

    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    positive_rate_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    pred_proba_thresholds = positive_rate_to_pred_probs(
        pred_probs=df["pred_prob"],
        positive_rate_thresholds=positive_rate_thresholds,
    )

    plot_sensitivity_by_time_to_outcome(
        labels=df["label"],
        y_hat_probs=df["pred_prob"],
        pred_proba_thresholds=pred_proba_thresholds,
        outcome_timestamps=df["timestamp_t2d_diag"],
        prediction_timestamps=df["timestamp"],
        bins=[0, 28, 182, 365, 730, 1825],
    )
