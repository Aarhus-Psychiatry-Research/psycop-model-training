"""Base charts."""
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_basic_chart(
    x_values: Iterable,
    y_values: Union[Iterable[int, float], Iterable[Iterable[int, float]]],
    x_title: str,
    y_title: str,
    plot_type: Union[list[str], str],
    labels: Optional[list[str]] = None,
    sort_x: Optional[Iterable[int]] = None,
    sort_y: Optional[Iterable[int]] = None,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    bar_count_values: Optional[Iterable] = None,
    y_limits: Optional[tuple[float, float]] = None,
    fig_size: Optional[tuple] = (5, 5),
    dpi: Optional[int] = 300,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot a simple chart using matplotlib. Options for sorting the x and y
    axis are available.

    Args:
        x_values (Iterable): The x values of the bar chart.
        y_values (Iterable): The y values of the bar chart.
        x_title (str): title of x axis
        y_title (str): title of y axis
        plot_type (Optional[Union[List[str], str]], optional): type of plots.
            Options are combinations of ["bar", "hbar", "line", "scatter"] Defaults to "bar".
        labels: (Optional[list[str]]): Optional labels to add to the plot(s).
        sort_x (Optional[Iterable[int]], optional): order of values on the x-axis. Defaults to None.
        sort_y (Optional[Iterable[int]], optional): order of values on the y-axis. Defaults to None.
        bar_count_values (Optional[Iterable], optional): Values to use for overlaid histogram of n in bins. Defaults to None.
        y_limits (Optional[tuple[float, float]], optional): y-axis limits. Defaults to None.
        fig_size (Optional[tuple], optional): figure size. Defaults to None.
        dpi (Optional[int], optional): dpi of figure. Defaults to 300.
        save_path (Optional[Path], optional): path to save figure. Defaults to None.
        flip_x_axis (bool, optional): Whether to flip the x axis. Defaults to False.
        flip_y_axis (bool, optional): Whether to flip the y axis. Defaults to False.

    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure
    """
    if isinstance(plot_type, str):
        plot_type = [plot_type]

    df = pd.DataFrame(
        {"x": x_values, "sort_x": sort_x, "sort_y": sort_y},
    )

    if sort_x is not None:
        df = df.sort_values(by=["sort_x"])

    if sort_y is not None:
        df = df.sort_values(by=["sort_y"])

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    axs = fig.subplots()

    if not isinstance(y_values[0], Iterable):
        y_values = [y_values]  # Make y_values an iterable

    plot_functions = {
        "bar": axs.bar,
        "hbar": axs.barh,
        "line": axs.plot,
        "scatter": axs.scatter,
    }

    # choose the first plot type as the one to use for legend
    label_plot_type = plot_type[0]

    label_plots = []
    for y_series in y_values:
        for p_type in plot_type:
            plot_function = plot_functions.get(p_type)
            plot = plot_function(df["x"], y_series, color="green")
            if p_type == label_plot_type:
                # need to one of the plot types for labelling
                label_plots.append(plot)
            if p_type == "hbar":
                plt.yticks(fontsize=7)

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xticks(fontsize=7)
    plt.xticks(rotation=45)

    if y_limits is not None:
        plt.ylim(y_limits)

    if flip_x_axis:
        plt.gca().invert_xaxis()
    if flip_y_axis:
        plt.gca().invert_yaxis()
    if labels is not None:
        plt.gca().figlegend(
            [plot[0] for plot in label_plots],
            [str(label) for label in labels],
            loc="upper left",
            bbox_to_anchor=(0.14, 0.95),
            frameon=True,
        )

    if bar_count_values is not None:
        # add additional y-axis for count
        bar_overlay = plt.gca().twinx()
        bar_overlay.bar(df["x"], bar_count_values, color="green")

        bar_overlay.set_ylabel("Number of observations")
        axs.set_zorder(bar_overlay.get_zorder() + 1)
        # axs.patch.set_visible(False)
        axs.set_facecolor("none")
        bar_overlay.set_facecolor("none")
        bar_overlay.bar_label(bar_overlay.bar(df["x"], bar_count_values))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.close()

    return save_path
