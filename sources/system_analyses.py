#!/usr/bin/env python3
# coding:utf-8

# default libraries
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
from collections import defaultdict

# installed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from scipy.stats import entropy
import colorcet as cc

# local libraries
from sources.dev_utils import type2category


def get_min_total_models(models: Path) -> Dict[str, int]:
    """
    Calculate the minimum total models from a given file path.

    Args:
        models (Path): Path to the models file.

    Returns:
        Dict[str, int]: A dictionary containing the minimum total for each model.
    """
    min_total = {}
    models_df = pd.read_csv(models, sep="\t", names=['name', 'path'])
    for idx, model_file in models_df["path"].items():
        with open(Path(model_file).resolve().as_posix()) as json_file:
            data = json.load(json_file)
            min_total[data["name"]] = data["func_units"][0]["parameters"]["min_total"]
    return min_total


def shannon_normalized(data: pd.DataFrame, min_required: Dict[str, int], nb_group: int) -> pd.Series:
    """
    Calculate the normalized Shannon entropy for each system type.

    Args:
        data (pd.DataFrame): DataFrame containing system data.
        min_required (Dict[str, int]): Minimum required values for each system.
        nb_group (int): Number of groups to consider.

    Returns:
        pd.Series: Series containing normalized Shannon entropy for each system type.
    """
    type2shannon = defaultdict(list)
    data["model_GF_list"] = data["model_GF"].str.split(', ')

    for sys_name in data["system name"].unique():
        df = data[data["system name"] == sys_name]
        family_counts = np.array(df['model_GF_list'].apply(len))
        total = len(set(df['model_GF_list'].explode().tolist()))
        proportions = family_counts / total
        shannon_index = entropy(proportions)

        max_shannon_index = np.log(len(family_counts))
        normalized_diversity = shannon_index / max_shannon_index if max_shannon_index > 0 else 0
        diversity_weighted = normalized_diversity * (len(family_counts) / min_required[sys_name])

        type2shannon[type2category[sys_name]].append((diversity_weighted, df.shape[0]))

    shannon = {"system type": [], "entropy": []}
    for k, values in type2shannon.items():
        entropies, weights = zip(*values)
        weighted_entropy = np.average(entropies, weights=weights)
        shannon["system type"].append(k)
        shannon["entropy"].append(weighted_entropy)

    shannon_series = pd.Series(shannon["entropy"], index=shannon["system type"])
    top_types = shannon_series.nlargest(nb_group - 1).index.tolist()
    shannon_top = {"system type": [], "entropy": []}
    for sys_type in top_types:
        shannon_top["system type"].append(sys_type)
        shannon_top["entropy"].append(shannon_series[sys_type])

    concat_o = []
    for k, v in type2shannon.items():
        if k not in top_types:
            concat_o += v

    o_entropy, o_weight = zip(*concat_o)
    o_weighted_entropy = np.average(o_entropy, weights=o_weight)
    shannon_top["system type"].append("Other")
    shannon_top["entropy"].append(o_weighted_entropy)

    return pd.Series(shannon_top["entropy"], index=shannon_top["system type"])


def get_fraction(values_list: pd.Series, total: int) -> float:
    """
    Calculate the fraction of unique values in a series.

    Args:
        values_list (pd.Series): Series containing values.
        total (int): Total number of unique values.

    Returns:
        float: Fraction of unique values.
    """
    values = set(values_list.explode().tolist())
    return len(values) / total


def calculate_fraction_for_system(data: pd.DataFrame, nb_group: int, key: str, field: str = "system name",
                                  reverse: bool = False) -> pd.Series:
    """
    Calculate the fraction for each system based on the organisms associated with it.

    Args:
        data (pd.DataFrame): DataFrame containing the system and organism data.
        nb_group (int): Number of groups to consider.
        key (str): Column name for the data to calculate fractions.
        field (str): Column name for the system grouping.
        reverse (bool): Whether to reverse the order of the plot bars.

    Returns:
        pd.Series: A series with fraction for each system.
    """
    data[f'{key}_list'] = data[key].str.split(', ')
    total = len(set(data[f'{key}_list'].explode().tolist()))
    fraction_per_system = data.groupby(field)[f'{key}_list'].apply(get_fraction, total=total)

    top_fraction = fraction_per_system.nlargest(nb_group - 1)
    data[f'fraction_{key}_grouped'] = data[field].apply(lambda x: x if x in top_fraction.index else 'Other')
    fraction_group_count = data.groupby(f'fraction_{key}_grouped')[f'{key}_list'].apply(get_fraction, total=total)

    sys_frac_order = list(fraction_group_count.sort_values(ascending=False).index)
    if 'Other' in sys_frac_order:
        sys_frac_order.remove('Other')
    sys_frac_order.append('Other')
    fraction_group_reorder = fraction_group_count.reindex(reversed(sys_frac_order) if reverse else sys_frac_order)

    return fraction_group_reorder


def plot_occurrences_and_fraction(data: pd.DataFrame, models: Path, nb_group: int, field: str = "system name",
                                  output: Path = None, kind: str = 'barh', reverse: bool = False,
                                  bar_colors: Tuple[str, ...] = None):
    """
    Generate a plot with multiple bar plots: system occurrences, occurrences weighted by organisms, and Shannon entropy.

    Args:
        data (pd.DataFrame): DataFrame containing the system data.
        models (Path): Path to the model to use.
        nb_group (int): Number of groups to display in the plots.
        field (str): Column name for grouping systems.
        output (Path, optional): Path to save the output plots.
        kind (str, optional): Type of plot (e.g., 'barh' for horizontal bar plots).
        reverse (bool, optional): Whether to reverse the order of the plot bars.
        bar_colors (Tuple[str, ...], optional): Colors for the bars.
    """
    if bar_colors is None:
        bar_colors = ("darkgray",) * 3

    def set_ax(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = ""):
        """Set the title and labels for a given axis."""
        base_size = 22
        ax.set_title(title, fontsize=base_size + 4)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=base_size + 2)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=base_size + 2)

        plt.setp(ax.get_xticklabels(), horizontalalignment='right')
        ax.tick_params(axis='x', labelsize=base_size, rotation=45)
        ax.tick_params(axis='y', labelsize=base_size)

    def reorder_and_plot(series: pd.Series, color: str, xlabel: str = "", ylabel: str = "",
                         rev: bool = False, output_name: str = None) -> pd.Series:
        """
        Reorder the series and plot it on an individual figure and/or on the main figure.

        Args:
            series (pd.Series): Data series to be plotted.
            color (str): Color for the bars.
            xlabel (str, optional): Label for the x-axis.
            ylabel (str, optional): Label for the y-axis.
            rev (bool, optional): Whether to reverse the order of the bars.
            output_name (str, optional): Name for saving the individual figure.

        Returns:
            pd.Series: Reordered series for plotting in the main figure.
        """
        order = list(series.sort_values(ascending=False).index)
        if 'Other' in order:
            order.remove('Other')
        order.append('Other')
        series_reordered = series.reindex(reversed(order) if rev else order)

        _, ax = plt.subplots(figsize=(10, 8))
        series_reordered.plot(kind=kind, color=color, ax=ax)
        set_ax(ax, title="", xlabel=xlabel, ylabel=ylabel)
        plt.tight_layout()
        plt.savefig(output / f"{output_name}.png", format='png', dpi=300)
        plt.savefig(output / f"{output_name}.svg", format='svg')
        plt.show()

        return series_reordered

    # Plot occurrences of systems
    type_counts = data[field].value_counts()
    top_types = type_counts.nlargest(nb_group - 1)
    data['cat_grouped'] = data[field].apply(lambda x: x if x in top_types.index else 'Other')
    type_grouped_counts = data['cat_grouped'].value_counts()
    reorder_and_plot(type_grouped_counts, bar_colors[0], ylabel="# Systems", xlabel=field,
                     rev=reverse, output_name="occurrences_systems")

    # Plot fractions weighted by organisms
    fraction_org_per_system = calculate_fraction_for_system(data, nb_group, "organism", field=field, reverse=reverse)
    reorder_and_plot(fraction_org_per_system, bar_colors[1], xlabel=field, ylabel="System frequency in genomes",
                     output_name="weighted_fraction_organisms")

    # Plot Shannon entropy
    min_total = get_min_total_models(models)
    shannon_entropy = shannon_normalized(data, min_required=min_total, nb_group=nb_group)
    reorder_and_plot(shannon_entropy, "orange", xlabel=field, ylabel="Shannon",
                     output_name="shannon_entropy")


def create_color_map(system_types: Set[str]) -> Dict[str, str]:
    """
    Create a consistent color mapping for system types using a color palette suitable for color blindness.
    The function ensures reproducibility by using a seed for random operations.

    Args:
        system_types (Set[str]): Set of system types for which to create a color map.

    Returns:
        Dict[str, str]: A dictionary mapping system types to colors.
    """
    # Ensure "Other" is included in system types
    if "Other" not in system_types:
        system_types.add("Other")

    num_types = len(system_types)

    # Choose a palette based on the number of types
    if num_types <= 8:
        # Use the 'colorblind' palette from Seaborn for <= 8 groups
        palette = sns.color_palette("colorblind", num_types)
    elif num_types <= 20:
        # Use 'tab20' from Matplotlib for 8-20 groups
        palette = sns.color_palette("tab20", num_types)
    else:
        # Use a sequential palette for more than 20 groups
        # You can use 'Blues', 'Oranges', or any other sequential palette from Seaborn/Matplotlib
        # palette = sns.color_palette('deep', num_types, as_cmap=False)  # Sequential palette (e.g., 'Blues')
        dec = 30
        palette = cc.glasbey_bw_minc_20_hue_330_100[dec:num_types + dec]

    # Create the color map by mapping system types to palette colors
    sorted_types = sorted(system_types)
    color_map = dict(zip(sorted_types, palette))

    return color_map


def spot_occurrence(data: pd.DataFrame, output: Path, field: str = "system name",
                    nb_cols: int = 50, nb_top: int = 19, keep_other_spots: bool = False) -> Tuple[pd.DataFrame, set]:
    """
    Generate a barplot showing the number of systems per spot, colored by system type.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        output (Path): Path to save the output file.
        field (str): Column name to group the data by (system name or type).
        nb_cols (int): Number of top spots to display, others grouped into "Other spots".
        nb_top (int): Number of top types to display, others grouped into "Other".
        keep_other_spots (bool): Whether to keep "Other spots" in the final output.

    Returns:
        Tuple[pd.DataFrame, set]: DataFrame of spot occurrences and set of top types.
    """
    # Split the multiple spots and remove duplicates
    tmp_data = data.assign(spots=data['spots'].str.split(', ')).explode('spots')
    data_expanded = tmp_data.drop_duplicates(subset=["system number", "spots"], ignore_index=False)

    # Count the occurrences of systems by spot
    spot_type_counts = data_expanded.groupby(['spots', field]).size().unstack(fill_value=0)
    spot_type_counts['total'] = spot_type_counts.sum(axis=1)
    top_spots = spot_type_counts.nlargest(nb_cols, 'total').index

    spot_type_counts.to_csv(output / "spot_type_counts.csv", index=True, sep="\t", header=True)

    # Group the rest under "Other spots"
    data_expanded['spot_grouped'] = data_expanded['spots'].apply(lambda x: x if x in top_spots else 'Other spots')

    # Recount the systems by 'spot_grouped' and 'type_grouped'
    spot_grouped_type_counts = data_expanded.groupby(['spot_grouped', field]).size().unstack(fill_value=0)
    spot_grouped_type_counts['total'] = spot_grouped_type_counts.sum(axis=1)

    # Sort the spots and ensure "Other spots" is at the end
    spot_order = list(spot_grouped_type_counts.sort_values(by='total', ascending=False).drop(columns='total').index)
    if 'Other spots' in spot_order:
        spot_order.remove('Other spots')
    if keep_other_spots:
        spot_order.append('Other spots')

    spot_grouped_type_counts = spot_grouped_type_counts.reindex(spot_order).drop(columns='total')

    # Get top types
    type_grouped_spot_counts = spot_grouped_type_counts.T.sum(axis=1)
    top_types = type_grouped_spot_counts.nlargest(nb_top).index.tolist()

    # Group non-top types into "Other"
    non_top = [col for col in spot_grouped_type_counts.columns if col not in top_types]
    spot_grouped_type_counts["Other"] = spot_grouped_type_counts[non_top].sum(axis=1)
    spot_grouped_type_counts.drop(non_top, axis=1, inplace=True)
    top_types.append("Other")

    # Remove "spot_" prefix from spot names
    spot_grouped_type_counts.index = spot_grouped_type_counts.index.str.replace("spot_", '')

    return spot_grouped_type_counts, set(top_types)


def plot_spot_occurrence(spot_data: pd.DataFrame, color_map: Dict, nb_color_legend: int, ax: plt.Axes = None) -> None:
    """
    Plot a stacked bar chart of spot occurrences colored by system type.

    Args:
        spot_data (pd.DataFrame): DataFrame containing spot occurrence data.
        color_map (Dict): Dictionary mapping system types to colors.
        nb_color_legend (int): Number of columns in the legend.
        ax (plt.Axes): The axes on which to plot.

    Returns:
        None
    """
    base_fontsize = 24
    colors = [color_map[system_type] for system_type in spot_data.columns]

    # Create the stacked bar plot on the provided axes
    spot_data.plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_xlabel('Spot', fontsize=base_fontsize + 2)
    ax.set_ylabel('# Systems', fontsize=base_fontsize + 2)
    ax.tick_params(axis='x', labelsize=base_fontsize)
    ax.tick_params(axis='y', labelsize=base_fontsize)

    # Customize the legend with larger font size
    ax.legend(title="System types", fontsize=base_fontsize + 2, title_fontsize=base_fontsize + 4,
              loc='upper right', ncol=nb_color_legend)

    # Adjust layout to prevent overlap
    plt.tight_layout()


def get_pie_charts_info(data: pd.DataFrame, field: str = "type_grouped", nb_spots: int = 5,
                        nb_slice: int = 10, top_spots: List[str] = None) -> Tuple[set, List[str], pd.DataFrame]:
    """
    Prepare data for generating pie charts for the top spots with the most systems.

    Args:
        data (pd.DataFrame): DataFrame containing the system data.
        field (str): Column name for grouping systems.
        nb_spots (int): Number of top spots to generate pie charts for.
        nb_slice (int): Number of top slices to show in each pie chart.
        top_spots (List[str]): List of top spots to consider.

    Returns:
        Tuple[set, List[str], pd.DataFrame]: Set of top types, list of top spots, and DataFrame of spot type counts.
    """
    # Split the multiple spots and remove duplicates
    tmp_data = data.assign(spots=data['spots'].str.split(', ')).explode('spots')
    data_expanded = tmp_data.drop_duplicates(subset=["system number", "spots"], ignore_index=False)

    # Count the occurrences of systems by spot
    spot_type_counts = data_expanded.groupby(['spots', field]).size().unstack(fill_value=0)
    spot_type_counts.index = spot_type_counts.index.str.replace("spot_", '')
    if '' in spot_type_counts.index:
        spot_type_counts = spot_type_counts.drop(index='')
    spot_type_counts['total'] = spot_type_counts.sum(axis=1)

    if top_spots is None:
        # Select the top spots
        top_spots = spot_type_counts.nlargest(nb_spots, 'total').index

    # Collect top types across the top spots
    top_types = set()
    for spot in top_spots:
        spot_data = spot_type_counts.loc[spot].drop('total')
        spot_data_filtered = spot_data[spot_data > 0]
        top_types.update(spot_data_filtered.nlargest(nb_slice).index.tolist())

    return top_types, top_spots, spot_type_counts


def plot_pie(spot_data: pd.DataFrame, color_map: Dict, ax: plt.Axes = None, nb_slice: int = 10,
             base_fontsize: int = 20) -> None:
    """
    Plot a pie chart for a given spot's system type distribution.

    Args:
        spot_data (pd.DataFrame): DataFrame containing system type data for a spot.
        color_map (Dict): Dictionary mapping system types to colors.
        ax (plt.Axes): The axes on which to plot.
        nb_slice (int): Number of top slices to show in the pie chart.
        base_fontsize (int): Base font size for text elements.

    Returns:
        None
    """
    # Filter out categories with 0 occurrences
    spot_data_filtered = spot_data[spot_data > 0]
    top_types = spot_data_filtered.nlargest(nb_slice)
    spot_top_type = spot_data_filtered.loc[top_types.index]
    other_sum = spot_data_filtered.drop(top_types.index).sum()
    spot_top_type['Other'] = other_sum

    colors = [color_map[system_type] for system_type in spot_top_type.index]

    # Plot the pie chart
    wedges, texts, autotexts = ax.pie(
        spot_top_type, labels=spot_top_type.index, colors=colors, autopct='%1.1f%%',
        startangle=90, labeldistance=1.4, pctdistance=0.85,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white', 'radius': 1.2}
    )

    # Draw lines between labels and pie slices
    for j, text in enumerate(texts):
        wedge = wedges[j]
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))

        line_start = (wedge.r * x, wedge.r * y)
        line_end = (1.5 * wedge.r * x, 1.5 * wedge.r * y)

        connection_line = ConnectionPatch(
            xyA=line_start, xyB=line_end, coordsA="data", coordsB="data", axesA=ax, color="black", linewidth=1
        )
        ax.add_artist(connection_line)

    # Adjust text properties for better readability
    for text in texts:
        text.set_fontsize(base_fontsize + 2)
    for autotext in autotexts:
        autotext.set_fontsize(base_fontsize)

    # Set title
    spot = spot_data.name
    ax.set_title(f"Spot: {spot}", fontsize=base_fontsize + 4, pad=40)


def plot_pie_charts(top_spots: list, spot_type_counts: pd.DataFrame, color_map: Dict, axes=None,
                    nb_slice: int = 10) -> None:
    """
    Generate pie charts for the top spots with the most systems, showing the distribution of system types.

    Args:
        top_spots (list): List of top spots to generate pie charts for.
        spot_type_counts (pd.DataFrame): DataFrame containing counts of system types per spot.
        color_map (Dict): Dictionary mapping system types to colors.
        axes (list of matplotlib.axes.Axes): Axes on which to plot each pie chart.
        nb_slice (int): Number of top slices to show in each pie chart.

    Returns:
        None
    """
    # Plot pie charts for each spot with polylines
    for i, spot in enumerate(top_spots):
        spot_data = spot_type_counts.loc[spot].drop('total')
        plot_pie(spot_data, color_map, axes[i], nb_slice, base_fontsize=24)

    plt.tight_layout()


def plot_spots_and_pies(data: pd.DataFrame, field: str = "type_grouped", nb_cols: int = 50,
                        nb_spots: int = 5, output: Path = None, figsize: Tuple[int, int] = (40, 20)) -> None:
    """
    Generate a plot with occurrences in spots and pie charts for the top spots.

    Args:
        data (pd.DataFrame): DataFrame containing the system data.
        field (str): Column name for grouping systems.
        nb_cols (int): Number of spots to show in the spot occurrence plot.
        nb_spots (int): Number of top spots to generate pie charts for.
        output (Path): Path to save the output plots.
        figsize (Tuple[int, int]): Size of the figure.

    Returns:
        None
    """
    nb_slice = 10
    nb_top_occ = 24

    # Get spot occurrences and top types
    spot_grouped_type_counts, top_types_occ = spot_occurrence(data, output, field=field, nb_cols=nb_cols,
                                                              nb_top=nb_top_occ, keep_other_spots=False)

    # Get pie chart information
    top_types_pie, top_spots, spot_type_counts = get_pie_charts_info(data, field, nb_spots, nb_slice)

    # Create a consistent color map for system types
    color_map = create_color_map(top_types_occ | top_types_pie)

    # Plot and save bar plot for spot occurrences
    nb_col_legend = len(top_types_occ) // 10 + 1
    fig, ax = plt.subplots(figsize=figsize)
    plot_spot_occurrence(spot_grouped_type_counts, color_map, nb_color_legend=nb_col_legend, ax=ax)
    ax.set_title("Occurrences in spots", fontsize=20)
    plt.tight_layout()
    plt.savefig(output / "spots_sys_distri.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Plot and save individual pie charts
    for i, spot in enumerate(top_spots):
        spot_data = spot_type_counts.loc[spot].drop('total')
        fig, ax = plt.subplots(figsize=figsize)
        plot_pie(spot_data, color_map, ax, nb_slice, base_fontsize=24)
        plt.tight_layout()
        if output:
            plt.savefig(output / f"spots_{spot}.png", format='png', dpi=300)
            plt.savefig(output / f"spots_{spot}.svg", format='svg')
        plt.close()

    # Create a combined plot with bar plot and pie charts
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, nb_spots, height_ratios=[1.25, 1], figure=fig)

    ax_spot_occurrences = fig.add_subplot(gs[0, :])  # Use all columns for the bar plot
    plot_spot_occurrence(spot_grouped_type_counts, color_map, nb_color_legend=nb_col_legend, ax=ax_spot_occurrences)

    # Create pie charts in the second row
    pie_axes = [fig.add_subplot(gs[1, i]) for i in range(nb_spots)]
    plot_pie_charts(top_spots, spot_type_counts, color_map, axes=pie_axes, nb_slice=nb_slice)

    plt.tight_layout()

    # Save the combined figure if output path is provided
    if output:
        plt.savefig(output / "spots_and_pies.png", format='png', dpi=300)
        plt.savefig(output / "spots_and_pies.svg", format='svg')

    plt.show()
