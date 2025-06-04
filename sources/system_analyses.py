#!/usr/bin/env python3
# coding:utf-8

# default libraries
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
from collections import defaultdict
import ast

# installed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from scipy.stats import entropy
# from bokeh.palettes import TolRainbow, Colorblind
# import squarify
# import colorcet as cc
# from bokeh.io import output_file, save, show
# from bokeh.plotting import figure
# from bokeh.models import LinearColorMapper, ColorBar, Span
# from bokeh.layouts import gridplot
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift

# local libraries
from sources.dev_utils import type2category


def create_color_map(system_types: Set[str]):
    """
    Create a consistent color mapping for system types, using an appropriate color palette for color blindness.

    Args:
        data (pd.DataFrame): DataFrame containing the system data.
        field (str): Column name for system types.

    Returns:
        dict: A dictionary mapping system types to colors.
    """
    num_types = len(system_types)
    if "Other" not in system_types:
        num_types += 1
        system_types.add("Other")

    if num_types <= 8:
        # Use the 'colorblind' palette from Seaborn for <= 8 groups
        palette = Colorblind[num_types]
    elif num_types <= 20:
        # Use 'tab20' from Matplotlib for 8-20 groups
        palette = TolRainbow[num_types]
    else:
        # Use a sequential palette for more than 20 groups
        # You can use 'Blues', 'Oranges', or any other sequential palette from Seaborn/Matplotlib
        # palette = sns.color_palette('deep', num_types, as_cmap=False)  # Sequential palette (e.g., 'Blues')
        dec = 30
        palette = cc.glasbey_bw_minc_20_hue_330_100[dec:num_types + dec]

    # Create the color map by mapping system types to palette colors
    color_map = dict(zip(sorted(system_types), palette))

    return color_map


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
                                  output: Path = None, kind: str = 'barh', reverse: bool = False, bar_colors: Tuple[str, ...] = None):
    """
    Generate a plot with multiple bar plots: system occurrences, occurrences weighted by organisms, and Shannon entropy.

    Args:
        data (pd.DataFrame): DataFrame containing the system data.
        nb_group (int): Number of groups to display in the plots.
        field (str): Column name for grouping systems.
        output (Path, optional): Path to save the output plots.
        figsize (Tuple[int, int], optional): Figure size for the plots.
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

        fig, ax = plt.subplots(figsize=(10, 8))
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

def spot_occurency(data: pd.DataFrame, output: Path, field: str = "system name",
                   nb_cols: int = 50, nb_top: int = 19, keep_other_spots: bool = False):
    """
    Generate a barplot showing the number of systems per spot, colored by system type.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        field (str): Column name to group the data by (system name or type).
        nb_cols (int): Number of top spots to display, others grouped into "Other spots".
        ax (matplotlib.axes.Axes): The axes on which to plot.

    Returns:
        None
    """
    # Split the multiple spots and remove duplicates
    tmp_data = data.assign(spots=data['spots'].str.split(', ')).explode('spots')
    data_expanded = tmp_data.drop_duplicates(subset=["system number", "spots"], ignore_index=False)

    # Count the occurrences of systems by spot
    spot_type_counts = data_expanded.groupby(['spots', field]).size().unstack(fill_value=0)
    # spot_type_counts = spot_type_counts.drop("")
    # Select the top spots
    # spot_type_counts = spot_type_counts.drop(index='')
    spot_type_counts['total'] = spot_type_counts.sum(axis=1)
    top_spots = spot_type_counts.nlargest(nb_cols, 'total').index

    spot_type_counts.to_csv(output / "spot_type_counts.csv", index=True, sep="\t", header=True)
    # Group the rest under "Other spots"
    data_expanded['spot_grouped'] = data_expanded['spots'].apply(lambda x: x if x in top_spots else 'Other spots')

    # Recount the systems by 'spot_grouped' and 'type_grouped'
    spot_grouped_type_counts = data_expanded.groupby(['spot_grouped', field]).size().unstack(fill_value=0)

    # Sort the spots and ensure "Other spots" is at the end
    spot_grouped_type_counts['total'] = spot_grouped_type_counts.sum(axis=1)
    spot_order = list(spot_grouped_type_counts.sort_values(by='total', ascending=False).drop(columns='total').index)
    if ('Other spots' in spot_order):
        spot_order.remove('Other spots')
    if keep_other_spots:
        spot_order.append('Other spots')

    spot_grouped_type_counts = spot_grouped_type_counts.reindex(spot_order).drop(columns='total')

    type_grouped_spot_counts = spot_grouped_type_counts.T.sum(axis=1)
    top_types = type_grouped_spot_counts.nlargest(nb_top).index.tolist()

    non_top = [c for c in spot_grouped_type_counts.columns if c not in top_types]
    spot_grouped_type_counts["Other"] = spot_grouped_type_counts[non_top].sum(axis=1)
    spot_grouped_type_counts.drop(non_top, axis=1, inplace=True)
    top_types.append("Other")
    spot_grouped_type_counts.index = spot_grouped_type_counts.index.str.replace("spot_", '', )
    # Reindex the data
    return spot_grouped_type_counts, set(top_types)


def plot_spot_occurency(spot_data: pd.DataFrame, color_map: dict, nb_color_legend: int, ax: plt.Axes = None):
    # Generate the colors for the system types using the color_map

    base_fontsize = 24
    colors = [color_map[system_type] for system_type in spot_data.columns]

    # Create the stacked barplot on the provided axes
    spot_data.plot(kind='bar', stacked=True, ax=ax, color=colors)
    # ax.set_title("# Systems per spot", fontsize=base_fontsize + 4)
    ax.set_xlabel('Spot', fontsize=base_fontsize + 2)
    ax.set_ylabel('#Systems', fontsize=base_fontsize + 2)
    ax.tick_params(axis='x', labelsize=base_fontsize)
    ax.tick_params(axis='y', labelsize=base_fontsize)

    # Customize the legend with larger font size
    ax.legend(title="System types", fontsize=base_fontsize + 2, title_fontsize=base_fontsize + 4,
              loc='upper right', ncol=nb_color_legend)

    # Adjust layout to prevent overlap
    plt.tight_layout()


def get_pie_charts_info(data: pd.DataFrame, field: str = "type_grouped", nb_spots: int = 5, nb_slice: int = 10,
                        top_spots: List[str] = None):
    """
    Generate pie charts for the top X spots with the most systems, showing the distribution of system types.

    Args:
        data (pd.DataFrame): DataFrame containing the system data.
        field (str): Column name for grouping systems (e.g., 'system name' or 'type_grouped').
        nb_spots (int): Number of top spots to generate pie charts for.
        axes (list of matplotlib.axes.Axes): Axes on which to plot each pie chart.

    Returns:
        None
    """
    # Split the multiple spots and remove duplicates
    tmp_data = data.assign(spots=data['spots'].str.split(', ')).explode('spots')
    data_expanded = tmp_data.drop_duplicates(subset=["system number", "spots"], ignore_index=False)

    # Count the occurrences of systems by spot
    spot_type_counts = data_expanded.groupby(['spots', field]).size().unstack(fill_value=0)
    spot_type_counts.index = spot_type_counts.index.str.replace("spot_", '', )
    spot_type_counts = spot_type_counts.drop(index='')
    spot_type_counts['total'] = spot_type_counts.sum(axis=1)
    if top_spots is None:
        # Select the top spots
        top_spots = spot_type_counts.nlargest(nb_spots, 'total').index

    # Plot pie charts for each spot with polylines (lines connecting labels to pie chart)
    top_types = set()
    for i, spot in enumerate(top_spots):
        spot_data = spot_type_counts.loc[spot].drop('total')

        # Filter out categories with 0 occurrences
        spot_data_filtered = spot_data[spot_data > 0]  # Only keep non-zero values
        top_types |= set(spot_data_filtered.nlargest(nb_slice).index.tolist())

    return top_types, top_spots, spot_type_counts


def plot_pie(spot_data: pd.DataFrame, color_map: dict, ax: plt.Axes = None, nb_slice: int = 10,
             base_fontsize: int = 20):
    # Filter out categories with 0 occurrences
    spot_data_filtered = spot_data[spot_data > 0]  # Only keep non-zero values
    top_types = spot_data_filtered.nlargest(nb_slice)
    spot_top_type = spot_data_filtered.loc[top_types.index]
    other_sum = spot_data_filtered.drop(top_types.index).sum()
    spot_top_type['Other'] = other_sum

    colors = [color_map[system_type] for system_type in spot_top_type.index]

    # Increase the radius of the pie chart (default is 1, increase slightly)
    wedges, texts, autotexts = ax.pie(spot_top_type, labels=spot_top_type.index,
                                      colors=colors, autopct='%1.1f%%',
                                      startangle=90, labeldistance=1.4, pctdistance=0.85,
                                      wedgeprops={'linewidth': 1, 'edgecolor': 'white', 'radius': 1.2})

    # Draw lines between labels and pie slices (polylines)
    for j, text in enumerate(texts):
        wedge = wedges[j]
        # Angle in radians of the center of the wedge
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(angle))  # Calculate X-coordinate for line
        y = np.sin(np.radians(angle))  # Calculate Y-coordinate for line

        # Starting point of the line (edge of the pie slice)
        line_start = (wedge.r * x, wedge.r * y)
        # Ending point of the line (where the label is)
        line_end = (1.5 * wedge.r * x, 1.5 * wedge.r * y)

        # Draw the line connecting the wedge to the label
        connection_line = ConnectionPatch(
            xyA=line_start, xyB=line_end, coordsA="data", coordsB="data", axesA=ax, color="black", linewidth=1
        )
        ax.add_artist(connection_line)

    # Adjust text properties for better readability
    for text in texts:
        text.set_fontsize(base_fontsize + 2)  # Augmenter la taille de police des labels (légende)
    for autotext in autotexts:
        autotext.set_fontsize(base_fontsize)  # Augmenter la taille de police des pourcentages

    # Increase title font size
    spot = spot_data.name
    ax.set_title(f"Spot: {spot}", fontsize=base_fontsize + 4, pad=40)  # Augmenter la taille de police du titre


def plot_pie_charts(top_spots, spot_type_counts, color_map: dict, axes=None, nb_slice: int = 10):
    """
    Generate pie charts for the top X spots with the most systems, showing the distribution of system types.

    Args:
        data (pd.DataFrame): DataFrame containing the system data.
        field (str): Column name for grouping systems (e.g., 'system name' or 'type_grouped').
        nb_spots (int): Number of top spots to generate pie charts for.
        axes (list of matplotlib.axes.Axes): Axes on which to plot each pie chart.

    Returns:
        None
    """

    # Plot pie charts for each spot with polylines (lines connecting labels to pie chart)
    for i, spot in enumerate(top_spots):
        spot_data = spot_type_counts.loc[spot].drop('total')
        plot_pie(spot_data, color_map, axes[i], nb_slice, base_fontsize=24)

    plt.tight_layout()


def plot_spots_and_pies(data: pd.DataFrame, field: str = "type_grouped", nb_cols: int = 50,
                        nb_spots: int = 5, output: Path = None, figsize: Tuple[int, int] = (40, 20)):
    """
    Generate a plot with occurrences in spots and pie charts for the top X spots.

    Args:
        data (pd.DataFrame): DataFrame containing the system data.
        color_map (dict): Color map for system types.
        field (str): Column name for grouping systems.
        nb_cols (int): Number of spots to show in the spot_occurency plot.
        nb_spots (int): Number of top spots to generate pie charts for.
        output (Path): Path to save the output plots.

    Returns:
        None
    """
    nb_slice = 10
    nb_top_occ = 24
    spot_grouped_type_counts, top_types_occ = spot_occurency(data, output, field=field, nb_cols=nb_cols,
                                                             nb_top=nb_top_occ, keep_other_spots=False)

    top_types_pie, top_spots, spot_type_counts = get_pie_charts_info(data, field, nb_spots, nb_slice)

    # Create a consistent color map for system types
    color_map = create_color_map(top_types_occ | top_types_pie)

    # First row: Barplot for spot occurrences
    nb_col_legend = len(top_types_occ) // 10 + 1
    fig, ax = plt.subplots(figsize=figsize)
    plot_spot_occurency(spot_grouped_type_counts, color_map, nb_color_legend=nb_col_legend, ax=ax)
    ax.set_title("Occurrences in spots", fontsize=20)
    plt.tight_layout()
    plt.savefig(output / f"spots_sys_distri.png", bbox_inches="tight", dpi=300)
    plt.close()

    for i, spot in enumerate(top_spots):
        spot_data = spot_type_counts.loc[spot].drop('total')
        fig, ax = plt.subplots(figsize=figsize)
        plot_pie(spot_data, color_map, ax, nb_slice, base_fontsize=24)
        plt.tight_layout()
        # plt.show()
        if output:
            plt.savefig(output / f"spots_{spot}.png", format='png', dpi=300)
            plt.savefig(output / f"spots_{spot}.svg", format='svg')

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, nb_spots, height_ratios=[1.25, 1], figure=fig)

    ax_spot_occurrences = fig.add_subplot(gs[0, :])  # Use all columns for the barplot
    plot_spot_occurency(spot_grouped_type_counts, color_map, nb_color_legend=nb_col_legend, ax=ax_spot_occurrences)
    # ax_spot_occurrences.set_title("Occurrences in spots", fontsize=20)

    # Second row: Pie charts
    pie_axes = [fig.add_subplot(gs[1, i]) for i in range(nb_spots)]  # Create one axis per pie chart
    plot_pie_charts(top_spots, spot_type_counts, color_map, axes=pie_axes, nb_slice=nb_slice)

    plt.tight_layout()

    # Save figure if output path is provided
    if output:
        plt.savefig(output / "spots_and_pies.png", format='png', dpi=300)
        plt.savefig(output / "spots_and_pies.svg", format='svg')

    plt.show()


def get_partition_color(sub_df, colors_dict: dict, threshold=0.5):
    partition_counts = sub_df["partition"].value_counts()
    dominant_partition = partition_counts.idxmax()
    dominant_proportion = partition_counts.max() / len(sub_df)
    if dominant_proportion >= threshold:
        return colors_dict.get(dominant_partition, colors_dict["undecided"])
    else:
        accessory_count = ((partition_counts["shell"] if "shell" in partition_counts else 0) +
                           (partition_counts["cloud"] if "cloud" in partition_counts else 0) +
                           (partition_counts["accessory"] if "accessory" in partition_counts else 0))
        if accessory_count / len(sub_df) >= threshold:
            return colors_dict["accessory"]
        else:
            persistent_shell = ((partition_counts["shell"] if "shell" in partition_counts else 0) +
                                (partition_counts["persistent"] if "persistent" in partition_counts else 0) +
                                (partition_counts["persistent|shell"] if "persistent|shell" in partition_counts else 0))
            persistent_cloud = ((partition_counts["cloud"] if "cloud" in partition_counts else 0) +
                                (partition_counts["persistent"] if "persistent" in partition_counts else 0) +
                                (partition_counts["persistent|cloud"] if "persistent|cloud" in partition_counts else 0))
            if persistent_shell / len(sub_df) >= threshold:
                return colors_dict["persistent|shell"]
            elif persistent_cloud / len(sub_df) >= threshold:
                return colors_dict["undecided"]
            else:
                return colors_dict["undecided"]


def plot_treemap(data: pd.DataFrame, field: str = "system type", figsize: Tuple[int, int] = (40, 20)):
    colors_dict = {
        "persistent": "#fcb423",
        "shell": "#08d824",  # vert clair
        "cloud": "#46b2f4",  # bleu clair
        "persistent|shell": "#006400",  # vert foncé
        "persistent|cloud": "#00008B",  # bleu foncé
        "accessory": "#a54eed",  # violet clair
        "undecided": "gray"
    }
    system_counts = data[field].value_counts()
    total_systems = len(data)
    data["proportion"] = data[field].map(system_counts) / total_systems

    # 4. Préparation des données pour le treemap
    sizes = data.groupby(field)["proportion"].first()  # Taille des boîtes
    labels = data.groupby(field)["proportion"].first().index
    colors = data.groupby(field).apply(lambda group: get_partition_color(group, colors_dict))

    # 5. Création du treemap
    fig, ax = plt.subplots(figsize=figsize)
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7, ax=ax, pad=True, text_kwargs={'fontsize': 20})

    for rectangle in ax.patches:
        system = rectangle.get_label()
        size = system_counts[system]
        x = rectangle.get_x() + rectangle.get_width() / 2
        y = rectangle.get_y() + rectangle.get_height() / 2 - 1.5
        ax.text(x, y, f"{size}", ha="center", va="top", fontsize=18, fontweight="bold", color="Black")

    # Ajout des légendes
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors_dict[k], label=k) for k in colors_dict.keys()]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1.0, 0.5), fontsize=20)
    ax.set_title("Treemap of type of systems present and in which partitions they are in majority", fontsize=24)

    plt.axis('off')
    plt.show()


def spot_heatmap(data: pd.DataFrame, nb_spots: int, nb_sys: int, field: str = "type_grouped",
                 figsize: Tuple[int, int] = (40, 20)):
    # Étape 1 : Calculer la fréquence des spots
    # Transformer la colonne 'spots' en listes
    data['spots'] = data['spots'].fillna("").apply(lambda x: x.split(', '))

    # Aplatir toutes les listes pour compter la fréquence des spots
    all_spots = [spot for sublist in data['spots'] for spot in sublist if spot != '']
    spot_counts = pd.Series(all_spots).value_counts()

    # Étape 2 : Limiter aux X spots les plus fréquents et regrouper les autres
    top_spots = spot_counts.head(nb_spots).index.tolist()
    data['spots'] = data['spots'].apply(lambda spots: [spot if spot in top_spots else 'other spots' for spot in spots])

    system_counts = data.explode('spots')[field].value_counts()

    # Garder les 40 systèmes les plus fréquents
    top_systems = system_counts.head(nb_sys).index.tolist()
    # Regrouper les systèmes non sélectionnés sous "other systems"
    data[field] = data[field].apply(lambda x: x if x in top_systems else 'other systems')

    # Étape 3 : Construire la matrice des pourcentages
    # Initialiser une matrice pour les pourcentages
    spot_matrix = pd.DataFrame(0.0, index=data[field].unique(), columns=top_spots + ['other spots'])

    # Remplir la matrice
    for sys_name, row in data.groupby(field)[["spots"]].sum().iterrows():
        total_spots = len(row['spots'])
        for spot in row['spots']:
            spot_matrix.at[sys_name, spot] += 1 / total_spots * 100

    # Combiner les systèmes regroupés sous "other systems"
    if 'other systems' in spot_matrix.index:
        other_systems_row = spot_matrix.loc[spot_matrix.index == 'other systems'].sum()
        spot_matrix = spot_matrix.drop(index='other systems')
        spot_matrix.loc['other systems'] = other_systems_row

    spot_matrix = spot_matrix.sort_index()

    base_fontsize = 22
    # Étape 4 : Visualisation de la heatmap
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.5)
    sns.heatmap(spot_matrix.T, annot=True, cmap="rocket_r", cbar_kws={'label': 'Pourcentage'}, linewidths=.5, fmt=".1f")
    plt.title("Heatmap des pourcentages des systèmes dans les spots", fontsize=base_fontsize + 6, pad=20)
    plt.xlabel(field, fontsize=base_fontsize + 2)
    plt.xticks(fontsize=base_fontsize)
    plt.ylabel("Spots", fontsize=base_fontsize + 2)
    plt.yticks(fontsize=base_fontsize)
    plt.tight_layout()
    plt.show()


def gen_plot_hotspot_identification(data: pd.DataFrame, nb_spot: int = 50, percentage: float = 80):
    # Étape 1 : Transformation en une matrice (pivot table)
    heatmap_data = data.pivot_table(
        index='spot_id', columns='system type', values='organism_intersection_count', aggfunc='sum', fill_value=0
    )

    # Étape 2 : Sélection des X meilleurs spots
    row_sums = heatmap_data.sum(axis=1)  # Somme totale par spot
    top_spots = row_sums.nlargest(nb_spot).index  # Les X meilleurs spots
    other_spots = heatmap_data[~heatmap_data.index.isin(top_spots)].sum(axis=0)  # Regrouper les autres

    # Étape 3 : Ajouter une ligne "Other Spots"
    filtered_heatmap_data = heatmap_data.loc[top_spots]
    filtered_heatmap_data.loc['Other Spots'] = other_spots
    filtered_heatmap_data.index = filtered_heatmap_data.index.str.replace('spot_', "")

    # Étape 4 : Trier les spots, mais garder "Other Spots" en bas
    row_sums_filtered = filtered_heatmap_data.sum(axis=1).drop('Other Spots')  # Exclure "Other Spots" temporairement
    filtered_heatmap_data = filtered_heatmap_data.loc[
        row_sums_filtered.sort_values(ascending=False).index.tolist() + ['Other Spots']]
    # Étape 5 : Calcul des sommes pour les barplots
    col_sums = filtered_heatmap_data.sum(axis=0)

    # Étape 6 : Calcul du seuil pour le pourcentage
    row_sums_sorted = row_sums_filtered.sort_values(ascending=False)  # Sommes triées
    cumulative_sum = row_sums_sorted.cumsum()
    total_sum = row_sums_sorted.sum()
    cutoff_index = cumulative_sum[cumulative_sum >= total_sum * (percentage / 100)].index[0]

    # Ajuster la position en fonction de l'ordre inversé des axes
    cutoff_position = list(reversed(filtered_heatmap_data.index)).index(cutoff_index) + 0.5

    # Conversion des données en matrices pour Bokeh
    data_matrix = filtered_heatmap_data.values
    x = np.arange(data_matrix.shape[1])
    y = np.arange(data_matrix.shape[0])

    heatmap_width = 1800
    heatmap_height = 870
    col_bar_height = 130
    row_bar_width = 130
    # Création de la heatmap
    heatmap_fig = figure(
        # title="Heatmap of Organism Intersection Counts",
        x_range=list(filtered_heatmap_data.columns),
        # y_range=list(filtered_heatmap_data.columns),
        y_range=list(reversed(filtered_heatmap_data.index)),
        # x_range=list(reversed(filtered_heatmap_data.index)),
        tools="hover,save,pan,box_zoom,reset,wheel_zoom",
        tooltips=[("Value", "@image")],
        width=heatmap_width,
        height=heatmap_height
    )
    title_size = "15pt"
    axes_size = "13pt"
    label_size = "11pt"
    heatmap_fig.xaxis.axis_label = "System Type"
    heatmap_fig.yaxis.axis_label = "Spot ID"
    heatmap_fig.xaxis.major_label_orientation = 1
    heatmap_fig.title.text_font_size = title_size  # Taille du titre
    heatmap_fig.xaxis.axis_label_text_font_size = axes_size  # Taille de l'étiquette de l'axe X
    heatmap_fig.yaxis.axis_label_text_font_size = axes_size  # Taille de l'étiquette de l'axe Y
    heatmap_fig.xaxis.major_label_text_font_size = label_size  # Taille des labels X
    heatmap_fig.yaxis.major_label_text_font_size = label_size  # Taille des labels Y

    # Ajout de l'image
    # Générer la palette rocket_r avec 256 couleurs
    rocket_r_palette = sns.color_palette("rocket_r", as_cmap=False, n_colors=128)
    # Convertir la palette en format hexadécimal utilisable par Bokeh
    rocket_r_palette_hex = [to_hex(color) for color in rocket_r_palette]
    mapper = LinearColorMapper(palette=rocket_r_palette_hex, low=data_matrix.min(), high=data_matrix.max())
    # mapper = LogColorMapper(palette=Reds256[::-1], low=max(1, data_matrix.min()), high=data_matrix.max())
    heatmap_fig.image(image=[data_matrix[::-1]], x=0, y=0, dw=len(x), dh=len(y), color_mapper=mapper)

    # ticker = FixedTicker(ticks=list(range(10, 151, 10)))  # Ticks de 10 à 150 avec un pas de 10

    # ColorBar pour la heatmap
    color_bar = ColorBar(
        color_mapper=mapper,
        # ticker=ticker,
        location=(0, 0),
        # title="Values",  # Titre de la ColorBar
        label_standoff=12  # Distance entre les labels et la barre
    )
    # color_bar.yaxis.axis_label_text_font_size = label_size
    color_bar.major_label_text_font_size = label_size
    heatmap_fig.add_layout(color_bar, 'right')

    # Barplot des colonnes
    col_bar_fig = figure(
        x_range=heatmap_fig.x_range,
        width=heatmap_width,
        height=col_bar_height,
        title="Sum per System",
        tools=""
    )
    col_bar_fig.vbar(
        x=filtered_heatmap_data.columns,
        top=col_sums.values,
        width=0.8,
        color="gray"
    )
    col_bar_fig.xaxis.visible = False
    col_bar_fig.title.text_font_size = axes_size
    col_bar_fig.yaxis.axis_label_text_font_size = label_size
    col_bar_fig.yaxis.major_label_text_font_size = label_size

    # Barplot des lignes
    row_bar_fig = figure(
        y_range=heatmap_fig.y_range,
        width=row_bar_width,
        height=heatmap_height,
        title="Sum per Spot",
        tools=""
    )
    row_bar_fig.hbar(
        y=list(filtered_heatmap_data.index),
        right=row_sums_filtered._append(pd.Series({'Other Spots': other_spots.sum()})).values,
        height=0.8,
        color="gray"
    )
    row_bar_fig.yaxis.visible = False
    row_bar_fig.x_range.flipped = True
    row_bar_fig.title.text_font_size = axes_size
    row_bar_fig.xaxis.axis_label_text_font_size = label_size
    row_bar_fig.xaxis.major_label_text_font_size = label_size
    row_bar_fig.xaxis.major_label_orientation = 1

    # Ajouter une ligne de délimitation pour le pourcentage
    cutoff_line = Span(location=cutoff_position, dimension='width', line_color='red', line_dash='dashed', line_width=2)
    row_bar_fig.add_layout(cutoff_line)

    # Assemblage des figures avec gridplot
    final_layout = gridplot([
        [None, col_bar_fig],
        [row_bar_fig, heatmap_fig]
    ])

    # Afficher la figure
    show(final_layout)
    return list(top_spots)[:6]


def parse_data_hotspot_identification(system: pd.DataFrame, spot2rgp: pd.DataFrame, rgp2genomes: pd.DataFrame):
    def inter_org(row):
        spot_sys_org_inter = set(row['genome']).intersection(set(row['organism']))
        return [org for org in row["genome"] if org in spot_sys_org_inter]

    def sort_list(row):
        return list(sorted(row.dropna().tolist()))

    spot_rgp_genome = spot2rgp.merge(rgp2genomes, left_on='rgp_id', right_on='region', how='left')
    spot_rgp_genome = spot_rgp_genome.groupby('spot_id')['genome'].apply(sort_list).reset_index()
    system["spots"] = system["spots"].str.split(', ')
    system_expanded = system.explode('spots')
    system_expanded = system_expanded.rename(columns={'spots': 'spot_id'})

    spot_system = spot_rgp_genome.merge(system_expanded, on='spot_id', how='inner')

    spot_system["organism"] = spot_system["organism"].str.split(', ')

    final_df = spot_system[['spot_id', 'system number', 'system name', 'system type', 'organism', 'genome']]
    # final_df = final_df.rename(columns={'genome': 'organisms_list'})
    return final_df


def prepare_scatter_data(data):
    def safe_eval(x):
        if isinstance(x, str):
            return set(ast.literal_eval(x))
        return set(x)

    def count_valid_organisms(row):
        spot_id = row['spot_id']
        genome_set = row['unique_spot_org']
        # Filtrer df_1 pour le spot_id actuel
        filtered_df = data[data['spot_id'] == spot_id]
        # Compter les organismes valides

        valid_count = sum(len(set(organisms) & genome_set) for organisms in filtered_df['organism'])
        return valid_count

    data['genome_list'] = data['genome'].apply(safe_eval)
    grouped = data.groupby('spot_id').agg(
        unique_spot_org=('genome_list', lambda x: set().union(*x)),
        system_id=('system number', set),
        system_list=('system name', set),
        unique_system_cat=('system type', lambda x: set(x))).reset_index()

    grouped['nb_spot_org'] = grouped['unique_spot_org'].apply(len)
    grouped['frequency'] = grouped['unique_spot_org'].apply(lambda x: len(x) / 941)
    grouped['nb_systems'] = grouped['system_id'].apply(len)
    grouped['nb_systems_genomes_level'] = grouped.apply(count_valid_organisms, axis=1)
    grouped['nb_type'] = grouped['system_list'].apply(len)
    grouped['diversity'] = grouped['unique_system_cat'].apply(len)

    grouped["unique_spot_org"] = grouped["unique_spot_org"].apply(lambda x: ", ".join(sorted(x)))
    grouped[["spot_id", 'nb_spot_org', 'nb_systems', 'nb_type', 'diversity', "nb_systems_genomes_level",
             'unique_spot_org', 'system_id', 'system_list', 'unique_system_cat'
             ]].to_csv(output_path/"hotspot_systems.tsv", sep="\t", index=False)

    X = grouped[['frequency', 'nb_systems', "nb_spot_org", "nb_systems_genomes_level", 'nb_type', 'diversity']]

    # model = KMeans(n_clusters=6, random_state=42, n_init=10)
    model = MeanShift(bandwidth=None,
                      seeds=None,
                      bin_seeding=False,
                      min_bin_freq=1,
                      cluster_all=True,
                      n_jobs=None,
                      max_iter=30)
    labels = model.fit_predict(X)
    return grouped, labels

def scatter_plot_hotspot(data, x_field="frequency", y_field='nb_systems', s_field="diversity", labels = None,
                         file_name = "hotspot_systems.png"):
    """
    Génère un scatterplot avec :
    - En X : la fréquence relative du spot dans les organismes.
    - En Y : le nombre de systèmes dans le spot.

    :param data: DataFrame avec les colonnes ['spot', 'nb_systems', 'organisms_list']
    :param total_organisms: Nombre total d'organismes
    """
    # Tracer le scatterplot
    plt.figure(figsize=(20, 16))
    plt.scatter(data[x_field], data[y_field], s=data[s_field] * 150, c=labels, cmap='rocket_r',
                alpha=0.7, edgecolors='k')
    # Ajouter les labels des points
    for i, row in data.iterrows():
        if row["nb_systems"] >= 1000 or row["frequency"] >= .4:
            plt.text(row[x_field], row[y_field], row['spot_id'].replace("spot_", ""),
                     fontsize=16, ha='center', va='center')

    plt.xlabel("Spot frequency", fontsize=20)
    plt.xticks(fontsize=16)
    plt.ylabel("Number of systems in spot", fontsize=20)
    plt.yticks(fontsize=16)
    # plt.yscale('log')
    # plt.title("Relation entre la fréquence des spots et le nombre de systèmes")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Ajouter une légende pour la taille des points
    handles, labels = plt.gca().get_legend_handles_labels()
    for size in np.linspace(min(data['diversity']), max(data['diversity']), 5):
        handles.append(plt.scatter([], [], s=size * 150, c='k', alpha=0.7, edgecolors='k'))
        labels.append(f'    {int(size)}')

    plt.legend(handles=handles, labels=labels, title="Number of categories", ncols=len(handles),
               bbox_to_anchor=(.5, 1.02),
               loc='lower center', fontsize=16, title_fontsize=18, labelspacing=2, frameon=False)
    plt.savefig(output_path/file_name, dpi=300)
    plt.show()


def hotspot_identification(system: pd.DataFrame, spot2rgp: pd.DataFrame, rgp2genomes: pd.DataFrame,
                           nb_spots: int = 50, percentage: int = 90, nb_slice: int = 10):
    data = parse_data_hotspot_identification(system, spot2rgp, rgp2genomes)
    grouped, labels = prepare_scatter_data(data)
    scatter_plot_hotspot(grouped, labels=labels)
    scatter_plot_hotspot(grouped, y_field="nb_systems_genomes_level", labels=labels,
                         file_name="hotspot_genome_lvl.png")
    # top_spots = gen_plot_hotspot_identification(data, nb_spots, percentage=percentage)
    # top_types_pie, _, spot_type_counts = get_pie_charts_info(system, "system type", nb_spots, nb_slice,
    #                                                          top_spots=top_spots)
    # fig, axes = plt.subplots(nrows=1, ncols=len(top_spots), figsize=(40, 20))
    # color_map = create_color_map(top_types_pie)
    # plot_pie_charts(top_spots, spot_type_counts, color_map, axes=axes, nb_slice=nb_slice)
    # plt.tight_layout()
    # plt.savefig(output_path / "hotspot_pie.png")
