#!/usr/bin/env python3
# coding: utf-8

# Default libraries
from pathlib import Path
from typing import Tuple

# Installed libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local libraries
from sources.dev_utils import type2category


def load_system_data(results_path: Path) -> pd.DataFrame:
    """
    Load and aggregate system data from multiple files into a single DataFrame.

    This function reads system data from multiple TSV files, aggregates the counts
    by system type, and combines them into a single DataFrame.

    Args:
        results_path (Path): Path to the directory containing system data files.

    Returns:
        pd.DataFrame: A DataFrame with aggregated system counts by type for each species.
    """
    df = pd.DataFrame()
    for system_result in results_path.glob("*/dfinder/systems.tsv"):
        species_name = system_result.parent.parent.name
        curr_df = pd.read_csv(system_result, sep="\t")
        group = curr_df.groupby("system name")["system number"].count().reset_index(name="count")
        group["system type"] = group["system name"].map(type2category)
        group_type = group.groupby("system type")["count"].sum()
        df = pd.concat([df, group_type], axis=1, sort=True).fillna(0)
        df = df.rename(columns={"count": species_name})

    return df


def generate_heatmap(
        df: pd.DataFrame,
        output_path: Path,
        figsize: Tuple[float, float],
        cbar_label: str,
        base_fontsize: int = 24,
        **kwargs
) -> None:
    """
    Generate and save a heatmap visualization of the given data.

    Args:
        df (pd.DataFrame): DataFrame containing data to visualize.
        output_path (Path): Path where the heatmap image will be saved.
        figsize (Tuple[float, float]): Size of the figure.
        cbar_label (str): Label for the color bar.
        base_fontsize (int): Base font size for text elements.
        **kwargs: Additional keyword arguments for seaborn heatmap.
    """
    plt.figure(figsize=figsize)
    sns.set_theme(font_scale=1.5)

    cbar_kws = {'shrink': 0.8}
    ax = sns.heatmap(df.T, cbar_kws=cbar_kws, annot_kws={"rotation": 90, "fontsize": base_fontsize - 5}, **kwargs)

    plt.xlabel("Systems", fontsize=base_fontsize + 2)
    plt.xticks(fontsize=base_fontsize)
    plt.ylabel("Species", fontsize=base_fontsize + 2)
    plt.yticks(fontsize=base_fontsize)

    # Adjust color bar font size
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=base_fontsize)
    colorbar.set_label(cbar_label, fontsize=base_fontsize + 2)

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.show()
