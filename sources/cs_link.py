#!/usr/bin/env python3
# coding: utf-8

import itertools
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sources.dev_utils import type2category


def parse_data(data: pd.DataFrame, systems: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Parse and merge data from DataFrames containing system and spot information.

    This function processes the input data to extract and merge information about systems,
    pangenomes, and spots, returning a DataFrame with the parsed data.

    Args:
        data (pd.DataFrame): DataFrame containing conserved spot data.
        systems (Dict[str, pd.DataFrame]): Dictionary of DataFrames containing system data for each species.

    Returns:
        pd.DataFrame: A DataFrame with parsed data.
    """

    def grouped_row(sub_df: pd.DataFrame) -> pd.Series:
        """
        Process a group of rows to extract system, pangenome, and spot information.

        Args:
            sub_df (pd.DataFrame): Subset of the DataFrame to process.

        Returns:
            pd.Series: Series containing processed data for a group.
        """
        spot_id = int(sub_df.spots.unique()[0].replace('spot_', ""))
        systems_list = sub_df["system name"].tolist()
        sys_id_list = sub_df["system number"].tolist()
        return pd.Series([spot_id, sys_id_list, systems_list], index=["Spot_ID", "sys_id", "systems"])

    def grouped_cs(sub_df: pd.DataFrame) -> pd.Series:
        """
        Process a group of rows to extract spot and system information for conserved systems.

        Args:
            sub_df (pd.DataFrame): Subset of the DataFrame to process.

        Returns:
            pd.Series: Series containing processed spot and system data for conserved systems.
        """
        spot_list = sub_df["Spot_ID"].tolist()
        pan_list = sub_df["Pangenome"].tolist()
        sys_id_lists = sub_df["sys_id"].apply(lambda x: x if isinstance(x, list) else []).tolist()
        systems_id_list = list(itertools.chain(*sys_id_lists))
        sys_lists = sub_df["systems"].apply(lambda x: x if isinstance(x, list) else []).tolist()
        systems_list = list(itertools.chain(*sys_lists))
        return pd.Series([spot_list, pan_list, systems_id_list, systems_list],
                         index=["Spot_ID", "Pangenome", "systems_id", "systems"])

    spots2systems_list = []
    for species, system_df in systems.items():
        system_df["spots"] = system_df["spots"].str.split(", ")
        system_df = system_df.explode("spots")
        spots_df = system_df.groupby("spots").apply(grouped_row).reset_index(drop=True)
        spots_df["Pangenome"] = species
        spots_df = spots_df.set_index(["Spot_ID", "Pangenome"])
        spots2systems_list.append(spots_df)

    spots2systems = pd.concat(spots2systems_list, axis="index")
    data = data.set_index(["Spot_ID", "Pangenome"])
    cs2systems = data.join(spots2systems, how="left").reset_index()
    cs2systems_group = (
        cs2systems.groupby(["Conserved_Spot_ID"])
        .apply(grouped_cs)
        .reset_index()
        .set_index(["Conserved_Spot_ID"], drop=True)
    )
    return cs2systems_group


def generate_tile_plot(pivot: pd.DataFrame, output: Path, base_fontsize: int = 18) -> None:
    """
    Generate and save a tile plot visualizing the presence of system categories across clusters.

    Args:
        pivot (pd.DataFrame): DataFrame containing data to visualize.
        output (Path): Path where the plot image will be saved.
        base_fontsize (int): Base font size for the plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(24, 20))

    sns.heatmap(
        pivot,
        annot=False,
        cmap="rocket_r",
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    ax.set_ylabel("Cluster", fontsize=base_fontsize + 2)
    ax.set_xlabel("System category", fontsize=base_fontsize + 2)
    ax.tick_params(axis='y', labelsize=base_fontsize, rotation=0)
    ax.tick_params(axis='x', labelsize=base_fontsize)

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=base_fontsize)
    colorbar.set_label("Presence (%)", fontsize=base_fontsize + 2)

    plt.tight_layout()
    plt.savefig(output.parent / f"{output.name}.png", format='png', dpi=300)
    plt.savefig(output.parent / f"{output.name}.svg", format='svg')
    plt.show()


def generate_tile_plot_visualization(data: pd.DataFrame, output: Path, base_fontsize: int = 18) -> None:
    """
    Generate and save tile plot visualizations of system data.

    This function generates a heatmap visualization of system category presence across clusters.

    Args:
        data (pd.DataFrame): DataFrame containing system data.
        output (Path): Path to the directory where plots will be saved.
        base_fontsize (int): Base font size for the plot.

    Returns:
        None
    """

    def calculate_percentage(sub_data: pd.DataFrame) -> pd.Series:
        """
        Calculate the percentage of each system type by cluster.

        Args:
            sub_data (pd.DataFrame): Subset of the DataFrame to process.

        Returns:
            pd.Series: Series containing percentage data for each system type.
        """
        total = sub_data["count"].sum()
        return sub_data.groupby("category")["count"].apply(lambda x: 100 * x / total)

    def get_pangenome_matrix(sub_df: pd.DataFrame) -> Tuple:
        """
        Generate a matrix of pangenome presence for each cluster.

        Args:
            sub_df (pd.DataFrame): Subset of the DataFrame to process.

        Returns:
            Tuple: A tuple containing unique sorted pangenomes present in the cluster.
        """
        pan_present = tuple(sorted(set(itertools.chain(*sub_df["Pangenome"].values))))
        return pan_present

    data_exp = data.explode('systems')
    data_exp["category"] = data_exp['systems'].map(type2category)
    result = (
        data_exp.groupby(['Conserved_Spot_ID', 'category'])
        .size()
        .reset_index(name='count')
    )
    result = (
        result.groupby(['Conserved_Spot_ID'])
        .apply(calculate_percentage)
        .reset_index()
        .drop('level_2', axis=1)
    )

    heatmap_data = (
        result.pivot(index='Conserved_Spot_ID', columns='category', values='count')
        .fillna(0)
    )

    pangenome_combinations = (
        data.groupby("Conserved_Spot_ID")
        .apply(get_pangenome_matrix)
    )
    pangenome_combinations.name = "combination"
    pangenome_combinations = pangenome_combinations.reset_index()
    grouped_ids = pangenome_combinations.groupby(by="combination")["Conserved_Spot_ID"].apply(list)

    sorted_groups = grouped_ids.reindex(
        sorted(grouped_ids.index, key=lambda d: (len(d), d), reverse=True)
    )
    ordered_ids = [cid for group in sorted_groups for cid in group]
    heatmap_data = heatmap_data.reindex(ordered_ids)
    heatmap_data = heatmap_data.dropna(thresh=(len(heatmap_data.columns)))

    generate_tile_plot(heatmap_data, output / "cs_link", base_fontsize=base_fontsize)
