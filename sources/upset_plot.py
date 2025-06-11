#!/usr/bin/env python3
# coding: utf-8

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet

from sources.dev_utils import type2category


def parse_data(data: pd.DataFrame, systems: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse and merge data from DataFrames containing system and spot information.

    This function processes the input data to extract and merge information about systems,
    pangenomes, and spots, returning a DataFrame with the parsed data.

    Args:
        data (pd.DataFrame): DataFrame containing conserved spot data.
        systems (Dict[str, pd.DataFrame]): Dictionary of DataFrames containing system data for each species.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A DataFrame with parsed data and a list of pangenomes.
    """

    def grouped_row(sub_df: pd.DataFrame) -> pd.Series:
        """
        Process a group of rows to extract system, pangenome, and spot information.

        Args:
            sub_df (pd.DataFrame): Subset of the DataFrame to process.

        Returns:
            pd.Series: Series containing processed data for a group.
        """
        sys = sub_df["systems_name"].str.split(',').explode().tolist()
        pangenomes = sub_df["Pangenome"].tolist()
        spots = sub_df["Spot ID"].tolist()
        return pd.Series(
            [sys, pangenomes, spots] + [p in pangenomes for p in pangenomes_list],
            index=["system", "pangenome", "spot"] + pangenomes_list
        )

    # Identify all spots associated with a system in the systems dictionary
    spots2systems_list = []
    for species, system_df in systems.items():
        system_df["Pangenome"] = species
        spots2systems_list.append(system_df)

    spots2systems_df = pd.concat(spots2systems_list, ignore_index=True)
    spots2systems_df["Spot ID"] = spots2systems_df["name"].str.replace("spot_", "").apply(int)

    joined_df = pd.merge(left=data, right=spots2systems_df, on=["Pangenome", "Spot ID"],
                         how="inner", validate="one_to_one")
    joined_df = joined_df.drop(columns=["name", "systems_ID"])
    pangenomes_list = joined_df['Pangenome'].unique().tolist()
    grouped_df = joined_df.groupby(["Conserved ID"]).apply(grouped_row)
    exploded_df = grouped_df.explode('system').reset_index()
    exploded_df["system_type"] = exploded_df["system"].map(type2category)
    return exploded_df, pangenomes_list


def generate_upset_plot_spots(
        cs_spots: pd.DataFrame,
        nb_spots: Dict[str, int],
        systems_data: Dict[str, pd.DataFrame],
        output: Path,
        base_font_size: int = 12
) -> None:
    """
    Generate and save an UpSet plot visualizing spot intersections across pangenomes.

    This function creates an UpSet plot to visualize the intersections of spots across different
    pangenomes, highlighting the presence of systems and their counts.

    Args:
        cs_spots (pd.DataFrame): DataFrame containing conserved spot data.
        nb_spots (Dict[str, int]): Dictionary of the number of spots per pangenome.
        systems_data (Dict[str, pd.DataFrame]): Dictionary of system data for each pangenome.
        output (Path): Path to save the output plot.
        base_font_size (int): Base font size for the plot.

    Returns:
        None
    """

    def grouped_row(sub_df: pd.DataFrame) -> pd.Series:
        """
        Process a group of rows to extract intersection and system information.

        Args:
            sub_df (pd.DataFrame): Subset of the DataFrame to process.

        Returns:
            pd.Series: Series containing processed intersection and system data for a group.
        """
        pangenomes = sub_df["Pangenome"].tolist()
        intersection = frozenset(pangenomes)
        spots = sub_df["Spot ID"].tolist()
        with_systems = []
        nb_systems = []
        for p, spot in zip(pangenomes, spots):
            spot2sys_df = systems_data[p]
            spot2sys_df = spot2sys_df.set_index("name")
            if f"spot_{spot}" in spot2sys_df.index.values:
                with_systems.append(True)
                shared_systems = spot2sys_df.loc[f"spot_{spot}"]["systems_ID"].split(",")
                inter2sys[intersection][p] |= set(shared_systems)
                nb_systems.append(len(shared_systems))
                shared_pan_systems[p] |= set(shared_systems)
            else:
                with_systems.append(False)
                nb_systems.append(0)

        cs_with_systems = any(with_systems)
        cs_nb_systems = sum(nb_systems)
        return pd.Series(
            [cs_with_systems, cs_nb_systems, pangenomes, spots, with_systems, nb_systems] +
            [p in pangenomes for p in pangenomes_list],
            index=["cs_with_systems", "cs_nb_sys", "pangenome", "spot", "with_systems", "nb_systems"] + pangenomes_list
        )

    count_spot = cs_spots.groupby('Pangenome')["Spot ID"].count()
    pangenomes_list = list(nb_spots.keys())
    inter2sys = defaultdict(lambda: defaultdict(set))
    shared_pan_systems = {p: set() for p in pangenomes_list}
    all_pan_systems = {
        p: set(systems_data[p]["systems_ID"].str.split(",").explode().unique())
        for p in pangenomes_list
    }

    binary_df = (
        cs_spots.groupby("Conserved ID")
        .apply(grouped_row)
        .reset_index()
        .set_index(pangenomes_list + ["Conserved ID"])
    )

    alone_df = pd.DataFrame(
        [
            [p == pangenome for p in pangenomes_list] +
            [pd.NA, True, len(all_pan_systems[pangenome]) - len(shared_pan_systems[pangenome]), [pangenome],
             pd.NA, pd.NA, pd.NA, nb_spot - count_spot[pangenome], pd.NA]
            for pangenome, nb_spot in nb_spots.items()
        ],
        columns=pangenomes_list + ["Conserved ID"] + binary_df.columns.tolist() + ["nb_spot", "nb_with_sys"]
    ).set_index(pangenomes_list + ["Conserved ID"])

    binary_df["nb_spot"] = binary_df["spot"].apply(len)
    binary_df["nb_with_sys"] = binary_df["with_systems"].apply(lambda x: sum(x))
    binary_df = pd.concat([binary_df, alone_df], axis="index").reset_index().set_index(pangenomes_list)

    binary_df.to_csv(output / "upset_data.tsv", sep="\t", index=True)

    upset = UpSet(
        binary_df,
        intersection_plot_elements=0,
        sort_by="cardinality",
        include_empty_subsets=False,
        element_size=None,
        show_counts=True
    )

    upset.add_stacked_bars(by="cs_with_systems", title="Conserved systems", elements=10)
    upset.add_stacked_bars(by="cs_with_systems", sum_over="nb_spot", colors=["darkblue"] * 15, title="# Spots")
    upset.add_stacked_bars(by="cs_with_systems", sum_over='cs_nb_sys', colors=["lightblue"] * 15, title="# Systems")

    fig = plt.figure(figsize=(24, 16))
    plot = upset.plot(fig=fig)

    plot["totals"].set_xlabel("# Spot clusters in species", fontsize=base_font_size + 2)
    plot["totals"].tick_params(labelsize=base_font_size)

    l, b, w, h = plot["totals"].get_position().bounds
    plot["totals"].set_position([l - 0.05, b, w, h])

    for annotation in plot["totals"].texts:
        annotation.set_fontsize(base_font_size)

    plot["matrix"].tick_params(labelsize=base_font_size)
    plot["extra0"].tick_params(labelsize=base_font_size)
    plot["extra0"].set_ylabel("# Spot clusters", fontsize=base_font_size + 2)
    plot["extra0"].legend(
        title="With defense systems",
        loc='upper right',
        fontsize=base_font_size,
        title_fontsize=base_font_size + 2,
        frameon=False,
        bbox_to_anchor=(1, 1.03)
    )

    for annotation in plot["extra0"].texts:
        annotation.set_fontsize(base_font_size)

    plot["extra1"].tick_params(labelsize=base_font_size)
    plot["extra1"].yaxis.label.set_size(base_font_size + 2)
    plot["extra1"].set_yscale("log")

    for annotation in plot["extra1"].texts:
        annotation.set_fontsize(base_font_size)

    plot["extra1"].get_legend().remove()
    plot["extra2"].tick_params(labelsize=base_font_size)
    plot["extra2"].yaxis.label.set_size(base_font_size + 2)
    plot["extra2"].set_yscale("log")

    for annotation in plot["extra2"].texts:
        annotation.set_fontsize(base_font_size)

    plot["extra2"].get_legend().remove()

    plt.savefig(output / 'upset_color_sys.png', format='png', dpi=300)
    plt.savefig(output / 'upset_color_sys.svg', format='svg')
    plt.show()
