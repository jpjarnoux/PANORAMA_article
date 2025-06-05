#!/usr/bin/env python3
# coding:utf-8

from pathlib import Path
from collections import defaultdict
from typing import Dict, Set

import itertools
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
from tqdm import tqdm

from sources.dev_utils import type2category as subtype2type


def parse_systems(sub_data: pd.DataFrame) -> pd.DataFrame:
    """Extract and filter unique gene family combinations from the given data.

    Args:
        sub_data (pd.DataFrame): DataFrame containing system data with columns for system number,
                                 system name, category, and gene family.

    Returns:
        pd.DataFrame: DataFrame containing filtered combinations of gene families.
    """

    def get_fam_comb(sub_df: pd.DataFrame) -> set:
        """Get unique gene family combinations for a subset of data.

        Args:
            sub_df (pd.DataFrame): Subset of the data.

        Returns:
            set: Unique gene family combinations.
        """
        return set(sub_df["gene family"].unique())

    # Group by system number, name, and category to get combinations
    sub_df = sub_data.groupby(["system number", "system name", "category"])[list(sub_data.columns)].apply(
        get_fam_comb).reset_index()
    sub_df.columns = ["ID", "name", "category", "combination"]

    # Sort combinations by size in descending order
    combs_list = sorted(sub_df["combination"].tolist(), key=len, reverse=True)

    # Filter combinations to retain maximal sets
    filtered_combs = defaultdict(lambda: {"id": set(), "names": set(), "categories": set()})

    for comb in combs_list:
        systems_df = sub_df[sub_df["combination"] == comb]
        ids = set(itertools.chain(*systems_df["ID"].str.split(", ").tolist()))
        names = set(systems_df["name"].tolist())
        categories = set(systems_df["category"].tolist())

        # Find the largest combination that contains the current combination
        largest_comb = frozenset(sorted(comb))
        for larger in filtered_combs:
            if comb.issubset(larger):
                largest_comb = larger
                break

        # Update the filtered combinations
        filtered_combs[largest_comb]["id"] |= ids
        filtered_combs[largest_comb]["names"] |= names
        filtered_combs[largest_comb]["categories"] |= categories

    return pd.DataFrame.from_dict(filtered_combs, orient="index")


def read_projection(file: Path) -> pd.DataFrame:
    """Read and process a projection file.

    Args:
        file (Path): Path to the projection file.

    Returns:
        pd.DataFrame: Processed DataFrame grouped by spots.
    """
    proj_df = pd.read_csv(file, sep="\t", header=0,
                          usecols=["system number", "system name", "gene family", "annotation", "spots"],
                          dtype={0: str})

    proj_df["category"] = proj_df["system name"].map(subtype2type)
    return proj_df.groupby("spots")[
        ['system number', 'system name', 'gene family', 'annotation', 'spots', 'category']].apply(parse_systems)


def get_data_projection(projection: Path) -> pd.DataFrame:
    """Read and concatenate projection files.

    Args:
        projection (Path): Path to the directory containing projection files.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all projection files.
    """
    spot_df_list = []
    for file in tqdm(list(projection.glob("*.tsv")), desc="Read projections"):
        spot_df = read_projection(file)
        if not spot_df.empty:
            spot_df["spot_sys_organisms"] = file.stem
            spot_df_list.append(spot_df.reset_index().rename(columns={"level_1": "combinations"}).set_index("spots"))

    return pd.concat(spot_df_list, axis="index")


def get_spot2orgs(tables: Path) -> Dict[str, Set[str]]:
    """Map spots to organisms.

    Args:
        tables (Path): Path to the directory containing organism tables.

    Returns:
        Dict[str, Set[str]]: Dictionary mapping spots to sets of organisms.
    """
    spot2orgs = defaultdict(set)
    for file in tqdm(list(tables.glob("*.tsv")), desc="Read tables"):
        org = file.stem
        table_df = pd.read_csv(file, sep="\t", header=0, usecols=["Spot"]).dropna(subset=["Spot"])
        for spot in table_df["Spot"].unique():
            spot2orgs[spot].add(org)

    return spot2orgs


def parse_data_scatter_plot(data: pd.DataFrame, spot2orgs: Dict[str, Set[str]], nb_genomes: int) -> pd.DataFrame:
    """Parse data for scatter plot generation.

    Args:
        data (pd.DataFrame): DataFrame containing spot data.
        spot2orgs (Dict[str, Set[str]]): Dictionary mapping spots to organisms.
        nb_genomes (int): Total number of genomes.

    Returns:
        pd.DataFrame: DataFrame with parsed data for scatter plot.
    """

    def grouped_data(sub_df: pd.DataFrame) -> pd.Series:
        """Group data by spot for analysis.

        Args:
            sub_df (pd.DataFrame): Subset of the data for a specific spot.

        Returns:
            pd.Series: Series containing grouped data metrics.
        """
        spot_id = sub_df.index.unique().tolist()[0]
        nb_spot_org = len(spot2orgs[spot_id])
        frequency = nb_spot_org / nb_genomes
        defense_spot_frequency = len(sub_df["spot_sys_organisms"].unique().tolist()) / nb_genomes
        nb_systems_org = len(sub_df["combinations"].tolist())
        nb_systems_pan = len(set(itertools.chain(*sub_df["id"])))
        diversity = len(set(itertools.chain(*sub_df["categories"])))
        return pd.Series([nb_spot_org, frequency, defense_spot_frequency, nb_systems_org, nb_systems_pan, diversity],
                         index=["nb spot genomes", "spot frequency", "defense spot frequency", "nb systems genomes",
                                "nb systems pangenome", "diversity"])

    return data.groupby("spots").apply(grouped_data)


def gen_scatter_plot(data: pd.DataFrame, x_field: str = "spot frequency", x_label: str = "Spot frequency",
                     y_field: str = 'nb systems pangenome', y_label: str = "Number of systems",
                     file_name: str = "hotspot_systems_pangenome.png", min_freq: float = 0.4,
                     min_y_for_text: float = 60, output: Path = Path(".")):
    """Generate a scatter plot from the given data.

    Args:
        data (pd.DataFrame): DataFrame containing data for plotting.
        x_field (str): Field to use for the x-axis.
        x_label (str): Label for the x-axis.
        y_field (str): Field to use for the y-axis.
        y_label (str): Label for the y-axis.
        file_name (str): Name of the file to save the plot.
        min_freq (float): Minimum frequency threshold for labeling spots.
        min_y_for_text (float): Minimum y-value threshold for labeling spots.
        output (Path): Path to the directory where the plot will be saved.
    """
    fig, ax = plt.subplots(figsize=(20, 16))
    plt.scatter(data[x_field], data[y_field], s=data["diversity"] * 150, c='deepskyblue', alpha=0.7, edgecolors='k')

    # Add horizontal and vertical lines at the thresholds
    plt.axhline(y=min_y_for_text, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=min_freq, color='red', linestyle='--', linewidth=2)

    # Shade the region where values exceed the thresholds
    # ax.add_patch(Rectangle((min_freq, min_y_for_text), max(data[x_field]) - min_freq + .02,
    #                        max(data[y_field]) - min_y_for_text + 12, alpha=0.3, color='red'))

    # Add labels to points
    for spot_id, row in data.iterrows():
        if row[y_field] >= min_y_for_text or row[x_field] >= min_freq:
            plt.text(row[x_field], row[y_field], spot_id.replace("spot_", ""), fontsize=16, ha='center', va='center')

    plt.xlabel(x_label, fontsize=20)
    plt.xticks(fontsize=16)
    plt.ylabel(y_label, fontsize=20)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add legend for point sizes
    handles, labels = plt.gca().get_legend_handles_labels()
    sizes = [1] + list(range(10, int(max(data['diversity'])) + 1, 10)) + [int(max(data['diversity']))]
    for size in sizes:
        handles.append(plt.scatter([], [], s=size * 150, c='deepskyblue', alpha=0.7, edgecolors='k'))
        labels.append(f'    {int(size)}')

    plt.legend(handles=handles, labels=labels, title="Number of categories", ncols=len(handles),
               bbox_to_anchor=(0.5, 1.02), loc='lower center', fontsize=16, title_fontsize=18,
               labelspacing=2, frameon=False)
    plt.savefig(output / file_name, dpi=300)
    plt.show()


def scatter_plot(data: pd.DataFrame, spot2orgs: Dict[str, Set[str]], nb_genomes: int, output: Path = Path(".")):
    """Generate and save scatter plots for the given data.

    Args:
        data (pd.DataFrame): DataFrame containing data for plotting.
        spot2orgs (Dict[str, Set[str]]): Dictionary mapping spots to organisms.
        nb_genomes (int): Total number of genomes.
        output (Path): Path to the directory where plots will be saved.
    """
    parsed_data = parse_data_scatter_plot(data, spot2orgs, nb_genomes)
    parsed_data.to_csv(output / "scatter_hotspot.tsv", sep="\t", index=True)
    gen_scatter_plot(parsed_data, x_field='defense spot frequency', x_label='Defense spot frequency', min_freq=.25,
                     y_field="nb systems pangenome", file_name="defense_hotspot_systems_pangenome.png", output=output)