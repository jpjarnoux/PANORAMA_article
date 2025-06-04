#!/usr/bin/env python3
# coding:utf-8

# default libraries
from pathlib import Path
from typing import Dict, FrozenSet, Set, Tuple

# installed libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# local libraries
from sources.dev_utils import type2category, cluster_low_sys


def gen_boxplot(count_df: pd.DataFrame, output: Path) -> Tuple[Set[str], Set[str]]:
    """
    Generates a boxplot based on the provided DataFrame and saves it as an image file.

    Args:
        count_df (pd.DataFrame): The DataFrame containing occurrence counts.
        output (Path): The output path where the plot image will be saved.

    Returns:
        tuple: A tuple containing the main group and other groups after clustering.
    """
    # Add a 'Group' column to the DataFrame
    count_df.reset_index(inplace=True)
    count_df.rename(columns={'index': 'systems'}, inplace=True)
    count_df['Group'] = count_df['systems'].map(type2category)

    # Summarize counts by group
    sum_df = count_df.groupby('Group').sum(numeric_only=True).T.sum(numeric_only=True).T
    sum_df.columns = ['Total']

    # Cluster the systems based on the threshold (0.7)
    main_group, others_group = cluster_low_sys(sum_df.to_dict(), 0.85)

    # Replace the group name with 'Other' for groups classified as 'others'
    for other in others_group:
        count_df["Group"] = count_df["Group"].replace(other, 'Other')

    # Reshape the DataFrame for plotting
    count_melt = pd.melt(count_df, id_vars=['systems', 'Group'], var_name='Condition', value_name='Occurrences')
    count_melt = count_melt.dropna()

    # Create the boxplot
    _, ax = plt.subplots(
        figsize=(20, 11.25),
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 3]}  # Ratio of 1:3 for ax[0] and ax[1]
    )
    sns.boxplot(x='Group', y='Occurrences', hue='Condition', data=count_melt, ax=ax[0], gap=.1)

    ax[0].set_ylim(ymin=105, ymax=130)

    # Rotate x-axis labels for better readability
    # ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right', fontsize=16)
    ax[0].set_yticks(ax[0].get_yticks())
    ax[0].set_yticklabels([f'{int(y)}' for y in ax[0].get_yticks()], fontsize=16)

    # Set x and y axis labels
    ax[0].set_xlabel('Systems type', fontsize=18)

    ax[0].set_ylabel('')

    # Position the legend outside the plot
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='x-large')

    sns.boxplot(x='Group', y='Occurrences', hue='Condition', data=count_melt, ax=ax[1], gap=.1)

    ax[1].set_ylim(ymin=0, ymax=75)
    # Rotate x-axis labels for better readability
    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].set_xticklabels([x.get_text() for x in ax[1].get_xticklabels()], rotation=45, ha='right', fontsize=16)
    ax[1].set_yticks(ax[1].get_yticks())
    ax[1].set_yticklabels([f'{int(y)}' for y in ax[1].get_yticks()], fontsize=16)

    # Set x and y axis labels
    ax[1].set_xlabel('Systems type', fontsize=18)

    ax[1].set_ylabel('Occurrences', fontsize=18)

    ax[1].get_legend().remove()

    # plt.show()
    # Save the plot as an image
    plt.savefig(output, transparent=False)
    plt.savefig(output.with_suffix('.svg'), format='svg')
    plt.show()
    return main_group, others_group


def gen_boxplot_pangenome(padloc: Dict[str, Set[FrozenSet[str]]], dfinder: Dict[str, Set[FrozenSet[str]]],
                          panorama: Dict[str, Set[FrozenSet[str]]], output: Path) -> Tuple[Set[str], Set[str]]:
    """
    Generates a boxplot comparing systems predicted in pangenome between different tools and PANORAMA.

    Args:
        padloc (Dict[str, Set[FrozenSet[str]]]): systems predicted by 'Padloc' at pangenome level.
        dfinder (Dict[str, Set[FrozenSet[str]]]): systems predicted by 'DefenseFinder' at pangenome level.
        panorama (Dict[str, Set[FrozenSet[str]]]): systems predicted by 'PANORAMA' with models from 'PADLOC' & 'DefenseFinder'
        output (Path): The output path where the boxplot image will be saved.

    Returns:
        tuple: A tuple containing the main group and other groups after clustering.
    """

    # Get the complete list of system_types (union of all dictionary keys)
    system_types = set()

    # Iterate over each tool's data directly
    for tool in [padloc, dfinder, panorama]:
        for res in tool.keys():
            system_types.add(res)

    # Convert the set to a sorted list
    system_types = sorted(list(system_types))

    # Create a DataFrame with system_types as the index
    count_df = pd.DataFrame(index=system_types)

    # For each tool, count the number of systems per system_type
    for name, tool in zip(["padloc", "dfinder", "panorama"], [padloc, dfinder, panorama]):
        count_df[name] = count_df.index.map(lambda x: len(tool.get(x, set())))

    # Generate and save the boxplot
    return gen_boxplot(count_df, output)
