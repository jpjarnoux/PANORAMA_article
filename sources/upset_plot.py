#!/usr/bin/env python3
# coding: utf-8

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from upsetplot import UpSet


def parse_data(cs_data: pd.DataFrame, systems: Dict[str, pd.DataFrame],
               total_spots: Dict[str, int] = None) -> pd.DataFrame:
    """
       Parse and merge data from DataFrames containing system and spot information.

       This function processes the input data to extract and merge information about systems,
       pangenomes, and spots, returning a DataFrame with the parsed data.

       Args:
           cs_data (pd.DataFrame): DataFrame containing conserved spot data.
           systems (Dict[str, pd.DataFrame]): Dictionary of DataFrames containing system data for each species.
           total_spots (Dict[str, int]): Dictionary of total number of spots for each species.

       Returns:
           Tuple[pd.DataFrame, List[str]]: A DataFrame with parsed data and a list of pangenomes.
    """
    pangenome_list = systems.keys()
    # Step 1: Prepare system data for each pangenome
    # We need: which systems are associated with which spots
    pangenome_spot_to_systems = {}  # {pangenome: {spot: set(system_numbers)}}

    for pangenome, sys_df in systems.items():
        sys_df = sys_df.copy()
        sys_df['spots'] = sys_df['spots'].fillna('')

        spot_to_systems = {}
        for _, row in sys_df.iterrows():
            system_number = row['system number']
            if row['spots']:
                spots = [s.strip() for s in row['spots'].split(',')]
                for spot in spots:
                    if spot not in spot_to_systems:
                        spot_to_systems[spot] = set()
                    spot_to_systems[spot].add(system_number)

        pangenome_spot_to_systems[pangenome] = spot_to_systems

    # Step 2: Track which spots are in conserved groups
    conserved_spots = {}  # {pangenome: set of spot_ids}
    for pg in pangenome_list:
        conserved_spots[pg] = set()

    for _, row in cs_data.iterrows():
        pg = row['Pangenome']
        spot_id = row['Spot_ID']
        if pg in conserved_spots:
            conserved_spots[pg].add(spot_id)

    # Step 3: Process conserved spots (grouped)
    result_rows = []

    for conserved_id, group in cs_data.groupby('Conserved_Spot_ID'):
        # Get pangenomes and spots in this conserved group
        pangenomes = group['Pangenome'].tolist()
        spots = group['Spot_ID'].tolist()

        # For each pangenome in the list, check presence
        pangenome_presence = {pg: (pg in pangenomes) for pg in pangenome_list}

        # Collect all unique systems across all spots in this conserved group
        all_systems = set()
        with_systems = []
        nb_systems = []

        for i, (pg, spot) in enumerate(zip(pangenomes, spots, strict=False)):
            spot_str = f"spot_{spot}"
            systems_set = pangenome_spot_to_systems.get(pg, {}).get(spot_str, set())
            nb_sys = len(systems_set)
            nb_systems.append(nb_sys)
            with_systems.append(nb_sys > 0)
            all_systems.update(systems_set)  # Add to global set

        # Calculate summary stats
        cs_nb_sys = len(all_systems)  # Unique systems across all spots
        cs_with_systems = cs_nb_sys > 0
        nb_spot = len(spots)
        nb_with_sys = sum(with_systems)

        # Build row dict with dynamic pangenome columns
        row = {}
        for pg in pangenome_list:
            row[pg] = pangenome_presence[pg]

        row.update({
            'Conserved ID': float(conserved_id),
            'cs_with_systems': cs_with_systems,
            'cs_nb_sys': cs_nb_sys,
            'pangenome': pangenomes,
            'spot': spots,
            'with_systems': with_systems,
            'nb_systems': nb_systems,
            'nb_spot': nb_spot,
            'nb_with_sys': nb_with_sys
        })

        result_rows.append(row)

    # Step 4: Add non-conserved spots for each pangenome
    for pg in pangenome_list:
        # Get all spots for this pangenome (0 to total_spots[pg] - 1)
        all_spots = set(range(total_spots[pg]))

        # Get non-conserved spots
        non_conserved = all_spots - conserved_spots[pg]

        if len(non_conserved) > 0:
            # Collect all unique systems for non-conserved spots
            all_systems = set()
            nb_with_sys = 0

            for spot_id in non_conserved:
                spot_str = f"spot_{spot_id}"
                systems_set = pangenome_spot_to_systems.get(pg, {}).get(spot_str, set())
                all_systems.update(systems_set)
                if len(systems_set) > 0:
                    nb_with_sys += 1

            # Build row for non-conserved spots
            row = {}
            for pg_col in pangenome_list:
                row[pg_col] = (pg_col == pg)

            row.update({
                'Conserved ID': np.nan,
                'cs_with_systems': len(all_systems) > 0,
                'cs_nb_sys': len(all_systems),  # Unique systems
                'pangenome': [pg],
                'spot': None,
                'with_systems': None,
                'nb_systems': None,
                'nb_spot': len(non_conserved),
                'nb_with_sys': nb_with_sys
            })

            result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    return result_df

def generate_upset_plot_spots(data: pd.DataFrame, pangenome_list: List[str], output, base_font_size: int = 12):
    binary_df = data.set_index(pangenome_list)
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
