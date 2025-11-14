#!/usr/bin/env python3
# coding: utf-8

# Default libraries
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, FrozenSet, Set, Tuple

# Installed libraries
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib_venn import venn3

# Local libraries
from sources.dev_utils import similar


def compare_sets(set_a: Set[FrozenSet[str]], set_b: Set[FrozenSet[str]], set_c: Set[FrozenSet[str]]) -> Dict:
    """Compare three sets and find unique and common elements among them.

    Args:
        set_a (Set[FrozenSet[str]]): First set of frozensets to compare.
        set_b (Set[FrozenSet[str]]): Second set of frozensets to compare.
        set_c (Set[FrozenSet[str]]): Third set of frozensets to compare.

    Returns:
        Dict: A dictionary with counts and frozensets categorized by their intersections.
    """
    # Create unique sets from nested lists
    unique_a = set(chain(*set_a))
    unique_b = set(chain(*set_b))
    unique_c = set(chain(*set_c))

    # Find intersections
    common_abc = unique_a & unique_b & unique_c
    common_ab = (unique_a & unique_b) - common_abc
    common_ac = (unique_a & unique_c) - common_abc
    common_bc = (unique_b & unique_c) - common_abc

    # Find unique elements
    unique_a -= (common_abc | common_ab | common_ac)
    unique_b -= (common_abc | common_ab | common_bc)
    unique_c -= (common_abc | common_ac | common_bc)

    def find_frozensets(category: Set[str]) -> Dict[str, Set[FrozenSet[str]]]:
        """Find frozensets associated with each category.

        Args:
            category (Set[str]): The set of elements to find.

        Returns:
            Dict[str, Set[FrozenSet[str]]]: A dictionary of frozensets categorized by their presence in the sets.
        """
        mapping = {"A": set_a, "B": set_b, "C": set_c}
        result = {k: set() for k in mapping}

        for key, group in mapping.items():
            for fs in group:
                if category & fs:
                    result[key].add(fs)

        return result

    # Build the dictionary of frozensets
    frozenset_dict = {
        "111": find_frozensets(common_abc),
        "110": find_frozensets(common_ab),
        "101": find_frozensets(common_ac),
        "011": find_frozensets(common_bc),
        "100": find_frozensets(unique_a),
        "010": find_frozensets(unique_b),
        "001": find_frozensets(unique_c),
    }

    # Return the results
    return {
        "counts": {
            "111": len(common_abc),
            "110": len(common_ab),
            "101": len(common_ac),
            "011": len(common_bc),
            "100": len(unique_a),
            "010": len(unique_b),
            "001": len(unique_c),
        },
        "frozensets": frozenset_dict,
    }


def get_common_specified_systems(padloc: Dict[str, Set[FrozenSet[str]]], dfinder: Dict[str, Set[FrozenSet[str]]],
                                 panorama: Dict[str, Set[FrozenSet[str]]]) -> Dict[str, int]:
    """Calculate common and specific systems among three tools.

    Args:
        padloc (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PADLOC.
        dfinder (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for DFinder.
        panorama (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PANORAMA.

    Returns:
        Dict: A dictionary containing subset counts.
    """
    sys_type_set = set(padloc.keys()).union(dfinder.keys(), panorama.keys())

    subset_dict = dict.fromkeys(["100", "010", "110", "001", "101", "011", "111"], 0)

    for sys_type in sys_type_set:
        padloc_systems = padloc.get(sys_type, set())
        dfinder_systems = dfinder.get(sys_type, set())
        panorama_systems = panorama.get(sys_type, set())

        comp_dict = compare_sets(padloc_systems, dfinder_systems, panorama_systems)

        for k, v in comp_dict['counts'].items():
            subset_dict[k] += v

    return subset_dict


def plot_venn_diagram(padloc: Dict[str, Set[FrozenSet[str]]], dfinder: Dict[str, Set[FrozenSet[str]]],
                      panorama: Dict[str, Set[FrozenSet[str]]], output: Path) -> None:
    """Generate and display a Venn diagram comparing the systems detected by different tools.

    Args:
        padloc (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PADLOC.
        dfinder (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for DFinder.
        panorama (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PANORAMA.
        output (Path): Path to save the Venn diagram.
    """
    n_padloc = sum([len(x) for x in padloc.values()])
    n_df = sum([len(x) for x in dfinder.values()])
    n_panorama = sum([len(x) for x in panorama.values()])
    print(f"\nNumber of pangenome systems: PADLOC {n_padloc}, DFINDER {n_df}, PANORAMA {n_panorama}")
    subset_dict = get_common_specified_systems(padloc, dfinder, panorama)

    # Create a figure and axis for the Venn diagram
    _, _ = plt.subplots(figsize=(20, 11.25))

    # Generate the Venn diagram
    venn = venn3(subsets=subset_dict, set_labels=(f'PADLOC ({n_padloc})', f'DefenseFinder ({n_df})',
                                                  f'PANORAMA ({n_panorama})'))
    for label in venn.set_labels:
        if label:
            label.set_fontsize(20)

    for subset in venn.subset_labels:
        if subset:
            subset.set_fontsize(14)

    print(f'PADLOC: {subset_dict["100"]}, DFinder: {subset_dict["010"]}, PANORAMA: {subset_dict["001"]}, '
          f'PADLOC & DFinder: {subset_dict["110"]}, PANORAMA & PADLOC: {subset_dict["101"]}, '
          f'PANORAMA & DFinder: {subset_dict["011"]}, ALL: {subset_dict["111"]}')

    plt.savefig(output / "Venn.png", transparent=False)
    plt.savefig(output / "Venn.svg", transparent=False, format="svg")
    plt.show()


def _build_similarity_graph(set1: Set[FrozenSet[str]], set2: Set[FrozenSet[str]], threshold: float) -> nx.Graph:
    """Build a similarity graph where elements of set1 and set2 are connected if they are similar.

    Args:
        set1 (Set[FrozenSet[str]]): First set of frozensets to compare.
        set2 (Set[FrozenSet[str]]): Second set of frozensets to compare.
        threshold (float): Threshold for similarity.

    Returns:
        nx.Graph: A networkx graph with nodes connected based on similarity.
    """
    graph = nx.Graph()

    # Add nodes
    for item in set1:
        graph.add_node((item, "set1"))  # Mark each node by its origin set
    for item in set2:
        graph.add_node((item, "set2"))

    # Add edges for similar elements
    for item1 in set1:
        for item2 in set2:
            if similar(item1, item2, threshold):
                graph.add_edge((item1, "set1"), (item2, "set2"))

    return graph


def _compute_common_and_specific(set1: Set[FrozenSet[str]], set2: Set[FrozenSet[str]],
                                 threshold: float) -> Tuple[int, int, int, int]:
    """Analyze connected components of the graph to determine common and specific elements.

    Args:
        set1 (Set[FrozenSet[str]]): First set of frozensets to compare.
        set2 (Set[FrozenSet[str]]): Second set of frozensets to compare.
        threshold (float): Threshold for similarity.

    Returns:
        tuple: Counts of common groups and specific elements in set1 and set2.
    """
    graph = _build_similarity_graph(set1, set2, threshold)
    components = list(nx.connected_components(graph))

    common_set_1 = 0
    common_set_2 = 0
    specific_count_set1 = 0
    specific_count_set2 = 0

    for component in components:
        count_set1 = sum(node[1] == "set1" for node in component)
        count_set2 = sum(node[1] == "set2" for node in component)

        if count_set1 > 0:
            if count_set2 > 0:
                common_set_1 += count_set1
                common_set_2 += count_set2
            else:
                specific_count_set1 += count_set1  # Group only in set1
        else:
            specific_count_set2 += count_set2  # Group only in set2

    return common_set_1, common_set_2, specific_count_set1, specific_count_set2


def count_system_similar(padloc: Dict[str, Set[FrozenSet[str]]],
                         dfinder: Dict[str, Set[FrozenSet[str]]],
                         panorama_padloc: Dict[str, Set[FrozenSet[str]]],
                         panorama_dfinder: Dict[str, Set[FrozenSet[str]]],
                         output: Path, threshold: float = 0.5) -> pd.DataFrame:
    """Count similar systems among different tools and save the results to a CSV file.

    Args:
        padloc (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for PADLOC.
        dfinder (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets for DFinder.
        panorama_padloc (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets
            for PANORAMA compared with PADLOC.
        panorama_dfinder (Dict[str, Set[FrozenSet[str]]]): Dictionary of system types and their sets
            for PANORAMA compared with DFinder.
        output (Path): Path to save the results.
        threshold (float): Threshold for similarity.
    """
    sys_type_set = set(padloc) | set(dfinder) | set(panorama_padloc) | set(panorama_dfinder)
    nb_padloc = len(list(chain(*padloc.values())))
    nb_dfinder = len(list(chain(*dfinder.values())))
    nb_panorama_padloc = len(list(chain(*panorama_padloc.values())))
    nb_panorama_dfinder = len(list(chain(*panorama_dfinder.values())))
    comp_dict = {
        "PADLOC & DFinder": defaultdict(float),
        "PADLOC & PANORAMA": defaultdict(float),
        "DFinder & PANORAMA": defaultdict(float),
    }

    for sys_type in sys_type_set:
        padloc_systems = padloc.get(sys_type, set())
        dfinder_systems = dfinder.get(sys_type, set())
        panorama_padloc_systems = panorama_padloc.get(sys_type, set())
        panorama_dfinder_systems = panorama_dfinder.get(sys_type, set())

        # Comparison PADLOC <-> DFinder
        common_padloc, common_dfinder, specific_padloc, specific_dfinder = _compute_common_and_specific(padloc_systems,
                                                                                                        dfinder_systems,
                                                                                                        threshold)
        comp_dict["PADLOC & DFinder"]["Common M1"] += common_padloc
        comp_dict["PADLOC & DFinder"]["Common M2"] += common_dfinder
        comp_dict["PADLOC & DFinder"]["Specific M1"] += specific_padloc
        comp_dict["PADLOC & DFinder"]["Specific M2"] += specific_dfinder

        # Comparison PADLOC <-> PANORAMA
        common_padloc, common_panorama, specific_padloc, specific_panorama = _compute_common_and_specific(
            padloc_systems,
            panorama_padloc_systems,
            threshold)
        comp_dict["PADLOC & PANORAMA"]["Common M1"] += common_padloc
        comp_dict["PADLOC & PANORAMA"]["Common M2"] += common_panorama
        comp_dict["PADLOC & PANORAMA"]["Specific M1"] += specific_padloc
        comp_dict["PADLOC & PANORAMA"]["Specific M2"] += specific_panorama
        # Comparison DFinder <-> PANORAMA
        common_dfinder, common_panorama, specific_dfinder, specific_panorama = _compute_common_and_specific(
            dfinder_systems,
            panorama_dfinder_systems,
            threshold)
        comp_dict["DFinder & PANORAMA"]["Common M1"] += common_dfinder
        comp_dict["DFinder & PANORAMA"]["Common M2"] += common_panorama
        comp_dict["DFinder & PANORAMA"]["Specific M1"] += specific_dfinder
        comp_dict["DFinder & PANORAMA"]["Specific M2"] += specific_panorama

    comp_dict["PADLOC & DFinder"]["% common"] = round((comp_dict["PADLOC & DFinder"]["Common M1"] +
                                                       comp_dict["PADLOC & DFinder"]["Common M2"]) / (
                                                              nb_padloc + nb_dfinder) * 100, 2)
    comp_dict["PADLOC & PANORAMA"]["% common"] = round((comp_dict["PADLOC & PANORAMA"]["Common M1"] +
                                                        comp_dict["PADLOC & PANORAMA"]["Common M2"]) / (
                                                               nb_padloc + nb_panorama_padloc) * 100, 2)
    comp_dict["DFinder & PANORAMA"]["% common"] = round((comp_dict["DFinder & PANORAMA"]["Common M1"] +
                                                         comp_dict["DFinder & PANORAMA"]["Common M2"]) / (
                                                                nb_dfinder + nb_panorama_dfinder) * 100, 2)

    res_df = pd.DataFrame.from_dict(comp_dict, orient="index")
    res_df.index = [f"PADLOC ({nb_padloc}) & DFinder ({nb_dfinder})",
                    f"PADLOC ({nb_padloc}) & PANORAMA ({nb_panorama_padloc})",
                    f"DFinder ({nb_dfinder}) & PANORAMA ({nb_panorama_dfinder})"]
    res_df = res_df[["Common M1", "Common M2", "% common", "Specific M1", "Specific M2"]]
    res_df.to_csv(output / "prediction_comparison.tsv", index=True, sep="\t")
    return res_df
